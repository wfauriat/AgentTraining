# agent.py — the growing LangGraph agent.
# smoke.py stays as the "hello world"; agent.py is what we extend each phase.

from datetime import datetime

from typing import Annotated, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langgraph.types import Command, interrupt

from config import (
    KEEP_ALIVE,
    MODEL,
    MODEL_TIMEOUT,
    NUM_CTX,
    PRUNE_THRESHOLD_TOKENS,
    SUMMARY_KEEP_TAIL,
    SUMMARY_MODEL,
)
from prompts import SUMMARIZER_SYSTEM, resolve_chat_system
from tools.docs import search_documents
from tools.files import (
    copy_file,
    delete_file,
    edit_file,
    find_files,
    list_directory,
    make_directory,
    move_file,
    read_file,
    write_file,
)
from tools.memory import forget, recall, remember, render_facts
from tools.python_sandbox import run_python
from tools.web import web_fetch, web_search


@tool
def get_current_time() -> str:
    """Return the current local time as an ISO 8601 string."""
    return datetime.now().isoformat(timespec="seconds")


TOOLS: list[BaseTool] = [
    get_current_time,
    read_file,
    write_file,
    edit_file,
    copy_file,
    move_file,
    list_directory,
    make_directory,
    find_files,
    delete_file,
    web_search,
    web_fetch,
    run_python,
    search_documents,
    remember,
    forget,
    recall,
]
llm = ChatOllama(
    model=MODEL,
    temperature=0,
    num_ctx=NUM_CTX,
    keep_alive=KEEP_ALIVE,
    client_kwargs={"timeout": MODEL_TIMEOUT},
).bind_tools(TOOLS)
summarizer_llm = ChatOllama(
    model=SUMMARY_MODEL,
    temperature=0,
    num_ctx=NUM_CTX,
    keep_alive=KEEP_ALIVE,
    client_kwargs={"timeout": MODEL_TIMEOUT},
)


# ── State with rolling summary ───────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


# ── Context-pruning helpers ──────────────────────────────────────────────

def _approx_tokens(messages: list) -> int:
    """1 token ≈ 4 chars heuristic. Fast and good enough for budget checks.
    Adds a small per-message overhead for the role/structure boilerplate."""
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(content) // 4 + 4
    return total


def _safe_split(messages: list) -> tuple[list, list]:
    """Return (to_summarize, to_keep).

    The split MUST be at a HumanMessage boundary — splitting between an
    AIMessage's tool_calls and its corresponding ToolMessages would leave
    the model with orphaned tool_call_ids and is fatal at the next call.

    Leading SystemMessages are always preserved at the front of to_keep.
    """
    head_system: list = []
    rest: list = []
    seen_non_system = False
    for m in messages:
        if not seen_non_system and isinstance(m, SystemMessage):
            head_system.append(m)
        else:
            seen_non_system = True
            rest.append(m)

    if len(rest) <= SUMMARY_KEEP_TAIL:
        return [], messages

    # Find the earliest HumanMessage in the last SUMMARY_KEEP_TAIL window.
    target_idx = max(0, len(rest) - SUMMARY_KEEP_TAIL)
    boundary = None
    for i in range(target_idx, len(rest)):
        if isinstance(rest[i], HumanMessage):
            boundary = i
            break

    if boundary is None or boundary == 0:
        return [], messages

    return rest[:boundary], head_system + rest[boundary:]


def prune_node(state: AgentState) -> dict:
    """Summarize older messages when the conversation exceeds the token budget.

    No-op when under threshold. Otherwise splits at a HumanMessage boundary,
    summarizes everything before it (incorporating any prior summary), and
    emits RemoveMessage entries so the add_messages reducer drops them.

    Prints a one-line trace to stdout when it actually summarizes — silent
    no-ops would otherwise leave the user wondering whether prune ever fires."""
    msgs = state["messages"]
    before_tokens = _approx_tokens(msgs)
    if before_tokens <= PRUNE_THRESHOLD_TOKENS:
        return {}

    to_summarize, kept = _safe_split(msgs)
    if not to_summarize:
        # Couldn't find a safe boundary; signal but don't summarize.
        # This is informational — common when the tail is all
        # tool_call/tool_result pairs without a HumanMessage break.
        print(
            f"\n  · prune skipped: {before_tokens} tokens but no safe "
            f"HumanMessage boundary in last {SUMMARY_KEEP_TAIL} messages"
        )
        return {}

    # Render the messages-to-summarize as a flat transcript inside one
    # HumanMessage. If we pass them as alternating Human/AI roles the model
    # often treats the last AI as a complete answer and emits nothing.
    def _label(m) -> str:
        cls = m.__class__.__name__.replace("Message", "")
        if hasattr(m, "tool_calls") and m.tool_calls:
            calls = "; ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            return f"{cls} [tool_calls: {calls}]"
        if hasattr(m, "name") and m.name:  # ToolMessage
            return f"{cls}({m.name})"
        return cls

    transcript_lines = [
        f"{_label(m)}: {str(m.content)[:600]}" for m in to_summarize
    ]
    transcript = "\n".join(transcript_lines)

    prior = state.get("summary", "")
    prior_section = f"Prior summary:\n{prior}\n\n" if prior else ""

    summary_input: list = [
        SystemMessage(content=SUMMARIZER_SYSTEM),
        HumanMessage(
            content=(
                f"{prior_section}Conversation transcript to summarize "
                f"(<= 200 tokens output, keep facts/file paths/decisions/tool "
                f"results, drop filler):\n\n{transcript}\n\nWrite the summary now."
            )
        ),
    ]

    response = summarizer_llm.invoke(summary_input)
    new_summary = str(response.content).strip()

    removals = [RemoveMessage(id=m.id) for m in to_summarize if getattr(m, "id", None)]
    after_tokens = _approx_tokens(kept) + (len(new_summary) // 4 + 4)
    print(
        f"\n  · context summarized: ~{before_tokens} → ~{after_tokens} tokens "
        f"(dropped {len(removals)} messages, {len(new_summary)} chars of summary)"
    )
    return {"messages": removals, "summary": new_summary}


def call_model(state: AgentState) -> dict:
    """Build the LLM input by prepending a fresh SystemMessage built from:
    (1) the active persona (resolved each call so /persona takes effect),
    (2) persistent facts injection, and (3) the rolling summary.

    Strips any pre-existing SystemMessage in state so the live persona is the
    sole authoritative one — this also tolerates threads from earlier code
    that seeded a SystemMessage into state."""
    base_msgs = [m for m in state["messages"] if not isinstance(m, SystemMessage)]

    parts: list[str] = [resolve_chat_system()]
    facts = render_facts()
    if facts:
        parts.append(f"[Persistent facts]\n{facts}")
    summary = state.get("summary", "")
    if summary:
        parts.append(f"[Earlier conversation summary]\n{summary}")

    msgs = [SystemMessage(content="\n\n".join(parts)), *base_msgs]
    return {"messages": [llm.invoke(msgs)]}


# Tools that mutate the world or run code. Each call gates on a human
# approval via interrupt() before the serial tool node executes it.
DESTRUCTIVE_TOOLS: set[str] = {
    "write_file",
    "run_python",
    "delete_file",
    "edit_file",
    "copy_file",
    "move_file",
}

_APPROVAL_YES = {"yes", "y", "true", "approve", "ok"}


def make_serial_tool_node(tools: list[BaseTool]):
    """Drop-in replacement for ToolNode that runs tool calls sequentially,
    with a HITL approval gate for destructive tools.

    Two phases — important because LangGraph re-runs a node body from the top
    on every resume(). To avoid duplicate side effects, ALL approvals are
    gathered first (phase 1, side-effect-free), then ALL tools execute
    (phase 2, runs exactly once).
    """
    by_name = {t.name: t for t in tools}

    def serial_tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_calls = list(getattr(last, "tool_calls", []) or [])

        # ── Phase 1: gather approvals via interrupt() ───────────────────
        # Each interrupt(payload) pauses the graph; on resume, returns the
        # value passed via Command(resume=...). Replays of this loop are
        # safe — interrupt() recalls prior resume values per call position.
        approvals: dict[str, str] = {}
        for call in tool_calls:
            if call["name"] in DESTRUCTIVE_TOOLS:
                decision = interrupt(
                    {
                        "type": "tool_approval",
                        "tool": call["name"],
                        "args": call["args"] or {},
                        "tool_call_id": call["id"],
                    }
                )
                approvals[call["id"]] = str(decision)

        # ── Phase 2: execute ────────────────────────────────────────────
        results: list[ToolMessage] = []
        for call in tool_calls:
            name = call["name"]
            args = call["args"] or {}

            if name in DESTRUCTIVE_TOOLS:
                decision = approvals.get(call["id"], "no").strip().lower()
                if decision not in _APPROVAL_YES:
                    results.append(
                        ToolMessage(
                            content=f"[denied by user: decision={decision!r}]",
                            name=name,
                            tool_call_id=call["id"],
                        )
                    )
                    continue

            tool_obj = by_name.get(name)
            if tool_obj is None:
                content = f"[error: unknown tool '{name}']"
            else:
                try:
                    content = tool_obj.invoke(args)
                except Exception as e:  # surface as tool result, let model recover
                    content = f"[error: {type(e).__name__}: {e}]"
            results.append(
                ToolMessage(content=str(content), name=name, tool_call_id=call["id"])
            )
        return {"messages": results}

    return serial_tool_node


# Graph is built but not compiled at module level — chat.py wants a different
# checkpointer. Callers compile with whichever saver fits their lifetime.
#
# Topology with the prune node:
#   START → prune → agent → (tools | END)
#                    ↑________|
#   tools → prune → agent ...
# prune is a no-op when under PRUNE_THRESHOLD_TOKENS, so the cost is one
# function call per super-step in the common case.
graph = StateGraph(AgentState)
graph.add_node("prune", prune_node)
graph.add_node("agent", call_model)
graph.add_node("tools", make_serial_tool_node(TOOLS))
graph.add_edge(START, "prune")
graph.add_edge("prune", "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "prune")


if __name__ == "__main__":
    # In-process MemorySaver is fine for this two-turn smoke test.
    app = graph.compile(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "demo"}}
    sys_msg = SystemMessage(
        content="You are a helpful assistant. Use tools when relevant. Be brief."
    )

    def run_turn(user_text: str, system: SystemMessage | None = None) -> None:
        new_msgs: list = []
        if system is not None:
            new_msgs.append(system)
        new_msgs.append(HumanMessage(content=user_text))
        # Cast to AgentState shape for the type checker — at runtime LangGraph
        # accepts any dict matching the TypedDict schema; only the static
        # checker needs the named type.
        state_in: AgentState = {"messages": new_msgs, "summary": ""}
        for step in app.stream(state_in, config=config, stream_mode="updates"):
            for node_name, update in step.items():
                print(f"\n── {node_name} ──")
                for msg in update.get("messages", []):
                    msg.pretty_print()

    print("=" * 60, "\nTurn 1\n", "=" * 60, sep="")
    run_turn(
        "Write a one-line haiku about debugging into a file named 'haiku.md'.",
        system=sys_msg,
    )
    print("\n" + "=" * 60, "\nTurn 2 — same thread_id, no filename hint given\n", "=" * 60, sep="")
    run_turn("Read the file you just created and tell me both its name and contents.")

    snapshot = app.get_state(config)
    print(f"\n[debug] thread 'demo' now holds {len(snapshot.values['messages'])} messages")
