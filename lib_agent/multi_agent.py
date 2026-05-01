# multi_agent.py — supervisor pattern over two specialized workers.
#
#   START → supervisor → (research_agent | code_agent | END)
#                          ↑__________________|
#
# Each worker is a compiled subgraph (its own model + tool set + tool node).
# The supervisor uses with_structured_output(Route) to pick the next worker
# or FINISH. HITL gates inside workers compose through subgraph boundaries:
# an interrupt() raised in code_agent's tool node propagates up to the parent
# graph's stream loop, and resume via Command flows back down.

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

from agent import get_current_time, make_serial_tool_node, prune_node
from config import MODEL, NUM_CTX
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
from tools.memory import forget, recall, remember
from tools.python_sandbox import run_python
from tools.web import web_fetch, web_search


# ── Worker subgraph factory ───────────────────────────────────────────────

class _WorkerState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _build_worker(tools: list[BaseTool], system_prompt: str):
    """Compile a tool-calling subgraph with its own LLM + tool subset.
    No checkpointer here — the parent supervisor's checkpointer covers the
    whole composition."""
    worker_llm = ChatOllama(model=MODEL, temperature=0, num_ctx=NUM_CTX).bind_tools(tools)

    def call_model(state: _WorkerState) -> dict:
        msgs = [SystemMessage(content=system_prompt), *state["messages"]]
        return {"messages": [worker_llm.invoke(msgs)]}

    g = StateGraph(_WorkerState)
    g.add_node("agent", call_model)
    g.add_node("tools", make_serial_tool_node(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


# ── Worker definitions ────────────────────────────────────────────────────

RESEARCH_TOOLS = [
    get_current_time,
    web_search,
    web_fetch,
    search_documents,
    recall,  # read-only access to persisted facts
]
RESEARCH_SYSTEM = (
    "You are the research_agent. Your tools: web_search, web_fetch, "
    "search_documents (local corpus), get_current_time.\n\n"
    "Look at the user's request and the conversation. Identify the part "
    "that needs lookup, search, or factual information — and answer THAT "
    "part using your tools. If the fact is already in the conversation, "
    "do not repeat it; just say you defer to the existing reply.\n\n"
    "Be concise (one short paragraph max). After your reply, control "
    "returns to a supervisor that may delegate to another worker."
)

CODE_TOOLS = [
    get_current_time,
    run_python,
    read_file,
    write_file,
    edit_file,
    copy_file,
    move_file,
    list_directory,
    make_directory,
    find_files,
    delete_file,
    remember,
    forget,
    recall,
]
CODE_SYSTEM = (
    "You are the code_agent. Your tools: run_python (Docker sandbox), "
    "read_file, write_file (workspace), get_current_time.\n\n"
    "Look at the user's request and the conversation. Identify the part "
    "that requires CODE EXECUTION, FILE READ, or FILE WRITE — and do it "
    "by CALLING your tools.\n\n"
    "CRITICAL RULES:\n"
    "- You MUST call a tool, not just describe what should happen.\n"
    "- If the user asked to write a file and you have the content from "
    "  the conversation, call write_file NOW with that content.\n"
    "- If the user asked to compute something, call run_python NOW.\n"
    "- Producing only a text reply about what 'would' happen is a failure.\n\n"
    "Be concise. After tool execution, give a one-sentence confirmation "
    "and control returns to a supervisor."
)

research_agent = _build_worker(RESEARCH_TOOLS, RESEARCH_SYSTEM)
code_agent = _build_worker(CODE_TOOLS, CODE_SYSTEM)


# ── Supervisor ────────────────────────────────────────────────────────────

WorkerChoice = Literal["research_agent", "code_agent", "FINISH"]
_ROUTE_KEYWORDS: tuple[str, ...] = ("research_agent", "code_agent", "FINISH")


SUPERVISOR_SYSTEM = """Pick the next worker in a multi-agent system AND
write a one-sentence directive for it.

WORKERS:
- research_agent: web search, web fetch, local document RAG. READ-ONLY.
- code_agent: Python sandbox, file read/write. REQUIRED for ANY file write,
  code execution, or workspace modification.

OUTPUT FORMAT — two lines exactly:
Line 1: keyword  (research_agent | code_agent | FINISH)
Line 2: one-sentence directive for that worker (omit if FINISH)

The directive on line 2 is critical: it tells the worker EXACTLY what to do.
For code_agent, the directive must say which tool to call and with what.

EXAMPLES:

User: "What time is it and save it to clock.txt"
First decision:
research_agent
Call get_current_time and report the current time.

After research_agent replies "Time is 16:30":
code_agent
Call write_file with filepath=clock.txt and content=16:30.

After code_agent confirms write:
FINISH

User: "Define photosynthesis from the corpus, then save to bio.md"
First decision:
research_agent
Call search_documents with query=photosynthesis and provide a one-sentence definition.

After research_agent provides the definition:
code_agent
Call write_file with filepath=bio.md and content set to the one-sentence definition just produced.

After code_agent writes:
FINISH

NOW DECIDE — output two lines (or just FINISH on one line). No extra text."""


# Use the no-thinking Qwen3 variant for supervisor decisions: thinking
# tokens get stripped from content by ChatOllama, sometimes leaving the
# message empty. Routing is short and doesn't need reasoning anyway.
SUPERVISOR_MODEL = "qwen3-nothink"
supervisor_llm = ChatOllama(model=SUPERVISOR_MODEL, temperature=0, num_ctx=NUM_CTX)


def _pick_keyword(text: str) -> str | None:
    """Return the first occurrence of a route keyword in text, or None."""
    text = text or ""
    # Prefer a clean single-token reply; fall back to substring match.
    stripped = text.strip().splitlines()[-1].strip() if text.strip() else ""
    if stripped in _ROUTE_KEYWORDS:
        return stripped
    for kw in _ROUTE_KEYWORDS:
        if kw in text:
            return kw
    return None


def _supervisor_context(messages: list) -> list:
    """Build a focused context for the supervisor: the system prompt, the
    most recent user request, and the most recent worker text reply (if any).

    Trimming the context dramatically improves routing reliability on a
    small local model — the full message history (with tool calls,
    intermediate AI messages, and tool results) is noise for a 3-way
    routing decision."""
    user_msg = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    last_reply = next(
        (
            m
            for m in reversed(messages)
            if isinstance(m, AIMessage) and m.content and not m.tool_calls
        ),
        None,
    )

    ctx: list = [SystemMessage(content=SUPERVISOR_SYSTEM)]
    if user_msg is not None:
        ctx.append(user_msg)
    if last_reply is not None:
        ctx.append(
            AIMessage(content=f"Most recent worker reply:\n{last_reply.content}")
        )
    return ctx


def _parse_supervisor(text: str) -> tuple[str | None, str]:
    """Pull (keyword, directive) out of the supervisor's two-line reply.
    Tolerates extra blank lines, leading think tags, or noise."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    keyword = None
    directive = ""
    for i, ln in enumerate(lines):
        kw = _pick_keyword(ln)
        if kw is not None:
            keyword = kw
            # Directive is the next non-empty line (if any)
            if i + 1 < len(lines):
                directive = lines[i + 1]
            break
    return keyword, directive


class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    supervisor_reason: str
    summary: str  # populated by prune_node when the conversation grows


def supervisor_node(state: SupervisorState) -> dict:
    response = supervisor_llm.invoke(_supervisor_context(state["messages"]))
    text = str(response.content or "")
    chosen, directive = _parse_supervisor(text)

    if chosen is None:
        has_ai = any(isinstance(m, AIMessage) for m in state["messages"])
        chosen = "FINISH" if has_ai else "research_agent"
        return {
            "next": chosen,
            "supervisor_reason": f"[fallback: no keyword in {text[:80]!r}]",
        }

    update: dict = {"next": chosen, "supervisor_reason": directive or text[:140].strip()}
    if chosen != "FINISH" and directive:
        # Inject the directive as a HumanMessage so the worker sees explicit
        # instructions (an 8B model often won't infer the next step from
        # conversation history alone).
        update["messages"] = [
            HumanMessage(content=f"[supervisor → {chosen}] {directive}")
        ]
    return update


def _make_worker_node(worker_app, name: str):
    """Wrap a compiled subgraph as a node. Returns only the NEW messages it
    produced so the parent state's add_messages reducer appends correctly."""

    def worker_node(state: SupervisorState) -> dict:
        prior_len = len(state["messages"])
        out = worker_app.invoke({"messages": state["messages"]})
        new_msgs = out["messages"][prior_len:]
        return {"messages": new_msgs}

    worker_node.__name__ = f"node_{name}"
    return worker_node


# ── Supervisor graph ──────────────────────────────────────────────────────


def _route(state: SupervisorState):
    nxt = state.get("next")
    return END if nxt in (None, "FINISH") else nxt


# Topology:
#   START → prune → supervisor → (research_agent | code_agent | END)
#                                  ↑___________________________|
# prune is shared with the single-agent graph (same prune_node, same
# AgentState-compatible shape — both have `messages` and `summary`).
graph = StateGraph(SupervisorState)
graph.add_node("prune", prune_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("research_agent", _make_worker_node(research_agent, "research_agent"))
graph.add_node("code_agent", _make_worker_node(code_agent, "code_agent"))
graph.add_edge(START, "prune")
graph.add_edge("prune", "supervisor")
graph.add_conditional_edges("supervisor", _route)
graph.add_edge("research_agent", "supervisor")
graph.add_edge("code_agent", "supervisor")
