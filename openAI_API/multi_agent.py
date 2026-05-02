# multi_agent.py — supervisor pattern over two specialized workers.
#
#   START → supervisor → (research_agent | code_agent | END)
#                          ↑__________________|
#
# Ported from lib_agent/ to the OpenAI-compatible endpoint. Topology and
# routing logic unchanged; only the LLM constructors moved from ChatOllama
# to ChatOpenAI. The keyword-router fallback is kept rather than swapping
# back to with_structured_output() — Mistral Small via vLLM/Ollama handles
# function-calling routes well, but the keyword path remains a robust
# safety net for thinking-token bleeds and other parser oddities.

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition

from agent import get_current_time, make_serial_tool_node, prune_node
from config import MODEL, MODEL_TIMEOUT, OPENAI_API_KEY, OPENAI_BASE_URL
from prompts import CODE_SYSTEM, RESEARCH_SYSTEM, SUPERVISOR_SYSTEM
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
    worker_llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        timeout=MODEL_TIMEOUT,
        max_retries=2,
    ).bind_tools(tools)

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
research_agent = _build_worker(RESEARCH_TOOLS, RESEARCH_SYSTEM)
code_agent = _build_worker(CODE_TOOLS, CODE_SYSTEM)


# ── Supervisor ────────────────────────────────────────────────────────────

WorkerChoice = Literal["research_agent", "code_agent", "FINISH"]
_ROUTE_KEYWORDS: tuple[str, ...] = ("research_agent", "code_agent", "FINISH")


# Supervisor uses the SAME model as workers. With a single OpenAI-compatible
# endpoint, there's no VRAM contention story to worry about (the server
# handles its own scheduling) — but using one model still simplifies the
# deployment surface and matches the lib_agent shape. Swap to a smaller
# routing-specialist model here if your endpoint serves multiple options.
SUPERVISOR_MODEL = MODEL
supervisor_llm = ChatOpenAI(
    model=SUPERVISOR_MODEL,
    temperature=0,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    timeout=MODEL_TIMEOUT,
    max_retries=2,
)


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
        # instructions (a small model often won't infer the next step from
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
