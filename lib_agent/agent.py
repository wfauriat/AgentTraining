# agent.py — the growing LangGraph agent.
# smoke.py stays as the "hello world"; agent.py is what we extend each phase.

from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import tools_condition

from config import MODEL
from tools.docs import search_documents
from tools.files import read_file, write_file
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
    web_search,
    web_fetch,
    run_python,
    search_documents,
]
llm = ChatOllama(model=MODEL, temperature=0).bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def make_serial_tool_node(tools: list[BaseTool]):
    """Drop-in replacement for ToolNode that runs tool calls sequentially.

    ToolNode dispatches in a thread pool, which races when tools have ordering
    deps (e.g. write_file then read_file on the same path emitted in one AI
    message). This version runs them in the order the model emitted, which
    matches the implicit dependency the model intended.

    Tradeoff: read-only tools that *could* run in parallel no longer do.
    Acceptable until we introduce per-tool parallelism declarations.
    """
    by_name = {t.name: t for t in tools}

    def serial_tool_node(state: MessagesState) -> dict:
        last = state["messages"][-1]
        results: list[ToolMessage] = []
        for call in getattr(last, "tool_calls", []) or []:
            name = call["name"]
            args = call["args"] or {}
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
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", make_serial_tool_node(TOOLS))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")


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
        for step in app.stream({"messages": new_msgs}, config=config, stream_mode="updates"):
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
