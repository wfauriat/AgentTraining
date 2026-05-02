# smoke.py — minimum LangGraph agent: one tool, streamed output.
#
# Conceptually identical to agent/loop.py:
#   START → agent → (tool? → tools → agent) → END
# The framework gives us state management, tool dispatch, and streaming
# in exchange for ~25 lines of declarative graph wiring.

from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import MODEL, OPENAI_API_KEY, OPENAI_BASE_URL


@tool
def get_current_time() -> str:
    """Return the current local time as an ISO 8601 string."""
    return datetime.now().isoformat(timespec="seconds")


TOOLS = [get_current_time]

# bind_tools attaches the tool schemas to every model call.
# temperature=0 keeps tool-calling deterministic enough for a smoke test.
llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
).bind_tools(TOOLS)


def call_model(state: MessagesState) -> dict:
    """The 'agent' node: one LLM call against the running message history."""
    return {"messages": [llm.invoke(state["messages"])]}


# MessagesState is a built-in TypedDict: {"messages": list[BaseMessage]}
# with an append reducer. State updates returned from nodes are merged in.
graph = StateGraph(MessagesState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
# tools_condition: routes to "tools" if last AI message has tool_calls, else END.
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
app = graph.compile()


if __name__ == "__main__":
    msgs = [
        SystemMessage(
            content="You are a helpful assistant. Use tools when relevant. Be brief."
        ),
        HumanMessage(content="What time is it right now?"),
    ]
    # stream_mode="values" emits the entire state after each super-step.
    # Alternatives: "updates" (just the diff), "messages" (token-level).
    for step in app.stream({"messages": msgs}, stream_mode="values"):
        step["messages"][-1].pretty_print()
