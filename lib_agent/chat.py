# chat.py — interactive REPL on top of the agent graph.
# Persists conversation state to ./checkpoints.sqlite, keyed by thread_id.
# Resume any past conversation with `python chat.py --thread <id>`.

import argparse
import sqlite3
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

from agent import graph
from observability import flush as flush_traces, make_callbacks

DB_PATH = Path(__file__).parent / "checkpoints.sqlite"
SYSTEM = SystemMessage(
    content="You are a helpful assistant. Use tools when relevant. Be brief."
)


def render_stream(stream_iter) -> None:
    """Render a multi-mode stream:
       - "messages" mode: token-by-token AI text, printed inline with flush
       - "updates"  mode: discrete events (tool calls assembled, tool results)

    Why both modes? Token chunks give the typing-effect UX. Updates give us
    the assembled tool-call args once they're complete (otherwise we'd see
    `{'fil → {'filep → {'filepath...` as the args streamed in)."""
    streaming = False  # are we mid-way through printing an AI text reply?

    for stream_mode, payload in stream_iter:
        if stream_mode == "messages":
            chunk, _meta = payload
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                if not streaming:
                    print("\nAssistant: ", end="", flush=True)
                    streaming = True
                print(chunk.content, end="", flush=True)

        elif stream_mode == "updates":
            for _node_name, update in payload.items():
                for msg in update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        # Text already streamed via "messages" — close the line first.
                        if streaming:
                            print()
                            streaming = False
                        for call in msg.tool_calls or []:
                            print(f"  [tool: {call['name']}({call['args']})]")
                    elif isinstance(msg, ToolMessage):
                        text = str(msg.content)
                        preview = (text[:200] + "…") if len(text) > 200 else text
                        print(f"  [{msg.name} → {preview}]")

    if streaming:
        print()  # close the line if the turn ended on an AI text reply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thread",
        default=None,
        help="thread_id to resume; defaults to a new uuid-based id",
    )
    args = parser.parse_args()
    thread_id = args.thread or f"chat-{uuid4().hex[:8]}"

    # check_same_thread=False: SqliteSaver may touch the connection from
    # internal worker threads when tool calls run concurrently.
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # idempotent — creates checkpoint tables on first run
    app = graph.compile(checkpointer=saver)

    # callbacks: Langfuse handler if LANGFUSE_* env vars are set, else empty.
    # The handler captures the full graph trace (model calls, tool calls,
    # streaming, latencies) and ships it to localhost:3000.
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": make_callbacks(),
    }

    # If the thread already has history, skip seeding the system message.
    snapshot = app.get_state(config)
    fresh = not snapshot.values.get("messages")
    seeded_count = 0 if fresh else len(snapshot.values["messages"])

    print(f"thread_id: {thread_id}  (db: {DB_PATH.name})")
    if fresh:
        print("(new thread — system prompt will be seeded on first message)")
    else:
        print(f"(resumed thread — {seeded_count} prior messages)")
    print("Type 'quit' to exit. Resume later with: python chat.py --thread", thread_id, "\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            flush_traces()
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("bye.")
            flush_traces()
            break

        new_msgs: list = []
        if fresh:
            new_msgs.append(SYSTEM)
            fresh = False
        new_msgs.append(HumanMessage(content=user_text))

        render_stream(
            app.stream(
                {"messages": new_msgs},
                config=config,
                stream_mode=["messages", "updates"],
            )
        )


if __name__ == "__main__":
    main()
