# chat.py — interactive REPL on top of the agent graph.
# Persists conversation state to ./checkpoints.sqlite, keyed by thread_id.
# Resume any past conversation with `python chat.py --thread <id>`.

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any
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
from langgraph.types import Command

from config import TURN_TIMEOUT
from observability import flush as flush_traces, make_callbacks
from prompts import CHAT_SYSTEM, PERSONA_PATH, resolve_chat_system


class TurnTimeout(Exception):
    """Raised when a single REPL turn exceeds TURN_TIMEOUT seconds of active
    streaming (HITL approval waits don't count — the deadline resets after
    each user decision)."""

DB_PATH = Path(__file__).parent / "checkpoints.sqlite"
# Note: the active persona is resolved per agent turn inside agent.call_model
# (not seeded once into state). /persona changes apply on the next message.
# Worker node names in the multi-agent graph. When updates arrive from these
# nodes, the AI message content was produced by an inner LLM call we never
# saw at token level — print it here, since streaming missed it.
_WORKER_NODES = {"research_agent", "code_agent"}


# ── slash-command dispatcher ──────────────────────────────────────────────
# Slash commands run deterministically without an LLM round-trip — typing
# /facts or /clear gives instant results, unlike "please show my facts"
# which would pay a model call.

_SLASH_NOT_HANDLED = object()  # sentinel: input wasn't a slash command
_SLASH_EXIT = "__SLASH_EXIT__"  # sentinel: exit the REPL


def _slash_help() -> str:
    return "\n".join([
        "available slash commands:",
        "  /help                       show this menu",
        "  /facts                      show persisted facts",
        "  /forget <key>               delete a stored fact",
        "  /threads                    list checkpoint thread_ids",
        "  /persona                    show active system prompt",
        "  /persona reset              restore the default persona",
        "  /persona edit               multi-line edit (blank line ends)",
        "  /persona load <path>        read persona from a file",
        "  /persona <inline text>      one-line persona replacement",
        "  /clear                      wipe the current thread (start fresh)",
        "  /quit  /exit  /q            leave the REPL",
        "(more coming as the UX pass progresses)",
    ])


def _slash_clear(app, config: RunnableConfig) -> str:
    """Drop the current thread's checkpoints. The next user message will
    re-seed the system prompt because get_state will return empty messages."""
    saver = getattr(app, "checkpointer", None)
    thread_id = config["configurable"]["thread_id"]
    if saver is not None and hasattr(saver, "delete_thread"):
        saver.delete_thread(thread_id)
        return f"cleared thread {thread_id!r}; next message starts fresh"
    return "[error: checkpointer does not support delete_thread]"


def _slash_facts() -> str:
    """Show the persistent facts.json contents. Read-only, no LLM round-trip."""
    from tools.memory import load_facts

    facts = load_facts()
    if not facts:
        return "no facts stored. (the agent stores facts via the `remember` tool when you say so)"
    lines = [f"  {k}: {v}" for k, v in sorted(facts.items())]
    return f"persistent facts ({len(facts)}):\n" + "\n".join(lines)


def _slash_forget(args: str) -> str:
    """Delete a fact by key. Reuses the same `forget` tool the agent uses."""
    if not args:
        return "usage: /forget <key>    (try /facts to see stored keys)"
    from tools.memory import forget as forget_tool

    return forget_tool.invoke({"key": args})


def _slash_persona(args: str) -> str:
    """
    /persona              show active persona
    /persona reset        delete persona.txt — back to prompts.CHAT_SYSTEM
    /persona edit         multi-line entry; blank line ends, /cancel aborts
    /persona load <path>  read alternate file; persist its contents to persona.txt
    /persona <inline>     single-line replacement (anything else after /persona)
    """
    args = args.strip()

    # Show current
    if not args:
        active = resolve_chat_system()
        is_override = PERSONA_PATH.exists() and active != CHAT_SYSTEM
        source = f"override → {PERSONA_PATH.name}" if is_override else "default → prompts.CHAT_SYSTEM"
        # Indent body for readability since slash output is already indented.
        body = "\n".join(f"    {ln}" for ln in active.splitlines())
        return f"active persona [{source}]:\n{body}"

    # Reset to default
    if args.lower() == "reset":
        if PERSONA_PATH.exists():
            try:
                PERSONA_PATH.unlink()
                return "persona reset to default"
            except Exception as e:
                return f"[error: {e}]"
        return "already at default (no persona.txt)"

    # Multi-line interactive edit
    if args.lower() == "edit":
        print("  · enter persona text below. blank line ends, /cancel aborts.")
        lines: list[str] = []
        while True:
            try:
                line = input("    > ")
            except EOFError:
                return "[edit cancelled (EOF)]"
            if line.strip().lower() == "/cancel":
                return "[edit cancelled]"
            if not line.strip():
                break
            lines.append(line)
        if not lines:
            return "no text entered; persona unchanged"
        return _write_persona("\n".join(lines))

    # Load from alternate path
    if args.lower().startswith("load "):
        path_str = args[5:].strip()
        if not path_str:
            return "usage: /persona load <path>"
        path = Path(path_str).expanduser()
        if not path.exists():
            return f"[error: file not found — {path}]"
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception as e:
            return f"[error reading {path}: {e}]"
        if not content:
            return f"[error: file is empty — {path}]"
        return _write_persona(content) + f" (loaded from {path})"

    # Inline single-line replacement
    return _write_persona(args)


def _write_persona(text: str) -> str:
    try:
        PERSONA_PATH.write_text(text.strip() + "\n", encoding="utf-8")
        return f"persona updated ({len(text)} chars). takes effect on next message."
    except Exception as e:
        return f"[error writing persona: {e}]"


def _slash_threads(config: RunnableConfig) -> str:
    """List thread_ids stored in checkpoints.sqlite, marking the current one.
    Pulls from the SQL store directly — same data `make list-threads` shows."""
    if not DB_PATH.exists():
        return "(no threads — db not yet created)"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id ORDER BY thread_id"
        ).fetchall()
    except Exception as e:
        return f"[error reading threads: {e}]"
    if not rows:
        return "(no threads)"
    current = config["configurable"]["thread_id"]
    lines = []
    for tid, n in rows:
        marker = "  ← current" if tid == current else ""
        lines.append(f"  {tid:<32}  {n} ckpts{marker}")
    return f"threads ({len(rows)}):\n" + "\n".join(lines)


def _handle_slash(line: str, app, config: RunnableConfig):
    """Dispatch a slash command. Returns:
       _SLASH_NOT_HANDLED — caller should proceed with normal LLM flow
       _SLASH_EXIT        — caller should break the REPL loop
       str                — printable feedback to display, then continue"""
    s = line.strip()
    if not s.startswith("/"):
        return _SLASH_NOT_HANDLED
    head, _, args = s.partition(" ")
    head = head.lower()
    args = args.strip()
    if head in {"/quit", "/exit", "/q"}:
        return _SLASH_EXIT
    if head == "/help":
        return _slash_help()
    if head == "/clear":
        return _slash_clear(app, config)
    if head == "/facts":
        return _slash_facts()
    if head == "/forget":
        return _slash_forget(args)
    if head == "/threads":
        return _slash_threads(config)
    if head == "/persona":
        return _slash_persona(args)
    return f"unknown slash command: {head}. type /help"


def _is_thread_fresh(app, config: RunnableConfig) -> bool:
    """A thread is fresh when its checkpoint store has no messages.
    Used to decide whether to prepend SYSTEM on the next turn."""
    snap = app.get_state(config)
    return not snap.values.get("messages")


def _ask_approval(payload: Any) -> str:
    """Interactive approval prompt. Returns 'yes' / 'no'.
    EOF (e.g. heredoc that ran out) is treated as 'no' for safety.

    `payload` is whatever was passed to `interrupt(...)` inside the agent's
    tool node — for our destructive-tool gate it's a dict, but it could also
    be a langgraph Interrupt object depending on version, so we duck-type."""
    payload = getattr(payload, "value", payload)
    tool = payload.get("tool", "?") if isinstance(payload, dict) else "?"
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    print(f"\n  ⚠ approval requested: {tool}({args})")
    try:
        ans = input("  approve? [y/N] ").strip().lower()
    except EOFError:
        print("  (no input — denied)")
        return "no"
    return "yes" if ans in {"y", "yes"} else "no"


def render_stream(stream_iter, deadline: float | None = None) -> None:
    """Render a multi-mode stream:
       - "messages" mode: token-by-token AI text, printed inline with flush
       - "updates"  mode: discrete events (tool calls assembled, tool results)

    Why both modes? Token chunks give the typing-effect UX. Updates give us
    the assembled tool-call args once they're complete (otherwise we'd see
    `{'fil → {'filep → {'filepath...` as the args streamed in)."""
    streaming = False  # are we mid-way through printing an AI text reply?

    for stream_mode, payload in stream_iter:
        # Cooperative deadline check: cancellation precision = chunk cadence.
        # When the model is streaming, this fires within milliseconds of the
        # deadline. If the model is mid-call (no chunks), we won't notice
        # until the underlying ChatOllama timeout fires (MODEL_TIMEOUT).
        if deadline is not None and time.monotonic() > deadline:
            try:
                stream_iter.close()
            except Exception:
                pass
            if streaming:
                print()  # close the current AI text line cleanly
            raise TurnTimeout()
        if stream_mode == "messages":
            chunk, meta = payload
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                # Skip the supervisor's structured-output stream — that JSON
                # is internal routing, not user-facing content.
                node = (meta or {}).get("langgraph_node")
                if node == "supervisor":
                    continue
                if not streaming:
                    print("\nAssistant: ", end="", flush=True)
                    streaming = True
                print(chunk.content, end="", flush=True)

        elif stream_mode == "updates":
            for node_name, update in payload.items():
                # __interrupt__ (and other meta channels) carry tuples, not dicts.
                # We surface interrupts via the post-stream snapshot, so skip here.
                if not isinstance(update, dict):
                    continue
                # Multi-agent: surface supervisor routing as a one-liner.
                if node_name == "supervisor":
                    nxt = update.get("next")
                    reason = update.get("supervisor_reason") or ""
                    if nxt:
                        if streaming:
                            print()
                            streaming = False
                        print(f"  [supervisor → {nxt}] {reason}")
                    continue
                is_worker = node_name in _WORKER_NODES
                for msg in update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if streaming:
                            print()
                            streaming = False
                        for call in msg.tool_calls or []:
                            print(f"  [tool: {call['name']}({call['args']})]")
                        # Worker subgraphs are invoked, not streamed, so their
                        # text content lands here (not via messages mode).
                        if is_worker and msg.content and str(msg.content).strip():
                            print(f"\n  ({node_name}): {msg.content}\n")
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
    parser.add_argument(
        "--multi",
        action="store_true",
        help="use the multi-agent supervisor graph (research_agent + code_agent)",
    )
    args = parser.parse_args()
    thread_id = args.thread or f"chat-{uuid4().hex[:8]}"

    if args.multi:
        from multi_agent import graph
    else:
        from agent import graph

    # check_same_thread=False: SqliteSaver may touch the connection from
    # internal worker threads when tool calls run concurrently.
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()  # idempotent — creates checkpoint tables on first run
    app = graph.compile(checkpointer=saver)

    # callbacks: Phoenix instrumentation is implicit; this stays empty.
    # recursion_limit bumped for multi-agent loops (supervisor↔worker cycles).
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": make_callbacks(),
        "recursion_limit": 50 if args.multi else 25,
    }

    # Initial freshness probe — the slash dispatcher uses _is_thread_fresh()
    # on each turn so /clear can reset behavior without flag plumbing.
    snapshot = app.get_state(config)
    initial_fresh = not snapshot.values.get("messages")
    seeded_count = 0 if initial_fresh else len(snapshot.values["messages"])

    print(f"thread_id: {thread_id}  (db: {DB_PATH.name})")
    if initial_fresh:
        print("(new thread — system prompt will be seeded on first message)")
    else:
        print(f"(resumed thread — {seeded_count} prior messages)")
    print("Type /help for commands, /quit to exit. Resume later with: python chat.py --thread", thread_id, "\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            flush_traces()
            break
        if not user_text:
            continue
        # Backward compat: bare 'quit' / 'exit' still work without leading /.
        if user_text.lower() in {"quit", "exit"}:
            print("bye.")
            flush_traces()
            break

        # Slash command? Handle deterministically, no LLM round-trip.
        slash = _handle_slash(user_text, app, config)
        if slash is _SLASH_EXIT:
            print("bye.")
            flush_traces()
            break
        if slash is not _SLASH_NOT_HANDLED:
            if slash:
                # Indent + blank line so slash output reads as system-level,
                # not as a continuation of either user input or agent reply.
                print()
                for line in str(slash).splitlines():
                    print(f"  · {line}")
                print()
            continue  # next user prompt

        # System prompt is no longer seeded into state; agent.call_model
        # prepends a fresh one (active persona + facts + summary) per turn.
        new_msgs: list = [HumanMessage(content=user_text)]

        # Interrupt-aware turn with a cooperative wall-clock cap.
        # The deadline resets after each HITL approval — user input time is
        # exogenous and shouldn't count against the agent's budget.
        inputs: Any = {"messages": new_msgs}
        deadline = time.monotonic() + TURN_TIMEOUT
        try:
            while True:
                render_stream(
                    app.stream(inputs, config=config, stream_mode=["messages", "updates"]),
                    deadline=deadline,
                )
                snapshot = app.get_state(config)
                pending = [
                    intr for task in snapshot.tasks for intr in (task.interrupts or [])
                ]
                if not pending:
                    break
                payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
                inputs = Command(resume=_ask_approval(payload))
                deadline = time.monotonic() + TURN_TIMEOUT  # restart budget post-approval
        except TurnTimeout:
            print(f"\n  · turn timed out after {TURN_TIMEOUT}s — aborted")
            print("  · partial state may be in the thread; /clear to reset if needed")
        except KeyboardInterrupt:
            print("\n  · turn interrupted by user")
        except Exception as e:
            # Catch model timeouts, ollama errors, etc. so the REPL never dies.
            print(f"\n  · turn failed ({type(e).__name__}): {e}")


if __name__ == "__main__":
    main()
