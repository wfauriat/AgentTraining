# lib_agent — companion guide

A walkthrough of the abstractions, the entry points, and the interfaces
between them. Where `specs.md` is reference, this is "if you want to
understand *how* it works, start here."

**Companion to:** `specs.md` (canonical reference).
**Status:** 2026-05-01.

---

## 1. The 30-second mental model

```
                   chat.py (REPL)
                        │
                        │  builds inputs, drives stream
                        ▼
                   app.stream(...)         ┐
                        │                  │
                        ▼                  │  graph state machine
        ┌──────── compiled graph ────────┐ │  (LangGraph)
        │     prune ─► agent ─► tools    │ │
        │                                │ │
        │  (multi: prune ─► supervisor ─►│ │
        │   research_agent | code_agent) │ │
        └────────────────────────────────┘ │
                        │                  ┘
                        │  yields chunks + state updates
                        ▼
                   render_stream
                        │
                        ▼
                   Rich console
                        │
        on tool calls: HITL approval prompt
        on completion: token footer + new status bar
```

Every concrete piece below ties back to one of these boxes.

---

## 2. The seven core abstractions

### 2.1 State

A `TypedDict` carried through every graph node. Two flavors:

```python
# agent.py — single-agent
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str

# multi_agent.py — supervisor + workers
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    supervisor_reason: str
    summary: str
```

Both share `messages` (with the `add_messages` reducer that appends and
de-dupes by id) and `summary` (overwrite-by-default). The supervisor adds
`next` (which worker to route to) and `supervisor_reason` (one-line
diagnostic for the routing decision).

`add_messages` is the reducer that makes "send only the new HumanMessage,
don't re-send the whole history" work. It also processes `RemoveMessage(id)`
entries by deleting messages with matching ids — that's how `prune_node`
shrinks the conversation.

### 2.2 Node

A function `state -> partial state`. Examples:

```python
# agent.py
def call_model(state: AgentState) -> dict:
    base = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    parts = [resolve_chat_system()]                   # active persona
    if facts := render_facts():                       # persistent KV
        parts.append(f"[Persistent facts]\n{facts}")
    if summary := state.get("summary", ""):           # rolling summary
        parts.append(f"[Earlier conversation summary]\n{summary}")
    msgs = [SystemMessage(content="\n\n".join(parts)), *base]
    return {"messages": [llm.invoke(msgs)]}

def prune_node(state: AgentState) -> dict:
    if _approx_tokens(state["messages"]) <= PRUNE_THRESHOLD_TOKENS:
        return {}                                     # no-op; common path
    to_summarize, kept = _safe_split(state["messages"])
    if not to_summarize:                              # boundary couldn't split
        return {}
    new_summary = summarizer_llm.invoke(...).content
    removals = [RemoveMessage(id=m.id) for m in to_summarize]
    return {"messages": removals, "summary": new_summary}
```

Nodes return state *updates*, not whole states. The graph engine merges via
the field reducers.

### 2.3 Edge

How nodes connect. Three flavors used in this code:

| Edge | Use |
|---|---|
| `add_edge(START, "prune")` | Static — always go from A to B |
| `add_edge("tools", "prune")` | Static back-edge to form the loop |
| `add_conditional_edges("agent", tools_condition)` | Dynamic — `tools_condition` looks at the last AI message; routes to "tools" if it has tool_calls, else END |
| `add_conditional_edges("supervisor", _route)` | Dynamic — `_route(state)` returns `state["next"]` (`research_agent` / `code_agent` / END) |

### 2.4 Tool

Anything decorated with `@tool` from `langchain_core.tools`:

```python
@tool
def write_file(filepath: str, content: str) -> str:
    """Write text into a file in the local workspace ..."""
    safe = _safe_path(filepath)
    if safe is None:
        return "[error: path outside workspace or invalid]"
    safe.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {safe.relative_to(...)}"
```

The decorator does three things:
1. Builds a Pydantic input schema from the type hints (`filepath: str` etc.).
2. Builds a JSON-schema-compatible description from the docstring.
3. Wraps the function as a Runnable that LangChain knows how to expose to
   the model via `bind_tools([...])`.

Tools return strings. Errors are *also strings* (e.g.
`"[error: path outside workspace or invalid]"`) — the model reads them
and decides whether to retry or pivot. Raising would still work (caught
upstream) but error-as-string is the canonical pattern in this codebase.

### 2.5 Tool node — `make_serial_tool_node`

The dispatcher between the agent's tool-call requests and actual tool
executions. Responsibilities:

1. Read the last AIMessage from state.
2. For each requested `tool_call`, look up the tool by name.
3. **HITL gate phase**: if any tool name is in `DESTRUCTIVE_TOOLS`,
   emit `interrupt(...)` to pause the graph and request user approval.
   This phase MUST be side-effect-free because LangGraph re-runs the node
   body on every `Command(resume=...)` resume.
4. **Execution phase**: after all approvals are gathered (the node has
   re-run as many times as there were destructive calls), execute each
   tool with its args, build a `ToolMessage(content=str(result),
   name=name, tool_call_id=call["id"])`.
5. Return `{"messages": [...tool_messages...]}` for the reducer to append.

This replaces LangGraph's prebuilt `ToolNode`. We needed to replace it
because: (a) prebuilt runs tools concurrently — racy when one tool's output
is the next tool's input; (b) prebuilt has no HITL hook in the same place.

### 2.6 Worker subgraph

A *compiled* graph used as a node in the parent supervisor graph. Built
by `_build_worker(tools, system_prompt)`:

```python
g = StateGraph(_WorkerState)               # WorkerState: {messages: ...}
g.add_node("agent", call_model)            # local LLM bound to `tools`
g.add_node("tools", make_serial_tool_node(tools))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")
return g.compile()                         # ready to invoke as a callable
```

A compiled graph has the standard Runnable interface — `.invoke()`,
`.stream()`, `.astream()`. The supervisor wraps each into a node:

```python
def worker_node(state):
    out = worker_app.invoke({"messages": state["messages"]})
    new_msgs = out["messages"][len(state["messages"]):]   # diff vs input
    return {"messages": new_msgs}
```

The `len(state["messages"])` slice returns only what the worker added,
which the parent's `add_messages` reducer appends to its state. This is
what makes interrupts compose through the boundary: an `interrupt()` raised
in the worker's tool node propagates up through `worker_app.invoke()`,
through `worker_node`, through `app.stream(...)`, all the way to chat.py.

### 2.7 Supervisor

A node, not a graph. One LLM call to decide the next worker (or FINISH):

```python
def supervisor_node(state) -> dict:
    response = supervisor_llm.invoke(_supervisor_context(state["messages"]))
    text = str(response.content or "")
    chosen, directive = _parse_supervisor(text)
    ...
    update = {"next": chosen, "supervisor_reason": directive or text[:140]}
    if chosen != "FINISH" and directive:
        # Inject directive as HumanMessage so worker has explicit instructions
        update["messages"] = [HumanMessage(content=f"[supervisor → {chosen}] {directive}")]
    return update
```

Three architectural choices worth highlighting:

1. **Focused context** (`_supervisor_context`) — the supervisor sees only
   `[system, last_user_msg, last_worker_reply]`, NOT the full history.
   Empirically essential on 8B: full history → 50%+ wrong routing decisions.
2. **Keyword routing** — `_pick_keyword(text)` finds `research_agent` /
   `code_agent` / `FINISH` in the response text. Tried `with_structured_output`
   first; was unreliable on 8B (frequent `None` returns). Plain content +
   keyword search is robust.
3. **Directive injection** — supervisor adds a HumanMessage to state with
   the next instruction. Without it, an 8B worker often re-derives intent
   incorrectly. With it, the worker has its task spelled out.

---

## 3. Entry points

### 3.1 `chat.py` — interactive REPL (primary)

Argument: `--thread <id>` to resume; `--multi` to use the supervisor graph.

Lifecycle:
1. Parse args, derive `thread_id` (uuid prefix if not given).
2. Import `agent.graph` or `multi_agent.graph` based on `--multi`.
3. `SqliteSaver(sqlite3.connect(DB_PATH))` + `.setup()` (idempotent).
4. `app = graph.compile(checkpointer=saver)`.
5. Build `RunnableConfig`: `{"configurable": {"thread_id": ...},
   "callbacks": make_callbacks(), "recursion_limit": 50 if --multi else 25}`.
6. While loop:
   - Print status bar (Rich rule, color-coded by ctx %).
   - `input("You: ")`.
   - If empty, `quit/exit`, or starts with `/`, handle and continue.
   - Otherwise: turn = `[HumanMessage(content=user_text)]`, snapshot
     pre-turn AI message ids for the footer's id-diff token sum.
   - Inner loop: `render_stream(app.stream(...))`; on interrupt, ask user,
     `Command(resume=...)`, loop. Caught: `TurnTimeout`, `KeyboardInterrupt`,
     any other Exception → friendly message, REPL stays alive.
   - Print footer, repeat.

### 3.2 `agent.py` — single-agent graph

Importable: `from agent import graph` (uncompiled). Module-level demo runs
when invoked as `__main__` with `MemorySaver` + a 2-turn smoke test.

Exposes for re-use: `graph`, `make_serial_tool_node`, `prune_node`,
`get_current_time`, `_approx_tokens`. `multi_agent.py` imports several
of these.

### 3.3 `multi_agent.py` — supervisor pattern

Importable: `from multi_agent import graph`. Same shape as agent.py's graph
attribute. Re-uses `prune_node` (same code, different state schema — both
have `messages` + `summary`).

### 3.4 `scripts/index_docs.py` — RAG indexer

Standalone script. Run via `python -m scripts.index_docs` or `make index`.

```
corpus/biology.md
   │
   ▼
MarkdownHeaderTextSplitter (split on ## headings)
   │
   ▼
RecursiveCharacterTextSplitter (sub-split if > TARGET_CHUNK_CHARS)
   │
   ▼
NomicEmbeddings.embed_documents (with "search_document: " prefix)
   │
   ▼
LanceDB write to ./vector_db/<TABLE_NAME>  (cosine similarity)
```

The RAG tool (`tools/docs.py:search_documents`) embeds queries with
"search_query: " prefix and runs a top-K=5 cosine search.

### 3.5 `eval/runner.py` — eval harness

Run via `python -m eval.runner [--filter ID] [--skip-categories CSV]` or
`make eval`. Loads `eval/golden.py:GOLDEN`, builds a fresh
`MemorySaver`-backed app, runs each case via `app.invoke(...)` (auto-approves
HITL via `Command(resume="yes")` in a wrapper loop), scores against six
gates, writes `eval/reports/run_YYYYMMDD_HHMMSS.json`.

### 3.6 `admin.py` — state CLI

Subcommands: `info`, `list-threads`, `list-facts`, `purge-thread <id>`,
`purge-all --yes`, `purge-facts --yes`, `purge-traces --yes`. Run via
`python -m admin <cmd>` or wrapped Make targets. Writes are gated on
`--yes` to prevent accidents.

### 3.7 `smoke.py` — frozen 1-tool baseline

The original 25-line agent we shipped on day 1. Kept as a reference for
the simplest possible LangGraph use. `smoke.py` is never imported by other
modules.

---

## 4. Interfaces between abstractions

### 4.1 chat.py ↔ graph

The graph is treated as a black-box `Runnable`. chat.py never reaches
inside — it sees:

```python
app: CompiledStateGraph
app.stream(input, config, stream_mode=["messages","updates"]) -> Iterator
app.get_state(config) -> StateSnapshot   # has .values, .next, .tasks
app.checkpointer.delete_thread(thread_id)
app.invoke(input, config) -> final state
```

The `input` is a dict matching the graph's state schema or a `Command(resume=...)`.

### 4.2 graph node ↔ tool

The agent node calls `llm.invoke(messages)` where `llm = ChatOllama(...).bind_tools(TOOLS)`.

The model returns an `AIMessage` whose `.tool_calls` is a list of
`{"name": str, "args": dict, "id": str, "type": "tool_call"}`. Our serial
tool node iterates these, looks up the `@tool`-decorated callable by name,
calls `tool_obj.invoke(args)` (which validates args via the auto-derived
Pydantic schema, then runs the function, returning a string).

The string result becomes a `ToolMessage(content=str, name=name,
tool_call_id=call["id"])`. The `tool_call_id` is mandatory — the model's
next call needs it to match results back to the original requests.

### 4.3 prompts ↔ chat.py / agent.py / multi_agent.py

`prompts.py` is the single source of truth for system prompts and the
persona-resolution function:

| Prompt | Used by |
|---|---|
| `CHAT_SYSTEM` | `agent.call_model` via `resolve_chat_system()` |
| `RESEARCH_SYSTEM` | `multi_agent._build_worker(RESEARCH_TOOLS, RESEARCH_SYSTEM)` |
| `CODE_SYSTEM` | `multi_agent._build_worker(CODE_TOOLS, CODE_SYSTEM)` |
| `SUPERVISOR_SYSTEM` | `multi_agent.supervisor_node._supervisor_context` |
| `SUMMARIZER_SYSTEM` | `agent.prune_node` |

`resolve_chat_system()` reads `persona.txt` if present, else `CHAT_SYSTEM`.
Called per-turn in `agent.call_model`, so `/persona` changes apply on the
next message without restart.

### 4.4 facts.json ↔ tools.memory ↔ agent.call_model

`facts.json` is owned by `tools/memory.py`. Three @tools (`remember`,
`forget`, `recall`) read/write it. A `render_facts()` helper formats the
contents as a bullet list for system-prompt injection.

`agent.call_model` calls `render_facts()` every turn and prepends the
result as part of the SystemMessage. The model sees the facts as ambient
context — it doesn't need to call `recall()` to access them. `recall()`
exists as a fallback for programmatic queries (e.g., from inside a worker).

### 4.5 chat.py ↔ HITL (interrupt + resume)

The full handshake:

```
1. Worker LLM emits AIMessage with tool_calls including a destructive one.
2. Serial tool node enters phase 1 (gather approvals).
3. For each destructive call, calls interrupt({tool, args, tool_call_id}).
4. interrupt() raises GraphInterrupt internally; LangGraph catches it,
   commits a checkpoint with state.next = ("tools",), state.tasks contains
   the pending interrupts, returns from app.stream().
5. chat.py's render loop completes the iteration normally (no special
   handling — it just yields events until exhausted).
6. After the stream ends, chat.py calls app.get_state(config). It checks
   snapshot.tasks[*].interrupts — if non-empty, surfaces to user.
7. _ask_approval prints the warning + reads "y/N", returns "yes"/"no".
8. chat.py calls app.stream(Command(resume="yes"|"no"), config).
9. LangGraph re-enters the tool node body from the top; the interrupt()
   call at the same position now returns the resume value.
10. Phase 1 continues to next destructive call (back to step 3) or
    completes. Phase 2 then executes all tools.
11. Stream resumes normally.
```

The two-phase pattern (gather then execute) is what makes this safe under
node replay. Without it, the first tool would run twice on the second
resume.

### 4.6 chat.py ↔ render_stream

`render_stream(stream_iter, deadline=None)` consumes the multi-mode stream:

| Stream mode | Payload | Render |
|---|---|---|
| `messages` | `(AIMessageChunk, metadata)` | If `metadata["langgraph_node"]` not in `{supervisor, prune}`, append `chunk.content` to assistant text inline. Cyan `Assistant` header on first chunk. |
| `updates` | `{node_name: state_update}` | Skip non-dict (interrupts). For `supervisor` node: in debug, print magenta routing line. For other nodes: iterate `update["messages"]`; for AIMessage with tool_calls (debug only): yellow tool line; for worker subgraph AIMessage with content (always): cyan worker header + content; for ToolMessage (debug only): green result preview. |

Cooperative deadline check at every iteration: if `time.monotonic() > deadline`,
close the stream iterator and raise `TurnTimeout`.

### 4.7 chat.py ↔ Phoenix (observability)

Indirect via OpenInference auto-instrumentation. `observability.setup()`:

1. Imports `phoenix.otel.register` and `LangChainInstrumentor`.
2. `register(project_name="lib_agent", endpoint=PHOENIX_ENDPOINT)` builds an
   OTel TracerProvider, attaches an HTTP exporter pointed at Phoenix.
3. `LangChainInstrumentor().instrument(tracer_provider=tracer_provider)`
   patches LangChain's Runnable internals so every node, LLM call, and
   tool invocation emits a span.
4. Monkey-patches `OpenInferenceTracer` to add no-op `on_interrupt` and
   `on_resume` (LangChain 1.x added these callback hooks; OpenInference
   hadn't caught up yet — without the patch, `langchain` logs errors on
   every HITL gate).

Tracing is silent at the agent layer — no callback wiring in
`RunnableConfig.callbacks`. The OTel Python SDK uses contextvars to
correlate spans, which makes it transparent to the agent code.

---

## 5. Representative flows

### 5.1 Single-agent turn — "What time is it?"

```
chat.py: user types "What time is it?"
  ├─ status bar prints (green rule, ctx 0%)
  ├─ build new_msgs = [HumanMessage("What time is it?")]
  ├─ deadline = time.monotonic() + 180s
  └─ app.stream({"messages": new_msgs}, config, stream_mode=["messages","updates"])
       │
       ├─ prune node fires
       │    └─ _approx_tokens(state) < 8000 → no-op return {}
       │
       ├─ agent node fires
       │    └─ call_model:
       │         resolve_chat_system() → CHAT_SYSTEM string
       │         render_facts() → "" (empty store)
       │         msgs = [SystemMessage(...persona...), HumanMessage("What time is it?")]
       │         llm.invoke(msgs) → AIMessage with tool_calls=[{name: get_current_time, args: {}}]
       │         return {"messages": [AIMessage]}
       │    yields ("messages", chunk) for any text content (none here)
       │    yields ("updates", {"agent": {"messages": [AIMessage]}})
       │       → render_stream: in debug, prints yellow [tool: get_current_time({})]
       │
       ├─ tools_condition fires: AIMessage has tool_calls → route to "tools"
       │
       ├─ tools node fires (make_serial_tool_node)
       │    └─ phase 1 (approval gather): get_current_time NOT in DESTRUCTIVE_TOOLS, skip
       │    └─ phase 2 (execute): get_current_time.invoke({}) → "2026-05-01T15:30:00"
       │    return {"messages": [ToolMessage("2026-05-01T15:30:00", name="get_current_time", tool_call_id="...")]}
       │    yields ("updates", {"tools": {"messages": [ToolMessage]}})
       │       → render_stream: in debug, prints green [get_current_time → 2026-05-01...]
       │
       ├─ tools → prune (graph back-edge)
       ├─ prune fires → no-op
       ├─ agent fires again
       │    └─ call_model: msgs now includes [system, user, ai_with_tool_calls, tool_message]
       │       llm.invoke(msgs) → AIMessage(content="The current time is 2026-05-01 ...")
       │    yields ("messages", chunk("The")), ("messages", chunk(" current")), ... (token streaming)
       │       → render_stream: prints cyan "Assistant" header + streams text inline
       │    yields ("updates", {"agent": {"messages": [final AIMessage]}})
       │
       └─ tools_condition: no tool_calls in last message → END
chat.py:
  ├─ snapshot = app.get_state(config) — no pending interrupts
  ├─ exit inner while loop
  └─ print dim footer "↳ in 1240 tok · out 87 tok · 4.2s"
  loop: print new status bar, prompt for next input
```

### 5.2 HITL approval — "Write 'hi' to test.txt"

Same as above through agent's first call_model, but the AIMessage has
`tool_calls=[{"name": "write_file", "args": {"filepath": "test.txt", "content": "hi"}}]`.

```
... agent emits AIMessage with write_file tool_call ...
tools node fires:
  phase 1 (approval gather):
    write_file IS in DESTRUCTIVE_TOOLS
    interrupt({"type": "tool_approval", "tool": "write_file", "args": {...}, "tool_call_id": "..."})
    → GraphInterrupt raised internally
    → checkpoint committed with state.tasks[0].interrupts = [Interrupt(value={"tool":"write_file",...})]
    → app.stream() returns

chat.py inner loop:
  render_stream(...) just exhausted normally (no chunks for the interrupt)
  snapshot = app.get_state(config)
  pending = [Interrupt(...)] from snapshot.tasks
  payload = pending[0].value
  decision = _ask_approval(payload)   # bold red prompt, reads y/N
  inputs = Command(resume=decision)
  deadline = time.monotonic() + 180s   # reset; user input doesn't count
  loop: app.stream(inputs, config, stream_mode=...)
    LangGraph re-enters the tools node body from the top
    First interrupt() at same position now returns "yes" or "no"
    No more destructive calls in this batch → phase 1 done
    Phase 2 executes:
      if approval == "yes" → write_file.invoke(...) → "Wrote 2 chars to test.txt"
      else → ToolMessage("[denied by user]", ...)
    yields ("updates", {"tools": {"messages": [ToolMessage]}})
    ... continues to agent for the final reply ...
```

### 5.3 Context summarization — when state crosses 8K tokens

```
Some agent turn N triggers prune to fire:
  prune_node:
    msgs = state["messages"]  (~8200 tokens)
    _approx_tokens(msgs) > 8000 → proceed
    to_summarize, kept = _safe_split(msgs)
      walks backward from last 6 messages, finds the earliest HumanMessage
      to_summarize = msgs[:boundary_idx]
      kept = msgs[boundary_idx:]
    if to_summarize is empty: print "prune skipped: no safe boundary", return {}
    transcript = "\n".join(f"{label(m)}: {m.content[:600]}" for m in to_summarize)
    summary_input = [
      SystemMessage(SUMMARIZER_SYSTEM),
      HumanMessage(f"... transcript: {transcript} ... write the summary now."),
    ]
    new_summary = summarizer_llm.invoke(summary_input).content.strip()
    print "  · context summarized: ~A → ~B tokens (dropped K msgs, L chars)"
    return {
      "messages": [RemoveMessage(id=m.id) for m in to_summarize],
      "summary": new_summary,
    }

Reducer applies:
  - RemoveMessage entries cause add_messages to drop the matching messages
  - "summary" overwrites the state field

agent.call_model on the same super-step (next iteration):
  base = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    (state["messages"] is now the trimmed `kept` list)
  parts = [resolve_chat_system(), facts_str (if any), f"[Earlier summary]\n{new_summary}"]
  msgs = [SystemMessage("\n\n".join(parts)), *base]
  llm.invoke(msgs) — model sees the summary + recent kept messages
```

The status bar's `summary` field flips from `—` to `✓` after the first
prune fires for a thread.

### 5.4 Multi-agent — "Search for X then save to file.txt"

```
chat.py imports multi_agent.graph instead of agent.graph
app.stream({"messages": [HumanMessage(...)]}, config, ...)
  prune (no-op)
  supervisor_node:
    _supervisor_context: trim to [SUPERVISOR_SYSTEM, last_user, last_reply (none)]
    supervisor_llm.invoke(...) → AIMessage with content like:
      "research_agent\nCall web_search with query=X and report results"
    chosen, directive = _parse_supervisor(text) → ("research_agent", "Call web_search...")
    update = {
      "next": "research_agent",
      "supervisor_reason": "Call web_search...",
      "messages": [HumanMessage("[supervisor → research_agent] Call web_search...")],
    }
    render: in debug, magenta routing line
  conditional edge _route(state) → "research_agent"
  research_agent node fires (the worker subgraph as a callable):
    invoke the compiled worker graph with state["messages"] as input
    INSIDE the worker:
      worker.agent: call worker_llm bound to RESEARCH_TOOLS
        emits AIMessage with tool_calls=[web_search]
      worker.tools (serial): web_search not destructive, no HITL, executes
        ToolMessage with results
      worker.agent: another call → AIMessage with content (final research reply)
      tools_condition: no tool_calls → END
    worker.invoke returns {"messages": [...all 4 worker-internal messages...]}
    worker_node returns {"messages": [...new messages only (slice)...]}
    render: cyan "research_agent" header + content
  back to supervisor (prune first, no-op)
  supervisor_node:
    new context: [SUPERVISOR_SYSTEM, last_user, last_reply (research_agent's reply)]
    decides: "code_agent\nCall write_file with filepath=file.txt and content=..."
    routes to code_agent
  code_agent worker fires:
    INSIDE the worker:
      worker.agent: emits AIMessage with tool_calls=[write_file (destructive)]
      worker.tools (serial):
        phase 1: interrupt() for write_file approval
        ← This interrupt PROPAGATES UP through all the layers:
          worker_app.invoke() raises (or returns paused state)
          worker_node's invoke gets the paused state
          parent app.stream() yields control
        chat.py: detects pending interrupt, asks user, sends Command(resume="yes")
        LangGraph re-enters AT THE WORKER NODE (not just the tool node!)
        Worker re-runs: agent → tool node → interrupt() returns "yes"
        Phase 2: write_file executes, returns ToolMessage
      worker.agent: final reply
    worker_node returns new messages
    render: cyan "code_agent" header + content
  supervisor decides: FINISH
  conditional edge _route → END
chat.py: footer + new status bar
```

---

## 6. Where to look for what

If you want to understand…

| | Read |
|---|---|
| The agent loop | `agent.py:graph` (lines ~239-256) |
| HITL safety semantics | `agent.py:make_serial_tool_node` two-phase block |
| How summarization works | `agent.py:prune_node` + `_safe_split` |
| How a tool gets defined | `tools/files.py:write_file` (canonical example) |
| How sandbox path-traversal blocks | `tools/files.py:_safe_path` |
| How `run_python` is sandboxed | `tools/python_sandbox.py` (Docker args block) |
| Multi-agent routing logic | `multi_agent.py:supervisor_node` + `_pick_keyword` + `_parse_supervisor` |
| How worker subgraphs compose | `multi_agent.py:_build_worker` + `_make_worker_node` |
| The persona override | `prompts.py:resolve_chat_system` + `chat.py:_slash_persona` |
| Persistent facts | `tools/memory.py` (storage) + `agent.call_model` (injection) |
| How streaming + Rich coexist | `chat.py:render_stream` |
| `/debug` toggle wiring | `chat.py:_DEBUG`, `_slash_debug`, conditionals in `render_stream` |
| Status bar coloring | `chat.py:_status_bar` + the `console.rule(..., style=color)` call site in `main()` |
| Per-turn token id-diff | `chat.py:_turn_footer` + the `prior_ids` snapshot in `main()` |
| Eval scoring | `eval/runner.py:score_case` |
| Phoenix instrumentation | `observability.py:setup` |
| Admin operations | `admin.py` (each `cmd_*` function) |

---

## 7. Reading order for a new contributor

If you're picking this up cold:

1. **`smoke.py`** (25 lines) — the simplest possible LangGraph agent. Same
   pattern as `agent.py` but with one tool and no extras. Shows the
   StateGraph + tools_condition skeleton without distraction.
2. **`agent.py:graph`** — the production single-agent. After smoke.py,
   what you're seeing here is "smoke.py with a prune node, a serial tool
   node that has HITL gates, a fancier call_model, and 17 tools."
3. **`tools/files.py`** — pick three tools (read_file, write_file,
   edit_file). Each is ~20 lines. Pattern is consistent for all 17 tools.
4. **`chat.py:main`** — the REPL. Bottom-up: see how `app.stream()` is
   driven, how interrupts are detected and resumed, how the deadline works.
5. **`multi_agent.py`** — once you understand `agent.py`, this is "two of
   those subgraphs as nodes in a third graph, plus a supervisor that
   routes." The supervisor's routing-decision LLM call is the only new
   primitive.
6. **`eval/runner.py`** — how regressions get caught.
7. **`observability.py`** — three lines of meaningful work; rest is
   monkey-patching for compatibility.

`prompts.py`, `embeddings.py`, `admin.py`, `Makefile` are mostly utilities;
read on demand.

---

## 8. Concept-to-line index

Because the user might come back asking "which file again?":

```
StateGraph              agent.py:240, multi_agent.py:185
add_messages            agent.py:53 (AgentState)
RemoveMessage           agent.py:204 (in prune_node return)
@tool                   tools/files.py (any), tools/memory.py:53
ToolMessage             agent.py:303 (constructed in serial tool node)
interrupt()             agent.py:271 (in serial tool node phase 1)
Command(resume=...)     chat.py:438 (in main loop)
SqliteSaver             chat.py:463
ChatOllama              agent.py:73, multi_agent.py:60
NomicEmbeddings         embeddings.py:13
LanceDB                 scripts/index_docs.py:46, tools/docs.py:38
add_node                agent.py:252-254, multi_agent.py:188-191
add_conditional_edges   agent.py:255 (tools_condition), multi_agent.py:191 (_route)
StatusBar render        chat.py:506
TurnTimeout             chat.py:30 (class), chat.py:393 (raise)
slash dispatch          chat.py:_handle_slash
Persona override        prompts.py:resolve_chat_system
Facts injection         agent.py:call_model (parts builder)
```

---

## 9. Things this guide deliberately doesn't explain

- **LangGraph's checkpoint internals** — the `MetadataChannel`, the
  `BaseCheckpointSaver` protocol. Black box for our purposes.
- **OpenInference span schema** — Phoenix renders it nicely, that's all
  we use.
- **Ollama's HTTP API** — `langchain-ollama` is the wrapper; we don't
  reach behind it.
- **httpx connection pooling** — relevant only if scaling, not learning.

If you ever need any of these, the docs are the source. This guide is
the bridge between "what is the code structure" and "what is the agent
*doing*."
