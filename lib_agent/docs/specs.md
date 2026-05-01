# lib_agent — specifications

**Status as of:** 2026-05-01 (end-of-day)
**Supersedes:** `assessment_2026-05-01.md`
**Companion:** `companion.md` (deeper conceptual walkthrough)

A local agentic AI REPL built on LangGraph, Ollama (Qwen3 8B), Phoenix, and a
small ad-hoc tool surface. Designed to demonstrate the full set of
primitives a Claude-Code-like agent needs, on consumer hardware (8 GB GPU).

---

## 1. Capabilities at a glance

| Capability | Mechanism |
|---|---|
| Tool-using agent loop | LangGraph `StateGraph` + `bind_tools` + serial dispatch node |
| Multi-agent (supervisor pattern) | Worker subgraphs + keyword-routing supervisor + directive injection |
| Streaming (token + tool events) | `app.stream(stream_mode=["messages","updates"])`, multi-mode interleave |
| Persistence across processes | `SqliteSaver` checkpointer on `checkpoints.sqlite` |
| Persistent facts (across threads) | `facts.json` flat KV; auto-injected into system context |
| Context summarization | `prune_node` runs before every agent step; summarizes at 8K-token threshold |
| HITL approval gates | `interrupt()` on destructive tools; resume via `Command(resume=...)` |
| Rich REPL UX | Colored panels, slash commands, status bar, per-turn token footer, `/debug` |
| Observability | Phoenix self-hosted on `:6006` via OpenInference auto-instrumentation |
| Eval harness | 30 cases × 6 gates (tool / error / contains / safe / latency / artifact) |
| Sandboxed code execution | Docker `--network none --read-only --memory 256m --cpus 0.5 --rm` |
| Sandboxed filesystem | `_safe_path` resolved-relative-to workspace root |
| Web access | Tavily search API + httpx + trafilatura extraction |
| RAG | LanceDB + nomic-embed-text with prefix-aware embeddings |

---

## 2. Repository layout

```
lib_agent/
├── chat.py                       interactive REPL — single + multi-agent dispatch via --multi
├── agent.py                      single-agent graph: AgentState, prune_node, call_model, serial tool node
├── multi_agent.py                supervisor pattern: SupervisorState, supervisor_node, worker subgraphs
├── prompts.py                    five system prompts + persona resolver
├── observability.py              Phoenix auto-instrumentation setup
├── admin.py                      CLI for purging state / threads / facts / traces
├── config.py                     all runtime constants in one file
├── embeddings.py                 NomicEmbeddings wrapper (adds search_query/search_document prefixes)
├── smoke.py                      one-tool reference agent (frozen, kept as baseline)
│
├── tools/
│   ├── files.py                  read_file, write_file, edit_file, copy_file, move_file,
│   │                             list_directory, make_directory, find_files, delete_file
│   ├── web.py                    web_search (Tavily), web_fetch (httpx + trafilatura)
│   ├── python_sandbox.py         run_python (Docker isolated)
│   ├── docs.py                   search_documents (LanceDB RAG)
│   └── memory.py                 remember, forget, recall + render_facts() helper
│
├── eval/
│   ├── golden.py                 30 test cases, 6 categories
│   ├── runner.py                 6-gate scorer + JSON report writer
│   └── reports/                  per-run JSON output (gitignored)
│
├── scripts/
│   └── index_docs.py             rebuilds the LanceDB vector index from corpus/
│
├── observability/
│   ├── docker-compose.phoenix.yml         Phoenix container (active)
│   └── docker-compose.langfuse.yml        reference, not running
│
├── corpus/biology.md             RAG knowledge corpus
├── workspace/                    sandboxed filesystem (gitignored)
├── vector_db/                    LanceDB store (gitignored)
├── checkpoints.sqlite            SqliteSaver state (gitignored)
├── facts.json                    persistent KV facts (gitignored)
├── persona.txt                   optional system-prompt override (gitignored, only if user creates)
│
├── Makefile                      chat / chat-multi / eval / index / info / purge-* / phoenix-* / install
├── requirements.txt
├── perspectives.md               forward-looking plan (pre-UX)
├── assessment_2026-05-01.md      vs-Claude-Code snapshot + post-UX postscript
├── specs.md                      this file
└── companion.md                  conceptual walkthrough
```

---

## 3. Tool inventory (17 tools)

| Tool | Module | Destructive? | Sandbox |
|---|---|---|---|
| `get_current_time` | `agent.py` | no | n/a |
| `read_file` | `tools/files.py` | no | `_safe_path` |
| `write_file` | `tools/files.py` | **yes** | `_safe_path` |
| `edit_file` | `tools/files.py` | **yes** | `_safe_path` + uniqueness-by-default |
| `copy_file` | `tools/files.py` | **yes** | `_safe_path` (both src & dst) |
| `move_file` | `tools/files.py` | **yes** | `_safe_path` (both src & dst) |
| `list_directory` | `tools/files.py` | no | `_safe_path` |
| `make_directory` | `tools/files.py` | no | `_safe_path` (idempotent mkdir -p) |
| `find_files` | `tools/files.py` | no | `_safe_path` + result cap |
| `delete_file` | `tools/files.py` | **yes** | `_safe_path`; refuses dirs |
| `web_search` | `tools/web.py` | no | Tavily API; key from `.env` |
| `web_fetch` | `tools/web.py` | no | httpx + trafilatura; truncated output |
| `run_python` | `tools/python_sandbox.py` | **yes** | Docker `--network none --read-only --memory 256m --cpus 0.5`, 30 s wall clock |
| `search_documents` | `tools/docs.py` | no | LanceDB cosine similarity; top-K=5 |
| `remember` | `tools/memory.py` | no | Writes `facts.json`; flat KV |
| `forget` | `tools/memory.py` | no | Removes one fact key |
| `recall` | `tools/memory.py` | no | Reads `facts.json` |

`DESTRUCTIVE_TOOLS = {write_file, edit_file, copy_file, move_file, delete_file, run_python}`.
Each call to one of these triggers `interrupt()` for HITL approval before
execution. Approval cached per-call via the two-phase serial tool node so
LangGraph node-replay-on-resume doesn't double-execute.

---

## 4. State model — five tiers

| Tier | Storage | Lifetime | Owner |
|---|---|---|---|
| `messages` | LangGraph state, in-memory + checkpoint | Per turn → persisted per checkpoint | `add_messages` reducer |
| `summary` | LangGraph state field | Survives until thread cleared | `prune_node` |
| Conversation checkpoints | `checkpoints.sqlite` | Until thread deleted | `SqliteSaver` |
| Persistent facts | `facts.json` | Until manually deleted | `tools/memory.py` |
| RAG corpus | `vector_db/` (LanceDB) | Until reindexed | `scripts/index_docs.py` |

System prompt (CHAT persona) is **not** part of state — `agent.call_model`
prepends a freshly-resolved one (persona + facts + summary) on every call.
This is what makes `/persona` changes apply to the next message without a
state migration.

---

## 5. Graph topology

### Single agent (`agent.py:graph`)

```
START → prune → agent → (tools | END)
              ↑________|
        tools → prune → agent ...
```

`prune` is no-op when below `PRUNE_THRESHOLD_TOKENS = 8000`. When over,
splits at a HumanMessage boundary, summarizes everything before it via the
summarizer LLM, emits `RemoveMessage` entries to drop them.

### Multi-agent (`multi_agent.py:graph`)

```
START → prune → supervisor → (research_agent | code_agent | END)
                              ↑___________________________|
        research_agent → supervisor
        code_agent → supervisor
```

- `supervisor` chooses the next worker via keyword routing
  (`research_agent` / `code_agent` / `FINISH`). Falls back to `research_agent`
  when no AI message yet, `FINISH` when one exists, on parse miss.
- `research_agent` and `code_agent` are compiled subgraphs each with their
  own LLM (bound to their tool subset) + serial tool node + tools_condition.
- Supervisor injects a directive as a HumanMessage when routing; worker
  reads it as the immediate task.

State: `SupervisorState` extends the base shape with `next: str` and
`supervisor_reason: str` for routing visibility, plus `summary: str` for
the same prune semantics as single-agent.

---

## 6. REPL — `chat.py`

### Slash commands (run without LLM round-trips)

| Command | Effect |
|---|---|
| `/help` | Show menu |
| `/facts` | Print persisted facts |
| `/forget <key>` | Delete one fact |
| `/threads` | List checkpoint threads, mark current |
| `/persona` | Show active system prompt |
| `/persona reset` | Delete `persona.txt`, fall back to default |
| `/persona edit` | Multi-line entry; blank line ends; `/cancel` aborts |
| `/persona load <path>` | Read alternate file; persist contents to `persona.txt` |
| `/persona <inline>` | One-line replacement |
| `/debug [on\|off\|toggle]` | Toggle verbose stream rendering |
| `/clear` | Drop the current thread's checkpoints |
| `/quit`, `/exit`, `/q` | Leave |

Bare `quit` / `exit` (no slash) also exit, for backward compat.

### Render modes

| Element | Default mode | Debug mode |
|---|---|---|
| User prompt | `You: ...` | same |
| Assistant text | cyan `Assistant` header + streamed content | same |
| Worker subgraph reply (multi-agent) | cyan `research_agent` / `code_agent` header + content | same |
| Tool calls | hidden | yellow `[tool: name(args)]` |
| Tool results | hidden | green `[name → preview]` |
| Supervisor routing | hidden | magenta `[supervisor → x]` + dim reason |
| HITL approval | bold red `⚠ approval requested` (always shown) | same |
| Status bar | Rich rule, color-coded by ctx % (green <50, yellow <75, red ≥75) | same |
| Per-turn footer | dim grey `↳ in N tok · out M tok · Ts (k model calls)` | same |

### Per-turn safety

| Layer | Cap | Behavior on breach |
|---|---|---|
| `MODEL_TIMEOUT` | 90 s per `ChatOllama.invoke` (httpx-level) | Raises; caught by chat's per-turn handler |
| `TURN_TIMEOUT` | 180 s wall clock per REPL turn (cooperative) | `TurnTimeout` raised at next stream chunk; caught; print "turn timed out", continue |
| `KeyboardInterrupt` | n/a | "turn interrupted by user", continue |
| Any other exception | n/a | "turn failed (Type): msg", continue |

REPL never dies on a turn-level error.

---

## 7. Configuration (`config.py`)

| Constant | Default | Purpose |
|---|---|---|
| `MODEL` | `"qwen3:8b"` | Ollama model used by all roles |
| `NUM_CTX` | `16384` | Per-request context window |
| `KEEP_ALIVE` | `-1` | Keep model loaded indefinitely (prevents idle-unload races) |
| `WORKSPACE_DIR` | `"./workspace"` | Filesystem sandbox root |
| `TAVILY_URL` | `https://api.tavily.com/search` | Web search |
| `MAX_TOOL_RESULT_CHARS` | `4000` | Truncate large tool outputs |
| `TOOL_TIMEOUT` | `30` | Wall clock for `run_python` |
| `MODEL_TIMEOUT` | `90` | httpx timeout for ChatOllama |
| `TURN_TIMEOUT` | `180` | REPL turn wall clock |
| `EMBED_MODEL` | `"nomic-embed-text"` | Embeddings (with prefixes) |
| `DB_PATH`, `TABLE_NAME` | `"./vector_db"`, `"course"` | LanceDB |
| `CORPUS_PATH` | `"./corpus/biology.md"` | RAG source |
| `TARGET_CHUNK_CHARS`, `TOP_K` | `2000`, `5` | Indexing + retrieval |
| `PRUNE_THRESHOLD_TOKENS` | `8000` | Trigger summarization |
| `SUMMARY_KEEP_TAIL` | `6` | Boundary-aware tail to keep intact |
| `SUMMARY_MODEL` | `MODEL` | Summarizer LLM (single model => no VRAM swap) |

`.env` (gitignored) provides `TAVILY_API_KEY`.

---

## 8. Eval harness (`eval/`)

### Run

```
python -m eval.runner
python -m eval.runner --filter <substring>
python -m eval.runner --skip-categories network
make eval / make eval-quick
```

### Output

Per-run JSON in `eval/reports/run_YYYYMMDD_HHMMSS.json`. Each case:

| Gate | Letter | Asserts |
|---|---|---|
| `tool_pass` | T | `expect_tool` was called (or no tool if `expect_tool=None`) |
| `tool_error_pass` | E | If `expect_tool_result_error=True`, the tool returned an error string |
| `text_pass` | C | All `expect_text_contains` substrings appear in final reply |
| `text_safe` | S | None of `expect_text_NOT_contains` appears (no leak) |
| `latency_pass` | L | Wall clock ≤ `max_seconds` |
| `artifact_pass` | A | `forbidden_artifact` path does NOT exist on disk |

`overall = T ∧ E ∧ C ∧ S ∧ L ∧ A`. Capital letter = pass, lowercase = fail.

30 cases across 6 categories: tool, multi_tool, network, rag, text, negative.
Negatives include: path traversal on read/write/ls/delete/cp/mv/edit, python
sandbox timeout, host filesystem isolation, edit-uniqueness safety.

---

## 9. Observability — Phoenix

Run:
```
docker compose -f observability/docker-compose.phoenix.yml up -d
```
UI: http://localhost:6006 — project `lib_agent`.

OpenInference auto-instrumentation captures every span: graph nodes, LLM
calls (with input/output tokens, prompt/eval timings), tool executions,
HITL interrupts. No callback wiring needed.

Scope: traces only. No metrics, no alerts. Eval scoring lives separate, in
JSON reports.

---

## 10. Operations — Makefile

```
make chat                    interactive REPL (single agent)
make chat-multi              interactive REPL (multi-agent supervisor)
make eval                    full 30-case suite
make eval-quick              skip network category
make index                   rebuild LanceDB from corpus/
make info                    show db / facts / phoenix status
make list-threads            list checkpoint thread_ids
make list-facts              dump facts.json
make purge-state             delete checkpoints.sqlite
make purge-facts             delete facts.json
make purge-traces            reset Phoenix volume
make phoenix-up / down       observability lifecycle
make phoenix-logs            tail container logs
make install                 pip install -r requirements.txt
make help                    menu (default target)
```

Surgical operations available via `python -m admin <subcommand>` —
`purge-thread <id>`, `info`, `list-threads`, `list-facts`,
`purge-all --yes`, `purge-facts --yes`, `purge-traces --yes`.

---

## 11. Operational gotchas (lessons from build)

1. **Ollama keep_alive**: `KEEP_ALIVE = -1` is required to prevent the
   "5-minute idle unload races the next request" hang.
2. **8 GB GPU = single-model rule**: Two distinct models (e.g.
   `qwen3:8b` + `qwen3-nothink`) cannot coexist in VRAM with `keep_alive=-1`
   — the second never loads. All roles use `MODEL`.
3. **Rich markup eats brackets**: `[yellow][tool: x][/yellow]` — the inner
   `[tool: x]` looks like a markup tag. Use `console.print(text,
   style="yellow", markup=False, highlight=False)` to keep brackets literal.
4. **Summarizer needs a transcript shape**: alternating Human/AI inputs
   to a summarizer LLM frequently return empty content. Render messages
   into a single transcript inside one HumanMessage.
5. **interrupt() replays the node body**: gather all approvals before any
   side effects. Two-phase pattern in `make_serial_tool_node`.
6. **`config["configurable"]` is optional in `RunnableConfig`**: access
   via `(cfg.get("configurable") or {}).get("thread_id", "?")`.
7. **Substring eval matching is brittle**: model wording varies between
   "timed out" and "timeout"; "1048576" vs "1,048,576". Prefer
   tool-result-error gates over text checks where possible.
8. **Supervisor on 8B over-routes**: explicit STOP RULE in prompt +
   directive injection mitigates; perfect FINISH-eagerness requires a
   bigger model.

---

## 12. Known limits

| Limit | Why | Mitigation |
|---|---|---|
| Multi-agent slow on 8 GB | Each supervisor↔worker round = full model call (~10-15 s on Qwen3 8B with thinking) | Use single-agent for daily work; multi-agent for pattern demos |
| Supervisor occasional hallucinated routing | 8B coordination drift | Hard cap not yet implemented (planned tier B) |
| RAG retrieval mediocre on short queries | nomic-embed-text on small corpus has tight distance bands | Phase 3 RAG — hybrid + reranker — postponed |
| Single-flat `facts.json` | No project / user hierarchy like CLAUDE.md | Tier B work |
| No hooks (pre/post tool call) | Not implemented | Tier C |
| No MCP server / client | Not implemented | Tier C |

---

## 13. What we deliberately don't do

- No `langchain.agents.AgentExecutor` — legacy, not where the framework is going.
- No callback handler in `RunnableConfig.callbacks` — Phoenix uses OTel
  context, not callbacks. Keeps the config dict clean.
- No automatic retry on tool failure — errors come back as `[error: ...]`
  strings and the model decides what to do.
- No hot-reload of code changes — must restart `chat.py` to pick up edits.
- No MCP — exposing tools to other agents is future work.

---

## 14. Where to read next

- **`companion.md`** for a deeper conceptual walkthrough — abstractions,
  entry points, interfaces between components, representative flows.
- **`perspectives.md`** for the original forward-looking plan (now mostly
  delivered).
- **`assessment_2026-05-01.md`** for the comparison vs Claude Code, plus
  the postscript covering everything that shipped after midday.
