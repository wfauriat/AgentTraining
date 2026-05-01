# lib_agent vs Claude Code — comparative assessment

**Date:** 2026-05-01, 11:51 CEST
**Repo state at writing:** post context-summarization + persistent-facts shipping;
pre cp/mv/edit tools; pre UX/UI pass. 23/23 eval passing.

This is a **snapshot** — a comparison at the level of *"which primitives exist?"*,
not *"how polished is each one?"*. After the next two milestones
(missing fs tools, then a significant UX/UI pass) we will produce a `specs.md`
synthesizing the final code-base surface and this snapshot will be retired.

---

## 1. Reasoning + tool-use core

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Iterative reason → act → observe loop | ✅ LangGraph StateGraph cycle | ✅ |
| Native tool calling (typed args, schema-derived) | ✅ `@tool` decorator, Pydantic schemas | ✅ |
| Structured output (Pydantic-typed responses) | ⚠️ Tested, found unreliable on 8B; replaced with keyword routing | ✅ Reliable on frontier models |
| Streaming (token-level + tool-event interleave) | ✅ Multi-mode stream | ✅ |
| Stop-condition handling | ✅ `tools_condition` (no tool_calls → END) | ✅ |

**Verdict.** Structurally complete. The only material gap is structured-output
reliability, which is a model-class issue (8B vs frontier), not a missing
primitive.

---

## 2. Tool surface

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| File read/write | ✅ `read_file`, `write_file` | ✅ Read, Write, Edit |
| Directory ops (ls/mkdir/find/delete) | ✅ 4 bash-style tools | ✅ Bash, Glob, LS |
| File copy/move | ❌ Not yet — slated for next task | ✅ |
| Smart string-replace edit | ❌ We overwrite full content — slated for next task | ✅ Edit tool's old_string/new_string |
| Code execution | ✅ `run_python` (Docker sandbox) | ✅ Bash (no Docker) |
| Web search | ✅ Tavily | ✅ WebSearch |
| Web fetch | ✅ httpx + trafilatura | ✅ WebFetch |
| Local document RAG | ✅ LanceDB + nomic-embed-text | ❌ No native RAG; relies on filesystem search |
| Time/date | ✅ `get_current_time` | ⚠️ Via Bash |

**Verdict.** Tool surface is broadly comparable. Notable holes on our side:
`cp`/`mv` and a *surgical-edit* tool — both addressed in the next task.
Claude Code lacks our RAG primitive (a deliberate design choice — they
prefer grep + read).

---

## 3. Sandbox / safety

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Path traversal defense | ✅ `_safe_path` resolved-relative-to check | ✅ |
| Code execution isolation | ✅ Docker `--network none --read-only --memory --cpus` | ⚠️ Bash on host (relies on user-level perms) |
| No shell injection | ✅ No `shell=True`, stdin pipe for code | ✅ Argument array, not string |
| Wall-clock timeouts | ✅ 30s on `run_python` | ✅ Configurable |
| HITL approval gate on destructive ops | ✅ `interrupt()` on write_file/run_python/delete_file | ✅ Permission prompts on Bash, Write, Edit |
| Resource caps (RAM/CPU) | ✅ Docker `--memory 256m --cpus 0.5` | ❌ Host process |

**Verdict.** lib_agent's *runtime* sandbox is actually stronger because we run
code in Docker; Claude Code runs Bash on the host. Path-traversal and HITL
stories are comparable.

---

## 4. State / persistence / context

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Multi-turn conversation state | ✅ `add_messages` reducer | ✅ |
| Cross-process resume | ✅ SqliteSaver + `thread_id` | ✅ Conversation history |
| Multiple parallel conversations | ✅ thread_id scoping | ✅ |
| Time-travel / replay | ✅ Built into LangGraph (no UX yet) | ⚠️ Limited |
| Context summarization on overflow | ✅ Just shipped: prune node + transcript-summarize | ✅ Auto-summarizes near limit |
| Persistent user/project facts (cross-thread) | ✅ Just shipped: `facts.json` + auto-injection | ✅ `CLAUDE.md` (richer: hierarchical, project + user, version-controlled) |
| Hierarchical memory (project root, sub-dirs, user-global) | ❌ Single flat KV | ✅ |
| Token-budget awareness inline | ❌ Phoenix shows it; no inline display | ✅ Status bar |

**Verdict.** State and persistence are present. The richness gap is on memory
hierarchy (one flat file vs Claude's nested CLAUDE.md system) and inline
budget visibility (the latter slated for the UX pass).

---

## 5. Multi-agent / orchestration

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Subagent spawning | ✅ Supervisor + 2 worker subgraphs | ✅ `Task` / `Agent` tool |
| Routing decisions | ✅ Keyword + directive injection | ✅ Native to the architecture |
| Subgraph composition | ✅ Workers compile to node-callable | ✅ |
| Parallel subagents | ⚠️ Sequential (1 GPU constraint, by design) | ✅ Genuinely parallel |
| HITL through subgraph boundaries | ✅ `interrupt()` propagates | ✅ |
| Specialist roles (research vs code) | ✅ Built two | ✅ Configurable per-task |

**Verdict.** Structurally we match. Practical parallelism is hardware-bounded
for us; the framework supports it.

---

## 6. Observability / eval / tracing

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Trace capture (model + tool spans, hierarchy) | ✅ Phoenix self-hosted, OpenInference auto-instrumentation | ✅ Anthropic-internal traces |
| Web UI for trace inspection | ✅ localhost:6006 | ⚠️ Limited end-user UI |
| Eval harness with golden cases | ✅ 23 cases, 6 gates per case | ❌ User builds their own |
| Negative tests / sandbox-defense regression | ✅ 6 cases | ❌ |
| Latency budgets per case | ✅ | ❌ |
| Artifact-existence assertions (file did/didn't land) | ✅ `forbidden_artifact` gate | ❌ |
| LLM-as-judge | ❌ Not yet | ❌ Out of scope |
| Per-trace scoring back to UI | ❌ Phoenix supports it; not wired | ❌ |

**Verdict.** This is where lib_agent is **actually ahead in concept**. The eval
harness with multi-gate scoring including security regressions has no direct
equivalent end-user-facing in Claude Code. Claude's traces exist but are
operator-side; ours are user-visible.

---

## 7. Developer ergonomics

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Interactive REPL | ✅ `chat.py` | ✅ The IDE/CLI |
| Resume named conversations | ✅ `--thread <id>` | ✅ Session management |
| Slash commands (`/command`) | ❌ Slated for UX pass | ✅ Many |
| Hot-edit system prompt | ❌ Slated for UX pass | ✅ Persona / personality settings |
| Custom personas / identity | ❌ Single base prompt | ✅ |
| Hooks (pre/post tool, etc.) | ❌ | ✅ |
| Plugin / extension model | ⚠️ Indirectly via @tool | ✅ MCP |
| MCP server / client | ❌ Noted in roadmap | ✅ Native |
| Settings / preferences UI | ❌ Make targets only | ✅ Settings.json + UI |

**Verdict.** Biggest tier of gaps. Most are slated for the UX pass plus future
MCP work. None require new primitives — they surface what's already there.

---

## 8. Model layer

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Multiple model providers | ⚠️ Currently single (Ollama Qwen3); LangChain abstracts the layer | ✅ Anthropic only by design |
| Per-component model choice | ✅ Supervisor uses qwen3-nothink, workers use qwen3:8b | ⚠️ One model per session |
| Local inference | ✅ 100% local Ollama | ❌ Cloud-only |
| Quantization-aware decisions | ✅ Q4 + KV cache size tuning | n/a |
| Context window per-call control | ✅ `num_ctx=16384` parameter | ✅ |

**Verdict.** Surprisingly, lib_agent has **more model flexibility** because
LangChain's adapter layer makes the provider swappable. Claude Code is
intentionally Anthropic-only.

---

## 9. Production hygiene

| Capability | `lib_agent` | Claude Code |
|---|---|---|
| Configurable timeouts | ✅ `TOOL_TIMEOUT`, `MODEL_TIMEOUT` (implicit) | ✅ |
| Tool-output truncation | ✅ `MAX_TOOL_RESULT_CHARS` | ✅ |
| Result caching | ❌ Noted as Phase 4 | ⚠️ Conversation cache only |
| Retry / circuit breaker | ⚠️ httpx defaults; no policy | ✅ |
| Structured logs | ⚠️ Phoenix is the log; no JSONL | ✅ |
| Cost tracking | n/a (local) | ✅ |

**Verdict.** Middle of the road. Caching and explicit retry policy are real
gaps; both are small and slotted for post-UX phases.

---

## Big-picture summary

```
Conceptual primitive coverage     : ~85% of Claude Code's surface
Polish, depth, edge-case handling : maybe 30–40% (8B model, single-developer)
Areas where we're AHEAD           : eval harness, sandbox isolation depth,
                                    model flexibility, RAG primitive
Areas where we're meaningfully behind:
  - hierarchical memory (CLAUDE.md vs facts.json)
  - surgical-edit tool (Edit vs full overwrite)        ← next task
  - file copy/move                                     ← next task
  - hooks
  - MCP integration
  - slash commands / personas / status bar             ← next next task (UX)
```

**Bottom line.** At the level of *"which primitives exist?"*, lib_agent covers
the same conceptual surface as Claude Code on roughly **85%** of dimensions,
with **deeper sandbox isolation** and **more eval discipline** than Claude
Code exposes to its users — capped by frontier-model-grade polish ceilings
imposed by the 8B model running underneath.

What we're missing is mostly **UX surface** (slash commands, personas,
hot-edit, status bar, hooks) plus a few specific tools (cp/mv, Edit-style
surgical edits). That's exactly what the next two milestones address, with
MCP and hierarchical memory as natural follow-ups.

If a hiring conversation went *"sketch the architecture of a Claude-Code-like
agent — what primitives does it need?"*, this codebase **is** that sketch with
working implementations. The remaining work is polish, not invention.

---

## Plan trail (for `specs.md` synthesis later)

After this snapshot date (2026-05-01), the planned shipping order is:

1. **Missing fs tools** — `copy_file`, `move_file`, surgical `edit_file`
   (string replacement). Add HITL gating where appropriate; add eval cases.
2. **UX/UI pass** — Rich panels for output, slash commands (`/facts`,
   `/forget`, `/persona`, `/threads`, `/quit`), hot-edit of system prompt,
   inline token budget display, status bar, status visibility for the
   summary/facts state. Likely couples with the planned `prompts.py`
   extraction so prompts can be hot-edited cleanly.
3. **`specs.md`** — full synthesis of the resulting code-base surface.

This file (`assessment_2026-05-01.md`) gets retired when `specs.md` lands.

---

## Postscript — what shipped after this snapshot (2026-05-01 evening)

This snapshot was written midday. By end of day the two follow-up
milestones called out in the plan trail above were both delivered.
Updated cell values, against the same comparison axes:

### Tool surface — gaps closed
- `copy_file`, `move_file` — both shipped, sandboxed via the same
  `_safe_path` defense, both in `DESTRUCTIVE_TOOLS` (HITL-gated).
- `edit_file` — surgical string replacement shipped with safety-by-default:
  `old_string` must appear **exactly once** unless `replace_all=True`. Same
  pattern as Claude Code's `Edit` tool. HITL-gated.
- 7 new eval cases: `cp_basic`, `mv_basic`, `edit_basic`, `neg_cp_traversal`,
  `neg_mv_traversal`, `neg_edit_traversal`, `neg_edit_uniqueness`.
- The fs-tool surface now matches Claude Code's at the primitive level
  (Read / Write / Edit / Bash{ls, mkdir, cp, mv, rm, glob}).

### Developer ergonomics — most gaps closed
- Slash commands all shipped: `/help`, `/clear`, `/quit` (+ `/exit`, `/q`),
  `/facts`, `/forget <key>`, `/threads`, `/persona` (5 modes: show / reset /
  edit / load / inline), `/debug [on|off|toggle]`.
- Hot-edit system prompt: `persona.txt` override, `agent.call_model`
  re-resolves on every call so changes take effect on the next message
  without restart.
- Status bar before every prompt: thread_id, ctx %, summary marker,
  facts count. Color-coded (green / yellow / red) by ctx pressure.
- Per-turn token footer aggregates input + output tokens across all model
  calls in a turn (single-agent: usually 2; multi-agent: 4-12).
- Rich panels: cyan `Assistant` headers, yellow tool-call lines (debug),
  green tool-result lines (debug), magenta supervisor routing (debug),
  red HITL approval prompts (always shown).
- `/debug` toggle: default mode is clean (only assistant text + footer +
  status); debug mode reveals tool calls, tool results, supervisor routing.
- Centralized `prompts.py` for all 5 system prompts; clean dependency
  target for the future "model-conditional prompts" work that's now an
  optional follow-up rather than urgent.

### Production hygiene — gaps closed
- `MODEL_TIMEOUT = 90` propagates to httpx so a hung Ollama daemon errors
  out cleanly instead of hanging the REPL.
- `TURN_TIMEOUT = 180` cooperative deadline checked at every stream chunk;
  raises `TurnTimeout` cleanly. HITL waits don't count against the budget.
- `KEEP_ALIVE = -1` keeps the model loaded indefinitely — fixes the
  "5-min idle → unload → reload-races-next-turn → hang" failure mode.
- Per-turn try/except in `chat.py` catches `TurnTimeout`,
  `KeyboardInterrupt`, and any other Exception — REPL never dies.

### State / persistence / context — closed gap on context summarization
- `prune_node` runs before every agent step; summarizes when state exceeds
  `PRUNE_THRESHOLD_TOKENS = 8000`. Visible feedback when it fires, with the
  summarizer's stream filtered out of the user-visible render so it doesn't
  pollute the assistant reply.
- Persistent facts (`facts.json`) shipped alongside, with auto-injection
  into `agent.call_model`'s system context every turn. Three tools
  (`remember` / `forget` / `recall`) plus `/facts`, `/forget`, `/persona`
  CLI counterparts.
- Multi-agent state ALSO uses `prune_node` (via SupervisorState which
  carries `summary` like AgentState).

### Multi-agent — adjusted for 8 GB local hardware
- Single-model multi-agent: supervisor + workers + summarizer all use
  `qwen3:8b` (one VRAM resident). Originally split across `qwen3:8b` and
  `qwen3-nothink` — that triggered VRAM contention with `keep_alive=-1`
  (the second model couldn't load while the first was pinned), causing
  90-second timeouts on subsequent turns. Lesson worth preserving.
- Supervisor's keyword-routing fallback handles Qwen3's thinking-token
  emptiness without needing a separate non-thinking model.

### Model layer
- Sticking with `qwen3:8b` everywhere. `qwen3-nothink` removed from active
  code paths (still installed for ad-hoc experiments).

### Observability — unchanged
- Phoenix still self-hosted; eval still 23 → 30 cases.

### Net updated big-picture summary

```
Conceptual primitive coverage     : ~95% of Claude Code's surface (was 85%)
Polish, depth, edge-case handling : ~50% (was 30-40%)
Areas where we're AHEAD (unchanged): eval discipline, sandbox isolation,
                                     model flexibility, RAG primitive
Areas still meaningfully behind:
  - hierarchical memory (CLAUDE.md vs facts.json — still flat)
  - hooks
  - MCP integration
  - native parallel subagent execution (ours queues on 1 GPU)
```

### Lessons captured during the late-day pass (worth surviving into specs)

1. **Ollama `keep_alive=-1`** — must be set explicitly to prevent silent
   5-minute idle unloads that race the next REPL turn.
2. **Two models on 8 GB VRAM with `keep_alive=-1`** — deadlock. Single
   model only on local hardware below ~16 GB.
3. **Rich markup eats nested brackets**: `[yellow][tool: x][/yellow]` —
   the inner `[tool: x]` looks like a markup tag and gets dropped. Pass
   `markup=False, highlight=False` along with `style="yellow"` to keep
   literal brackets.
4. **Summarizer LLM with alternating Human/AI input returns empty** —
   wrap as a single transcript inside one HumanMessage to give the model
   a clean "now produce a reply" stance.
5. **`interrupt()` replays the node body** — gather all approvals first
   (side-effect-free), then execute (runs once). Two-phase serial tool node.
6. **`config["configurable"]["thread_id"]`** — `configurable` is optional in
   `RunnableConfig` per the type stubs; access via `(cfg.get("configurable")
   or {}).get("thread_id", "?")` to keep type checker quiet.
7. **Supervisor over-routing on 8B** — explicit STOP RULE in supervisor
   prompt + directive injection + few-shot examples covering read-only
   paths reduces but doesn't eliminate. Hard cap is the safety net.

### Closing note

Original assessment said "the remaining work is polish, not invention".
By end of day that polish is done. `specs.md` and `companion.md` (written
alongside this postscript) supersede this snapshot.
