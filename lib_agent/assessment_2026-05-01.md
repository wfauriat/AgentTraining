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
