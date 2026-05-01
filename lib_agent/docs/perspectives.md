# lib_agent — perspectives

Snapshot of what's done, deferred, and next. Written 2026-05-01 at the close
of Phase 2.

## Where we stand

A LangGraph agent built side-by-side with the hand-rolled `agent/`:

- **State graph** with serial tool node (race-free), `add_messages` reducer,
  `tools_condition` routing.
- **7 tools**: `get_current_time`, `read_file`, `write_file`, `web_search`,
  `web_fetch`, `run_python` (Docker sandbox), `search_documents` (LanceDB +
  nomic-embed-text RAG).
- **Persistence** via SqliteSaver; cross-process conversation resume by
  `thread_id`.
- **Streaming** in `chat.py`: token-level (`messages` mode) interleaved with
  discrete tool events (`updates` mode).
- **Observability**: Phoenix self-hosted on `:6006`, OpenInference
  auto-instrumentation; full graph + LLM + tool spans for every run.
- **Eval harness**: 17 cases, 6 categories (tool, multi_tool, network, rag,
  text, negative). Six independent gates: tool selected, tool errored when
  expected, text contains required, text does NOT contain forbidden, latency
  under budget, no forbidden artifact on disk.

## Deferred — Phase 3 (better RAG)

Postponed by choice. Comes back later with:

- Hybrid retrieval (BM25 + vector)
- Cross-encoder reranking (`bge-reranker-v2-m3`)
- Query rewriting / HyDE
- ragas metrics on the existing eval set
- The dataset and graph already exist; this phase mostly adds new retrieval
  nodes and re-evaluates with the harness.

Re-entry cost: low — eval harness is the comparison anchor.

## Next — ranked by learning value × effort

### Tier A — high payoff

1. **HITL `interrupt` on destructive tools** *(half-day)*
   - Wrap `write_file` and `run_python` calls in a `interrupt(...)` gate.
   - `chat.py` surfaces the interrupt as `[approve? y/n]`.
   - Resume via `app.invoke(Command(resume="yes"), config=...)`.
   - Demonstrates LangGraph's pause-resume across processes — the killer
     feature unique to this framework, and a real production pattern.

2. **Multi-agent supervisor** *(1 day)*
   - `research_agent` subgraph with `web_search` + `web_fetch` + `search_documents`.
   - `code_agent` subgraph with `run_python` + file tools.
   - `supervisor` graph routes user requests, dispatches to one or both
     workers, synthesizes their outputs.
   - Subgraphs compile to node-compatible callables — composition is free.
   - Single shared local model; queue serialization fits 8 GB VRAM budget.
   - Single most CV-relevant pattern we haven't touched.

3. **Structured outputs + planner node** *(half-day)*
   - `model.with_structured_output(Plan)` for an explicit decomposition step.
   - Pydantic schemas as the planner's contract; downstream nodes consume
     `state["plan"].steps`.
   - Solves the "model decides write_file content before run_python returns"
     issue surfaced earlier by removing the parallel-commit window.

### Tier B — useful but less unique

4. **LLM-as-judge eval** *(half-day)*
   - Qwen3 grades free-form replies on rubric criteria; scores pushed to
     Phoenix as span annotations.
   - Catches "called extra tool we didn't need" and "answer is technically
     right but evasive" — issues the rule-based gates miss.

5. **Caching** *(half-day)*
   - `set_llm_cache(SQLiteCache(...))` for model calls.
   - Embedding cache for indexing iterations.
   - Pays off during eval re-runs (5–10× speedup on cache hits).

6. **MCP server** *(half-day)*
   - Expose the 7 tools as a Model Context Protocol server.
   - Claude Desktop, Cursor, etc. consume them.
   - Striking portfolio demo; small code change.

### Tier C — specialized

7. **Pydantic AI port** of Phase 1 — taste a different framework.
8. **DSPy** on a single prompt against the eval set — measurable optimization.
9. **LoRA / QLoRA fine-tune** of Qwen3 on tool-call traces — fixes
   model-class issues like the `\n`-escape bug at the source. Tight on 8 GB.

## Chain-of-thought variants (specifically)

| Pattern              | Verdict for this stack                                       |
|----------------------|--------------------------------------------------------------|
| ReAct                | Already implicit in your tool-calling loop. Marginal value to make explicit. |
| Plan-and-execute     | Effectively the supervisor pattern simplified. Subsumed by #2. |
| Self-critique pass   | Cheap (two extra nodes); real quality lift on hard tasks. Worth a side experiment after #2. |
| Tree-of-thoughts     | Big lifts on hard reasoning, but expensive (N× model calls). Niche. |

## Chosen path

**#1 (HITL) → #2 (multi-agent supervisor)**

Sequence reasoning:
- HITL exercises `interrupt` and persistence-across-pause — two LangGraph
  features the current code doesn't touch.
- Supervisor exercises subgraph composition — the third major feature.
- Together they exhaust LangGraph's distinctive surface area.
- Eval harness regresses both for free as we build.
- Tier B items (#3 structured outputs, #4 LLM-judge, #5 caching) drop out
  naturally during the supervisor build — we'll feel the pain that motivates
  them, then add them on demand.

## Branch hygiene

This direction is a real complexity bump (HITL state, subgraph composition,
multi-model orchestration). Develop on a feature branch so the Phase 1+2
artifact remains a clean reference for comparison.

## Open considerations

- **Single-model contention under multi-agent**: two worker subgraphs both
  calling `qwen3:8b` on a single GPU will serialize at the Ollama layer.
  Acceptable for learning; document the latency impact.
- **HITL UX**: terminal `[y/n]` is fine for now; if we later want a web UI,
  the same `interrupt` mechanism plumbs to any frontend that can `POST /resume`.
- **Eval coverage gap**: the harness has no multi-agent cases yet. We'll add
  them as #2 progresses — ideally a case per supervisor routing decision.
