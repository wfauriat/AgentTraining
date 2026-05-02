# AgentTraining

A personal training repo for learning agentic AI from the ground up on
consumer hardware (8 GB GPU, Ollama, Qwen3 8B). Two complete agents live
side-by-side, built in sequence so the second can lean on lessons from the
first.

```
AgentTraining/
├── chat_test.py        first scratch — one Ollama HTTP call
├── my_loop.py          second scratch — hand-rolled tool loop, ~50 lines
├── agent_starter.md    original spec that drove the agent/ build
├── agent/              artefact #1 — hand-rolled, no framework
└── lib_agent/          artefact #2 — LangGraph + LangChain rebuild
```

The two scratches at the top (`chat_test.py`, `my_loop.py`) are the
"hello world" warm-ups: one chat call, then one chat call inside a loop
with one tool. They are kept on purpose as the lowest rung of the ladder.

---

## `agent/` — hand-rolled, no framework

A minimal-but-complete agent built sequentially from `agent_starter.md`.
Goal: understand every byte of the agent loop without a framework hiding
the wiring.

- `loop.py` (under 100 lines): `send → branch on tool calls → repeat`,
  capped by `MAX_TURNS`, every tool result appended (errors included so
  the model can recover), per-tool wall-clock timeout.
- 7 tools, one module each (`tools/`): `get_current_time`, `web_search`
  (Tavily), `web_fetch` (httpx + trafilatura), `read_file` / `write_file`
  (workspace-sandboxed), `run_python` (Docker `--network none --read-only`,
  30 s wall clock), `search_documents` (LanceDB + `nomic-embed-text` RAG),
  plus `finish` for explicit termination.
- Pydantic input schemas per tool, auto-converted to the model-facing tool
  spec.
- Path-traversal defenses on the filesystem tools, tested in
  `test_tools.py`.
- JSONL session log per run under `agent/logs/` (one event per line:
  user, model call, tool call, reply — with latencies). The first thing
  to look at when the agent does something weird.

Build order, configuration, safety notes, and the `nomic-embed-text`
caveats are all in [`agent/readme.md`](agent/readme.md). The original
spec is [`agent_starter.md`](agent_starter.md).

Run from `agent/`:

```bash
pip install -r requirements.txt
ollama pull qwen3:8b nomic-embed-text
docker pull python:3.11-slim
python main.py
```

---

## `lib_agent/` — LangGraph + LangChain rebuild

Same hardware target, same model, but rebuilt on top of LangGraph to
demonstrate the full set of primitives a Claude-Code-class agent needs
on consumer hardware.

What the framework version adds over `agent/`:

| Capability | Mechanism |
|---|---|
| Tool-using agent loop | LangGraph `StateGraph` + `bind_tools` + serial tool node |
| Multi-agent | Supervisor + `research_agent` / `code_agent` worker subgraphs |
| Streaming | `app.stream(stream_mode=["messages","updates"])` interleave |
| Persistence across processes | `SqliteSaver` on `checkpoints.sqlite` |
| Persistent facts (across threads) | `facts.json`, auto-injected into the system prompt |
| Context summarization | `prune_node` summarizes at 8K-token threshold |
| HITL approval gates | Two-phase `interrupt()` on destructive tools, `Command(resume=...)` |
| Rich REPL UX | Slash commands, status bar, per-turn token footer, `/debug` |
| Observability | Phoenix on `:6006` via OpenInference auto-instrumentation |
| Eval harness | 30 cases × 6 gates (tool / error / contains / safe / latency / artifact) |

17 tools (the same 7 as `agent/` plus an enriched filesystem surface,
memory tools, and richer error contracts). Single-agent and supervisor
graphs share the same `prune_node` and serial tool node.

Run from `lib_agent/`:

```bash
make install
make chat            # single-agent REPL
make chat-multi      # multi-agent supervisor REPL
make eval            # full 30-case suite
make phoenix-up      # start observability
```

`lib_agent/docs/` is the documentation hub:

- [`specs.md`](lib_agent/docs/specs.md) — canonical reference: every
  capability, tool, config knob, gotcha.
- [`companion.md`](lib_agent/docs/companion.md) — conceptual
  walkthrough: abstractions, interfaces between components,
  representative end-to-end flows (single-agent turn, HITL approval,
  context summarization, multi-agent dispatch).
- [`perspectives.md`](lib_agent/docs/perspectives.md) — original
  forward-looking plan (now mostly delivered).
- [`assessment_2026-05-01.md`](lib_agent/docs/assessment_2026-05-01.md) —
  comparison vs Claude Code.
- [`port_to_openai.md`](lib_agent/docs/port_to_openai.md) — notes on
  swapping the local Ollama backend for the OpenAI API.

---

## Reading order

If you're picking the repo up cold and want to follow the actual
learning arc:

1. `chat_test.py` — confirm Ollama works.
2. `my_loop.py` — one tool, one loop, no abstractions.
3. `agent_starter.md` — the spec that everything below was built from.
4. `agent/` — read `loop.py`, then one tool module, then `agent/readme.md`.
5. `lib_agent/smoke.py` — the simplest possible LangGraph agent (~25 lines).
6. `lib_agent/docs/companion.md` §1–§3 — the mental model.
7. `lib_agent/agent.py` — production single-agent: `smoke.py` plus prune,
   serial tool node with HITL, 17 tools.
8. `lib_agent/multi_agent.py` — two of those subgraphs as nodes in a
   third graph, with a routing supervisor.
9. `lib_agent/eval/runner.py` — how regressions get caught.

---

## Requirements

- NVIDIA GPU with 8 GB+ VRAM (CPU-only works but is slow).
- Python 3.10+ for `agent/`, 3.11+ for `lib_agent/`.
- [Ollama](https://ollama.com), with `qwen3:8b` and `nomic-embed-text` pulled.
- Docker, for the `run_python` sandbox.
- Tavily API key in each artefact's `.env` (`TAVILY_API_KEY=...`),
  for web search.

`.env` files are gitignored in both artefacts.

---

## License

Personal learning project. Use however you like.
