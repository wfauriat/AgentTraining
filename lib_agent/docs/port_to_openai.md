# Port: Ollama / Qwen3 → OpenAI-compatible endpoint / Mistral Small

**Target deployment:** air-gapped (private network, no public internet).
**Source:** `lib_agent/` as of 2026-05-01 (post UX/UI pass).
**Endpoint shape assumed:** OpenAI-compatible `/v1/chat/completions` (likely
vLLM or `llama.cpp` server). Optionally `/v1/embeddings`.

This document is a **stepwise port checklist**. Run on a connected machine
first (Phase 0–5), package for air-gap (Phase 6), deploy and validate
(Phase 7). Each step has explicit file:line touch points where possible.

The eval harness is the deployment validation tool — if 30/30 cases still
pass after the port, the port worked.

---

## Phase 0 — preflight on the connected machine (~15 min)

Goal: confirm endpoint shape and capabilities before touching code.

- [ ] **Identify the served model name**:
  ```bash
  curl -s "$OPENAI_BASE_URL/models" | jq .
  ```
  Expect at least one model object with `"id": "mistral-small-..."` (exact
  name varies). Record this string — it's what you'll set `MODEL` to.

- [ ] **Confirm chat completions work** (raw HTTP, no LangChain yet):
  ```bash
  curl -s "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model": "<NAME>", "messages": [{"role":"user","content":"say ok"}]}' | jq .
  ```
  Expect a `choices[0].message.content` string.

- [ ] **Test tool calling at the wire level**:
  ```bash
  curl -s "$OPENAI_BASE_URL/chat/completions" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model":"<NAME>",
      "messages":[{"role":"user","content":"what time is it?"}],
      "tools":[{"type":"function","function":{"name":"get_time","description":"current time","parameters":{"type":"object","properties":{}}}}]
    }' | jq '.choices[0].message'
  ```
  Expect `tool_calls` populated. If not — the endpoint doesn't expose tool
  calling and the entire approach changes (you'd need a JSON-mode prompt
  pattern instead). **This is the load-bearing capability.**

- [ ] **Test embeddings endpoint** (optional path A):
  ```bash
  curl -s "$OPENAI_BASE_URL/embeddings" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"<EMBED_NAME>","input":"hello"}' | jq '.data[0].embedding | length'
  ```
  - If 200 OK with a vector → **path A** (OpenAIEmbeddings).
  - If 404 / not implemented → **path B** (HuggingFaceEmbeddings local).

- [ ] **Test streaming** (`"stream": true`). Confirm SSE works.
  Optional but useful — if streaming doesn't work, drop `stream_mode="messages"`
  from chat.py's render call and only use `"updates"`. Token-by-token UX
  is sacrificed; nothing else breaks.

- [ ] **Note timeouts you observe**. vLLM is fast; per-call latency is
  often <2s for tool selection, <10s for prose generation. Set `MODEL_TIMEOUT`
  and `TURN_TIMEOUT` accordingly (suggested: 30 / 90 — half the Ollama values).

---

## Phase 1 — minimal swap of LLM provider (~30 min)

```bash
git checkout -b openai-port
```

### File changes

- [ ] **`requirements.txt`**:
  - Remove: `langchain-ollama>=0.3`, `langfuse>=2.50,<3` (already removed)
  - Add: `langchain-openai>=0.3`
  - Keep everything else.

- [ ] **`config.py`** — top of file:
  ```python
  import os
  MODEL = os.getenv("LIB_AGENT_MODEL", "mistral-small-latest")
  OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://internal-llm.corp/v1")
  OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")  # vLLM accepts any string
  # Drop these (Ollama-specific; vLLM/llama.cpp manage their own):
  # NUM_CTX = ...   ← remove
  # KEEP_ALIVE = ...← remove
  ```
  Adjust later in the file:
  ```python
  TOOL_TIMEOUT = 30        # unchanged, Docker sandbox cap
  MODEL_TIMEOUT = 30       # was 90; vLLM is fast
  TURN_TIMEOUT = 90        # was 180; halve since model is faster
  ```

- [ ] **`agent.py`** — top imports:
  - Remove: `from langchain_ollama import ChatOllama`
  - Add:    `from langchain_openai import ChatOpenAI`
  - Update the `from config import (...)` block: remove `KEEP_ALIVE` and
    `NUM_CTX`, add `OPENAI_API_KEY, OPENAI_BASE_URL`.

  Around the two `ChatOllama(...)` constructions (`llm` and
  `summarizer_llm`):
  ```python
  llm = ChatOpenAI(
      model=MODEL,
      temperature=0,
      base_url=OPENAI_BASE_URL,
      api_key=OPENAI_API_KEY,
      timeout=MODEL_TIMEOUT,
      max_retries=2,
  ).bind_tools(TOOLS)

  summarizer_llm = ChatOpenAI(
      model=SUMMARY_MODEL,
      temperature=0,
      base_url=OPENAI_BASE_URL,
      api_key=OPENAI_API_KEY,
      timeout=MODEL_TIMEOUT,
      max_retries=2,
  )
  ```

- [ ] **`multi_agent.py`** — same import swap, same constructor pattern at
  two sites: `worker_llm` (in `_build_worker`) and `supervisor_llm`.

- [ ] **`smoke.py`** — also has `ChatOllama`. Either port it for
  consistency or delete (it was a Day-1 reference; no longer used).

### Validation

- [ ] `pip install -r requirements.txt` (on connected machine).
- [ ] `python -c "import agent, multi_agent, chat; print('imports OK')"`
- [ ] `make chat` — single quick prompt: `What time is it?`. Confirm tool
  call fires, reply renders.

If single-agent works at this point, **the core of the port is done**.

---

## Phase 2 — embeddings (~20–60 min depending on path)

### Path A — endpoint serves `/v1/embeddings`

- [ ] **`embeddings.py`** — replace the entire file:
  ```python
  # embeddings.py — embeddings via the same OpenAI-compatible endpoint.
  from langchain_openai import OpenAIEmbeddings
  from config import EMBED_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL

  embeddings = OpenAIEmbeddings(
      model=EMBED_MODEL,
      base_url=OPENAI_BASE_URL,
      api_key=OPENAI_API_KEY,
  )
  ```
  Drop the `NomicEmbeddings` subclass entirely.

  **About prefixes**: `nomic-embed-text` needed `search_query:` /
  `search_document:` prefixes. The replacement embedding model probably
  does NOT (most modern embedders don't). Check the served model's docs.
  If it does (e.g., `intfloat/e5-mistral-7b-instruct`, BGE), wrap into a
  subclass like the original `NomicEmbeddings` but with the new prefixes.

- [ ] **`config.py`** — set `EMBED_MODEL` to whatever the endpoint serves
  (e.g., `"BAAI/bge-base-en-v1.5"` or whatever model name appears in
  `/v1/models`).

- [ ] **`tools/docs.py`** — change:
  - `from embeddings import NomicEmbeddings` → `from embeddings import embeddings`
  - In `_ensure_ready`, replace `_embedder = NomicEmbeddings(model=EMBED_MODEL)` with
    `_embedder = embeddings`

- [ ] **`scripts/index_docs.py`** — same swap pattern.

### Path B — endpoint doesn't serve embeddings (use local HuggingFace)

- [ ] **`requirements.txt`** add:
  ```
  langchain-huggingface>=0.1
  sentence-transformers>=3.0
  ```

- [ ] **`embeddings.py`**:
  ```python
  from langchain_huggingface import HuggingFaceEmbeddings
  from pathlib import Path

  EMBED_MODEL = "BAAI/bge-base-en-v1.5"
  CACHE_DIR = str(Path(__file__).parent / "_models")

  embeddings = HuggingFaceEmbeddings(
      model_name=EMBED_MODEL,
      cache_folder=CACHE_DIR,
      model_kwargs={"device": "cpu"},  # or "cuda" if GPU available
      encode_kwargs={"normalize_embeddings": True},
  )
  ```

- [ ] **Pre-stage the model on the connected machine**:
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; \
             SentenceTransformer('BAAI/bge-base-en-v1.5', cache_folder='./_models')"
  ```
  This populates `_models/` with the weights. Add `_models/` to your
  air-gap transfer payload.

### After either path

- [ ] **Reindex the corpus**:
  ```bash
  rm -rf vector_db/
  make index
  ```
  The vector dimension and similarity behavior change with the embedder, so
  the old index is unusable.

- [ ] **Smoke RAG**: `make chat` →
  ```
  Use the local corpus to define photosynthesis in one sentence.
  ```
  Should call `search_documents` and return relevant content.

---

## Phase 3 — web tools decision (~15 min)

The web tools (`tools/web.py`: `web_search`, `web_fetch`) need internet.
Three options:

### 3.A — Drop them entirely (recommended for air-gap)

- [ ] **`agent.py`** — `TOOLS` list: remove `web_search`, `web_fetch`.
- [ ] **`multi_agent.py`** — `RESEARCH_TOOLS`: remove the same.
- [ ] **Adjust `RESEARCH_SYSTEM` prompt** in `prompts.py` to drop mentions
  of web tools. Example: `"Your tools: search_documents, get_current_time, recall."`
- [ ] **`eval/golden.py`** — remove the two `network` cases (`web_search`,
  `web_fetch_wikipedia`) OR keep them and always invoke
  `--skip-categories network`.
- [ ] **`tools/web.py`** — leave on disk for reference, just unimported.
- [ ] **`requirements.txt`** — `trafilatura`, `python-dotenv` can be dropped
  if no other consumer uses them. `httpx` stays (langchain depends on it).

### 3.B — Replace with internal corporate tools

- [ ] Write new `@tool`-decorated functions for whatever's available
  (Confluence, internal search, internal Wikipedia mirror). Pattern in
  `tools/web.py:web_fetch` is the template. Keep the same input-validation
  and `[error: ...]` return idioms.

### 3.C — Keep but mark as no-op

- [ ] In `tools/web.py`, replace each tool body with
  `return "[error: web access not available in this deployment]"`. Keeps
  the tool surface stable (no model-prompt changes needed) but model
  learns the tool fails.

---

## Phase 4 — eval calibration (~30 min)

The 30 golden cases were tuned to Qwen3 8B's wording. Mistral Small
phrases differently. Run once and tune.

- [ ] **First baseline run**:
  ```bash
  make purge-state
  python -m eval.runner --skip-categories network > /tmp/port_eval_1.log 2>&1
  grep -E '^(running|gates|  [a-z]|[0-9]+/|report)' /tmp/port_eval_1.log
  ```

- [ ] **For each failing case**, inspect:
  - **`l` (latency) fail** → adjust `max_seconds` in golden.py for that case.
    vLLM is fast; if a case used to take 30s and now takes 8s, tighten the
    budget (catches regressions earlier). If a case fails because Mistral is
    *more* verbose (longer reply = more output tokens to render), loosen.
  - **`c` (text_contains) fail** → the model phrased the answer differently.
    Either tighten the prompt to demand the exact phrase, or relax the
    expectation (e.g., remove the `expect_text_contains` and rely on
    other gates).
  - **`t` (tool selection) fail** → the model picked a different tool or
    didn't call one. Update the prompt or `expect_tool`.

- [ ] **Re-run until 30/30 (or 28/30 if you skip network)**.

- [ ] **Optional: restore `with_structured_output` for supervisor**.
  We replaced it with keyword-routing because Qwen3 8B was unreliable.
  Mistral Small via vLLM probably handles it fine. Quick test in
  `multi_agent.py:supervisor_node`:
  ```python
  # Try this first; if it works reliably, drop _pick_keyword + _parse_supervisor:
  supervisor_llm = ChatOpenAI(...).with_structured_output(Route, method="function_calling")
  ```
  If it works, the supervisor logic simplifies considerably.

---

## Phase 5 — observability sanity check (~5 min)

Phoenix should "just work" — OpenInference instruments LangChain Runnables
regardless of model provider. Verify:

- [ ] `docker compose -f observability/docker-compose.phoenix.yml up -d`
- [ ] After `make eval`, check `http://localhost:6006` for trace tree.
  ChatOpenAI spans should appear with same shape as ChatOllama did.

- [ ] **Strip the OpenInference monkey-patch** if no longer needed.
  In `observability.py:setup`, the patch for `on_interrupt`/`on_resume`
  was added because OpenInference's tracer didn't implement them. Test
  whether the upstream caught up — if yes, remove the patch. If no,
  leave it (it's a no-op until the dispatcher calls those hooks).

---

## Phase 6 — offline staging (~1–2 hours)

Goal: produce a single tarball that brings the agent to a machine with no
internet.

- [ ] **Wheelhouse**: on a connected machine with same Python (3.12) and
  glibc as the target:
  ```bash
  rm -rf wheelhouse && mkdir wheelhouse
  pip download -r requirements.txt -d wheelhouse/
  ```

- [ ] **Docker images**:
  ```bash
  docker pull arizephoenix/phoenix:latest
  docker pull python:3.11-slim
  docker save arizephoenix/phoenix:latest python:3.11-slim \
    -o docker-images.tar
  ```

- [ ] **Embedding model** (only path B):
  ```bash
  # Ran in Phase 2.B — confirm _models/ is present
  du -sh _models/    # should be ~500 MB for bge-base
  ```

- [ ] **RAG corpus**: confirm `corpus/` is committed to the repo.

- [ ] **Compose tarball**:
  ```bash
  tar czf lib_agent_offline.tar.gz \
    --exclude='.venv' \
    --exclude='checkpoints.sqlite*' \
    --exclude='vector_db' \
    --exclude='workspace' \
    --exclude='facts.json' \
    --exclude='session_transcript.md' \
    --exclude='.git' \
    lib_agent/ wheelhouse/ docker-images.tar
  ```
  (Adjust the `lib_agent/` prefix if you tar from inside the repo.)

- [ ] **Document the deploy procedure** in a short `deploy.sh` or
  `DEPLOY.md`:
  ```bash
  # On target machine:
  tar xzf lib_agent_offline.tar.gz
  cd lib_agent

  python3 -m venv .venv
  .venv/bin/pip install --no-index --find-links=../wheelhouse -r requirements.txt

  docker load -i ../docker-images.tar

  # Start observability (optional, depends on policy)
  docker compose -f observability/docker-compose.phoenix.yml up -d

  # Build RAG index
  python -m scripts.index_docs

  # Smoke test
  make eval --skip-categories network
  ```

---

## Phase 7 — deploy + validate (~30 min on target)

- [ ] **Smoke imports**: `python -c "import agent, multi_agent, chat"` — must
  succeed before any further test.

- [ ] **Endpoint reachability**: from inside the target machine,
  ```bash
  curl -sf "$OPENAI_BASE_URL/models" >/dev/null && echo OK || echo FAIL
  ```

- [ ] **Single-agent smoke**: `make chat` → `What time is it?` → quit.

- [ ] **Multi-agent smoke**: `make chat-multi` → "Use the corpus to define
  photosynthesis, then save to bio.txt" → approve write → confirm file
  exists in `workspace/`.

- [ ] **Full eval**:
  ```bash
  make purge-state
  python -m eval.runner --skip-categories network
  ```
  Expected: 28/28 pass (or 30/30 if you kept network behind an internal
  search backend). Any failure → triage with the gate-letter diagnosis.

- [ ] **Phoenix UI check**: open `http://target:6006` (port-forwarded if
  needed) and confirm trace tree renders.

- [ ] **Persistence check**: `make chat`, `My favorite color is teal`,
  remember triggers, quit. `make chat` again, ask "what's my favorite
  color?" — should answer "teal" via auto-injection.

---

## Reference: file-by-file touch list

| File | Change | Phase |
|---|---|---|
| `requirements.txt` | swap ollama→openai; add HF if path B | 1, 2 |
| `config.py` | drop KEEP_ALIVE/NUM_CTX; add OPENAI_*, MODEL update; tighten timeouts | 1 |
| `agent.py` | imports + 2 ChatOpenAI constructors | 1 |
| `multi_agent.py` | imports + 2 ChatOpenAI constructors | 1 |
| `embeddings.py` | replace NomicEmbeddings with OpenAIEmbeddings or HFEmbeddings | 2 |
| `tools/docs.py` | use new `embeddings` module export | 2 |
| `scripts/index_docs.py` | use new `embeddings` module export | 2 |
| `agent.py` (TOOLS) | remove web tools (3.A) | 3 |
| `multi_agent.py` (RESEARCH_TOOLS) | remove web tools (3.A) | 3 |
| `prompts.py` (RESEARCH_SYSTEM) | drop web mentions if 3.A | 3 |
| `eval/golden.py` | remove network cases or keep with skip flag | 3, 4 |
| `eval/golden.py` | tune `expect_text_contains` and `max_seconds` | 4 |
| `multi_agent.py` (supervisor) | optional: restore with_structured_output | 4 |
| `observability.py` | optional: drop OpenInference monkey-patch if upstream fixed | 5 |
| `tools/web.py` | leave or replace per 3.B | 3 |
| `smoke.py` | port or delete | 1 |
| `chat.py` | **no changes** — fully provider-agnostic | — |
| `tools/files.py` | **no changes** | — |
| `tools/python_sandbox.py` | **no changes** | — |
| `tools/memory.py` | **no changes** | — |
| `prompts.py` | **no changes** (except possibly RESEARCH_SYSTEM in 3.A) | — |
| `admin.py`, `Makefile` | **no changes** | — |

---

## Risks / known unknowns to surface during port

1. **vLLM tool-calling format quirks.** Some vLLM builds emit `tool_calls`
   in a slightly non-standard shape that LangChain's OpenAI adapter
   parses fine but logs warnings about. Confirm by checking Phoenix's
   span attributes: `llm.output_messages.0.message.tool_calls.0.tool_call.function.name`
   should be the tool name, not a JSON-encoded blob.

2. **Streaming SSE compatibility.** vLLM streaming is well-supported; some
   `llama.cpp` server builds had buggy SSE in older versions. If
   `stream_mode="messages"` produces no token-level chunks but
   `"updates"` works fine, drop `"messages"` from chat.py's stream call.

3. **Context window claim vs. actual.** Mistral Small's nominal context is
   32K, but the served deployment may set lower. If you hit silent
   truncation: lower `PRUNE_THRESHOLD_TOKENS` from 8000 to (deployed_ctx /
   2). The eval will surface this as flaky-looking model behavior on long
   threads.

4. **Internal CA certs for HTTPS endpoints.** If `OPENAI_BASE_URL` uses
   HTTPS with a private CA, set:
   ```python
   import httpx
   client = httpx.Client(verify="/etc/ssl/certs/internal-ca.pem")
   llm = ChatOpenAI(..., http_client=client)
   ```
   Same for embeddings if path A.

5. **Auth headers beyond Bearer**. Some internal proxies want `X-Api-Key`
   or similar. `ChatOpenAI(default_headers={"X-Api-Key": "..."})` handles it.

6. **Embedding dimension mismatch with old vector_db**. Always reindex
   after switching embedders. Stale `vector_db/` will return random matches
   (or crash with a dimension assertion).

7. **Tool descriptions tuned to Qwen3 quirks.** The "MUST call a tool"
   language in `CODE_SYSTEM` and the `\n`-anti-double-escape clause in
   `run_python` were Qwen3-8B-specific. They don't *break* Mistral but they
   are noise. Worth re-reading after port and trimming where the rationale
   doesn't apply.

8. **Phoenix trace volume.** vLLM's faster turnaround means more spans per
   wall-clock minute. Phoenix v2 server is fine at low scale but watch
   memory if you do extensive multi-day eval runs.

---

## Suggested session structure for the port

When you fire up another session on a connected machine:

1. Have it read `port_to_openai.md` (this file) first.
2. Tell it: "Port phase 1 is the minimum — do it and run a smoke test."
3. Then phase 2, then phase 4 (calibrate eval).
4. Phase 3 (web decision) is a five-minute conversation about "do you
   need them or drop them?".
5. Phases 6 and 7 are operational, can happen later when you actually
   have hardware to deploy onto.

The model in that session won't have this build's history; **that's fine**.
This file plus `docs/specs.md` plus `docs/companion.md` give it everything it
needs to execute the port without context. The session should chdir into the
repo root (where `agent.py`, `chat.py` etc. live) and reference
`docs/port_to_openai.md` from there.

If something during the port surprises you (a behavior different from
what's documented here), update this file as you go — it becomes the
record of "what we learned about the actual deployment."
