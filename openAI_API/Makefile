# Convenience targets. Run `make help` for the menu.

PY := ../lib_agent/.venv/bin/python
PIP := ../lib_agent/.venv/bin/pip
PHOENIX_COMPOSE := observability/docker-compose.phoenix.yml

.DEFAULT_GOAL := help

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' Makefile | awk -F':.*## ' '{printf "  %-22s %s\n", $$1, $$2}'

# ── REPL ──────────────────────────────────────────────────────────────────

chat:  ## interactive REPL (single-agent)
	$(PY) chat.py

chat-multi:  ## interactive REPL (multi-agent supervisor)
	$(PY) chat.py --multi

# ── eval ──────────────────────────────────────────────────────────────────

eval:  ## run the full eval suite
	$(PY) -m eval.runner

eval-quick:  ## eval skipping network and negative-timeout categories
	$(PY) -m eval.runner --skip-categories network

# ── RAG / corpus ──────────────────────────────────────────────────────────

index:  ## (re)build the LanceDB vector index from corpus/
	$(PY) -m scripts.index_docs

# ── persisted state ───────────────────────────────────────────────────────

info:  ## show checkpoint db + phoenix status
	$(PY) -m admin info

list-threads:  ## list checkpoint thread_ids
	$(PY) -m admin list-threads

list-facts:  ## list persisted facts (facts.json)
	$(PY) -m admin list-facts

purge-state:  ## drop the checkpoints.sqlite database
	$(PY) -m admin purge-all --yes

purge-facts:  ## drop facts.json
	$(PY) -m admin purge-facts --yes

purge-traces:  ## nuke phoenix traces (volume reset)
	$(PY) -m admin purge-traces --yes

# ── observability stack ───────────────────────────────────────────────────

phoenix-up:  ## start the phoenix container
	docker compose -f $(PHOENIX_COMPOSE) up -d

phoenix-down:  ## stop the phoenix container (keeps the volume)
	docker compose -f $(PHOENIX_COMPOSE) down

phoenix-logs:  ## tail the phoenix container logs
	docker compose -f $(PHOENIX_COMPOSE) logs -f phoenix

# ── deps ──────────────────────────────────────────────────────────────────

install:  ## (re)install python dependencies into .venv
	$(PIP) install -r requirements.txt

.PHONY: help chat chat-multi eval eval-quick index info list-threads list-facts \
        purge-state purge-facts purge-traces phoenix-up phoenix-down phoenix-logs install
