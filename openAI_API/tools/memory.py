# tools/memory.py — persistent flat-KV facts store.
#
# A different memory tier from conversation state:
#   - checkpoints.sqlite : per-thread conversation messages (LangGraph SqliteSaver)
#   - state["summary"]   : per-thread rolling summary of pruned messages
#   - facts.json         : per-USER, per-PROJECT, across-all-threads facts
#                          (this file)
#   - vector_db/         : indexed knowledge corpus for RAG
#
# Use facts for things the user wants the agent to remember forever
# (preferences, project context, glossary terms). Facts are auto-injected
# into the model's system context on every turn — see agent.call_model.

import json
from pathlib import Path

from langchain_core.tools import tool

FACTS_PATH = Path(__file__).parent.parent / "facts.json"


def load_facts() -> dict[str, str]:
    """Read facts.json into a dict. Returns {} if missing or unreadable."""
    if not FACTS_PATH.exists():
        return {}
    try:
        data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_facts(facts: dict[str, str]) -> None:
    FACTS_PATH.write_text(
        json.dumps(facts, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def render_facts() -> str:
    """Format current facts as a bullet list for system-prompt injection.
    Returns the empty string if no facts exist."""
    facts = load_facts()
    if not facts:
        return ""
    lines = [f"- {k}: {v}" for k, v in sorted(facts.items())]
    return "\n".join(lines)


@tool
def remember(key: str, value: str) -> str:
    """Store a persistent fact (preference, project context, glossary term).
    Facts survive process restarts and are visible across every conversation.

    Args:
        key: short slug naming the fact (e.g. 'favorite_color', 'project').
        value: the fact's value — any short text.
    """
    facts = load_facts()
    facts[key] = value
    _save_facts(facts)
    return f"Remembered {key!r}: {value}"


@tool
def forget(key: str) -> str:
    """Delete a previously remembered fact.

    Args:
        key: the fact's name.
    """
    facts = load_facts()
    if key in facts:
        del facts[key]
        _save_facts(facts)
        return f"Forgot {key!r}"
    return f"No fact named {key!r} was stored."


@tool
def recall(key: str = "") -> str:
    """Retrieve a stored fact by key, or list all keys if no key given.

    Args:
        key: the fact's name. If empty, returns the list of all keys.
    """
    facts = load_facts()
    if not facts:
        return "[no facts stored]"
    if not key:
        return "Stored fact keys: " + ", ".join(sorted(facts))
    if key in facts:
        return f"{key}: {facts[key]}"
    return f"[no fact named {key!r}]"
