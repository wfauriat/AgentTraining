#!/usr/bin/env python3
"""Convert a Claude Code session JSONL into a readable Markdown transcript.

Usage:
    python transcript_to_md.py [--input PATH] [--output PATH] [--include-thinking] [--max-tool-result-chars N]

Defaults to converting the most recent session under
~/.claude/projects/-home-ai-user-Documents-Sandbox-AgentTraining-lib-agent/.

The Claude Code JSONL has 9 event types observed in this session:
  user, assistant, system, attachment, ai-title, file-history-snapshot,
  permission-mode, queue-operation, last-prompt
We render user + assistant; everything else is metadata and gets skipped.

Within an assistant message, content is a list of blocks:
  text         → rendered as the assistant's reply prose
  tool_use     → rendered as a code block with the tool name + JSON args
  thinking     → skipped by default (toggle with --include-thinking)

Within a user message, content is either a plain string or a list whose
blocks can be tool_result entries (truncated for readability).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

DEFAULT_PROJECT_DIR = Path.home() / ".claude/projects/-home-ai-user-Documents-Sandbox-AgentTraining-lib-agent"


def _newest_jsonl(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no *.jsonl in {directory}")
    return candidates[0]


def _format_timestamp(ts: str) -> str:
    """ISO 8601 → short local-zone display."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n…[truncated at {max_chars} chars; full output in JSONL]"


def _render_user(entry: dict, max_tool_result_chars: int) -> str:
    """User-typed content gets a visually prominent treatment so it's easy
    to scan past the much larger volume of tool-call traffic. Tool-result
    entries (LangChain's "user" role for tool returns) are de-emphasized in
    a collapsible block."""
    msg = entry.get("message", {})
    content = msg.get("content", "")

    # Plain user prompt — most common case.
    if isinstance(content, str):
        cleaned = content.strip()
        if cleaned.startswith("<local-command-caveat>"):
            return None  # skip the caveat-only entries
        if cleaned.startswith("<command-name>"):
            # User ran a slash-style local command (e.g., /context); concise note.
            cmd = cleaned.split(">")[1].split("<")[0] if ">" in cleaned else "?"
            return f"\n---\n\n## ▶ User local command  `/{cmd}`\n"
        # Real user prompt: HR + ## header + visible marker for scrolling.
        return f"\n---\n\n## ▶ You\n\n{cleaned}\n"

    # Array content — usually tool_result blocks.
    parts: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "tool_result":
            tool_name = block.get("tool_use_id", "?")
            result = block.get("content", "")
            if isinstance(result, list):
                result = "\n".join(b.get("text", "") for b in result if isinstance(b, dict))
            result = _truncate(str(result), max_tool_result_chars)
            parts.append(f"<details><summary>tool result ({tool_name[:8]})</summary>\n\n```\n{result}\n```\n\n</details>")
        elif btype == "text":
            parts.append(block.get("text", ""))
    if not parts:
        return None
    # Tool-result-only user entries: rendered de-emphasized, no big header.
    return "\n\n".join(parts) + "\n"


def _render_assistant(entry: dict, include_thinking: bool, max_tool_result_chars: int) -> str:
    msg = entry.get("message", {})
    content = msg.get("content", [])
    parts: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(text)
        elif btype == "tool_use":
            name = block.get("name", "?")
            args = block.get("input", {})
            args_str = json.dumps(args, indent=2, ensure_ascii=False)
            args_str = _truncate(args_str, max_tool_result_chars)
            parts.append(f"**Tool call: `{name}`**\n\n```json\n{args_str}\n```")
        elif btype == "thinking":
            if include_thinking:
                t = block.get("thinking", "").strip()
                if t:
                    parts.append(f"<details><summary>thinking</summary>\n\n{t}\n\n</details>")
    if not parts:
        return None
    return "### Assistant\n\n" + "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="JSONL file (default: newest in project dir)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "session_transcript.md")
    parser.add_argument("--include-thinking", action="store_true", help="include assistant <thinking> blocks (verbose)")
    parser.add_argument("--max-tool-result-chars", type=int, default=1500, help="truncate large tool outputs")
    args = parser.parse_args()

    src = args.input or _newest_jsonl(DEFAULT_PROJECT_DIR)

    out_lines: list[str] = []
    out_lines.append(f"# Claude Code session transcript")
    out_lines.append("")
    out_lines.append(f"**Source:** `{src}`  ")
    out_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    out_lines.append(f"**Thinking blocks:** {'included' if args.include_thinking else 'omitted'}  ")
    out_lines.append(f"**Tool output truncation:** {args.max_tool_result_chars} chars")
    out_lines.append("")
    out_lines.append("---")
    out_lines.append("")

    counts = {"user": 0, "assistant": 0, "skipped": 0, "errors": 0}
    last_ts: str | None = None

    with src.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                counts["errors"] += 1
                continue

            etype = entry.get("type")
            ts = entry.get("timestamp")
            block: str | None = None

            if etype == "user":
                # Skip pure-meta entries (caveats with no real content)
                if entry.get("isMeta"):
                    counts["skipped"] += 1
                    continue
                block = _render_user(entry, args.max_tool_result_chars)
                if block is not None:
                    counts["user"] += 1
            elif etype == "assistant":
                block = _render_assistant(entry, args.include_thinking, args.max_tool_result_chars)
                if block is not None:
                    counts["assistant"] += 1
            else:
                counts["skipped"] += 1
                continue

            if block is None:
                counts["skipped"] += 1
                continue

            # Light timestamp header at hour boundaries to chunk the transcript visually
            if ts and (last_ts is None or ts[:13] != last_ts[:13]):
                out_lines.append(f"\n<small>*— {_format_timestamp(ts)} —*</small>\n")
                last_ts = ts

            out_lines.append(block)
            out_lines.append("")  # spacer

    out_lines.append("\n---\n")
    out_lines.append(
        f"*{counts['user']} user turns · {counts['assistant']} assistant turns · "
        f"{counts['skipped']} non-content events skipped"
        f"{f' · {counts['errors']} JSON errors' if counts['errors'] else ''}*"
    )

    args.output.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote {args.output}  ({args.output.stat().st_size:,} bytes)")
    print(f"counts: {counts}")


if __name__ == "__main__":
    main()
