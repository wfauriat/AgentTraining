# eval/runner.py — runs the golden set against the agent, scores rule-based,
# writes a JSON report and a one-line-per-case CLI summary. Phoenix captures
# the traces automatically (via observability.setup); failures are clickable
# in the UI for root-cause inspection.
#
# Usage from lib_agent/:
#   python -m eval.runner
#   python -m eval.runner --filter rag
#   python -m eval.runner --skip-categories network

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from agent import graph
from observability import flush, setup as setup_obs

from .golden import GOLDEN

SYSTEM = SystemMessage(
    content="You are a helpful assistant. Use tools when relevant. Be brief."
)


def extract_tool_calls(messages: list) -> list[str]:
    """All tool names invoked across the message history (in order)."""
    return [
        call["name"]
        for m in messages
        if isinstance(m, AIMessage) and m.tool_calls
        for call in m.tool_calls
    ]


def extract_tool_results(messages: list) -> list[str]:
    """All ToolMessage contents (str) — what the model saw back from each tool."""
    return [str(m.content) for m in messages if isinstance(m, ToolMessage)]


def get_final_text(messages: list) -> str:
    """Last non-empty AI text reply in the history."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and str(m.content).strip():
            return str(m.content)
    return ""


def score_case(
    case: dict, tool_calls: list[str], tool_results: list[str], final_text: str, elapsed: float
) -> dict:
    """Five independent gates. All must pass for `overall` to be True."""
    # 1. correct tool was (or wasn't) called
    expect = case.get("expect_tool")
    if expect is None:
        tool_pass = len(tool_calls) == 0
    else:
        tool_pass = expect in tool_calls

    # 2. tool returned an error string when expected (negative tests)
    if case.get("expect_tool_result_error"):
        # our tools encode failures as "[error: ...]" strings
        tool_error_pass = any(r.lower().startswith("[error") or "[error" in r.lower() for r in tool_results)
    else:
        tool_error_pass = True

    # 3. final text contains all required substrings
    needles = case.get("expect_text_contains") or []
    haystack = final_text.lower()
    text_pass = all(s.lower() in haystack for s in needles)

    # 4. final text contains NONE of the forbidden substrings (no leaks)
    forbidden = case.get("expect_text_NOT_contains") or []
    text_safe = not any(s.lower() in haystack for s in forbidden)

    # 5. latency under budget (if any)
    budget = case.get("max_seconds")
    latency_pass = budget is None or elapsed <= budget

    # 6. forbidden artifact: a path that MUST NOT exist after the test
    #    (catches "sandbox bypassed" regardless of what the agent claims)
    forbidden = case.get("forbidden_artifact")
    artifact_pass = forbidden is None or not os.path.exists(forbidden)

    overall = tool_pass and tool_error_pass and text_pass and text_safe and latency_pass and artifact_pass

    return {
        "tool_pass": tool_pass,
        "tool_error_pass": tool_error_pass,
        "text_pass": text_pass,
        "text_safe": text_safe,
        "latency_pass": latency_pass,
        "artifact_pass": artifact_pass,
        "overall": overall,
        "tool_calls_seen": tool_calls,
        "tool_results_preview": [r[:160] for r in tool_results],
        "final_text_preview": final_text[:240],
    }


def run_one(app, case: dict) -> dict:
    thread_id = f"eval-{case['id']}-{uuid4().hex[:6]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    msgs = [SYSTEM, HumanMessage(content=case["prompt"])]
    t0 = time.time()
    state = app.invoke({"messages": msgs}, config=config)
    elapsed = time.time() - t0
    tool_calls = extract_tool_calls(state["messages"])
    tool_results = extract_tool_results(state["messages"])
    final = get_final_text(state["messages"])
    score = score_case(case, tool_calls, tool_results, final, elapsed)
    return {
        "id": case["id"],
        "category": case["category"],
        "prompt": case["prompt"],
        "expect_tool": case.get("expect_tool"),
        "expect_text_contains": case.get("expect_text_contains"),
        "expect_text_NOT_contains": case.get("expect_text_NOT_contains"),
        "expect_tool_result_error": case.get("expect_tool_result_error"),
        "max_seconds": case.get("max_seconds"),
        "elapsed_s": round(elapsed, 2),
        "thread_id": thread_id,
        **score,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=None, help="substring filter on case ids")
    parser.add_argument(
        "--skip-categories", default="", help="comma-separated categories to skip (e.g. network)"
    )
    args = parser.parse_args()

    setup_obs()  # enable Phoenix auto-instrumentation
    app = graph.compile(checkpointer=MemorySaver())

    skip = {s for s in args.skip_categories.split(",") if s}
    cases = [c for c in GOLDEN if not args.filter or args.filter in c["id"]]
    cases = [c for c in cases if c["category"] not in skip]

    print(f"running {len(cases)} cases (project: lib_agent → http://localhost:6006)")
    print("gates:  T=tool E=tool_error C=text_contains S=text_safe L=latency A=artifact  (lowercase = fail)\n")
    results: list[dict] = []
    for c in cases:
        prefix = f"  {c['id']:28s}  ({c['category']:10s})  "
        print(prefix, end="", flush=True)
        try:
            r = run_one(app, c)
            mark = "✓" if r["overall"] else "✗"
            tool_seen = ",".join(r["tool_calls_seen"]) or "none"
            # build a one-letter-per-gate summary so failures are diagnosable
            gates = "".join([
                "T" if r["tool_pass"] else "t",
                "E" if r["tool_error_pass"] else "e",
                "C" if r["text_pass"] else "c",
                "S" if r["text_safe"] else "s",
                "L" if r["latency_pass"] else "l",
                "A" if r["artifact_pass"] else "a",
            ])
            print(f"{mark} {gates}  {r['elapsed_s']:5.1f}s  tools=[{tool_seen}]")
            results.append(r)
        except Exception as e:
            print(f"!  exception: {type(e).__name__}: {e}")
            results.append({**c, "error": str(e), "overall": False})

    flush()  # push any buffered traces to Phoenix before exit

    passed = sum(1 for r in results if r.get("overall"))
    print(f"\n{passed}/{len(results)} passed")

    reports_dir = Path("eval/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
