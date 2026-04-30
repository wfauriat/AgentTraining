# tools/python_sandbox.py — execute Python inside a hardened Docker container.
#
# Sandbox guarantees (set on every invocation):
#   --rm           container removed on exit (no leftover state)
#   --network none no internet, no host network
#   --read-only    rootfs is immutable from inside
#   --memory 256m  hard memory cap
#   --cpus 0.5     half a core
#   stdin pipe     code is fed on stdin to `python -` (no temp file on host)
# Wall-clock cap: subprocess.run(timeout=TOOL_TIMEOUT) — kills runaway code.

import subprocess

from langchain_core.tools import tool

from config import MAX_TOOL_RESULT_CHARS, TOOL_TIMEOUT


@tool
def run_python(code: str) -> str:
    """Execute Python code in an isolated Docker sandbox and return stdout.

    Use print() to produce output you want to read back. Important constraints:
      - NO filesystem access. Cannot read or write workspace files. To process
        a file's contents, use read_file first and pass the text into the code.
      - NO network access. No HTTP, no DNS.
      - State does NOT persist across calls — each call is a fresh container.
      - Hard limits: 256 MB memory, 0.5 CPU, 30 s wall clock.

    Args:
        code: Python source to execute (sent on stdin to `python -`).
    """
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "-i",
                "--network", "none",
                "--memory", "256m",
                "--cpus", "0.5",
                "--read-only",
                "python:3.11-slim",
                "python", "-",
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"[error: code execution timed out after {TOOL_TIMEOUT}s]"
    except FileNotFoundError:
        return "[error: docker not found on PATH]"
    except Exception as e:
        return f"[error: {type(e).__name__}: {e}]"

    output = result.stdout or ""
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    output = output.strip() or "[ran with no output]"

    if len(output) > MAX_TOOL_RESULT_CHARS:
        output = (
            output[:MAX_TOOL_RESULT_CHARS]
            + f"\n…[truncated at {MAX_TOOL_RESULT_CHARS} chars]"
        )
    return output
