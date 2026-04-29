# tools/python_sandbox.py — Python execution inside a Docker container
import subprocess

from pydantic import BaseModel, Field, ValidationError

from config import TOOL_TIMEOUT


# ── Argument schemas ───────────────────────────────────────────────────────

class RunPythonArgs(BaseModel):
    code: str = Field(..., min_length=1, description="The Python code to execute as a script.")


# ── Tool implementation ────────────────────────────────────────────────────

def run_python(code: str) -> str:
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
        return f"[error: code execution timed out after {TOOL_TIMEOUT} seconds]"
    except Exception as e:
        return f"[error: {e}]"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    return output or "[ran with no output]"


# ── Schema sent to Ollama ──────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code in an isolated sandbox and return whatever was printed "
                "to stdout. Use print() to produce output. The sandbox has NO access to the "
                "host filesystem — it cannot read or write files in the workspace. To work "
                "with files in the workspace, use read_file and write_file tools and pass "
                "content as code arguments. The sandbox also has no network. State does "
                "not persist between calls."
            ),
            "parameters": RunPythonArgs.model_json_schema(),
        },
    }
]


# ── Dispatch ───────────────────────────────────────────────────────────────

def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "run_python":
        try:
            args = RunPythonArgs(**arguments)
        except ValidationError as e:
            return f"[error: invalid arguments for run_python — {e}]"
        return run_python(args.code)

    return f"[error: unknown tool '{tool_name}']"