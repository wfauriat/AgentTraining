import subprocess

from config import TOOL_TIMEOUT

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
            input=code,                  # pipe code to stdin
            capture_output=True,         # capture stdout/stderr
            text=True,                   # strings, not bytes
            timeout=TOOL_TIMEOUT,        # kill after 30s
        )
    except subprocess.TimeoutExpired:
        return "[error: code execution timed out after 30 seconds]"
    except Exception as e:
        return f"[error: {e}]"

    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"

    return output or "[ran with no output]"


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
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to be executed by python as a script in a single string.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

def dispatch(tool_name: str, arguments: dict) -> str:
    if tool_name == "run_python":
        code = arguments.get("code", "")
        if not code:
            return "[error: run_python requires a code argument]"
        return run_python(code)

    return f"[error: unknown tool '{tool_name}']"

