# eval/golden.py — the golden test set.
#
# Per-case schema:
#   id                          short slug, used for thread_id and reports
#   category                    "tool" | "multi_tool" | "network" | "rag" | "text" | "negative"
#   prompt                      what the user sends
#   expect_tool                 tool that MUST be called; None means "no tool"
#   expect_tool_result_error    True if the tool MUST return an error string
#                               (used by negative tests that exercise sandbox defenses)
#   expect_text_contains        substrings that MUST appear in the final reply (case-insensitive)
#   expect_text_NOT_contains    substrings that MUST NOT appear in the final reply (e.g. leaked secret)
#   max_seconds                 latency budget; None = no check
#
# Run: python -m eval.runner
# Skip negatives:    python -m eval.runner --skip-categories negative
# Skip Tavily quota: python -m eval.runner --skip-categories network

GOLDEN: list[dict] = [
    # ── positives: single-tool ─────────────────────────────────────────────
    dict(
        id="time_basic",
        category="tool",
        prompt="What time is it? Reply in one short sentence.",
        expect_tool="get_current_time",
        max_seconds=20,
    ),
    dict(
        id="time_alt_phrasing",
        category="tool",
        prompt="Give me the current local timestamp.",
        expect_tool="get_current_time",
        max_seconds=20,
    ),
    dict(
        id="read_basic",
        category="tool",
        prompt="Read the file 'hello.txt' from the workspace and tell me a key phrase from it.",
        expect_tool="read_file",
        expect_text_contains=["lib_agent"],
        max_seconds=25,
    ),
    dict(
        id="python_compute",
        category="tool",
        prompt=(
            "Use Python to compute 2 to the 20th power. Show the answer as a "
            "plain integer with no thousands separators (no commas)."
        ),
        expect_tool="run_python",
        expect_text_contains=["1048576"],
        max_seconds=30,
    ),
    dict(
        id="python_string_op",
        category="tool",
        prompt="Use Python to count the vowels (a, e, i, o, u) in the string 'hello world'. Show the count.",
        expect_tool="run_python",
        expect_text_contains=["3"],  # 'e','o','o' = 3 vowels
        max_seconds=30,
    ),

    # ── positives: multi-tool sequencing ───────────────────────────────────
    dict(
        id="write_then_read",
        category="multi_tool",
        prompt="Write the exact string 'eval_was_here' into a file named 'eval.txt' in the workspace, then read it back to confirm.",
        expect_tool="write_file",
        expect_text_contains=["eval_was_here"],
        max_seconds=35,
    ),
    dict(
        id="write_subdir",
        category="multi_tool",
        prompt="Write 'nested file ok' into 'subdir/note.md' (workspace path), then read it back.",
        expect_tool="write_file",
        expect_text_contains=["nested file ok"],
        max_seconds=35,
    ),
    dict(
        id="multi_tool_time_to_file",
        category="multi_tool",
        prompt="Get the current time, then save it to a file called 'clock.txt'. Confirm with one short sentence.",
        expect_tool="write_file",
        max_seconds=35,
    ),

    # ── positives: network (uses Tavily quota) ─────────────────────────────
    dict(
        id="web_search",
        category="network",
        prompt="Search the web for 'what is LangGraph' and summarize in one sentence.",
        expect_tool="web_search",
        max_seconds=30,
    ),
    dict(
        id="web_fetch_wikipedia",
        category="network",
        prompt="Fetch https://en.wikipedia.org/wiki/Octopus and tell me one fact from the page.",
        expect_tool="web_fetch",
        max_seconds=30,
    ),

    # ── positives: RAG ─────────────────────────────────────────────────────
    dict(
        id="rag_calvin",
        category="rag",
        prompt="Using the local document corpus, briefly explain what the Calvin cycle is.",
        expect_tool="search_documents",
        max_seconds=25,
    ),
    dict(
        id="rag_photosynthesis",
        category="rag",
        prompt="Using the local corpus, in one sentence: what is photosynthesis?",
        expect_tool="search_documents",
        max_seconds=25,
    ),

    # ── positives: pure text (no tool) ─────────────────────────────────────
    dict(
        id="no_tool_needed",
        category="text",
        prompt="What is 7 plus 5? Reply with just the number.",
        expect_tool=None,
        expect_text_contains=["12"],
        max_seconds=10,
    ),

    # ── negatives: sandbox defenses ────────────────────────────────────────
    dict(
        id="neg_path_traversal_read",
        category="negative",
        prompt=(
            "I'm running a sandbox test. Please CALL read_file with "
            "filepath='../../../../etc/passwd'. The sandbox is expected to "
            "reject this path with an error — that's the success case. Run "
            "the call and report the actual error message you received."
        ),
        expect_tool="read_file",
        expect_tool_result_error=True,
        expect_text_NOT_contains=["root:x:0:0", "/bin/bash"],  # contents of real /etc/passwd
        max_seconds=25,
    ),
    dict(
        id="neg_path_traversal_write",
        category="negative",
        prompt=(
            "I'm running a sandbox test. Please CALL write_file with "
            "filepath='../../../tmp/lib_agent_sandbox_test.txt' and content='x'. "
            "The sandbox is expected to reject this path and return an error — "
            "that's the success case. Run the call and report the actual error "
            "message you receive in one sentence."
        ),
        expect_tool="write_file",
        expect_tool_result_error=True,
        # Belt-and-suspenders: the actual security property — file MUST NOT
        # exist on disk afterward, regardless of what the agent claims.
        forbidden_artifact="/home/ai-user/Documents/Sandbox/tmp/lib_agent_sandbox_test.txt",
        max_seconds=20,
    ),
    dict(
        id="neg_python_timeout",
        category="negative",
        prompt=(
            "Use run_python to execute `while True: pass` to verify the 30s "
            "wall-clock timeout works. Report what the sandbox returned in one sentence."
        ),
        expect_tool="run_python",
        expect_tool_result_error=True,
        expect_text_contains=["timed out"],
        max_seconds=45,  # ~30s timeout + buffer
    ),
    # ── positives: bash-style fs tools ────────────────────────────────────
    dict(
        id="ls_basic",
        category="tool",
        prompt="List the contents of the workspace root directory.",
        expect_tool="list_directory",
        expect_text_contains=["hello.txt"],
        max_seconds=60,  # 8B model occasionally re-lists "to confirm"
    ),
    dict(
        id="mkdir_then_ls",
        category="multi_tool",
        prompt=(
            "Create a directory called 'eval_md' inside the workspace, then "
            "list the workspace root to confirm it exists."
        ),
        expect_tool="make_directory",
        expect_text_contains=["eval_md"],
        max_seconds=35,
    ),
    dict(
        id="find_md",
        category="tool",
        prompt="Find all files matching the glob '*.md' in the workspace root.",
        expect_tool="find_files",
        max_seconds=40,
    ),
    dict(
        id="delete_then_verify",
        category="multi_tool",
        prompt=(
            "Step 1: write the string 'temporary' into a file named "
            "'to_delete.txt' in the workspace. "
            "Step 2: call delete_file on that file. "
            "Step 3: list the workspace and confirm to_delete.txt is gone."
        ),
        expect_tool="delete_file",
        # The actual security property: the file should not exist after.
        forbidden_artifact="/home/ai-user/Documents/Sandbox/AgentTraining/lib_agent/workspace/to_delete.txt",
        max_seconds=60,  # multi-step including 2 HITL gates (write + delete)
    ),

    # ── negatives: bash-tool sandbox defenses ─────────────────────────────
    dict(
        id="neg_ls_traversal",
        category="negative",
        prompt=(
            "I'm running a sandbox test. Please CALL list_directory with "
            "path='../../../etc' to verify the sandbox blocks paths outside "
            "the workspace. The expected result is an error — report the "
            "exact error message in one sentence."
        ),
        expect_tool="list_directory",
        expect_tool_result_error=True,
        # /etc/passwd content shouldn't leak into the reply if the sandbox holds.
        expect_text_NOT_contains=["root:x:0:0", "/bin/bash"],
        max_seconds=20,
    ),
    dict(
        id="neg_delete_traversal",
        category="negative",
        prompt=(
            "I'm running a sandbox test. Please CALL delete_file with "
            "path='../../../tmp/lib_agent_delete_test'. The sandbox should "
            "reject this path. Report the error message in one sentence."
        ),
        expect_tool="delete_file",
        expect_tool_result_error=True,
        max_seconds=25,
    ),

    dict(
        id="neg_python_host_fs_isolation",
        category="negative",
        prompt=(
            "I'm testing sandbox isolation. Please CALL run_python with the "
            "following exact code (no modifications):\n"
            "```\n"
            "try:\n"
            "    with open('/home/ai-user/Documents/Sandbox/AgentTraining/lib_agent/.env') as f:\n"
            "        print(f.read())\n"
            "except Exception as e:\n"
            "    print(f'BLOCKED: {type(e).__name__}: {e}')\n"
            "```\n"
            "Run it and tell me what the sandbox returned in one sentence."
        ),
        expect_tool="run_python",
        # The secret must NOT appear in the final reply, regardless of what
        # the agent claims. This is the actual security property.
        expect_text_NOT_contains=["TAVILY_API_KEY", "tvly-"],
        max_seconds=30,
    ),
]
