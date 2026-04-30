# eval/golden.py — the golden test set.
#
# Each case is a dict:
#   id                    short slug, used for thread_id and reports
#   category              "tool" | "multi_tool" | "network" | "rag" | "text"
#   prompt                what the user sends
#   expect_tool           name of a tool that MUST be called; None means "no tool"
#   expect_text_contains  list of substrings that must appear in the final reply (case-insensitive); None = no check
#
# Categories let us skip groups: e.g. --skip-categories network avoids Tavily quota.
# Keep cases hermetic: don't depend on prior conversation state. workspace/hello.txt
# and vector_db/ must exist (run scripts.index_docs first).

GOLDEN: list[dict] = [
    dict(
        id="time_basic",
        category="tool",
        prompt="What time is it? Reply in one short sentence.",
        expect_tool="get_current_time",
        expect_text_contains=None,
    ),
    dict(
        id="read_basic",
        category="tool",
        prompt="Read the file 'hello.txt' from the workspace and tell me a key phrase from it.",
        expect_tool="read_file",
        expect_text_contains=["lib_agent"],
    ),
    dict(
        id="write_then_read",
        category="multi_tool",
        prompt="Write the exact string 'eval_was_here' into a file named 'eval.txt' in the workspace, then read it back to confirm.",
        expect_tool="write_file",
        expect_text_contains=["eval_was_here"],
    ),
    dict(
        id="web_search",
        category="network",
        prompt="Search the web for 'what is LangGraph' and summarize in one sentence.",
        expect_tool="web_search",
        expect_text_contains=None,
    ),
    dict(
        id="python_compute",
        category="tool",
        prompt="Use Python to compute 2 to the 20th power. Show the answer.",
        expect_tool="run_python",
        expect_text_contains=["1048576"],
    ),
    dict(
        id="rag_calvin",
        category="rag",
        prompt="Using the local document corpus, briefly explain what the Calvin cycle is.",
        expect_tool="search_documents",
        expect_text_contains=None,
    ),
    dict(
        id="no_tool_needed",
        category="text",
        prompt="What is 7 plus 5? Reply with just the number.",
        expect_tool=None,  # explicitly: no tool should be called
        expect_text_contains=["12"],
    ),
    dict(
        id="multi_tool_time_to_file",
        category="multi_tool",
        prompt="Get the current time, then save it to a file called 'clock.txt'. Confirm with one short sentence.",
        expect_tool="write_file",
        expect_text_contains=None,
    ),
]
