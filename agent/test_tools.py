from unittest.mock import patch, MagicMock

from tools.meta import get_current_time
from tools.web import web_search, web_fetch, dispatch
from tools.files import read_file, safe_path
from tools.python import run_python
from config import MAX_TOOL_RESULT_TOKENS

def test_get_current_time_returns_string():
    assert isinstance(get_current_time(), str)
  
def test_web_search_formats_results():
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "results": [
            {"title": "Title 1",
             "content": "Content 1",
             "url": "http://url1.com"},
            {"title": "Title 2",
             "content": "Content 2",
             "url": "http://url2.com"}
        ]
    }
    with patch("httpx.post", return_value=fake_response):
        result = web_search("anything")

    assert isinstance(result, str)
    assert "Title 1" in result
    assert "Title 2" in result
    assert "http://url1.com" in result

def test_web_search_handles_empty_results():
    fake_response = MagicMock()
    fake_response.json.return_value = {"results": []}

    with patch("httpx.post", return_value=fake_response):
        result = web_search("anything")
    
    assert "no results found" in result.lower()

def test_web_fetch_truncates_long_content():
    fake_response = MagicMock()
    fake_response.headers = {"content-type": "text/html"}
    fake_response.text = "<html>...</html>" 

    long_text = "a" * 10000
    with patch("httpx.get", return_value=fake_response), \
        patch("tools.web.trafilatura.extract", return_value=long_text):
        result = web_fetch("http://example.com")

    assert len(result) == MAX_TOOL_RESULT_TOKENS  
    assert result.startswith("a")

def test_dispatch_unknown_tool_returns_error():
    result = dispatch("nonexistent_tool", {})
    assert "error" in result.lower()

def test_read_file_rejects_traversal():
    result = read_file("../etc/passwd")
    assert "error" in result.lower() or "forbidden" in result.lower()

def test_safe_path_rejects_dangerous_patterns():
    dangerous = [
        "../etc/passwd",                 # parent traversal
        "../../etc/passwd",              # multiple parents
        "/etc/passwd",                   # absolute path (joining replaces base)
        "notes/../../../etc/passwd",     # traversal hidden after safe prefix
        "./../../etc/passwd",            # current-dir prefix doesn't help
        "subdir/../../outside.txt",      # escapes after a real subdir
    ]
    for path in dangerous:
        result = safe_path(path)
        assert result is None, f"safe_path failed to reject: {path!r} -> {result}"


def test_safe_path_accepts_legitimate_paths():
    legitimate = [
        "notes.txt",
        "subdir/notes.txt",
        "deeply/nested/path/file.md",
        "notes/../notes.txt",     # technically uses .. but resolves inside
    ]
    for path in legitimate:
        result = safe_path(path)
        assert result is not None, f"safe_path wrongly rejected: {path!r}"

def test_python_in_sandbox():
    testcode = "print(1+1)"
    assert "2" in run_python(testcode)

def main():
    tests = [
        test_get_current_time_returns_string,
        test_web_search_formats_results,
        test_web_search_handles_empty_results,
        test_web_fetch_truncates_long_content,
        test_dispatch_unknown_tool_returns_error,
        test_read_file_rejects_traversal,
        test_safe_path_rejects_dangerous_patterns,
        test_safe_path_accepts_legitimate_paths,
        test_python_in_sandbox
    ]
    
    passed = failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")

if __name__ == "__main__":
    main()