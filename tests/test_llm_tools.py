import json

from code_agent.tools.llm_tools import build_patch_repair_prompt, build_task_prompt, call_llm_for_patch, extract_unified_diff


def test_extract_unified_diff_from_plain_text():
    diff = """Some notes
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
"""
    assert extract_unified_diff(diff).startswith("diff --git a/foo.py b/foo.py")


def test_extract_unified_diff_returns_none_for_placeholder_text():
    assert extract_unified_diff("LLM integration is not configured.") is None


def test_build_task_prompt_includes_user_request_and_context():
    prompt = build_task_prompt(
        repo_dir="/repo",
        user_task="add triangle area support",
        repo_context="--- file: src/area.py ---\nprint('hello')",
        test_command="python -m pytest",
    )

    assert "add triangle area support" in prompt
    assert "--- file: src/area.py ---" in prompt
    assert "Return only a valid unified diff" in prompt


def test_build_patch_repair_prompt_includes_apply_error():
    prompt = build_patch_repair_prompt(
        repo_dir="/repo",
        bad_patch="diff --git a/a.py b/a.py",
        apply_stderr="error: corrupt patch at line 54",
        current_files="--- file: a.py ---\n   1: print('hello')",
    )

    assert "error: corrupt patch at line 54" in prompt
    assert "diff --git a/a.py b/a.py" in prompt
    assert "1: print('hello')" in prompt
    assert "corrected unified diff" in prompt


def test_call_llm_for_patch_without_provider_uses_manual_mode():
    response = call_llm_for_patch("fix this")

    assert "Use `--api deepseek` or `--api openai`" in response


def test_call_llm_for_patch_uses_openai_compatible_response(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n"
                        }
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["auth"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = call_llm_for_patch("fix this", provider="deepseek")

    assert response.startswith("diff --git")
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["auth"] == "Bearer test-key"
