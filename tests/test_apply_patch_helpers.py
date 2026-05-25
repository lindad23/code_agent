from code_agent.nodes.apply_patch import _apply_unified_diff_fallback, _extract_touched_files, _format_current_files


def test_extract_touched_files_from_diff():
    diff = """diff --git a/src/area.py b/src/area.py
--- a/src/area.py
+++ b/src/area.py
diff --git a/tests/test_area.py b/tests/test_area.py
--- a/tests/test_area.py
+++ b/tests/test_area.py
"""

    assert _extract_touched_files(diff) == ["src/area.py", "tests/test_area.py"]


def test_format_current_files_includes_line_numbers(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    area = source / "area.py"
    area.write_text("first\nsecond\n", encoding="utf-8")
    diff = "diff --git a/src/area.py b/src/area.py\n"

    formatted = _format_current_files(str(tmp_path), diff)

    assert "--- file: src/area.py ---" in formatted
    assert "   1: first" in formatted
    assert "   2: second" in formatted


def test_apply_unified_diff_fallback_finds_shifted_context(tmp_path):
    target = tmp_path / "sample.py"
    target.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    diff = """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -10,2 +10,3 @@
 two
 three
+four
"""

    assert _apply_unified_diff_fallback(str(tmp_path), diff)
    assert target.read_bytes() == b"one\r\ntwo\r\nthree\r\nfour\r\n"
