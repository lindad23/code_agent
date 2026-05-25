from __future__ import annotations

from code_agent.state import CodeAgentState
from code_agent.tools.file_tools import write_text


def _tail(text: str | None, max_chars: int = 8000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def analyze_failure(state: CodeAgentState) -> dict:
    stdout = _tail(state.get("test_stdout"))
    stderr = _tail(state.get("test_stderr"))
    returncode = state.get("test_returncode")
    summary = f"""Test command failed with return code {returncode}.

--- stdout tail ---
{stdout}

--- stderr tail ---
{stderr}
"""
    results_root = state.get("results_root", "./results")
    write_text(f"{results_root}/failure_summary.md", summary)
    return {"failure_summary": summary}
