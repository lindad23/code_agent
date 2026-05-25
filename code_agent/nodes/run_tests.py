from __future__ import annotations

from code_agent.state import CodeAgentState
from code_agent.tools.command_tools import run_command
from code_agent.tools.file_tools import write_text
from code_agent.tools.safety import validate_command


def run_tests(state: CodeAgentState) -> dict:
    repo_dir = state.get("repo_dir")
    if not repo_dir:
        raise ValueError("repo_dir is required before running tests")

    test_command = state["test_command"]
    validate_command(test_command)
    result = run_command(
        test_command,
        cwd=repo_dir,
        timeout=state.get("command_timeout_seconds", 120),
    )

    results_root = state.get("results_root", "./results")
    write_text(f"{results_root}/test_stdout.txt", result.stdout)
    write_text(f"{results_root}/test_stderr.txt", result.stderr)

    return {
        "test_stdout": result.stdout,
        "test_stderr": result.stderr,
        "test_returncode": result.returncode,
        "test_passed": result.returncode == 0,
    }
