from __future__ import annotations

from typing import TypedDict


class CodeAgentState(TypedDict, total=False):
    repo_url: str
    workspace_root: str
    results_root: str
    repo_dir: str | None
    fresh_clone: bool
    user_task: str | None
    repo_context_file: str | None

    test_command: str
    test_passed: bool | None
    test_stdout: str | None
    test_stderr: str | None
    test_returncode: int | None

    failure_summary: str | None
    patch_suggestion: str | None
    patch_file: str | None
    patch_prompt_file: str | None
    patch_apply_stdout: str | None
    patch_apply_stderr: str | None
    patch_repair_prompt_file: str | None
    patch_repair_attempts: int
    max_patch_repair_attempts: int
    api_provider: str | None
    llm_model: str | None
    llm_temperature: float
    llm_max_tokens: int

    allow_apply_patch: bool
    patch_applied: bool
    debug_attempts: int
    max_debug_attempts: int

    command_timeout_seconds: int
    final_status: str | None


def initial_state(
    *,
    repo_url: str,
    user_task: str | None = None,
    workspace_root: str = "./workspaces",
    fresh_clone: bool = False,
    results_root: str = "./results",
    test_command: str = "python -m pytest -q --tb=short",
    api_provider: str | None = None,
    llm_model: str | None = None,
    llm_temperature: float = 0.2,
    llm_max_tokens: int = 4096,
    allow_apply_patch: bool = False,
    max_debug_attempts: int = 1,
    max_patch_repair_attempts: int = 1,
    command_timeout_seconds: int = 120,
) -> CodeAgentState:
    return {
        "repo_url": repo_url,
        "workspace_root": workspace_root,
        "results_root": results_root,
        "repo_dir": None,
        "fresh_clone": fresh_clone,
        "user_task": user_task,
        "repo_context_file": None,
        "test_command": test_command,
        "test_passed": None,
        "test_stdout": None,
        "test_stderr": None,
        "test_returncode": None,
        "failure_summary": None,
        "patch_suggestion": None,
        "patch_file": None,
        "patch_prompt_file": None,
        "patch_apply_stdout": None,
        "patch_apply_stderr": None,
        "patch_repair_prompt_file": None,
        "patch_repair_attempts": 0,
        "max_patch_repair_attempts": max_patch_repair_attempts,
        "api_provider": api_provider,
        "llm_model": llm_model,
        "llm_temperature": llm_temperature,
        "llm_max_tokens": llm_max_tokens,
        "allow_apply_patch": allow_apply_patch,
        "patch_applied": False,
        "debug_attempts": 0,
        "max_debug_attempts": max_debug_attempts,
        "command_timeout_seconds": command_timeout_seconds,
        "final_status": None,
    }
