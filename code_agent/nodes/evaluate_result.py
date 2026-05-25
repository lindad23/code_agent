from __future__ import annotations

import json
from datetime import datetime, timezone

from code_agent.state import CodeAgentState
from code_agent.tools.file_tools import write_text


def evaluate_result(state: CodeAgentState) -> dict:
    if state.get("user_task") and not state.get("patch_applied"):
        final_status = "task_patch_not_applied"
    elif state.get("test_passed") is True:
        final_status = "passed"
    elif state.get("patch_applied"):
        final_status = "failed_after_patch"
    else:
        final_status = "failed_patch_suggested"

    summary = {
        "final_status": final_status,
        "repo_url": state.get("repo_url"),
        "repo_dir": state.get("repo_dir"),
        "user_task": state.get("user_task"),
        "test_command": state.get("test_command"),
        "api_provider": state.get("api_provider"),
        "llm_model": state.get("llm_model"),
        "test_returncode": state.get("test_returncode"),
        "test_passed": state.get("test_passed"),
        "debug_attempts": state.get("debug_attempts", 0),
        "patch_applied": state.get("patch_applied", False),
        "patch_apply_stdout": state.get("patch_apply_stdout"),
        "patch_apply_stderr": state.get("patch_apply_stderr"),
        "patch_repair_attempts": state.get("patch_repair_attempts", 0),
        "max_patch_repair_attempts": state.get("max_patch_repair_attempts", 1),
        "patch_repair_prompt_file": state.get("patch_repair_prompt_file"),
        "patch_prompt_file": state.get("patch_prompt_file"),
        "patch_file": state.get("patch_file"),
        "repo_context_file": state.get("repo_context_file"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_text(
        f"{state.get('results_root', './results')}/final_state.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )
    return {"final_status": final_status}
