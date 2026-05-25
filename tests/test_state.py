from code_agent.state import initial_state


def test_initial_state_contains_minimal_defaults():
    state = initial_state(repo_url="https://github.com/example/project.git")

    assert state["repo_url"] == "https://github.com/example/project.git"
    assert state["workspace_root"] == "./workspaces"
    assert state["results_root"] == "./results"
    assert state["repo_dir"] is None
    assert state["fresh_clone"] is False
    assert state["user_task"] is None
    assert state["repo_context_file"] is None
    assert state["test_command"] == "python -m pytest -q --tb=short"
    assert state["api_provider"] is None
    assert state["llm_model"] is None
    assert state["allow_apply_patch"] is False
    assert state["patch_applied"] is False
    assert state["patch_apply_stdout"] is None
    assert state["patch_apply_stderr"] is None
    assert state["patch_repair_attempts"] == 0
    assert state["max_patch_repair_attempts"] == 1
    assert state["debug_attempts"] == 0
    assert state["max_debug_attempts"] == 1
    assert state["final_status"] is None
