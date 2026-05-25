from __future__ import annotations

from code_agent.state import CodeAgentState
from code_agent.tools.git_tools import clone_repo as clone_repo_to_workspace


def clone_repo(state: CodeAgentState) -> dict:
    repo_dir = clone_repo_to_workspace(
        state["repo_url"],
        state["workspace_root"],
        timeout=max(300, state.get("command_timeout_seconds", 120)),
        fresh=state.get("fresh_clone", False),
    )
    return {"repo_dir": str(repo_dir)}
