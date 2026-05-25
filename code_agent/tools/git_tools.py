from __future__ import annotations

import re
import shutil
import os
import stat
from pathlib import Path

from code_agent.tools.command_tools import CommandResult, run_command
from code_agent.tools.file_tools import ensure_dir


def repo_name_from_url(repo_url: str) -> str:
    cleaned = repo_url.rstrip("/").removesuffix(".git")
    name = re.split(r"[/\\:]", cleaned)[-1]
    if not name:
        raise ValueError(f"Cannot infer repo name from URL: {repo_url}")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _remove_tree(path: Path) -> None:
    def handle_remove_error(func, failed_path, exc_info):
        try:
            os.chmod(failed_path, stat.S_IWRITE)
            func(failed_path)
        except OSError:
            raise exc_info[1]

    shutil.rmtree(path, onerror=handle_remove_error)


def clone_repo(repo_url: str, workspace_root: str | Path, timeout: int = 300, *, fresh: bool = False) -> Path:
    workspace = ensure_dir(workspace_root)
    repo_dir = workspace / repo_name_from_url(repo_url)
    if fresh and repo_dir.exists():
        resolved_workspace = workspace.resolve()
        resolved_repo = repo_dir.resolve()
        if resolved_workspace == resolved_repo or resolved_workspace not in resolved_repo.parents:
            raise ValueError(f"Refusing to remove path outside workspace: {resolved_repo}")
        _remove_tree(resolved_repo)
    if (repo_dir / ".git").exists():
        return repo_dir
    result = run_command(["git", "clone", repo_url, str(repo_dir)], cwd=workspace, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{result.stderr or result.stdout}")
    return repo_dir


def git_apply(
    repo_dir: str | Path,
    patch_file: str | Path,
    timeout: int = 120,
    *,
    ignore_whitespace: bool = False,
) -> CommandResult:
    command = ["git", "apply"]
    if ignore_whitespace:
        command.extend(["--ignore-space-change", "--ignore-whitespace"])
    command.append(str(Path(patch_file).resolve()))
    return run_command(command, cwd=repo_dir, timeout=timeout)


def rev_parse_head(repo_dir: str | Path, timeout: int = 30) -> str | None:
    result = run_command(["git", "rev-parse", "HEAD"], cwd=repo_dir, timeout=timeout)
    if result.returncode != 0:
        return None
    return result.stdout.strip()
