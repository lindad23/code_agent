from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from fnmatch import fnmatch
from pathlib import Path

from code_agent.tools.file_tools import ensure_dir, write_text


def download_huggingface_dataset(dataset_id: str, subset: str | None, cache_dir: str | Path):
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install the experiment dependencies first."
        ) from exc

    cache = ensure_dir(cache_dir)
    download_config = DownloadConfig(max_retries=5, resume_download=True)
    if subset:
        return load_dataset(dataset_id, subset, cache_dir=str(cache), download_config=download_config)
    return load_dataset(dataset_id, cache_dir=str(cache), download_config=download_config)


def download_huggingface_repository(repo_id: str, local_dir: str | Path, *, repo_type: str = "model") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "The `huggingface-hub` package is required. Install the experiment dependencies first."
        ) from exc

    target = ensure_dir(local_dir)
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(target))
    return target


def _repository_cache_key(repo_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id.strip("/")) or "repository"
    digest = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def _copy_repository_snapshot(source: Path, target: Path, *, allow_patterns: Sequence[str] | None = None) -> None:
    if allow_patterns is None:
        shutil.copytree(
            source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".code_agent_complete.json"),
        )
        return

    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if not path.is_file() or path.name == ".code_agent_complete.json":
            continue
        relative = path.relative_to(source)
        relative_posix = relative.as_posix()
        if not any(fnmatch(relative_posix, pattern) for pattern in allow_patterns):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def stage_huggingface_repository(
    repo_id: str,
    workspace_dir: str | Path,
    cache_root: str | Path,
    *,
    repo_type: str = "model",
    allow_patterns: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Return a run-local repository copy sourced from a reusable download cache."""
    target = Path(workspace_dir).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        shared = ensure_dir(cache_root) / repo_type / _repository_cache_key(repo_id)
        return target, shared

    shared = ensure_dir(Path(cache_root) / repo_type / _repository_cache_key(repo_id))
    marker = shared / ".code_agent_complete.json"
    if not marker.exists():
        download_huggingface_repository(repo_id, shared, repo_type=repo_type)
        write_text(
            marker,
            json.dumps({"repo_id": repo_id, "repo_type": repo_type}, indent=2),
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    _copy_repository_snapshot(shared, target, allow_patterns=allow_patterns)
    return target, shared


def download_huggingface_model_repository(model_id: str, local_dir: str | Path) -> Path:
    return download_huggingface_repository(model_id, local_dir, repo_type="model")
