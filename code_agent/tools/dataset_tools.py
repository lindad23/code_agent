from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from code_agent.tools.file_tools import ensure_dir, write_text


def download_huggingface_dataset(dataset_id: str, subset: str | None, cache_dir: str | Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required. Install the experiment dependencies first."
        ) from exc

    cache = ensure_dir(cache_dir)
    if subset:
        return load_dataset(dataset_id, subset, cache_dir=str(cache))
    return load_dataset(dataset_id, cache_dir=str(cache))


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


def stage_huggingface_repository(
    repo_id: str,
    workspace_dir: str | Path,
    cache_root: str | Path,
    *,
    repo_type: str = "model",
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
    shutil.copytree(
        shared,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(marker.name),
    )
    return target, shared


def download_huggingface_model_repository(model_id: str, local_dir: str | Path) -> Path:
    return download_huggingface_repository(model_id, local_dir, repo_type="model")
