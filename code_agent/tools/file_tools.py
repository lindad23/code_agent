from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def safe_resolve_path(root: str | Path, path: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.resolve()
    if root_path != resolved and root_path not in resolved.parents:
        raise ValueError(f"Path escapes root: {resolved}")
    return resolved
