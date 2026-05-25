from __future__ import annotations

from pathlib import Path


DEFAULT_INCLUDE_SUFFIXES = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".json",
    ".ini",
    ".cfg",
}

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
}


def collect_repo_context(
    repo_dir: str | Path,
    *,
    max_files: int = 40,
    max_file_chars: int = 6000,
    max_total_chars: int = 30000,
) -> str:
    root = Path(repo_dir).resolve()
    chunks: list[str] = []
    total = 0
    file_count = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in DEFAULT_EXCLUDED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in DEFAULT_INCLUDE_SUFFIXES:
            continue
        if file_count >= max_files or total >= max_total_chars:
            break

        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if len(content) > max_file_chars:
            content = content[:max_file_chars] + "\n...[truncated]\n"

        chunk = f"\n--- file: {relative} ---\n{content}\n"
        if total + len(chunk) > max_total_chars:
            break
        chunks.append(chunk)
        total += len(chunk)
        file_count += 1

    return "".join(chunks).strip()
