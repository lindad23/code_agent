from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int
    elapsed_time: float


def split_command(command: str) -> list[str]:
    return shlex.split(command, posix=False)


def run_command(command: str | list[str], cwd: str | Path | None = None, timeout: int = 120) -> CommandResult:
    start = time.monotonic()
    args = split_command(command) if isinstance(command, str) else command
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            elapsed_time=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\nCommand timed out after {timeout} seconds.",
            returncode=124,
            elapsed_time=time.monotonic() - start,
        )
