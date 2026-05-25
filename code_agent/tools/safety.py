from __future__ import annotations

import re


_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
]


def validate_command(command: str) -> None:
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.search(command):
            raise ValueError(f"Refusing dangerous command: {command}")
