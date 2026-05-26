from __future__ import annotations

import sys
from typing import TextIO


class CliProgress:
    """Render small step-based progress output without affecting JSON stdout."""

    def __init__(
        self,
        total_steps: int,
        *,
        stream: TextIO | None = None,
        enabled: bool = True,
        width: int = 20,
    ) -> None:
        self.total_steps = max(1, total_steps)
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self.width = width
        self.current_step = 0
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self._closed = False
        self._rendered_length = 0

    def add_steps(self, count: int) -> None:
        if count > 0:
            self.total_steps += count

    def update(self, label: str) -> None:
        if not self.enabled or self._closed:
            return
        self.current_step += 1
        completed_steps = min(self.current_step - 1, self.total_steps)
        self._render(completed_steps, f"{self.current_step}/{self.total_steps} {label}")

    def finish(self, label: str = "completed") -> None:
        if not self.enabled or self._closed:
            return
        self._render(self.total_steps, label)
        if self._interactive:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def fail(self, label: str = "failed") -> None:
        if not self.enabled or self._closed:
            return
        completed_steps = min(self.current_step - 1, self.total_steps)
        self._render(completed_steps, label)
        if self._interactive:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _render(self, completed_steps: int, label: str) -> None:
        filled = round(self.width * completed_steps / self.total_steps)
        bar = "#" * filled + "-" * (self.width - filled)
        prefix = "\r" if self._interactive else ""
        suffix = "" if self._interactive else "\n"
        rendered = f"[{bar}] {label}"
        padding = " " * max(0, self._rendered_length - len(rendered)) if self._interactive else ""
        self.stream.write(f"{prefix}{rendered}{padding}{suffix}")
        self.stream.flush()
        self._rendered_length = len(rendered)
