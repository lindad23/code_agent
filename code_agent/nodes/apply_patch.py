from __future__ import annotations

import re
from pathlib import Path

from code_agent.state import CodeAgentState
from code_agent.tools.file_tools import write_text
from code_agent.tools.git_tools import git_apply
from code_agent.tools.llm_tools import build_patch_repair_prompt, call_llm_for_patch, extract_unified_diff


def _extract_touched_files(diff: str) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", diff, re.MULTILINE):
        path = match.group(2).strip()
        if path not in files:
            files.append(path)
    return files


def _format_current_files(repo_dir: str, diff: str, *, max_file_chars: int = 12000) -> str:
    root = Path(repo_dir).resolve()
    chunks: list[str] = []
    for relative in _extract_touched_files(diff):
        candidate = (root / relative).resolve()
        if root != candidate and root not in candidate.parents:
            continue
        if not candidate.exists() or not candidate.is_file():
            chunks.append(f"\n--- file: {relative} ---\n[file does not exist]\n")
            continue
        try:
            content = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            chunks.append(f"\n--- file: {relative} ---\n[binary or non-utf8 file skipped]\n")
            continue
        if len(content) > max_file_chars:
            content = content[:max_file_chars] + "\n...[truncated]\n"
        numbered = "\n".join(f"{index:4}: {line}" for index, line in enumerate(content.splitlines(), start=1))
        chunks.append(f"\n--- file: {relative} ---\n{numbered}\n")
    return "".join(chunks).strip()


def _detect_line_ending(path: Path) -> str:
    data = path.read_bytes()
    crlf_count = data.count(b"\r\n")
    lf_count = data.count(b"\n")
    if crlf_count and crlf_count >= max(1, lf_count // 2):
        return "\r\n"
    return "\n"


def _dominant_line_ending(repo_dir: str, diff: str) -> str:
    root = Path(repo_dir).resolve()
    endings: list[str] = []
    for relative in _extract_touched_files(diff):
        candidate = (root / relative).resolve()
        if root != candidate and root not in candidate.parents:
            continue
        if candidate.exists() and candidate.is_file():
            endings.append(_detect_line_ending(candidate))
    if endings and endings.count("\r\n") >= endings.count("\n"):
        return "\r\n"
    return "\n"


def _write_line_ending_adjusted_patch(results_root: str, diff: str, line_ending: str, suffix: str) -> str:
    normalized = diff.replace("\r\n", "\n").replace("\r", "\n")
    content = normalized.replace("\n", line_ending)
    target = Path(results_root) / f"generated_{suffix}.patch"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8"))
    return str(target)


def _try_git_apply(repo_dir: str, patch_file: str, timeout: int):
    result = git_apply(repo_dir, patch_file, timeout=timeout)
    if result.returncode == 0:
        return result
    whitespace_result = git_apply(
        repo_dir,
        patch_file,
        timeout=timeout,
        ignore_whitespace=True,
    )
    if whitespace_result.returncode == 0:
        return whitespace_result
    return result


def _parse_full_file_patch(diff: str) -> dict[str, list[str]] | None:
    files: dict[str, list[str]] = {}
    current_path: str | None = None
    current_lines: list[str] | None = None
    in_full_file_hunk = False

    for raw_line in diff.splitlines():
        file_match = re.match(r"^diff --git a/(.+?) b/(.+?)$", raw_line)
        if file_match:
            if current_path and current_lines is not None:
                files[current_path] = current_lines
            current_path = file_match.group(2).strip()
            current_lines = []
            in_full_file_hunk = False
            continue

        if current_path is None:
            continue

        hunk_match = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw_line)
        if hunk_match:
            if hunk_match.group(1) != "1" or hunk_match.group(2) != "1":
                return None
            in_full_file_hunk = True
            continue

        if not in_full_file_hunk:
            continue
        if raw_line.startswith("\\ No newline at end of file"):
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++ "):
            current_lines.append(raw_line[1:])
        elif raw_line.startswith(" "):
            current_lines.append(raw_line[1:])
        elif raw_line.startswith("-") and not raw_line.startswith("--- "):
            continue

    if current_path and current_lines is not None:
        files[current_path] = current_lines
    return files or None


def _apply_full_file_patch_fallback(repo_dir: str, diff: str) -> bool:
    parsed = _parse_full_file_patch(diff)
    if not parsed:
        return False

    root = Path(repo_dir).resolve()
    for relative, lines in parsed.items():
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            return False
        line_ending = _detect_line_ending(target) if target.exists() else _dominant_line_ending(repo_dir, diff)
        content = line_ending.join(lines) + line_ending
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
    return True


def _parse_unified_diff_hunks(diff: str) -> dict[str, list[list[tuple[str, str]]]]:
    files: dict[str, list[list[tuple[str, str]]]] = {}
    current_path: str | None = None
    current_hunk: list[tuple[str, str]] | None = None

    for raw_line in diff.splitlines():
        file_match = re.match(r"^diff --git a/(.+?) b/(.+?)$", raw_line)
        if file_match:
            if current_path and current_hunk:
                files.setdefault(current_path, []).append(current_hunk)
            current_path = file_match.group(2).strip()
            current_hunk = None
            continue

        if current_path is None:
            continue

        if raw_line.startswith("@@ "):
            if current_hunk:
                files.setdefault(current_path, []).append(current_hunk)
            current_hunk = []
            continue

        if current_hunk is None:
            continue
        if raw_line.startswith("\\ No newline at end of file"):
            continue
        if raw_line.startswith((" ", "+", "-")) and not raw_line.startswith(("+++ ", "--- ")):
            current_hunk.append((raw_line[0], raw_line[1:]))

    if current_path and current_hunk:
        files.setdefault(current_path, []).append(current_hunk)
    return files


def _find_block(lines: list[str], block: list[str], preferred_start: int = 0) -> int | None:
    if not block:
        return preferred_start

    def matches_at(index: int) -> bool:
        return lines[index : index + len(block)] == block

    window_start = max(0, preferred_start - 5)
    window_end = min(len(lines) - len(block), preferred_start + 5)
    for index in range(window_start, window_end + 1):
        if matches_at(index):
            return index

    for index in range(0, len(lines) - len(block) + 1):
        if matches_at(index):
            return index
    return None


def _apply_unified_diff_fallback(repo_dir: str, diff: str) -> bool:
    parsed = _parse_unified_diff_hunks(diff)
    if not parsed:
        return False

    root = Path(repo_dir).resolve()
    pending_writes: list[tuple[Path, bytes]] = []
    for relative, hunks in parsed.items():
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            return False
        if not target.exists() or not target.is_file():
            return False

        line_ending = _detect_line_ending(target)
        original = target.read_text(encoding="utf-8").splitlines()
        updated = list(original)
        search_start = 0
        for hunk in hunks:
            old_block = [line for tag, line in hunk if tag in {" ", "-"}]
            new_block = [line for tag, line in hunk if tag in {" ", "+"}]
            index = _find_block(updated, old_block, search_start)
            if index is None:
                return False
            updated[index : index + len(old_block)] = new_block
            search_start = index + len(new_block)

        content = line_ending.join(updated) + line_ending
        pending_writes.append((target, content.encode("utf-8")))

    for target, content in pending_writes:
        target.write_bytes(content)
    return True


def apply_patch(state: CodeAgentState) -> dict:
    debug_attempts = state.get("debug_attempts", 0) + 1
    if not state.get("allow_apply_patch", False):
        return {
            "patch_applied": False,
            "debug_attempts": debug_attempts,
        }

    repo_dir = state.get("repo_dir")
    patch_file = state.get("patch_file")
    patch_suggestion = state.get("patch_suggestion") or ""
    current_diff = extract_unified_diff(patch_suggestion)
    results_root = state.get("results_root", "./results")
    if not repo_dir or not patch_file or not current_diff:
        stderr = "No valid unified diff was available to apply."
        write_text(f"{results_root}/apply_patch_stderr.txt", stderr)
        return {
            "patch_applied": False,
            "patch_apply_stdout": "",
            "patch_apply_stderr": stderr,
            "debug_attempts": debug_attempts,
        }

    current_patch_file = patch_file
    repair_attempts = 0
    max_repairs = state.get("max_patch_repair_attempts", 1)
    last_stdout = ""
    last_stderr = ""
    repair_prompt_file = None

    while True:
        timeout = state.get("command_timeout_seconds", 120)
        result = _try_git_apply(repo_dir, current_patch_file, timeout)
        if result.returncode != 0:
            line_ending = _dominant_line_ending(repo_dir, current_diff)
            if line_ending != "\n":
                adjusted_patch_file = _write_line_ending_adjusted_patch(
                    results_root,
                    current_diff,
                    line_ending,
                    f"line_endings_{repair_attempts}",
                )
                adjusted_result = _try_git_apply(repo_dir, adjusted_patch_file, timeout)
                if adjusted_result.returncode == 0:
                    result = adjusted_result
                    current_patch_file = adjusted_patch_file
                else:
                    last_stderr = adjusted_result.stderr
                    write_text(f"{results_root}/apply_patch_line_endings_stderr.txt", last_stderr)
            if _apply_full_file_patch_fallback(repo_dir, current_diff):
                result = type(result)(
                    stdout="Applied full-file patch fallback after git apply failed.",
                    stderr="",
                    returncode=0,
                    elapsed_time=result.elapsed_time,
                )
            elif _apply_unified_diff_fallback(repo_dir, current_diff):
                result = type(result)(
                    stdout="Applied unified diff fallback after git apply failed.",
                    stderr="",
                    returncode=0,
                    elapsed_time=result.elapsed_time,
                )
        last_stdout = result.stdout
        last_stderr = result.stderr
        write_text(f"{results_root}/apply_patch_stdout.txt", last_stdout)
        write_text(f"{results_root}/apply_patch_stderr.txt", last_stderr)

        if result.returncode == 0:
            return {
                "patch_applied": True,
                "patch_file": current_patch_file,
                "patch_suggestion": current_diff,
                "patch_apply_stdout": last_stdout,
                "patch_apply_stderr": last_stderr,
                "patch_repair_attempts": repair_attempts,
                "debug_attempts": debug_attempts,
            }

        if (
            repair_attempts >= max_repairs
            or not state.get("api_provider")
            or str(state.get("api_provider")).lower() in {"none", "manual", "off"}
        ):
            break

        repair_attempts += 1
        repair_prompt = build_patch_repair_prompt(
            repo_dir=repo_dir,
            bad_patch=current_diff,
            apply_stderr=last_stderr or "git apply failed without stderr.",
            current_files=_format_current_files(repo_dir, current_diff),
        )
        repair_prompt_file = write_text(f"{results_root}/patch_repair_prompt_{repair_attempts}.md", repair_prompt)
        repaired_suggestion = call_llm_for_patch(
            repair_prompt,
            provider=state.get("api_provider"),
            model=state.get("llm_model"),
            temperature=state.get("llm_temperature", 0.2),
            max_tokens=state.get("llm_max_tokens", 4096),
        )
        repaired_diff = extract_unified_diff(repaired_suggestion)
        if not repaired_diff:
            last_stderr = "Patch repair did not return a valid unified diff."
            write_text(f"{results_root}/apply_patch_stderr.txt", last_stderr)
            return {
                "patch_applied": False,
                "patch_apply_stdout": last_stdout,
                "patch_apply_stderr": last_stderr,
                "patch_repair_prompt_file": str(repair_prompt_file),
                "patch_repair_attempts": repair_attempts,
                "debug_attempts": debug_attempts,
            }

        current_diff = repaired_diff
        current_patch_file = str(write_text(f"{results_root}/generated_repaired_{repair_attempts}.patch", current_diff))

    return {
        "patch_applied": False,
        "patch_file": current_patch_file,
        "patch_suggestion": current_diff,
        "patch_apply_stdout": last_stdout,
        "patch_apply_stderr": last_stderr,
        "patch_repair_prompt_file": str(repair_prompt_file) if repair_prompt_file else None,
        "patch_repair_attempts": repair_attempts,
        "debug_attempts": debug_attempts,
    }
