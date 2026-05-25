from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path


SUPPORTED_PROVIDERS = {"deepseek", "openai"}


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_patch_prompt(*, repo_dir: str, failure_summary: str, test_command: str) -> str:
    return f"""You are helping fix a failing Python repository.

Repository path:
{repo_dir}

Test command:
{test_command}

Failure summary:
{failure_summary}

Please inspect the repository and return only a valid unified diff that can be applied with `git apply`.
"""


def build_task_prompt(*, repo_dir: str, user_task: str, repo_context: str, test_command: str) -> str:
    return f"""You are helping modify a Python repository according to a user request.

Repository path:
{repo_dir}

User request:
{user_task}

Test command that will be run after your patch:
{test_command}

Repository context:
{repo_context}

Return only a valid unified diff that can be applied with `git apply`.
Do not include explanations, markdown fences, or prose outside the diff.
"""


def build_patch_repair_prompt(
    *,
    repo_dir: str,
    bad_patch: str,
    apply_stderr: str,
    current_files: str = "",
) -> str:
    return f"""You generated a unified diff for this repository, but `git apply` rejected it.

Repository path:
{repo_dir}

git apply error:
{apply_stderr}

Current contents of files touched by the patch, with line numbers:
{current_files or "No current file contents were available."}

Rejected patch:
{bad_patch}

Return only a corrected unified diff that can be applied with `git apply`.
Keep the intended code changes the same, but fix malformed hunk headers, line counts, context, and any other diff-format problems.
Use the current file contents and line numbers above as the source of truth for hunk headers and context lines.
Prefer full-file hunks for each touched file instead of small local hunks. For example, use a hunk that starts at line 1 and replaces the full current file content with the full desired file content. This is more reliable than trying to guess local hunk offsets.
Do not include fake `index 1111111..2222222` lines.
Do not include explanations, markdown fences, or prose outside the diff.
"""


def _provider_settings(provider: str, model: str | None) -> tuple[str, str, str]:
    normalized = provider.lower()
    if normalized == "deepseek":
        api_key_name = "DEEPSEEK_API_KEY"
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        resolved_model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        endpoint = f"{base_url}/chat/completions"
    elif normalized == "openai":
        api_key_name = "OPENAI_API_KEY"
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        resolved_model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        endpoint = f"{base_url}/chat/completions"
    else:
        raise ValueError(f"Unsupported API provider: {provider}")

    api_key = os.environ.get(api_key_name)
    if not api_key:
        raise ValueError(f"{api_key_name} is not set. Put it in .env or export it before running.")
    return endpoint, api_key, resolved_model


def call_llm(
    prompt: str,
    *,
    provider: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    system_prompt: str = "You are a senior Python debugging agent.",
    timeout: int = 120,
) -> str:
    load_env_file()
    endpoint, api_key, resolved_model = _provider_settings(provider, model)
    payload = {
        "model": resolved_model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{provider} API request failed with HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{provider} API request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{provider} API request timed out after {timeout} seconds.") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{provider} API response did not contain a chat message.") from exc


def call_llm_for_patch(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    if not provider or provider.lower() in {"none", "manual", "off"}:
        return (
            "LLM integration is not enabled for this run.\n\n"
            "Use `--api deepseek` or `--api openai` to call a model automatically. "
            "A patch prompt was saved for manual use."
        )
    return call_llm(
        prompt,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt="You are a senior Python debugging agent. Return only a valid unified diff.",
    )


def extract_unified_diff(text: str) -> str | None:
    fence_match = re.search(r"```(?:diff|patch)?\s*(diff --git .+?)```", text, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else text.strip()
    if "diff --git " in candidate and "\n--- " in candidate and "\n+++ " in candidate:
        start = candidate.find("diff --git ")
        return candidate[start:].strip() + "\n"
    return None
