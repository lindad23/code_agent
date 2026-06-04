from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import ast
import hashlib
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from code_agent.experiments.agent import (
    PROJECT_ROOT,
    _environment_cache_key,
    _environment_cache_root,
    _environment_cache_spec,
    _environment_python,
    _install_experiment_dependencies,
    _install_torch_runtime,
    _record_cached_environment,
    _record_verified_runtime,
    _query_gpu_memory_snapshots,
    _resolve_hardware_profile,
    _run_process,
    _run_process_streaming,
    _select_cached_environment,
    _select_execution_gpu,
    _select_previous_completed_environment,
    _try_clone_current_environment,
    _verify_torch_runtime,
    _write_environment_file,
)
from code_agent.experiments.models import ExperimentRequest, ExperimentRunState, unique_run_id
from code_agent.tools.file_tools import ensure_dir, safe_resolve_path, write_text
from code_agent.tools.llm_tools import call_llm


@dataclass(frozen=True)
class MaterializedResource:
    role: str
    name: str
    location: str
    status: str
    local_path: str | None = None
    cache_path: str | None = None
    cache_status: str | None = None
    cache_key: str | None = None
    error: str | None = None


class GenericCommandExecutionError(RuntimeError):
    def __init__(self, message: str, command_results: list[dict[str, Any]]):
        super().__init__(message)
        self.command_results = command_results


class GenericExecutionSpecValidationError(ValueError):
    pass


MAX_EXPERIMENT_TABLE_RECORDS = 12


class GenericJsonResponseError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        label: str,
        prompt: str,
        raw_response: str,
        repair_responses: list[str],
    ):
        super().__init__(message)
        self.label = label
        self.prompt = prompt
        self.raw_response = raw_response
        self.repair_responses = repair_responses


def _extract_json_object(text: str, *, label: str) -> dict[str, Any]:
    payload = text.strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"The {label} did not return a JSON object.")
    parsed = json.loads(payload[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"The {label} must be a JSON object.")
    return parsed


def _repair_json_response(
    *,
    request: ExperimentRequest,
    label: str,
    prompt: str,
    raw_response: str,
    parse_error: str,
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    repair_prompt = f"""Repair this malformed JSON response.

The previous response for {label} could not be parsed as JSON.

Parse error:
{parse_error}

Original task prompt:
{prompt}

Malformed response:
{raw_response}

Return only one valid JSON object. Do not include markdown fences or explanations.
Preserve the intended fields and content, fixing only JSON syntax and escaping issues.
If the malformed response appears truncated or too large, return a compact equivalent JSON object instead of preserving
the oversized matrix verbatim. Prefer shorter generated code, relative resource paths, and a small real subset of
experiment_table records.
"""
    repaired = call_llm(
        repair_prompt,
        provider=request.api_provider,
        model=request.llm_model,
        temperature=0.0,
        max_tokens=max_tokens,
        system_prompt="You repair malformed JSON. Return strict JSON only.",
        timeout=request.plan_timeout_seconds,
    )
    return _extract_json_object(repaired, label=f"repaired {label}"), repaired


def _extract_or_repair_json_object(
    response: str,
    *,
    request: ExperimentRequest,
    label: str,
    prompt: str,
    max_tokens: int,
) -> tuple[dict[str, Any], str | None]:
    try:
        return _extract_json_object(response, label=label), None
    except (JSONDecodeError, ValueError) as exc:
        parse_error = str(exc)

    repair_responses: list[str] = []
    raw_to_repair = response
    for _attempt in range(2):
        repair_prompt = f"""Repair this malformed JSON response.

The previous response for {label} could not be parsed as JSON.

Parse error:
{parse_error}

Original task prompt:
{prompt}

Malformed response:
{raw_to_repair}

Return only one valid JSON object. Do not include markdown fences or explanations.
Preserve the intended fields and content, fixing only JSON syntax and escaping issues.
"""
        repaired_response = call_llm(
            repair_prompt,
            provider=request.api_provider,
            model=request.llm_model,
            temperature=0.0,
            max_tokens=max_tokens,
            system_prompt="You repair malformed JSON. Return strict JSON only.",
            timeout=request.plan_timeout_seconds,
        )
        repair_responses.append(repaired_response)
        try:
            return _extract_json_object(repaired_response, label=f"repaired {label}"), repaired_response
        except (JSONDecodeError, ValueError) as exc:
            parse_error = str(exc)
            raw_to_repair = repaired_response

    raise GenericJsonResponseError(
        f"The {label} returned malformed JSON and automatic repair failed: {parse_error}",
        label=label,
        prompt=prompt,
        raw_response=response,
        repair_responses=repair_responses,
    )


def build_generic_experiment_prompt(request: ExperimentRequest) -> str:
    return f"""Design a domain-specific ML experiment plan from the structured user input.

Do not assume Hugging Face Transformers, text classification, GLUE, or any fixed executor.
Infer the task domain, resource types, repositories, datasets, metrics, command structure,
implementation strategy, ablations, and result parsing requirements from the input.

User task:
{request.task}

Structured resource context:
{request.resource_context or "(none)"}

Return only one JSON object with exactly these top-level keys:
task_type, resources, environment, implementation_plan, experiment_matrix, execution_plan,
metrics, result_artifacts, risks.

Requirements:
- task_type should describe the actual domain, e.g. time_series_forecasting, vision_classification,
  retrieval, language_modeling, reinforcement_learning, or another specific domain.
- resources must preserve every user-provided resource name and URL/path. If a resource URL/path is empty,
  propose a concrete candidate and explain why.
- implementation_plan must describe what code should be changed and how the improved idea should be implemented.
- experiment_matrix must include all requested baselines, benchmarks, seeds/repeats, ablations, and settings.
- execution_plan must list concrete but non-destructive shell commands or scripts to run, plus working directories.
- metrics must include the requested evaluation indexes or suitable metrics when the user left them empty.
- risks must mention missing resources, incompatible repositories, unclear scripts, or expensive runs.
- Do not include markdown fences.
"""


def request_generic_experiment_plan(request: ExperimentRequest) -> tuple[dict[str, Any], str, str]:
    prompt = build_generic_experiment_prompt(request)
    max_tokens = 20000
    response = call_llm(
        prompt,
        provider=request.api_provider,
        model=request.llm_model,
        temperature=0.1,
        max_tokens=max_tokens,
        system_prompt=(
            "You are a general ML experiment planner. You must not force tasks into a fixed benchmark family. "
            "Return strict JSON only."
        ),
        timeout=request.plan_timeout_seconds,
    )
    plan, repaired_response = _extract_or_repair_json_object(
        response,
        request=request,
        label="generic experiment planner",
        prompt=prompt,
        max_tokens=max_tokens,
    )
    combined_response = response if repaired_response is None else f"{response}\n\n===== repaired JSON =====\n{repaired_response}"
    return plan, prompt, combined_response


def _request_resources(request: ExperimentRequest) -> list[tuple[str, str, str]]:
    resources: list[tuple[str, str, str]] = []
    baseline_resources = request.baseline_resources or {"baseline": request.baseline_url}
    benchmark_resources = request.benchmark_resources or {"benchmark": request.benchmark_url}
    resources.extend(("baseline", name, location) for name, location in baseline_resources.items())
    resources.extend(("benchmark", name, location) for name, location in benchmark_resources.items())
    return resources


def _resource_dir_name(role: str, name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name.strip())
    return f"{role}_{cleaned or 'resource'}"


def _resource_cache_root(workspace_root: Path) -> Path:
    return ensure_dir(workspace_root / "asset_cache" / "resources")


def _resource_cache_key(role: str, name: str, location: str, kind: str, fingerprint: str | None = None) -> str:
    payload = {
        "role": role,
        "name": name,
        "location": location,
        "kind": kind,
        "fingerprint": fingerprint,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    slug_source = f"{role}_{name}"
    slug = "".join(char if char.isalnum() or char in "._-" else "_" for char in slug_source.strip())
    return f"{slug or 'resource'}-{digest}"


def _cache_marker(shared: Path) -> Path:
    return shared / ".code_agent_resource_complete.json"


def _is_complete_cached_resource(shared: Path) -> bool:
    return shared.exists() and _cache_marker(shared).exists()


def _write_resource_cache_marker(shared: Path, *, role: str, name: str, location: str, kind: str, cache_key: str) -> None:
    write_text(
        _cache_marker(shared),
        json.dumps(
            {
                "role": role,
                "name": name,
                "location": location,
                "kind": kind,
                "cache_key": cache_key,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )


def _copy_cached_resource_to_run(shared: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        shared,
        destination,
        ignore=shutil.ignore_patterns(".code_agent_resource_complete.json"),
    )


def _replace_cache_dir(staging: Path, shared: Path) -> None:
    if shared.exists():
        shutil.rmtree(shared)
    shared.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(shared))


def _is_git_resource(location: str) -> bool:
    normalized = location.strip().lower()
    return normalized.startswith(("https://github.com/", "http://github.com/", "git@")) or normalized.endswith(".git")


def _verify_http_resource(location: str, *, timeout: int = 20) -> None:
    request = urllib.request.Request(location, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            fallback = urllib.request.Request(location, method="GET")
            with urllib.request.urlopen(fallback, timeout=timeout):
                return
        raise


def _local_resource_fingerprint(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return f"file:{resolved}:{stat.st_size}:{stat.st_mtime_ns}"
    entries: list[str] = []
    for item in sorted(resolved.rglob("*")):
        if not item.is_file():
            continue
        stat = item.stat()
        entries.append(f"{item.relative_to(resolved).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return f"dir:{resolved}:{digest}"


def _stage_local_resource_in_cache(
    *,
    role: str,
    name: str,
    location: str,
    local_candidate: Path,
    resource_cache_root: Path,
) -> tuple[Path, str, str]:
    fingerprint = _local_resource_fingerprint(local_candidate)
    cache_key = _resource_cache_key(role, name, location, "local", fingerprint)
    shared = resource_cache_root / "local" / cache_key
    cache_status = "hit" if _is_complete_cached_resource(shared) else "miss"
    if cache_status == "miss":
        with tempfile.TemporaryDirectory(prefix="resource-local-", dir=str(ensure_dir(shared.parent))) as temp_dir:
            staging = Path(temp_dir) / "resource"
            if local_candidate.is_dir():
                shutil.copytree(local_candidate, staging)
            else:
                ensure_dir(staging)
                shutil.copy2(local_candidate, staging / local_candidate.name)
            _write_resource_cache_marker(staging, role=role, name=name, location=location, kind="local", cache_key=cache_key)
            _replace_cache_dir(staging, shared)
    return shared, cache_key, cache_status


def _stage_git_resource_in_cache(
    *,
    role: str,
    name: str,
    location: str,
    resource_cache_root: Path,
    timeout: int,
) -> tuple[Path | None, str, str, str | None]:
    cache_key = _resource_cache_key(role, name, location, "git")
    shared = resource_cache_root / "git" / cache_key
    cache_status = "hit" if _is_complete_cached_resource(shared) else "miss"
    if cache_status == "hit":
        return shared, cache_key, cache_status, None

    with tempfile.TemporaryDirectory(prefix="resource-git-", dir=str(ensure_dir(shared.parent))) as temp_dir:
        staging = Path(temp_dir) / "repo"
        attempts = _git_clone_attempt_commands(location, staging)
        failures: list[subprocess.CompletedProcess[str] | subprocess.TimeoutExpired] = []
        for command in attempts:
            if staging.exists():
                shutil.rmtree(staging)
            try:
                result = subprocess.run(
                    command,
                    cwd=resource_cache_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                failures.append(exc)
                continue
            if result.returncode == 0:
                break
            failures.append(result)
        else:
            return None, cache_key, "failed", _format_git_clone_failure(location, failures)
        _write_resource_cache_marker(staging, role=role, name=name, location=location, kind="git", cache_key=cache_key)
        _replace_cache_dir(staging, shared)
    return shared, cache_key, cache_status, None


def _git_clone_attempt_commands(location: str, staging: Path) -> list[list[str]]:
    return [
        ["git", "clone", "--depth", "1", "--filter=blob:none", location, str(staging)],
        ["git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1", "--filter=blob:none", location, str(staging)],
        ["git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1", location, str(staging)],
    ]


def _process_failure_text(failure: subprocess.CompletedProcess[str] | subprocess.TimeoutExpired) -> str:
    if isinstance(failure, subprocess.TimeoutExpired):
        output = failure.output.decode("utf-8", errors="replace") if isinstance(failure.output, bytes) else failure.output
        stderr = failure.stderr.decode("utf-8", errors="replace") if isinstance(failure.stderr, bytes) else failure.stderr
        return f"command timed out after {failure.timeout} seconds\n{output or ''}\n{stderr or ''}".strip()
    return "\n".join(part for part in (failure.stdout.strip(), failure.stderr.strip()) if part)


def _looks_like_missing_git_resource(message: str) -> bool:
    lowered = message.lower()
    missing_markers = [
        "repository not found",
        "not found",
        "authentication failed",
        "could not read username",
        "repository does not exist",
        "the requested url returned error: 404",
    ]
    transport_markers = [
        "early eof",
        "rpc failed",
        "gnutls recv error",
        "tls packet",
        "remote end hung up",
        "connection reset",
        "connection timed out",
        "failed to connect",
        "http/2",
        "curl 56",
    ]
    return any(marker in lowered for marker in missing_markers) and not any(marker in lowered for marker in transport_markers)


def _format_git_clone_failure(
    location: str,
    failures: list[subprocess.CompletedProcess[str] | subprocess.TimeoutExpired],
) -> str:
    details = "\n\n".join(
        f"[git clone attempt {index}]\n{_process_failure_text(failure)}"
        for index, failure in enumerate(failures, start=1)
    ).strip()
    if _looks_like_missing_git_resource(details):
        return f"指定网址不存在或无权限访问: {location}\n{details}"
    return (
        f"Git 资源下载失败，但这不等于网址不存在: {location}\n"
        "git clone 在传输过程中失败，常见原因是 GitHub 网络/TLS 中断、代理不稳定或大仓库 pack 下载被断开。\n"
        f"{details}"
    )


def _http_resource_filename(location: str, name: str) -> str:
    parsed = urllib.parse.urlparse(location)
    filename = Path(urllib.parse.unquote(parsed.path)).name
    if not filename:
        filename = "".join(char if char.isalnum() or char in "._-" else "_" for char in name.strip()) or "resource"
    return filename


def _download_http_resource(location: str, target: Path, *, timeout: int) -> None:
    request = urllib.request.Request(location, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _stage_http_resource_in_cache(
    *,
    role: str,
    name: str,
    location: str,
    resource_cache_root: Path,
    timeout: int,
) -> tuple[Path | None, str, str, str | None]:
    cache_key = _resource_cache_key(role, name, location, "http")
    shared = resource_cache_root / "http" / cache_key
    cache_status = "hit" if _is_complete_cached_resource(shared) else "miss"
    if cache_status == "hit":
        return shared, cache_key, cache_status, None

    try:
        _verify_http_resource(location, timeout=min(20, timeout))
    except Exception as exc:
        return None, cache_key, "failed", f"指定网址不存在: {location}\n{exc}"

    with tempfile.TemporaryDirectory(prefix="resource-http-", dir=str(ensure_dir(shared.parent))) as temp_dir:
        staging = ensure_dir(Path(temp_dir) / "resource")
        target = staging / _http_resource_filename(location, name)
        try:
            _download_http_resource(location, target, timeout=timeout)
        except Exception as exc:
            return None, cache_key, "failed", f"指定网址不存在: {location}\n{exc}"
        _write_resource_cache_marker(staging, role=role, name=name, location=location, kind="http", cache_key=cache_key)
        _replace_cache_dir(staging, shared)
    return shared, cache_key, cache_status, None


def _materialize_resource(
    role: str,
    name: str,
    location: str,
    resources_dir: Path,
    resource_cache_root: Path,
    timeout: int,
) -> MaterializedResource:
    location = location.strip()
    if not location:
        return MaterializedResource(role=role, name=name, location=location, status="needs_ai_resolution")

    destination = resources_dir / _resource_dir_name(role, name)
    local_candidate = Path(location).expanduser()
    if local_candidate.exists():
        shared, cache_key, cache_status = _stage_local_resource_in_cache(
            role=role,
            name=name,
            location=location,
            local_candidate=local_candidate,
            resource_cache_root=resource_cache_root,
        )
        _copy_cached_resource_to_run(shared, destination)
        return MaterializedResource(
            role=role,
            name=name,
            location=location,
            status="copied_local",
            local_path=str(destination),
            cache_path=str(shared),
            cache_status=cache_status,
            cache_key=cache_key,
        )

    if _is_git_resource(location):
        shared, cache_key, cache_status, error = _stage_git_resource_in_cache(
            role=role,
            name=name,
            location=location,
            resource_cache_root=resource_cache_root,
            timeout=timeout,
        )
        if error or shared is None:
            return MaterializedResource(
                role=role,
                name=name,
                location=location,
                status="failed",
                cache_status=cache_status,
                cache_key=cache_key,
                error=error,
            )
        _copy_cached_resource_to_run(shared, destination)
        return MaterializedResource(
            role=role,
            name=name,
            location=location,
            status="cloned_git",
            local_path=str(destination),
            cache_path=str(shared),
            cache_status=cache_status,
            cache_key=cache_key,
        )

    if location.startswith(("https://", "http://")):
        shared, cache_key, cache_status, error = _stage_http_resource_in_cache(
            role=role,
            name=name,
            location=location,
            resource_cache_root=resource_cache_root,
            timeout=timeout,
        )
        if error or shared is None:
            return MaterializedResource(
                role=role,
                name=name,
                location=location,
                status="failed",
                cache_status=cache_status,
                cache_key=cache_key,
                error=error,
            )
        _copy_cached_resource_to_run(shared, destination)
        return MaterializedResource(
            role=role,
            name=name,
            location=location,
            status="downloaded_http",
            local_path=str(destination),
            cache_path=str(shared),
            cache_status=cache_status,
            cache_key=cache_key,
        )

    return MaterializedResource(
        role=role,
        name=name,
        location=location,
        status="failed",
        error=f"指定网址不存在: {location}",
    )


def materialize_generic_resources(request: ExperimentRequest, workspace: Path) -> list[MaterializedResource]:
    resources_dir = ensure_dir(workspace / "resources")
    resource_cache_root = _resource_cache_root(request.workspace_root)
    resources = [
        _materialize_resource(role, name, location, resources_dir, resource_cache_root, request.timeout_seconds)
        for role, name, location in _request_resources(request)
    ]
    failures = [resource.error for resource in resources if resource.status == "failed" and resource.error]
    if failures:
        raise RuntimeError("\n\n".join(failures))
    return resources


def _summarize_materialized_resources(resources: list[MaterializedResource]) -> str:
    lines: list[str] = []
    for resource in resources:
        lines.append(
            json.dumps(
                {
                    "role": resource.role,
                    "name": resource.name,
                    "location": resource.location,
                    "status": resource.status,
                    "local_path": resource.local_path,
                    "cache_path": resource.cache_path,
                    "cache_status": resource.cache_status,
                    "cache_key": resource.cache_key,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _format_ast_arguments(args: ast.arguments) -> str:
    parts: list[str] = []
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        value = arg.arg
        if default is not None:
            value += "=..."
        parts.append(value)
    if args.vararg is not None:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        value = arg.arg
        if default is not None:
            value += "=..."
        parts.append(value)
    if args.kwarg is not None:
        parts.append("**" + args.kwarg.arg)
    return "(" + ", ".join(parts) + ")"


def _config_attribute_reads(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    parameter_names = {
        arg.arg
        for arg in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        if arg.arg != "self"
    }
    attributes: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in parameter_names:
            attributes.add(f"{node.value.id}.{node.attr}")
    return sorted(attributes)


def _python_file_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    parent = path.parent.name.lower()
    priority_terms = (
        "model",
        "models",
        "main",
        "train",
        "run",
        "exp",
        "experiment",
        "data",
        "dataset",
        "loader",
        "config",
    )
    priority = 0 if any(term in name or term in parent for term in priority_terms) else 1
    return priority, path.as_posix()


def _summarize_python_file_interfaces(path: Path, root: Path) -> list[str]:
    try:
        if path.stat().st_size > 400_000:
            return []
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError:
        relative_path = path.name

    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
            init = next((item for item in methods if item.name == "__init__"), None)
            forward = next((item for item in methods if item.name == "forward"), None)
            if init is not None:
                attrs = _config_attribute_reads(init)
                attr_suffix = f"; reads {', '.join(attrs[:16])}" if attrs else ""
                lines.append(f"{relative_path}: class {node.name}.__init__{_format_ast_arguments(init.args)}{attr_suffix}")
            if forward is not None:
                lines.append(f"{relative_path}: class {node.name}.forward{_format_ast_arguments(forward.args)}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            interesting_names = {
                "main",
                "train",
                "fit",
                "evaluate",
                "eval",
                "test",
                "run",
                "build_model",
                "get_model",
                "load_model",
                "get_args",
                "parse_args",
            }
            if node.name.lower() in interesting_names:
                lines.append(f"{relative_path}: def {node.name}{_format_ast_arguments(node.args)}")
    return lines


def _summarize_resource_python_interfaces(resources: list[MaterializedResource]) -> str:
    summaries: list[str] = []
    for resource in resources:
        if not resource.local_path:
            continue
        root = Path(resource.local_path)
        if not root.exists() or not root.is_dir():
            continue
        python_files = sorted(root.rglob("*.py"), key=_python_file_priority)
        resource_lines: list[str] = []
        for path in python_files[:60]:
            if any(part in {".git", "__pycache__", ".venv", "venv", "site-packages"} for part in path.parts):
                continue
            resource_lines.extend(_summarize_python_file_interfaces(path, root))
            if len(resource_lines) >= 80:
                break
        if resource_lines:
            summaries.append(
                json.dumps(
                    {
                        "role": resource.role,
                        "name": resource.name,
                        "local_path": resource.local_path,
                        "interfaces": resource_lines[:80],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    return "\n".join(summaries) or "(no Python interfaces detected)"


def _resource_template_aliases(resources: list[MaterializedResource], workspace: Path) -> list[dict[str, str]]:
    aliases: list[dict[str, str]] = []
    workspace_root = workspace.resolve()
    for resource in resources:
        if resource.local_path:
            local_path = Path(resource.local_path).resolve()
            try:
                relative_path = local_path.relative_to(workspace_root).as_posix()
            except ValueError:
                relative_path = f"resources/{Path(resource.local_path).name}"
        else:
            relative_path = f"resources/{_resource_dir_name(resource.role, resource.name)}"
        aliases.append(
            {
                "role": resource.role,
                "name": resource.name,
                "path": relative_path,
            }
        )
    return aliases


def build_generic_execution_template(resources: list[MaterializedResource], workspace: Path) -> dict[str, Any]:
    resource_aliases = _resource_template_aliases(resources, workspace)
    first_resource = resource_aliases[0]["path"] if resource_aliases else "resources/<resource_name>"
    return {
        "files": [
            {
                "path": "my_main.py",
                "content": (
                    "# Fill with concise Python code. The script must accept --cell-json and --output-json, "
                    "read the cell params, use real resources from the cell target, and write JSON metrics."
                ),
            }
        ],
        "commands": [],
        "experiment_table": {
            "entrypoint": {
                "cwd": ".",
                "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                "timeout_seconds": 3600,
                "env": {},
                "gpu_policy": {
                    "strategy": "greedy",
                    "min_free_memory_mb": 4096,
                    "reserve_memory_mb": 1024,
                    "max_workers_per_gpu": 1,
                },
            },
            "resource_aliases": resource_aliases,
            "records": [
                {
                    "id": "real_subset_cell_001",
                    "target": {
                        "code_path": "my_main.py",
                        "resources": [first_resource],
                    },
                    "params": {
                        "seed": 42,
                    },
                    "expected_metrics": ["primary_metric", "runtime_seconds"],
                }
            ],
        },
        "expected_outputs": ["generic_metrics.json"],
        "notes": "Fill this template with a compact real resource-backed subset.",
    }


def build_generic_execution_prompt(
    request: ExperimentRequest,
    *,
    plan: dict[str, Any],
    resources: list[MaterializedResource],
    workspace: Path,
    results_dir: Path,
    template: dict[str, Any] | None = None,
) -> str:
    template = template or build_generic_execution_template(resources, workspace)
    return f"""Create an executable specification for this generic ML experiment.

You are not limited to a particular benchmark, framework, or repository. Use the experiment plan,
the materialized resources, and the user's improved idea to create a minimal but real execution
workflow that can run inside the workspace.

Workspace root:
{workspace}

Results directory:
{results_dir}

Materialized resources, one JSON object per line:
{_summarize_materialized_resources(resources)}

Detected Python interfaces in materialized resources:
{_summarize_resource_python_interfaces(resources)}

Local execution spec template JSON:
{json.dumps(template, ensure_ascii=False, indent=2)}

Experiment plan JSON:
{json.dumps(plan, ensure_ascii=False, indent=2)}

User task:
{request.task}

Return only one JSON object with these keys:
files, commands, experiment_table, expected_outputs, notes.

Template requirements:
- Fill the local execution spec template above. Keep the same top-level keys and same experiment_table shape.
- Use `experiment_table.resource_aliases` as the only place to list resource paths. In each record, refer to those
  resources by relative path or resource name. Do not repeat absolute workspace/results paths inside every record.
- Do not output a generated-cell script or a separate experiment_table file. The local template is the contract.

Schema:
- files: list of objects with path and content. Paths must be relative to the workspace. Use these files for
  generated experiment drivers, adapters, result parsers, or lightweight implementation code. Adapters may bridge
  repository APIs, but they must not replace materialized baseline algorithms with newly written stand-ins.
- commands: list of setup commands with name, cwd, argv, timeout_seconds, and optional env. This may be empty.
  argv must be a JSON array, not a shell string. Do not put the experiment matrix loop here.
- experiment_table: object with entrypoint and records.
  - entrypoint: object with cwd, argv, timeout_seconds, and optional env. The generated entrypoint should normally be
    `my_main.py`, and argv should include placeholders such as
    ["python", "my_main.py", "--cell-json", "{{cell_file}}", "--output-json", "{{output_file}}"].
    It may include gpu_policy with strategy="greedy", min_free_memory_mb, reserve_memory_mb,
    estimated_cell_memory_mb, max_workers_per_gpu, and max_parallel_cells when the per-cell GPU footprint is known.
  - records: list of experiment records. Each record is exactly one experiment to execute and should include:
    id, target, params, and expected_metrics. target must include the relevant generated code path and/or materialized
    resource paths. params must contain the arguments the entrypoint needs to run this cell.
  Keep records compact: include at most {MAX_EXPERIMENT_TABLE_RECORDS} records in this JSON response. Use relative resource paths such as
  `resources/benchmark_ETTm1` or resource names, not repeated absolute paths. If the full matrix is larger, choose a
  representative real subset and let notes describe the omitted matrix dimensions.
- expected_outputs: list of relative workspace paths or absolute results-dir paths expected after execution.
- notes: short notes about what the execution covers and what remains manual.

Safety requirements:
- Do not use shell strings, pipes, redirection, command separators, rm, sudo, chmod, chown, curl pipes, or destructive commands.
- Do not use bash/sh scripts. Put loops and orchestration in generated Python scripts and run them with python.
- Do not write outside the workspace or the provided results directory.
- Prefer `python` for generated scripts. You may use `pip install -r ...` only when a repository requires dependencies.
- Do not install `torch` in generated generic commands; the experiment runtime selects and verifies a host-compatible
  PyTorch build before execution. Install only task-specific non-torch dependencies here.
- If user-specified resources were materialized, the workflow must run a real resource-backed experiment using those
  cloned repositories and datasets. Do not use DummyBaseline, random synthetic data, placeholder metrics, or notes that
  ask the user to replace dummy code later.
- For materialized baseline resources, import, execute, patch, subclass, or otherwise call the code in the baseline
  resource. Do not define a new DLinear/PatchTST/etc. wrapper as a substitute for the baseline repository. If the
  repository API is unclear, make the entrypoint fail with a clear JSON error instead of silently substituting a toy
  implementation.
- Use the detected Python interface summary above when constructing repository classes/functions. Do not guess model
  constructors. If a class takes a single config-like argument and the summary lists attributes read from it, build a
  simple argparse.Namespace/object with those attributes before instantiating it.
- The generated entrypoint must read resource locations from the cell JSON `target.resources`; do not rely only on a
  params field such as `resource_dir` and do not hard-code a resource layout outside the table target.
- If the full matrix would be too expensive, run a smaller real subset using the actual cloned model/data resources.
- Do not generate a separate script whose only job is to create the experiment_table. The execution spec itself should
  contain the compact records list.
- Keep the complete JSON response under 12000 tokens. Long Python files should be concise and avoid duplicated code.
- The generated entrypoint must read the cell JSON, run that real experiment, and write output JSON containing status,
  metrics, artifacts, and the executed params. The executor will aggregate per-cell outputs.
- For latency or memory measurement, use a real batch from the loaded dataset when available instead of a random tensor.
- Ensure the entrypoint or executor produces a JSON summary at `{results_dir / "generic_metrics.json"}`.
- Do not include markdown fences.
"""


def request_generic_execution_spec(
    request: ExperimentRequest,
    *,
    plan: dict[str, Any],
    resources: list[MaterializedResource],
    workspace: Path,
    results_dir: Path,
) -> tuple[dict[str, Any], str, str]:
    template = build_generic_execution_template(resources, workspace)
    write_text(
        results_dir / "generic_execution_template.json",
        json.dumps(template, ensure_ascii=False, indent=2),
    )
    prompt = build_generic_execution_prompt(
        request,
        plan=plan,
        resources=resources,
        workspace=workspace,
        results_dir=results_dir,
        template=template,
    )
    max_tokens = 20000
    response = call_llm(
        prompt,
        provider=request.api_provider,
        model=request.llm_model,
        temperature=0.1,
        max_tokens=max_tokens,
        system_prompt=(
            "You are a general ML experiment implementation agent. Return strict JSON only. "
            "Never force the task into a fixed Hugging Face or text-classification executor."
        ),
        timeout=request.plan_timeout_seconds,
    )
    spec, repaired_response = _extract_or_repair_json_object(
        response,
        request=request,
        label="generic execution implementer",
        prompt=prompt,
        max_tokens=max_tokens,
    )
    _validate_generic_execution_spec_shape(spec, label="generic execution spec")
    combined_response = response if repaired_response is None else f"{response}\n\n===== repaired JSON =====\n{repaired_response}"
    return spec, prompt, combined_response


def _tail_file(path: str | Path | None, *, max_chars: int = 12000) -> str:
    if path is None:
        return ""
    target = Path(path)
    if not target.exists():
        return ""
    text = target.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _summarize_command_results(command_results: list[dict[str, Any]]) -> str:
    records: list[dict[str, Any]] = []
    for result in command_results:
        record = dict(result)
        record["stdout_tail"] = _tail_file(record.get("stdout_file"))
        record["stderr_tail"] = _tail_file(record.get("stderr_file"))
        records.append(record)
    return json.dumps(records, ensure_ascii=False, indent=2)


def build_generic_debug_prompt(
    request: ExperimentRequest,
    *,
    plan: dict[str, Any],
    resources: list[MaterializedResource],
    previous_spec: dict[str, Any],
    command_results: list[dict[str, Any]],
    failure_summary: str,
    workspace: Path,
    results_dir: Path,
    attempt: int,
) -> str:
    template = build_generic_execution_template(resources, workspace)
    return f"""Repair the executable specification for this generic ML experiment.

The previous execution attempt failed. Diagnose the failure from the command records and stdout/stderr tails,
then return a corrected executable specification. Stay within the same safety rules and task domain.

Debug attempt:
{attempt}

Workspace root:
{workspace}

Results directory:
{results_dir}

Failure summary:
{failure_summary}

Materialized resources, one JSON object per line:
{_summarize_materialized_resources(resources)}

Detected Python interfaces in materialized resources:
{_summarize_resource_python_interfaces(resources)}

Local execution spec template JSON:
{json.dumps(template, ensure_ascii=False, indent=2)}

Experiment plan JSON:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Previous execution spec JSON:
{json.dumps(previous_spec, ensure_ascii=False, indent=2)}

Command results with output tails:
{_summarize_command_results(command_results)}

User task:
{request.task}

Return only one JSON object with these keys:
files, commands, experiment_table, expected_outputs, notes.

Rules:
- Fix the actual failure, for example wrong path, missing import, wrong script arguments, missing lightweight adapter,
  incorrect output path, or overly ambitious full-run command.
- Prefer repairing the generated one-click entrypoint and experiment_table records. Each table record must be one real
  experiment cell, and the entrypoint must accept `--cell-json` and `--output-json`.
- Fill the local execution spec template. Keep at most {MAX_EXPERIMENT_TABLE_RECORDS} records and use relative resource
  paths or resource names in records. Do not repeat absolute workspace/results paths in every record.
- Preserve or correct gpu_policy when it helps the executor greedily use multiple relatively idle GPUs without
  overcommitting memory.
- Keep experiment_table compact: at most 12 records, with relative resource paths. Do not enumerate a large full matrix
  in JSON and do not add a generated-cell script just to expand the matrix.
- If validation failed because the previous spec used dummy/synthetic data or ignored cloned resources, replace it with
  a real resource-backed workflow. Use the materialized repository and dataset paths directly.
- If the full matrix is too expensive, run a smaller real subset using actual cloned model/data resources, not dummy data.
- For materialized baseline resources, repair by importing, executing, patching, subclassing, or otherwise calling the
  baseline repository code. Do not replace the baseline with a generated stand-in class or a lightweight wrapper that
  merely shares the baseline name.
- Use the detected Python interface summary above to repair repository imports and constructors. Do not keep retrying
  positional/keyword argument guesses when the source signature shows a config-like object or a different call shape.
- The entrypoint must read resource locations from the cell JSON `target.resources`; each record should declare the
  baseline and benchmark resources it needs.
- The entrypoint or executor must produce a JSON summary at `{results_dir / "generic_metrics.json"}`.
- Do not use shell strings, pipes, redirection, command separators, rm, sudo, chmod, chown, curl pipes, or destructive commands.
- Do not use bash/sh scripts. Put loops and orchestration in generated Python scripts and run them with python.
- Do not write outside the workspace or the provided results directory.
- argv must be a JSON array, not a shell string.
- Do not include markdown fences.
"""


def request_generic_debug_execution_spec(
    request: ExperimentRequest,
    *,
    plan: dict[str, Any],
    resources: list[MaterializedResource],
    previous_spec: dict[str, Any],
    command_results: list[dict[str, Any]],
    failure_summary: str,
    workspace: Path,
    results_dir: Path,
    attempt: int,
) -> tuple[dict[str, Any], str, str]:
    prompt = build_generic_debug_prompt(
        request,
        plan=plan,
        resources=resources,
        previous_spec=previous_spec,
        command_results=command_results,
        failure_summary=failure_summary,
        workspace=workspace,
        results_dir=results_dir,
        attempt=attempt,
    )
    max_tokens = 20000
    response = call_llm(
        prompt,
        provider=request.api_provider,
        model=request.llm_model,
        temperature=0.1,
        max_tokens=max_tokens,
        system_prompt=(
            "You are a general ML experiment debugging agent. Return strict JSON only. "
            "Repair the execution spec without changing the task into a fixed benchmark family."
        ),
        timeout=request.plan_timeout_seconds,
    )
    spec, repaired_response = _extract_or_repair_json_object(
        response,
        request=request,
        label="generic execution debugger",
        prompt=prompt,
        max_tokens=max_tokens,
    )
    _validate_generic_execution_spec_shape(spec, label="debug execution spec")
    combined_response = response if repaired_response is None else f"{response}\n\n===== repaired JSON =====\n{repaired_response}"
    return spec, prompt, combined_response


def _validate_generic_execution_spec_shape(spec: dict[str, Any], *, label: str) -> None:
    if not isinstance(spec.get("files"), list):
        raise ValueError(f"The {label} must contain list-valued files.")
    if "commands" not in spec:
        spec["commands"] = []
    if not isinstance(spec.get("commands"), list):
        raise ValueError(f"The {label} must contain list-valued commands.")
    if "experiment_table" in spec and spec["experiment_table"] not in (None, []):
        _validate_experiment_table_template_contract(spec, label=label)
    for file in spec.get("files", []):
        if not isinstance(file, dict):
            continue
        path = str(file.get("path", "")).lower()
        if "generate" in path and "cell" in path:
            raise ValueError(f"The {label} must not generate a separate experiment-table expansion script: {path}")


def _validate_experiment_table_template_contract(spec: dict[str, Any], *, label: str) -> None:
    _entrypoint, records = _extract_experiment_table(spec)
    if len(records) > MAX_EXPERIMENT_TABLE_RECORDS:
        raise ValueError(
            f"The {label} experiment_table has {len(records)} records; "
            f"the template contract allows at most {MAX_EXPERIMENT_TABLE_RECORDS}."
        )
    absolute_path_pattern = re.compile(r'(?<![A-Za-z0-9_./-])/(?:hard_data|home|tmp|mnt|workspace|workspaces|results)/')
    for index, record in enumerate(records, start=1):
        record_text = json.dumps(record, ensure_ascii=False)
        if absolute_path_pattern.search(record_text):
            raise ValueError(
                f"The {label} experiment_table record {index} contains absolute paths. "
                "Use template resource_aliases plus relative resource paths in records."
            )


def _write_generated_files(spec: dict[str, Any], workspace: Path) -> list[dict[str, str]]:
    written: list[dict[str, str]] = []
    for item in spec.get("files", []):
        if not isinstance(item, dict):
            raise ValueError("Each generated file entry must be an object.")
        raw_path = str(item.get("path", "")).strip()
        if not raw_path:
            raise ValueError("Generated file entries require a non-empty path.")
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Generated file {raw_path} must provide string content.")
        target = safe_resolve_path(workspace, raw_path)
        write_text(target, content)
        written.append({"path": raw_path, "absolute_path": str(target)})
    return written


def _spec_text(spec: dict[str, Any]) -> str:
    return json.dumps(spec, ensure_ascii=False, sort_keys=True).lower()


def _execution_surface_text(spec: dict[str, Any]) -> str:
    surface = {
        "files": spec.get("files", []),
        "commands": spec.get("commands", []),
        "entrypoint": spec.get("entrypoint"),
        "experiment_table": spec.get("experiment_table"),
        "expected_outputs": spec.get("expected_outputs", []),
    }
    return json.dumps(surface, ensure_ascii=False, sort_keys=True).lower()


def _real_resource_backed_resources(resources: list[MaterializedResource]) -> list[MaterializedResource]:
    return [
        resource
        for resource in resources
        if resource.location.strip() and resource.local_path and Path(resource.local_path).exists()
    ]


def _generated_file_contents(spec: dict[str, Any]) -> list[tuple[str, str]]:
    contents: list[tuple[str, str]] = []
    for file in spec.get("files", []):
        if not isinstance(file, dict):
            continue
        path = str(file.get("path", "")).strip()
        content = file.get("content")
        if isinstance(content, str):
            contents.append((path, content))
    return contents


def _dummy_data_violations(spec: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    synthetic_assignment = re.compile(
        r"\b(?:train|val|valid|test|x|y|data|dataset|features|labels|inputs|targets)[A-Za-z0-9_]*\s*=\s*"
        r"(?:torch|np|numpy)\.(?:randn|rand|random|normal|uniform)\s*\(",
        re.IGNORECASE,
    )
    tensor_dataset_random = re.compile(
        r"\b(?:TensorDataset|Dataset|DataLoader)\s*\([^)]*(?:torch|np|numpy)\.(?:randn|rand|random|normal|uniform)\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    random_dataframe = re.compile(
        r"\b(?:DataFrame|pd\.DataFrame)\s*\([^)]*(?:np|numpy)\.(?:randn|rand|random|normal|uniform)\s*\(",
        re.IGNORECASE | re.DOTALL,
    )
    constant_metric_output = re.compile(
        r"\b(?:metrics|result|output)\s*=\s*\{[^{}]*(?:['\"](?:MSE|MAE|accuracy|loss|score)['\"]\s*:\s*[0-9.]+)[^{}]*\}",
        re.IGNORECASE | re.DOTALL,
    )
    for path, content in _generated_file_contents(spec):
        for pattern_name, pattern in (
            ("synthetic random data assignment", synthetic_assignment),
            ("synthetic random TensorDataset/DataLoader", tensor_dataset_random),
            ("synthetic random DataFrame", random_dataframe),
            ("constant placeholder metrics", constant_metric_output),
        ):
            if pattern.search(content):
                violations.append(f"{pattern_name} in {path or '<generated file>'}")
    return violations


def _baseline_resources(resources: list[MaterializedResource]) -> list[MaterializedResource]:
    return [
        resource
        for resource in _real_resource_backed_resources(resources)
        if resource.role.strip().lower() == "baseline"
    ]


def _resource_names_and_tokens(resources: list[MaterializedResource]) -> list[str]:
    tokens: list[str] = []
    for resource in resources:
        candidates = [resource.name, Path(resource.local_path or "").name]
        for candidate in candidates:
            normalized = str(candidate).strip().lower()
            if normalized:
                tokens.append(normalized)
            tokens.extend(part for part in re.split(r"[^a-zA-Z0-9]+", str(candidate).lower()) if len(part) >= 3)
    return sorted(set(tokens), key=len, reverse=True)


def _experiment_table_target_resources(spec: dict[str, Any]) -> list[str]:
    table = spec.get("experiment_table")
    if not isinstance(table, dict):
        return []
    records = table.get("records")
    if not isinstance(records, list):
        return []
    resources: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        target = record.get("target")
        if not isinstance(target, dict):
            continue
        raw_resources = target.get("resources")
        if isinstance(raw_resources, list):
            resources.extend(str(resource) for resource in raw_resources if str(resource).strip())
    return resources


def _entrypoint_generated_contents(spec: dict[str, Any]) -> str:
    entrypoint, _records = _extract_experiment_table(spec)
    argv = entrypoint.get("argv") if isinstance(entrypoint, dict) else None
    argv_paths = {
        Path(str(part)).as_posix().lstrip("./")
        for part in argv or []
        if str(part).endswith((".py", ".pyw"))
    }
    contents: list[str] = []
    for path, content in _generated_file_contents(spec):
        normalized = Path(path).as_posix().lstrip("./")
        if not argv_paths or normalized in argv_paths or Path(normalized).name in {Path(item).name for item in argv_paths}:
            contents.append(content)
    return "\n".join(contents)


def _resource_contract_violations(spec: dict[str, Any], resources: list[MaterializedResource]) -> list[str]:
    violations: list[str] = []
    baseline_resources = _baseline_resources(resources)
    target_resources = _experiment_table_target_resources(spec)
    if target_resources:
        entrypoint_content = _entrypoint_generated_contents(spec).lower()
        if "target" not in entrypoint_content or "resources" not in entrypoint_content:
            violations.append("entrypoint does not read cell target.resources")

    if not baseline_resources:
        return violations

    execution_text = _execution_surface_text(spec)
    standin_phrases = [
        "does not import baseline code",
        "does not import the baseline code",
        "does not use baseline code",
        "defines lightweight wrappers",
        "lightweight wrappers for",
        "actual baseline repositories should be used",
        "actual baseline repo should be used",
        "for production use, the actual baseline",
        "stand-in baseline",
        "stand in baseline",
        "baseline stand-in",
        "toy baseline",
        "approximate baseline",
        "baseline approximation",
        "reimplements the baseline",
        "reimplement the baseline",
    ]
    for phrase in standin_phrases:
        if phrase in execution_text:
            violations.append(f"baseline stand-in admission: {phrase}")

    generated_content = "\n".join(content for _path, content in _generated_file_contents(spec))
    generated_lower = generated_content.lower()
    integration_signals = [
        "sys.path",
        "importlib",
        "subprocess",
        "runpy",
        "baseline_path",
        "baseline_repo",
        "target_resources",
        'target["resources"]',
        "target['resources']",
        'target.get("resources"',
        "target.get('resources'",
        'cell.get("target"',
        "cell.get('target'",
    ]
    has_repo_integration_signal = any(signal.lower() in generated_lower for signal in integration_signals)
    if not has_repo_integration_signal:
        baseline_tokens = [re.escape(token) for token in _resource_names_and_tokens(baseline_resources)]
        for token in baseline_tokens:
            class_pattern = re.compile(
                rf"\bclass\s+[A-Za-z_][A-Za-z0-9_]*{token}[A-Za-z0-9_]*(?:wrapper|baseline|model|net|network)\b",
                re.IGNORECASE,
            )
            factory_pattern = re.compile(
                rf"\bdef\s+(?:get|build|create|make)_[A-Za-z0-9_]*{token}[A-Za-z0-9_]*(?:model|baseline)\b",
                re.IGNORECASE,
            )
            if class_pattern.search(generated_content) or factory_pattern.search(generated_content):
                violations.append("generated baseline-named stand-in without repository integration")
                break
    return violations


def validate_real_resource_execution_spec(spec: dict[str, Any], resources: list[MaterializedResource]) -> None:
    real_resources = _real_resource_backed_resources(resources)
    if not real_resources:
        return

    text = _spec_text(spec)
    execution_text = _execution_surface_text(spec)
    banned_patterns = [
        "dummybaseline",
        "dummy baseline",
        "dummy data",
        "random synthetic",
        "user must replace",
        "replace the dummy",
        "--smoke",
        "smoke test",
        "preflight workflow",
        "smoke/preflight",
        "#!/bin/bash",
        "#!/bin/sh",
    ]
    violations = [pattern for pattern in banned_patterns if pattern in execution_text]
    violations.extend(_dummy_data_violations(spec))
    violations.extend(_resource_contract_violations(spec, resources))
    if any(str(file.get("path", "")).strip().lower().endswith((".sh", ".bash")) for file in spec.get("files", []) if isinstance(file, dict)):
        violations.append("shell script file")
    for command in spec.get("commands", []):
        if not isinstance(command, dict):
            continue
        argv = command.get("argv", [])
        if isinstance(argv, list) and argv:
            executable = Path(str(argv[0])).name.lower()
            if executable in {"bash", "sh", "zsh"}:
                violations.append(f"shell command: {executable}")

    missing_resources: list[str] = []
    for resource in real_resources:
        local_path = str(Path(resource.local_path).resolve())
        path_name = Path(local_path).name.lower()
        relative_resource_path = f"resources/{path_name}"
        resource_name = resource.name.lower()
        if (
            local_path.lower() not in text
            and path_name not in text
            and relative_resource_path not in text
            and resource_name not in text
            and resource.location.lower() not in text
        ):
            missing_resources.append(f"{resource.role}:{resource.name} -> {local_path}")

    if violations or missing_resources:
        details: list[str] = []
        if violations:
            details.append("banned dummy/synthetic/preflight patterns: " + ", ".join(sorted(set(violations))))
        if missing_resources:
            details.append("materialized resources not referenced by the execution spec: " + "; ".join(missing_resources))
        raise GenericExecutionSpecValidationError(
            "Generic execution spec is not a real resource-backed experiment. " + " | ".join(details)
        )


def _normalize_command_argv(argv: Any) -> list[str]:
    if not isinstance(argv, list) or not argv:
        raise ValueError("Generic command argv must be a non-empty list.")
    normalized = [str(part) for part in argv]
    executable = Path(normalized[0]).name
    if executable in {"python", "python3"}:
        return [sys.executable, *normalized[1:]]
    if executable in {"pip", "pip3"}:
        return [sys.executable, "-m", "pip", *normalized[1:]]
    allowed = {Path(sys.executable).name, "nvidia-smi"}
    if executable not in allowed:
        raise ValueError(f"Generic command is not allowed: {normalized[0]}")
    return normalized


def _validate_command(command: dict[str, Any], workspace: Path) -> tuple[str, Path, list[str], int, dict[str, str]]:
    name = str(command.get("name", "command")).strip() or "command"
    cwd = safe_resolve_path(workspace, str(command.get("cwd", ".")))
    argv = _normalize_command_argv(command.get("argv"))
    timeout = int(command.get("timeout_seconds") or 3600)
    timeout = max(1, timeout)
    env_value = command.get("env") or {}
    if not isinstance(env_value, dict):
        raise ValueError(f"Generic command {name} env must be an object.")
    env = {str(key): str(value) for key, value in env_value.items()}
    banned_tokens = {"rm", "sudo", "chmod", "chown", "mkfs", "dd", "shutdown", "reboot"}
    for part in argv:
        if "\n" in part or "\r" in part:
            raise ValueError(f"Generic command {name} contains a newline in argv.")
        if Path(part).name in banned_tokens:
            raise ValueError(f"Generic command {name} contains a banned token: {part}")
    return name, cwd, argv, timeout, env


def _extract_experiment_table(spec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table = spec.get("experiment_table")
    if table in (None, []):
        return {}, []

    if isinstance(table, list):
        entrypoint = spec.get("entrypoint") or {}
        records = table
    elif isinstance(table, dict):
        entrypoint = table.get("entrypoint") or spec.get("entrypoint") or {}
        records = table.get("records") or table.get("rows") or table.get("experiments") or []
    else:
        raise ValueError("experiment_table must be an object or a list.")

    if not isinstance(entrypoint, dict):
        raise ValueError("experiment_table entrypoint must be an object.")
    if not isinstance(records, list):
        raise ValueError("experiment_table records must be a list.")
    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"experiment_table record {index} must be an object.")
        normalized_records.append(record)
    if normalized_records and not entrypoint:
        raise ValueError("experiment_table with records requires an entrypoint.")
    if normalized_records and (not isinstance(entrypoint.get("argv"), list) or not entrypoint.get("argv")):
        raise ValueError("experiment_table entrypoint requires non-empty list-valued argv.")
    return entrypoint, normalized_records


def _format_entrypoint_argv(
    argv: list[str],
    *,
    cell_file: Path,
    output_file: Path,
    cell_id: str,
    index: int,
    results_dir: Path,
    workspace: Path,
) -> list[str]:
    replacements = {
        "cell_file": str(cell_file),
        "output_file": str(output_file),
        "cell_id": cell_id,
        "index": str(index),
        "results_dir": str(results_dir),
        "workspace": str(workspace),
    }
    formatted: list[str] = []
    for part in argv:
        value = str(part)
        for key, replacement in replacements.items():
            value = value.replace("{" + key + "}", replacement)
        formatted.append(value)
    return formatted


def _parse_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        return {"parse_error": str(exc), "raw_tail": _tail_file(path, max_chars=4000)}


def _write_table_metrics_if_missing(
    results_dir: Path,
    table_results_file: Path,
    table_results: list[dict[str, Any]],
    *,
    force: bool = False,
) -> None:
    metrics_file = results_dir / "generic_metrics.json"
    if metrics_file.exists():
        if not force:
            return
        try:
            existing = json.loads(metrics_file.read_text(encoding="utf-8"))
        except JSONDecodeError:
            existing = None
        if not (
            isinstance(existing, dict)
            and existing.get("execution_mode") == "experiment_table"
            and "table_results_file" in existing
        ):
            return
    completed = [record for record in table_results if record.get("returncode") == 0]
    failed = [record for record in table_results if record.get("returncode") != 0]
    summary = {
        "execution_mode": "experiment_table",
        "num_records": len(table_results),
        "completed_records": len(completed),
        "failed_records": len(failed),
        "table_results_file": str(table_results_file),
        "records": [
            {
                "id": record.get("cell_id"),
                "returncode": record.get("returncode"),
                "output_file": record.get("output_file"),
                "output": record.get("parsed_output"),
            }
            for record in table_results
        ],
    }
    write_text(metrics_file, json.dumps(summary, ensure_ascii=False, indent=2))


def _table_gpu_policy(spec: dict[str, Any], entrypoint: dict[str, Any]) -> dict[str, Any]:
    policy = spec.get("gpu_policy") or entrypoint.get("gpu_policy") or {}
    if not isinstance(policy, dict):
        return {}
    return policy


def _number_from_mapping(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _record_estimated_gpu_memory_mb(record: dict[str, Any]) -> int | None:
    direct = _number_from_mapping(
        record,
        ("estimated_gpu_memory_mb", "gpu_memory_mb", "required_gpu_memory_mb"),
    )
    if direct is not None:
        return direct
    for nested_key in ("params", "target", "resources"):
        nested = record.get(nested_key)
        if isinstance(nested, dict):
            nested_value = _number_from_mapping(
                nested,
                ("estimated_gpu_memory_mb", "gpu_memory_mb", "required_gpu_memory_mb"),
            )
            if nested_value is not None:
                return nested_value
    return None


def _visible_gpu_indices_from_env(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            return None
        indices.add(int(part))
    return indices or None


def _build_greedy_gpu_slots(
    spec: dict[str, Any],
    entrypoint: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    hardware: Any,
    cuda_visible_devices: str | None,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    policy = _table_gpu_policy(spec, entrypoint)
    if policy.get("strategy") == "serial" or policy.get("enabled") is False:
        return [None], {"strategy": "serial", "reason": "gpu policy disabled"}
    if getattr(hardware, "accelerator", None) != "cuda":
        return [None], {"strategy": "serial", "reason": "non-cuda hardware"}

    snapshots = _query_gpu_memory_snapshots()
    if not snapshots:
        return [None], {"strategy": "serial", "reason": "no gpu memory snapshots"}

    visible_indices = _visible_gpu_indices_from_env(cuda_visible_devices)
    visible_snapshots = [
        snapshot
        for snapshot in snapshots
        if visible_indices is None or snapshot.index in visible_indices
    ]
    if not visible_snapshots:
        visible_snapshots = snapshots

    min_free_memory_mb = int(policy.get("min_free_memory_mb") or 4096)
    min_free_ratio = float(policy.get("min_free_ratio") or 0.2)
    candidates = [
        snapshot
        for snapshot in visible_snapshots
        if snapshot.free_memory_mb >= min_free_memory_mb
        and (snapshot.total_memory_mb <= 0 or snapshot.free_memory_mb / snapshot.total_memory_mb >= min_free_ratio)
    ]
    if not candidates:
        candidates = [max(visible_snapshots, key=lambda item: (item.free_memory_mb, -item.used_memory_mb, -item.index))]

    candidates = sorted(candidates, key=lambda item: (item.free_memory_mb, -item.used_memory_mb, -item.index), reverse=True)
    policy_estimate = _number_from_mapping(policy, ("estimated_cell_memory_mb", "estimated_gpu_memory_mb", "gpu_memory_mb"))
    record_estimates = [estimate for record in records if (estimate := _record_estimated_gpu_memory_mb(record)) is not None]
    estimated_cell_memory_mb = policy_estimate or (max(record_estimates) if record_estimates else None)
    reserve_memory_mb = int(policy.get("reserve_memory_mb") or 1024)
    max_workers_per_gpu = int(policy.get("max_workers_per_gpu") or (8 if estimated_cell_memory_mb else 1))
    max_parallel_cells = int(policy.get("max_parallel_cells") or len(records) or 1)

    slots: list[dict[str, Any] | None] = []
    slot_details: list[dict[str, Any]] = []
    for snapshot in candidates:
        if estimated_cell_memory_mb:
            available = max(0, snapshot.free_memory_mb - reserve_memory_mb)
            slot_count = max(1, available // estimated_cell_memory_mb)
        else:
            slot_count = 1
        slot_count = max(1, min(slot_count, max_workers_per_gpu))
        for slot_index in range(slot_count):
            if len(slots) >= max_parallel_cells:
                break
            slot = {
                "selected_gpu_index": snapshot.index,
                "cuda_visible_devices": str(snapshot.index),
                "free_memory_mb": snapshot.free_memory_mb,
                "used_memory_mb": snapshot.used_memory_mb,
                "total_memory_mb": snapshot.total_memory_mb,
                "slot_index": slot_index,
                "selection_policy": "greedy_free_memory_worker_pool",
            }
            slots.append(slot)
            slot_details.append(slot)
        if len(slots) >= max_parallel_cells:
            break

    return slots or [None], {
        "strategy": "greedy_gpu_worker_pool",
        "policy": policy,
        "estimated_cell_memory_mb": estimated_cell_memory_mb,
        "reserve_memory_mb": reserve_memory_mb,
        "max_workers_per_gpu": max_workers_per_gpu,
        "max_parallel_cells": max_parallel_cells,
        "snapshots": [
            {
                "index": snapshot.index,
                "free_memory_mb": snapshot.free_memory_mb,
                "used_memory_mb": snapshot.used_memory_mb,
                "total_memory_mb": snapshot.total_memory_mb,
            }
            for snapshot in snapshots
        ],
        "slots": slot_details,
    }


def _build_experiment_cell_command(
    *,
    entrypoint: dict[str, Any],
    cell: dict[str, Any],
    raw_cell_id: str,
    cell_id: str,
    index: int,
    cell_file: Path,
    output_file: Path,
    workspace: Path,
    results_dir: Path,
    request: ExperimentRequest,
    environment_prefix: Path,
    hardware: Any,
    log_index: int,
    attempt: int,
    gpu_slot: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str], int]:
    entrypoint_command = {
        "name": f"experiment_cell_{index:04d}_{cell_id}",
        "cwd": entrypoint.get("cwd", "."),
        "argv": _format_entrypoint_argv(
            [str(part) for part in entrypoint.get("argv", [])],
            cell_file=cell_file,
            output_file=output_file,
            cell_id=raw_cell_id,
            index=index,
            results_dir=results_dir,
            workspace=workspace,
        ),
        "timeout_seconds": entrypoint.get("timeout_seconds") or cell.get("timeout_seconds") or request.timeout_seconds,
        "env": entrypoint.get("env") or {},
    }
    name, cwd, argv, timeout, env = _validate_command(entrypoint_command, workspace)
    if argv and argv[0] == sys.executable:
        argv = [str(_environment_python(environment_prefix)), *argv[1:]]
    argv = _adapt_torch_pip_install_argv(argv, hardware)
    log_name = _command_log_name(name)
    stdout_file = results_dir / f"generic_attempt_{attempt:02d}_command_{log_index:02d}_{log_name}_stdout.txt"
    stderr_file = results_dir / f"generic_attempt_{attempt:02d}_command_{log_index:02d}_{log_name}_stderr.txt"
    env_overrides = dict(env)
    if gpu_slot is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu_slot["cuda_visible_devices"])
    return (
        {
            "index": log_index,
            "name": name,
            "kind": "experiment_table_record",
            "cell_index": index,
            "cell_id": raw_cell_id,
            "cwd": str(cwd),
            "argv": argv,
            "stdout_file": str(stdout_file),
            "stderr_file": str(stderr_file),
            "cell_file": str(cell_file),
            "output_file": str(output_file),
            "gpu_slot": gpu_slot,
        },
        env_overrides,
        min(timeout, request.timeout_seconds),
    )


def _run_experiment_cell_process(command_record: dict[str, Any], env_overrides: dict[str, str], timeout: int) -> dict[str, Any]:
    result = _run_process_streaming(
        command_record["argv"],
        cwd=Path(command_record["cwd"]),
        timeout=timeout,
        stdout_file=Path(command_record["stdout_file"]),
        stderr_file=Path(command_record["stderr_file"]),
        relay_stream=None,
        env_overrides=env_overrides,
    )
    output_file = Path(command_record["output_file"])
    parsed_output = _parse_json_file(output_file)
    output_contract_error = None
    if not output_file.exists():
        output_contract_error = f"Experiment table record {command_record['cell_id']} did not write output JSON: {output_file}"
    elif isinstance(parsed_output, dict) and "parse_error" in parsed_output:
        output_contract_error = f"Experiment table record {command_record['cell_id']} wrote invalid output JSON: {output_file}"
    return {
        **command_record,
        "returncode": result.returncode,
        "parsed_output": parsed_output,
        "output_contract_error": output_contract_error,
    }


def _run_experiment_table(
    spec: dict[str, Any],
    *,
    workspace: Path,
    results_dir: Path,
    request: ExperimentRequest,
    environment_prefix: Path,
    hardware: Any,
    gpu_selection: dict[str, Any] | None,
    command_results: list[dict[str, Any]],
    attempt: int,
) -> None:
    entrypoint, records = _extract_experiment_table(spec)
    if not records:
        return

    table_file = write_text(
        results_dir / f"generic_experiment_table_attempt_{attempt:02d}.json",
        json.dumps({"entrypoint": entrypoint, "records": records}, ensure_ascii=False, indent=2),
    )
    cells_dir = ensure_dir(results_dir / f"generic_experiment_cells_attempt_{attempt:02d}")
    table_results: list[dict[str, Any]] = []
    gpu_slots, schedule = _build_greedy_gpu_slots(
        spec,
        entrypoint,
        records,
        hardware=hardware,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    schedule_file = write_text(
        results_dir / f"generic_gpu_schedule_attempt_{attempt:02d}.json",
        json.dumps(schedule, ensure_ascii=False, indent=2),
    )
    prepared_cells: list[dict[str, Any]] = []

    for cell_index, cell in enumerate(records, start=1):
        raw_cell_id = str(cell.get("id") or f"cell_{cell_index:04d}").strip() or f"cell_{cell_index:04d}"
        cell_id = _command_log_name(raw_cell_id)
        cell_file = cells_dir / f"{cell_index:04d}_{cell_id}_input.json"
        output_file = cells_dir / f"{cell_index:04d}_{cell_id}_output.json"
        write_text(
            cell_file,
            json.dumps(
                {
                    "id": raw_cell_id,
                    "index": cell_index,
                    "workspace": str(workspace),
                    "results_dir": str(results_dir),
                    "table_file": str(table_file),
                    "gpu_schedule_file": str(schedule_file),
                    "record": cell,
                    "target": cell.get("target", {}),
                    "params": cell.get("params", {}),
                    "expected_metrics": cell.get("expected_metrics", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )
        prepared_cells.append(
            {
                "index": cell_index,
                "cell": cell,
                "raw_cell_id": raw_cell_id,
                "cell_id": cell_id,
                "cell_file": cell_file,
                "output_file": output_file,
            }
        )

    next_cell = 0
    next_log_index = len(command_results) + 1
    running: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    available_slots: list[dict[str, Any] | None] = list(gpu_slots) or [gpu_selection]
    table_results_file = results_dir / f"generic_experiment_table_results_attempt_{attempt:02d}.json"
    failure_record: dict[str, Any] | None = None
    max_workers = max(1, min(len(available_slots), len(prepared_cells)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_cell < len(prepared_cells) or running:
            while failure_record is None and next_cell < len(prepared_cells) and available_slots:
                prepared = prepared_cells[next_cell]
                gpu_slot = available_slots.pop(0)
                command_record, env_overrides, timeout = _build_experiment_cell_command(
                    entrypoint=entrypoint,
                    cell=prepared["cell"],
                    raw_cell_id=prepared["raw_cell_id"],
                    cell_id=prepared["cell_id"],
                    index=prepared["index"],
                    cell_file=prepared["cell_file"],
                    output_file=prepared["output_file"],
                    workspace=workspace,
                    results_dir=results_dir,
                    request=request,
                    environment_prefix=environment_prefix,
                    hardware=hardware,
                    log_index=next_log_index,
                    attempt=attempt,
                    gpu_slot=gpu_slot,
                )
                next_log_index += 1
                next_cell += 1
                future = executor.submit(_run_experiment_cell_process, command_record, env_overrides, timeout)
                running[future] = {"command_record": command_record, "gpu_slot": gpu_slot}

            if not running:
                break
            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                running_context = running.pop(future)
                available_slots.append(running_context["gpu_slot"])
                try:
                    record = future.result()
                except Exception as exc:
                    command_record = dict(running_context["command_record"])
                    record = {
                        **command_record,
                        "returncode": -1,
                        "parsed_output": None,
                        "output_contract_error": str(exc),
                    }
                command_results.append(record)
                table_results.append(record)
                write_text(table_results_file, json.dumps(table_results, ensure_ascii=False, indent=2))
                if record.get("returncode") != 0 or record.get("output_contract_error") is not None:
                    failure_record = record

    table_results = sorted(table_results, key=lambda item: int(item.get("cell_index") or item.get("index") or 0))
    write_text(
        table_results_file,
        json.dumps(table_results, ensure_ascii=False, indent=2),
    )
    _write_table_metrics_if_missing(results_dir, table_results_file, table_results, force=True)
    if failure_record is not None:
        failure_message = failure_record.get("output_contract_error") or (
            f"Generic experiment table record failed on attempt {attempt}: {failure_record.get('cell_id')}."
        )
        raise GenericCommandExecutionError(
            f"{failure_message} See {failure_record.get('stderr_file')}",
            command_results,
        )


def _looks_like_torch_requirement(part: str) -> bool:
    return re.match(r"^torch($|[<>=!~])", part) is not None


def _adapt_torch_pip_install_argv(argv: list[str], hardware: Any) -> list[str]:
    if len(argv) < 5 or argv[1:4] != ["-m", "pip", "install"]:
        return argv
    if "--index-url" in argv or "-i" in argv:
        return argv
    install_targets = [part for part in argv[4:] if not part.startswith("-")]
    if not any(_looks_like_torch_requirement(part) for part in install_targets):
        return argv
    if any(not _looks_like_torch_requirement(part) for part in install_targets):
        return argv
    if getattr(hardware, "accelerator", None) != "cuda" or not getattr(hardware, "torch_index_url", None):
        return argv
    return [*argv, "--index-url", str(hardware.torch_index_url)]


def _ensure_current_torch_runtime(
    hardware: Any,
    *,
    workspace: Path,
    results_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    if getattr(hardware, "accelerator", None) != "cuda":
        return None
    environment_prefix = Path(sys.prefix)
    verify_timeout = min(300, timeout_seconds)
    try:
        runtime = _verify_torch_runtime(
            environment_prefix,
            hardware,
            cwd=workspace,
            timeout=verify_timeout,
        )
    except Exception as first_error:
        install_timeout = min(1800, timeout_seconds)
        install, uninstall = _install_torch_runtime(
            environment_prefix,
            hardware,
            cwd=workspace,
            timeout=install_timeout,
        )
        write_text(results_dir / "torch_install_stdout.txt", (uninstall.stdout if uninstall else "") + install.stdout)
        write_text(results_dir / "torch_install_stderr.txt", (uninstall.stderr if uninstall else "") + install.stderr)
        if install.returncode != 0:
            raise RuntimeError(
                "PyTorch CUDA runtime verification failed and automatic reinstall did not complete. "
                f"Initial verification error: {first_error}. See {results_dir / 'torch_install_stderr.txt'}"
            )
        runtime = _verify_torch_runtime(
            environment_prefix,
            hardware,
            cwd=workspace,
            timeout=verify_timeout,
        )
    write_text(results_dir / "torch_runtime.json", json.dumps(runtime, ensure_ascii=False, indent=2))
    return runtime


def _prepare_generic_environment(
    request: ExperimentRequest,
    *,
    workspace: Path,
    results_dir: Path,
    run_id: str,
    hardware: Any,
) -> tuple[Path, Any, dict[str, Any]]:
    environment_file = _write_environment_file(workspace / "environment.yml", request)
    environment_cache_root = _environment_cache_root(request.workspace_root)
    requested_cache_spec = _environment_cache_spec(request, hardware, environment_file)
    requested_cache_key = _environment_cache_key(requested_cache_spec)
    cached_environment = _select_cached_environment(environment_cache_root, requested_cache_key)
    if cached_environment is None:
        cached_environment = _select_previous_completed_environment(request, requested_cache_key)
    reused_cached_environment = cached_environment is not None
    environment_prefix = (
        cached_environment[0]
        if cached_environment is not None
        else environment_cache_root / f"env-{requested_cache_key}"
    )
    cloned_current_environment = False
    cloned_runtime: dict[str, Any] | None = None

    if reused_cached_environment:
        setup = subprocess.CompletedProcess(
            ["conda", "env", "reuse", "--prefix", str(environment_prefix)],
            0,
            f"Reusing cached experiment environment: {environment_prefix}\n",
            "",
        )
    else:
        cloned = _try_clone_current_environment(
            request,
            hardware,
            environment_prefix=environment_prefix,
            cache_root=environment_cache_root,
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
        )
        if cloned is not None:
            setup, cloned_runtime = cloned
            cloned_current_environment = True
        elif environment_prefix.exists():
            setup = _run_process(
                [
                    "conda",
                    "env",
                    "update",
                    "--prefix",
                    str(environment_prefix),
                    "--file",
                    str(environment_file),
                    "--prune",
                ],
                cwd=PROJECT_ROOT,
                timeout=request.timeout_seconds,
            )
        else:
            setup = _run_process(
                [
                    "conda",
                    "env",
                    "create",
                    "--prefix",
                    str(environment_prefix),
                    "--file",
                    str(environment_file),
                ],
                cwd=PROJECT_ROOT,
                timeout=request.timeout_seconds,
            )
    write_text(results_dir / "environment_stdout.txt", setup.stdout)
    write_text(results_dir / "environment_stderr.txt", setup.stderr)
    if setup.returncode != 0:
        raise RuntimeError(f"Conda environment creation failed. See {results_dir / 'environment_stderr.txt'}")

    if reused_cached_environment or cloned_current_environment:
        torch_stdout = (
            f"Reusing PyTorch runtime from cached environment: {environment_prefix}\n"
            if reused_cached_environment
            else f"Reusing PyTorch runtime cloned from current environment: {environment_prefix}\n"
        )
        torch_install = subprocess.CompletedProcess(
            ["pip", "install", "torch", "reuse"],
            0,
            torch_stdout,
            "",
        )
        torch_uninstall = None
    else:
        torch_install, torch_uninstall = _install_torch_runtime(
            environment_prefix,
            hardware,
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
        )
    write_text(results_dir / "torch_install_stdout.txt", (torch_uninstall.stdout if torch_uninstall else "") + torch_install.stdout)
    write_text(results_dir / "torch_install_stderr.txt", (torch_uninstall.stderr if torch_uninstall else "") + torch_install.stderr)
    if torch_install.returncode != 0:
        raise RuntimeError(f"PyTorch installation failed. See {results_dir / 'torch_install_stderr.txt'}")

    dependencies = _install_experiment_dependencies(
        environment_prefix,
        cwd=PROJECT_ROOT,
        timeout=request.timeout_seconds,
    )
    write_text(results_dir / "dependencies_stdout.txt", dependencies.stdout)
    write_text(results_dir / "dependencies_stderr.txt", dependencies.stderr)
    if dependencies.returncode != 0:
        raise RuntimeError(f"Experiment dependency installation failed. See {results_dir / 'dependencies_stderr.txt'}")

    runtime = cloned_runtime or _verify_torch_runtime(
        environment_prefix,
        hardware,
        cwd=PROJECT_ROOT,
        timeout=request.timeout_seconds,
    )
    if cloned_runtime is not None:
        runtime = _verify_torch_runtime(
            environment_prefix,
            hardware,
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
        )
    runtime_file = write_text(results_dir / "torch_runtime.json", json.dumps(runtime, ensure_ascii=False, indent=2))
    verified_hardware = _record_verified_runtime(request.hardware_profile_file, hardware, runtime)
    actual_cache_spec = _environment_cache_spec(request, verified_hardware, environment_file, runtime=runtime)
    actual_cache_key = _environment_cache_key(actual_cache_spec)
    environment_cache_metadata = _record_cached_environment(
        environment_cache_root,
        environment_prefix,
        cache_key=actual_cache_key,
        requested_cache_key=requested_cache_key,
        spec=actual_cache_spec,
        run_id=run_id,
        environment_file=environment_file,
        hardware=verified_hardware,
        runtime=runtime,
        reused=reused_cached_environment,
    )
    environment_cache_file = write_text(
        results_dir / "environment_cache.json",
        json.dumps(environment_cache_metadata, ensure_ascii=False, indent=2),
    )
    return environment_prefix, verified_hardware, {
        "environment_file": str(environment_file),
        "environment_prefix": str(environment_prefix),
        "environment_cache_file": str(environment_cache_file),
        "torch_runtime_file": str(runtime_file),
    }


def _command_log_name(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in name.strip())
    return cleaned.strip("._-") or "command"


def _run_generic_commands(
    spec: dict[str, Any],
    *,
    workspace: Path,
    results_dir: Path,
    request: ExperimentRequest,
    environment_prefix: Path,
    hardware: Any,
    attempt: int = 1,
) -> list[dict[str, Any]]:
    gpu_selection = _select_execution_gpu(hardware, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    if gpu_selection is not None:
        write_text(results_dir / "gpu_selection.json", json.dumps(gpu_selection, ensure_ascii=False, indent=2))
    python_executable = str(_environment_python(environment_prefix))

    command_results: list[dict[str, Any]] = []
    for index, command in enumerate(spec.get("commands", []), start=1):
        if not isinstance(command, dict):
            raise ValueError("Each generic command entry must be an object.")
        name, cwd, argv, timeout, env = _validate_command(command, workspace)
        if argv and argv[0] == sys.executable:
            argv = [python_executable, *argv[1:]]
        argv = _adapt_torch_pip_install_argv(argv, hardware)
        log_name = _command_log_name(name)
        stdout_file = results_dir / f"generic_attempt_{attempt:02d}_command_{index:02d}_{log_name}_stdout.txt"
        stderr_file = results_dir / f"generic_attempt_{attempt:02d}_command_{index:02d}_{log_name}_stderr.txt"
        env_overrides = dict(env)
        if gpu_selection is not None:
            env_overrides["CUDA_VISIBLE_DEVICES"] = str(gpu_selection["cuda_visible_devices"])
        result = _run_process_streaming(
            argv,
            cwd=cwd,
            timeout=min(timeout, request.timeout_seconds),
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            env_overrides=env_overrides,
        )
        record = {
            "index": index,
            "name": name,
            "cwd": str(cwd),
            "argv": argv,
            "returncode": result.returncode,
            "stdout_file": str(stdout_file),
            "stderr_file": str(stderr_file),
        }
        command_results.append(record)
        if result.returncode != 0:
            raise GenericCommandExecutionError(
                f"Generic command failed on attempt {attempt}: {name}. See {stderr_file}",
                command_results,
            )
    _run_experiment_table(
        spec,
        workspace=workspace,
        results_dir=results_dir,
        request=request,
        environment_prefix=environment_prefix,
        hardware=hardware,
        gpu_selection=gpu_selection,
        command_results=command_results,
        attempt=attempt,
    )
    return command_results


def _write_json_response_error_files(
    run_results: Path,
    *,
    prefix: str,
    error: GenericJsonResponseError,
) -> dict[str, Path]:
    files = {
        "prompt": write_text(run_results / f"{prefix}_prompt.md", error.prompt),
        "response": write_text(run_results / f"{prefix}_response.txt", error.raw_response),
        "error": write_text(run_results / f"{prefix}_json_error.txt", str(error)),
    }
    for index, repair_response in enumerate(error.repair_responses, start=1):
        files[f"repair_{index}"] = write_text(
            run_results / f"{prefix}_repair_response_attempt_{index:02d}.txt",
            repair_response,
        )
    return files


def run_generic_experiment_agent(
    request: ExperimentRequest,
    *,
    execute: bool = True,
    progress_callback=None,
) -> ExperimentRunState:
    if progress_callback is not None:
        progress_callback("initialize")
    run_id = unique_run_id(request.run_name) if request.run_name_is_prefix else request.run_name or unique_run_id()
    workspace = ensure_dir(request.workspace_root / run_id)
    run_results = ensure_dir(request.results_root / run_id)
    request_file = write_text(
        run_results / "request.json",
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2),
    )
    state = ExperimentRunState(status="generic_planning", run_id=run_id, request_file=str(request_file))
    write_text(run_results / "state.json", state.model_dump_json(indent=2))

    try:
        if progress_callback is not None:
            progress_callback("request_generic_plan")
        plan, prompt, response = request_generic_experiment_plan(request)
        prompt_file = write_text(run_results / "generic_plan_prompt.md", prompt)
        response_file = write_text(run_results / "generic_plan_response.txt", response)
        plan_file = write_text(run_results / "generic_plan.json", json.dumps(plan, ensure_ascii=False, indent=2))
        state.plan_prompt_file = str(prompt_file)
        state.plan_response_file = str(response_file)
        state.plan_file = str(plan_file)
        state.status = "generic_planned"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        if not execute:
            return state

        if progress_callback is not None:
            progress_callback("materialize_resources")
        state.status = "generic_materializing"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        resources = materialize_generic_resources(request, workspace)
        resources_file = write_text(
            run_results / "generic_resources.json",
            json.dumps([asdict(resource) for resource in resources], ensure_ascii=False, indent=2),
        )

        if progress_callback is not None:
            progress_callback("request_generic_execution")
        state.status = "generic_implementation_spec"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        try:
            spec, execution_prompt, execution_response = request_generic_execution_spec(
                request,
                plan=plan,
                resources=resources,
                workspace=workspace,
                results_dir=run_results,
            )
        except GenericJsonResponseError as exc:
            files = _write_json_response_error_files(run_results, prefix="generic_execution", error=exc)
            state.implementation_prompt_file = str(files["prompt"])
            state.implementation_response_file = str(files["response"])
            state.implementation_error_file = str(files["error"])
            write_text(run_results / "state.json", state.model_dump_json(indent=2))
            raise
        execution_prompt_file = write_text(run_results / "generic_execution_prompt.md", execution_prompt)
        execution_response_file = write_text(run_results / "generic_execution_response.txt", execution_response)
        execution_spec_file = write_text(
            run_results / "generic_execution_spec.json",
            json.dumps(spec, ensure_ascii=False, indent=2),
        )
        state.implementation_prompt_file = str(execution_prompt_file)
        state.implementation_response_file = str(execution_response_file)
        state.implementation_file = str(execution_spec_file)

        if progress_callback is not None:
            progress_callback("prepare_environment")
        state.status = "generic_preparing_environment"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        hardware = _resolve_hardware_profile(
            request.hardware_profile_file,
            refresh=request.refresh_hardware_profile,
        )
        hardware_file = write_text(run_results / "hardware.json", json.dumps(asdict(hardware), indent=2))
        environment_prefix, hardware, environment_info = _prepare_generic_environment(
            request,
            workspace=workspace,
            results_dir=run_results,
            run_id=run_id,
            hardware=hardware,
        )
        hardware_file = write_text(run_results / "hardware.json", json.dumps(asdict(hardware), indent=2))
        state.hardware_file = str(hardware_file)
        state.environment_file = environment_info["environment_file"]
        state.environment_prefix = environment_info["environment_prefix"]
        state.environment_cache_file = environment_info["environment_cache_file"]
        state.torch_runtime_file = environment_info["torch_runtime_file"]
        write_text(run_results / "state.json", state.model_dump_json(indent=2))

        current_spec = spec
        generated_files_file: Path | None = None
        command_results: list[dict[str, Any]] = []
        execution_attempts: list[dict[str, Any]] = []
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                validate_real_resource_execution_spec(current_spec, resources)
            except GenericExecutionSpecValidationError as exc:
                failure_record = {
                    "attempt": attempt,
                    "status": "failed_validation",
                    "error": str(exc),
                    "commands": [],
                }
                execution_attempts.append(failure_record)
                write_text(
                    run_results / f"generic_validation_failure_attempt_{attempt:02d}.json",
                    json.dumps(failure_record, ensure_ascii=False, indent=2),
                )
                if attempt >= max_attempts:
                    raise
                if progress_callback is not None:
                    progress_callback("debug_generic_execution")
                state.status = "generic_debugging"
                write_text(run_results / "state.json", state.model_dump_json(indent=2))
                try:
                    debug_spec, debug_prompt, debug_response = request_generic_debug_execution_spec(
                        request,
                        plan=plan,
                        resources=resources,
                        previous_spec=current_spec,
                        command_results=[
                            {
                                "name": "pre_execution_real_resource_validation",
                                "returncode": 1,
                            }
                        ],
                        failure_summary="Pre-execution validation failed: " + str(exc),
                        workspace=workspace,
                        results_dir=run_results,
                        attempt=attempt,
                    )
                except GenericJsonResponseError as json_exc:
                    files = _write_json_response_error_files(
                        run_results,
                        prefix=f"generic_debug_attempt_{attempt:02d}",
                        error=json_exc,
                    )
                    state.implementation_prompt_file = str(files["prompt"])
                    state.implementation_response_file = str(files["response"])
                    state.implementation_error_file = str(files["error"])
                    write_text(run_results / "state.json", state.model_dump_json(indent=2))
                    raise
                debug_prompt_file = write_text(run_results / f"generic_debug_prompt_attempt_{attempt:02d}.md", debug_prompt)
                debug_response_file = write_text(
                    run_results / f"generic_debug_response_attempt_{attempt:02d}.txt",
                    debug_response,
                )
                debug_spec_file = write_text(
                    run_results / f"generic_debug_spec_attempt_{attempt:02d}.json",
                    json.dumps(debug_spec, ensure_ascii=False, indent=2),
                )
                state.implementation_prompt_file = str(debug_prompt_file)
                state.implementation_response_file = str(debug_response_file)
                state.implementation_file = str(debug_spec_file)
                current_spec = debug_spec
                continue

            generated_files = _write_generated_files(current_spec, workspace)
            generated_files_file = write_text(
                run_results / f"generic_generated_files_attempt_{attempt:02d}.json",
                json.dumps(generated_files, ensure_ascii=False, indent=2),
            )
            state.implementation_workspace_file = str(generated_files_file)

            if progress_callback is not None:
                progress_callback("run_generic_commands")
            state.status = "generic_running" if attempt == 1 else "generic_debug_running"
            write_text(run_results / "state.json", state.model_dump_json(indent=2))
            try:
                command_results = _run_generic_commands(
                    current_spec,
                    workspace=workspace,
                    results_dir=run_results,
                    request=request,
                    environment_prefix=environment_prefix,
                    hardware=hardware,
                    attempt=attempt,
                )
                execution_attempts.append(
                    {
                        "attempt": attempt,
                        "status": "completed",
                        "generated_files_file": str(generated_files_file),
                        "commands": command_results,
                    }
                )
                break
            except GenericCommandExecutionError as exc:
                command_results = exc.command_results
                failure_record = {
                    "attempt": attempt,
                    "status": "failed",
                    "generated_files_file": str(generated_files_file),
                    "error": str(exc),
                    "commands": command_results,
                }
                execution_attempts.append(failure_record)
                write_text(
                    run_results / f"generic_failure_attempt_{attempt:02d}.json",
                    json.dumps(failure_record, ensure_ascii=False, indent=2),
                )
                if attempt >= max_attempts:
                    raise
                if progress_callback is not None:
                    progress_callback("debug_generic_execution")
                state.status = "generic_debugging"
                write_text(run_results / "state.json", state.model_dump_json(indent=2))
                try:
                    debug_spec, debug_prompt, debug_response = request_generic_debug_execution_spec(
                        request,
                        plan=plan,
                        resources=resources,
                        previous_spec=current_spec,
                        command_results=command_results,
                        failure_summary=str(exc),
                        workspace=workspace,
                        results_dir=run_results,
                        attempt=attempt,
                    )
                except GenericJsonResponseError as json_exc:
                    files = _write_json_response_error_files(
                        run_results,
                        prefix=f"generic_debug_attempt_{attempt:02d}",
                        error=json_exc,
                    )
                    state.implementation_prompt_file = str(files["prompt"])
                    state.implementation_response_file = str(files["response"])
                    state.implementation_error_file = str(files["error"])
                    write_text(run_results / "state.json", state.model_dump_json(indent=2))
                    raise
                debug_prompt_file = write_text(run_results / f"generic_debug_prompt_attempt_{attempt:02d}.md", debug_prompt)
                debug_response_file = write_text(
                    run_results / f"generic_debug_response_attempt_{attempt:02d}.txt",
                    debug_response,
                )
                debug_spec_file = write_text(
                    run_results / f"generic_debug_spec_attempt_{attempt:02d}.json",
                    json.dumps(debug_spec, ensure_ascii=False, indent=2),
                )
                state.implementation_prompt_file = str(debug_prompt_file)
                state.implementation_response_file = str(debug_response_file)
                state.implementation_file = str(debug_spec_file)
                current_spec = debug_spec
        else:
            raise RuntimeError("Generic execution did not run any attempts.")

        execution_summary = {
            "status": "generic_completed",
            "run_id": run_id,
            "resources_file": str(resources_file),
            "execution_spec_file": str(execution_spec_file),
            "generated_files_file": str(generated_files_file) if generated_files_file is not None else None,
            "attempts": execution_attempts,
            "commands": command_results,
            "expected_outputs": current_spec.get("expected_outputs", []),
            "notes": current_spec.get("notes", ""),
        }
        summary_file = write_text(
            run_results / "generic_execution_summary.json",
            json.dumps(execution_summary, ensure_ascii=False, indent=2),
        )
        metrics_candidate = run_results / "generic_metrics.json"
        state.metrics_file = str(metrics_candidate if metrics_candidate.exists() else summary_file)
        state.report_file = str(summary_file)
        state.stdout_file = next(
            (str(command["stdout_file"]) for command in command_results if "stdout_file" in command),
            None,
        )
        state.stderr_file = next(
            (str(command["stderr_file"]) for command in command_results if "stderr_file" in command),
            None,
        )
        state.status = "generic_completed"
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)
        write_text(run_results / "generic_error.txt", str(exc))

    write_text(run_results / "state.json", state.model_dump_json(indent=2))
    return state
