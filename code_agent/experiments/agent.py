from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, TextIO

import yaml

from code_agent.experiments.implementer import (
    ImplementationGenerationError,
    build_implementation_prompt,
    request_implementation,
)
from code_agent.experiments.models import ExperimentRequest, ExperimentRunState, default_run_id
from code_agent.experiments.planner import prepare_plan_request, request_experiment_plan
from code_agent.tools.file_tools import ensure_dir, write_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ProgressCallback = Callable[[str], None]
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_INSTALL_ATTEMPTS = 3


@dataclass(frozen=True)
class HardwareProfile:
    accelerator: str
    gpu_name: str | None = None
    driver_version: str | None = None
    reported_cuda_version: str | None = None
    torch_index_url: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    verified_device_name: str | None = None
    verified_at: str | None = None


@dataclass(frozen=True)
class GpuMemorySnapshot:
    index: int
    free_memory_mb: int
    used_memory_mb: int
    total_memory_mb: int


def _configured_conda_url_channels() -> list[str]:
    try:
        result = subprocess.run(
            ["conda", "config", "--show", "channels", "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError):
        return []
    if result.returncode != 0:
        return []

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return []

    channels = data.get("channels", [])
    if not isinstance(channels, list):
        return []
    return [
        channel
        for channel in channels
        if isinstance(channel, str) and channel.startswith(("https://", "http://"))
    ]


def _write_environment_file(path: Path, request: ExperimentRequest) -> Path:
    channels = _configured_conda_url_channels()
    environment = {
        "dependencies": [
            f"python={request.environment_python}",
            "pip",
            "git",
        ],
    }
    if channels:
        environment = {"channels": [*channels, "nodefaults"], **environment}
    return write_text(path, yaml.safe_dump(environment, sort_keys=False))


def _read_environment_definition(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _torch_requirement_for_cache(hardware: HardwareProfile, runtime: dict | None = None) -> str:
    version = runtime.get("torch_version") if runtime else hardware.torch_version
    return f"torch=={version}" if version else "torch>=2.4"


def _environment_cache_spec(
    request: ExperimentRequest,
    hardware: HardwareProfile,
    environment_file: Path,
    *,
    runtime: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "environment_python": request.environment_python,
        "environment": _read_environment_definition(environment_file),
        "accelerator": hardware.accelerator,
        "torch_requirement": _torch_requirement_for_cache(hardware, runtime),
        "torch_index_url": hardware.torch_index_url if hardware.accelerator == "cuda" else None,
    }


def _environment_cache_key(spec: dict) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _environment_cache_root(workspace_root: Path) -> Path:
    return ensure_dir(workspace_root / "asset_cache" / "environments")


def _environment_registry_file(cache_root: Path) -> Path:
    return cache_root / "registry.json"


def _read_environment_registry(cache_root: Path) -> dict:
    registry_file = _environment_registry_file(cache_root)
    if not registry_file.exists():
        return {"schema_version": 1, "latest": {}}
    try:
        data = json.loads(registry_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"schema_version": 1, "latest": {}}
    if not isinstance(data, dict):
        return {"schema_version": 1, "latest": {}}
    latest = data.get("latest")
    if not isinstance(latest, dict):
        data["latest"] = {}
    data.setdefault("schema_version", 1)
    return data


def _select_cached_environment(cache_root: Path, cache_key: str) -> tuple[Path, dict] | None:
    registry = _read_environment_registry(cache_root)
    entry = registry.get("latest", {}).get(cache_key)
    if not isinstance(entry, dict) or not entry.get("prefix"):
        return None
    prefix = Path(entry["prefix"])
    if not _environment_python(prefix).exists():
        return None
    return prefix, entry


def _select_previous_completed_environment(request: ExperimentRequest, cache_key: str) -> tuple[Path, dict] | None:
    results_root = Path(request.results_root).expanduser()
    if not results_root.exists():
        return None
    state_files = sorted(
        results_root.glob("*/state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for state_file in state_files:
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(state, dict) or state.get("status") != "completed":
            continue
        environment_prefix = Path(str(state.get("environment_prefix", "")))
        environment_file = Path(str(state.get("environment_file", "")))
        hardware_file = Path(str(state.get("hardware_file", "")))
        runtime_file = Path(str(state.get("torch_runtime_file", "")))
        if not all(path.exists() for path in (environment_file, hardware_file, runtime_file)):
            continue
        if not _environment_python(environment_prefix).exists():
            continue
        try:
            hardware_data = json.loads(hardware_file.read_text(encoding="utf-8"))
            runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(hardware_data, dict) or not isinstance(runtime, dict):
            continue
        allowed = set(HardwareProfile.__dataclass_fields__)
        try:
            hardware = HardwareProfile(**{key: value for key, value in hardware_data.items() if key in allowed})
        except TypeError:
            continue
        previous_spec = _environment_cache_spec(request, hardware, environment_file, runtime=runtime)
        if _environment_cache_key(previous_spec) != cache_key:
            continue
        return environment_prefix, {
            "source": "previous_completed_run",
            "run_id": state.get("run_id"),
            "state_file": str(state_file),
        }
    return None


def _record_cached_environment(
    cache_root: Path,
    environment_prefix: Path,
    *,
    cache_key: str,
    requested_cache_key: str,
    spec: dict,
    run_id: str,
    environment_file: Path,
    hardware: HardwareProfile,
    runtime: dict,
    reused: bool,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1,
        "cache_key": cache_key,
        "requested_cache_key": requested_cache_key,
        "run_id": run_id,
        "environment_prefix": str(environment_prefix),
        "environment_file": str(environment_file),
        "reused": reused,
        "updated_at": now,
        "spec": spec,
        "hardware": asdict(hardware),
        "runtime": runtime,
    }
    shared_metadata_file = write_text(
        environment_prefix / ".code_agent_environment.json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
    )
    metadata["shared_metadata_file"] = str(shared_metadata_file)

    registry = _read_environment_registry(cache_root)
    registry.setdefault("latest", {})
    registry["latest"][cache_key] = {
        "prefix": str(environment_prefix),
        "metadata_file": str(shared_metadata_file),
        "run_id": run_id,
        "updated_at": now,
    }
    write_text(_environment_registry_file(cache_root), json.dumps(registry, ensure_ascii=False, indent=2))
    return metadata


def _run_process(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _run_nvidia_smi(command: list[str], *, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    result = _run_process(command, cwd=PROJECT_ROOT, timeout=timeout)
    if result.returncode == 0:
        return result
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    try:
        fallback = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=environment,
        )
    except (FileNotFoundError, subprocess.SubprocessError, TimeoutError):
        return result
    return fallback if fallback.returncode == 0 else result


def _detect_hardware() -> HardwareProfile:
    try:
        result = _run_nvidia_smi(["nvidia-smi"], timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return HardwareProfile(accelerator="cpu")
    if result.returncode != 0:
        return HardwareProfile(accelerator="cpu")

    details = _run_nvidia_smi(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        timeout=10,
    )
    detail_parts = [part.strip() for part in details.stdout.splitlines()[0].split(",")] if details.stdout.strip() else []
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", result.stdout)
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", result.stdout)
    return HardwareProfile(
        accelerator="cuda",
        gpu_name=detail_parts[0] if detail_parts else "NVIDIA GPU",
        driver_version=detail_parts[1] if len(detail_parts) > 1 else (driver_match.group(1) if driver_match else None),
        reported_cuda_version=cuda_match.group(1) if cuda_match else None,
        torch_index_url=PYTORCH_CUDA_INDEX_URL,
    )


def _parse_gpu_memory_snapshot(row: str) -> GpuMemorySnapshot | None:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) != 4:
        return None
    try:
        return GpuMemorySnapshot(
            index=int(parts[0]),
            free_memory_mb=int(parts[1]),
            used_memory_mb=int(parts[2]),
            total_memory_mb=int(parts[3]),
        )
    except ValueError:
        return None


def _query_gpu_memory_snapshots() -> list[GpuMemorySnapshot]:
    try:
        result = _run_nvidia_smi(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    snapshots: list[GpuMemorySnapshot] = []
    for line in result.stdout.splitlines():
        snapshot = _parse_gpu_memory_snapshot(line)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def _visible_gpu_indices(value: str | None) -> set[int] | None:
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


def _select_execution_gpu(hardware: HardwareProfile, *, cuda_visible_devices: str | None = None) -> dict | None:
    if hardware.accelerator != "cuda":
        return None
    snapshots = _query_gpu_memory_snapshots()
    if not snapshots:
        return None

    visible_indices = _visible_gpu_indices(cuda_visible_devices)
    candidates = [
        snapshot
        for snapshot in snapshots
        if visible_indices is None or snapshot.index in visible_indices
    ]
    if not candidates:
        candidates = snapshots
    selected = max(candidates, key=lambda item: (item.free_memory_mb, -item.used_memory_mb, -item.index))
    return {
        "selected_gpu_index": selected.index,
        "cuda_visible_devices": str(selected.index),
        "free_memory_mb": selected.free_memory_mb,
        "used_memory_mb": selected.used_memory_mb,
        "total_memory_mb": selected.total_memory_mb,
        "selection_policy": "max_free_memory_mb",
    }


def _read_hardware_profile(path: str | Path) -> HardwareProfile | None:
    profile_file = Path(path).expanduser()
    if not profile_file.exists():
        return None
    data = yaml.safe_load(profile_file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or data.get("accelerator") not in {"cpu", "cuda"}:
        return None
    if data["accelerator"] == "cuda" and not data.get("torch_index_url"):
        return None
    allowed = set(HardwareProfile.__dataclass_fields__)
    return HardwareProfile(**{key: value for key, value in data.items() if key in allowed})


def _write_hardware_profile(path: str | Path, hardware: HardwareProfile) -> Path:
    return write_text(path, yaml.safe_dump(asdict(hardware), sort_keys=False, allow_unicode=True))


def _resolve_hardware_profile(path: str | Path, *, refresh: bool = False) -> HardwareProfile:
    if not refresh:
        cached = _read_hardware_profile(path)
        if cached is not None:
            return cached
    detected = _detect_hardware()
    _write_hardware_profile(path, detected)
    return detected


def _record_verified_runtime(path: str | Path, hardware: HardwareProfile, runtime: dict) -> HardwareProfile:
    verified = replace(
        hardware,
        torch_version=runtime.get("torch_version"),
        torch_cuda_version=runtime.get("torch_cuda_version"),
        verified_device_name=runtime.get("device_name"),
        verified_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_hardware_profile(path, verified)
    return verified


def _environment_python(environment_prefix: Path) -> Path:
    executable = "python.exe" if os.name == "nt" else "bin/python"
    return environment_prefix / executable


def _install_torch_runtime(
    environment_prefix: Path,
    hardware: HardwareProfile,
    *,
    cwd: Path,
    timeout: int,
) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str] | None]:
    python = str(_environment_python(environment_prefix))
    uninstall = None
    torch_requirement = f"torch=={hardware.torch_version}" if hardware.torch_version else "torch>=2.4"
    command = [
        python,
        "-m",
        "pip",
        "install",
        torch_requirement,
        "--retries",
        "10",
        "--timeout",
        "120",
    ]
    if hardware.accelerator == "cuda":
        uninstall = _run_process([python, "-m", "pip", "uninstall", "-y", "torch"], cwd=cwd, timeout=timeout)
        command.extend(["--index-url", hardware.torch_index_url or PYTORCH_CUDA_INDEX_URL])
    attempts: list[subprocess.CompletedProcess[str]] = []
    for attempt in range(1, TORCH_INSTALL_ATTEMPTS + 1):
        install = _run_process(command, cwd=cwd, timeout=timeout)
        attempts.append(install)
        if install.returncode == 0:
            return install, uninstall

    install = attempts[-1]
    combined_stdout = "\n\n".join(
        f"--- torch install attempt {index} stdout ---\n{attempt.stdout}"
        for index, attempt in enumerate(attempts, start=1)
    )
    combined_stderr = "\n\n".join(
        f"--- torch install attempt {index} stderr ---\n{attempt.stderr}"
        for index, attempt in enumerate(attempts, start=1)
    )
    install = subprocess.CompletedProcess(install.args, install.returncode, combined_stdout, combined_stderr)
    return install, uninstall


def _install_experiment_dependencies(environment_prefix: Path, *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return _run_process(
        [str(_environment_python(environment_prefix)), "-m", "pip", "install", "-e", f"{PROJECT_ROOT}[experiment]"],
        cwd=cwd,
        timeout=timeout,
    )


def _verify_torch_runtime(environment_prefix: Path, hardware: HardwareProfile, *, cwd: Path, timeout: int) -> dict:
    python = str(_environment_python(environment_prefix))
    inspect = _run_process(
        [
            python,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch_version': torch.__version__, "
                "'cuda_available': torch.cuda.is_available(), "
                "'torch_cuda_version': torch.version.cuda, "
                "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
            ),
        ],
        cwd=cwd,
        timeout=timeout,
    )
    if inspect.returncode != 0:
        raise RuntimeError(f"Unable to validate PyTorch runtime: {inspect.stderr.strip()}")
    runtime = json.loads(inspect.stdout.strip())
    if hardware.accelerator == "cuda" and not runtime["cuda_available"]:
        raise RuntimeError("NVIDIA GPU detected, but the installed PyTorch runtime cannot access CUDA.")
    return runtime


def _stream_pipe(pipe: BinaryIO, targets: list[TextIO], output: list[str]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            chunk = pipe.read1(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if not text:
                continue
            output.append(text)
            for target in targets:
                target.write(text)
                target.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            output.append(tail)
            for target in targets:
                target.write(tail)
                target.flush()
    finally:
        pipe.close()


def _run_process_streaming(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    stdout_file: Path,
    stderr_file: Path,
    relay_stream: TextIO | None = sys.stderr,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if env_overrides:
        environment.update(env_overrides)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    stdout: list[str] = []
    stderr: list[str] = []
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    stderr_file.parent.mkdir(parents=True, exist_ok=True)
    with stdout_file.open("w", encoding="utf-8", newline="") as stdout_target, stderr_file.open(
        "w", encoding="utf-8", newline=""
    ) as stderr_target:
        stdout_targets = [stdout_target, relay_stream] if relay_stream is not None else [stdout_target]
        stderr_targets = [stderr_target, relay_stream] if relay_stream is not None else [stderr_target]
        threads = [
            threading.Thread(target=_stream_pipe, args=(process.stdout, stdout_targets, stdout), daemon=True),
            threading.Thread(target=_stream_pipe, args=(process.stderr, stderr_targets, stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            returncode = -1
            message = f"\nProcess timed out after {timeout} seconds.\n"
            stderr.append(message)
            stderr_target.write(message)
            stderr_target.flush()
            if relay_stream is not None:
                relay_stream.write(message)
                relay_stream.flush()
        for thread in threads:
            thread.join()
    return subprocess.CompletedProcess(command, returncode, "".join(stdout), "".join(stderr))


def run_experiment_agent(
    request: ExperimentRequest,
    *,
    plan_only: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> ExperimentRunState:
    if progress_callback is not None:
        progress_callback("initialize")
    run_id = request.run_name or f"{default_run_id()}-{uuid.uuid4().hex[:8]}"
    workspace = ensure_dir(request.workspace_root / run_id)
    run_results = ensure_dir(request.results_root / run_id)
    request_file = write_text(
        run_results / "request.json",
        json.dumps(request.model_dump(mode="json"), ensure_ascii=False, indent=2),
    )
    state = ExperimentRunState(status="planning", run_id=run_id, request_file=str(request_file))
    write_text(run_results / "state.json", state.model_dump_json(indent=2))

    try:
        if progress_callback is not None:
            progress_callback("request_plan")
        model_id, dataset_id, prompt = prepare_plan_request(request)
        prompt_file = write_text(run_results / "plan_prompt.md", prompt)
        state.plan_prompt_file = str(prompt_file)
        state.status = "requesting_plan"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        try:
            plan, response = request_experiment_plan(
                request,
                model_id=model_id,
                dataset_id=dataset_id,
                prompt=prompt,
            )
        except Exception as exc:
            error_file = write_text(run_results / "plan_api_error.txt", str(exc))
            state.plan_error_file = str(error_file)
            raise
        response_file = write_text(run_results / "plan_response.txt", response)
        plan_file = write_text(run_results / "plan.json", plan.model_dump_json(indent=2))
        state.plan_file = str(plan_file)
        state.plan_response_file = str(response_file)
        state.status = "planned"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        if plan_only:
            return state

        if progress_callback is not None:
            progress_callback("implement_improvement")
        implementation_prompt = build_implementation_prompt(plan, request.task)
        implementation_prompt_file = write_text(run_results / "implementation_prompt.md", implementation_prompt)
        state.implementation_prompt_file = str(implementation_prompt_file)
        state.status = "implementing"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        try:
            implementation_source, implementation_response = request_implementation(
                request,
                plan,
                prompt=implementation_prompt,
            )
        except ImplementationGenerationError as exc:
            if exc.responses:
                implementation_response_file = write_text(
                    run_results / "implementation_response.txt",
                    "\n\n".join(
                        f"===== attempt {index} =====\n{response}"
                        for index, response in enumerate(exc.responses, start=1)
                    ),
                )
                state.implementation_response_file = str(implementation_response_file)
            if exc.sources:
                write_text(run_results / "implementation_invalid.py", exc.sources[-1])
            implementation_error_file = write_text(
                run_results / "implementation_api_error.txt",
                "\n".join(exc.errors) if exc.errors else str(exc),
            )
            state.implementation_error_file = str(implementation_error_file)
            write_text(run_results / "state.json", state.model_dump_json(indent=2))
            raise
        except Exception as exc:
            implementation_error_file = write_text(run_results / "implementation_api_error.txt", str(exc))
            state.implementation_error_file = str(implementation_error_file)
            raise
        implementation_response_file = write_text(
            run_results / "implementation_response.txt",
            implementation_response,
        )
        implementation_file = write_text(run_results / "improvement.py", implementation_source)
        implementation_workspace_file = write_text(workspace / "generated" / "improvement.py", implementation_source)
        state.implementation_response_file = str(implementation_response_file)
        state.implementation_file = str(implementation_file)
        state.implementation_workspace_file = str(implementation_workspace_file)
        state.status = "implemented"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))

        if progress_callback is not None:
            progress_callback("prepare_environment")
        hardware = _resolve_hardware_profile(
            request.hardware_profile_file,
            refresh=request.refresh_hardware_profile,
        )
        hardware_file = write_text(run_results / "hardware.json", json.dumps(asdict(hardware), indent=2))
        environment_file = _write_environment_file(workspace / "environment.yml", request)
        environment_cache_root = _environment_cache_root(request.workspace_root)
        requested_cache_spec = _environment_cache_spec(request, hardware, environment_file)
        requested_cache_key = _environment_cache_key(requested_cache_spec)
        cached_environment = _select_cached_environment(environment_cache_root, requested_cache_key)
        if cached_environment is None:
            cached_environment = _select_previous_completed_environment(request, requested_cache_key)
        reused_cached_environment = cached_environment is not None
        if cached_environment is not None:
            environment_prefix = cached_environment[0]
        else:
            environment_prefix = environment_cache_root / f"env-{requested_cache_key}"
        state.hardware_file = str(hardware_file)
        state.environment_file = str(environment_file)
        state.environment_prefix = str(environment_prefix)
        state.status = "preparing_environment"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))

        if reused_cached_environment:
            setup = subprocess.CompletedProcess(
                ["conda", "env", "reuse", "--prefix", str(environment_prefix)],
                0,
                f"Reusing cached experiment environment: {environment_prefix}\n",
                "",
            )
        elif environment_prefix.exists():
            setup_command = [
                "conda",
                "env",
                "update",
                "--prefix",
                str(environment_prefix),
                "--file",
                str(environment_file),
                "--prune",
            ]
            setup = _run_process(setup_command, cwd=PROJECT_ROOT, timeout=request.timeout_seconds)
        else:
            setup_command = [
                "conda",
                "env",
                "create",
                "--prefix",
                str(environment_prefix),
                "--file",
                str(environment_file),
            ]
            setup = _run_process(setup_command, cwd=PROJECT_ROOT, timeout=request.timeout_seconds)
        write_text(run_results / "environment_stdout.txt", setup.stdout)
        write_text(run_results / "environment_stderr.txt", setup.stderr)
        if setup.returncode != 0:
            raise RuntimeError(f"Conda environment creation failed. See {run_results / 'environment_stderr.txt'}")

        if reused_cached_environment:
            torch_install = subprocess.CompletedProcess(
                ["pip", "install", "torch", "reuse"],
                0,
                f"Reusing PyTorch runtime from cached environment: {environment_prefix}\n",
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
        torch_setup_output = (torch_uninstall.stdout if torch_uninstall else "") + torch_install.stdout
        torch_setup_error = (torch_uninstall.stderr if torch_uninstall else "") + torch_install.stderr
        write_text(run_results / "torch_install_stdout.txt", torch_setup_output)
        write_text(run_results / "torch_install_stderr.txt", torch_setup_error)
        if torch_install.returncode != 0:
            raise RuntimeError(f"PyTorch installation failed. See {run_results / 'torch_install_stderr.txt'}")
        dependencies = _install_experiment_dependencies(
            environment_prefix,
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
        )
        write_text(run_results / "dependencies_stdout.txt", dependencies.stdout)
        write_text(run_results / "dependencies_stderr.txt", dependencies.stderr)
        if dependencies.returncode != 0:
            raise RuntimeError(f"Experiment dependency installation failed. See {run_results / 'dependencies_stderr.txt'}")
        runtime = _verify_torch_runtime(
            environment_prefix,
            hardware,
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
        )
        hardware = _record_verified_runtime(request.hardware_profile_file, hardware, runtime)
        write_text(hardware_file, json.dumps(asdict(hardware), indent=2))
        runtime_file = write_text(run_results / "torch_runtime.json", json.dumps(runtime, indent=2))
        state.torch_runtime_file = str(runtime_file)
        actual_cache_spec = _environment_cache_spec(request, hardware, environment_file, runtime=runtime)
        actual_cache_key = _environment_cache_key(actual_cache_spec)
        environment_cache_metadata = _record_cached_environment(
            environment_cache_root,
            environment_prefix,
            cache_key=actual_cache_key,
            requested_cache_key=requested_cache_key,
            spec=actual_cache_spec,
            run_id=run_id,
            environment_file=environment_file,
            hardware=hardware,
            runtime=runtime,
            reused=reused_cached_environment,
        )
        environment_cache_file = write_text(
            run_results / "environment_cache.json",
            json.dumps(environment_cache_metadata, ensure_ascii=False, indent=2),
        )
        state.environment_cache_file = str(environment_cache_file)

        if progress_callback is not None:
            progress_callback("run_experiment")
        state.status = "running"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        stdout_file = run_results / "experiment_stdout.txt"
        stderr_file = run_results / "experiment_stderr.txt"
        gpu_selection = _select_execution_gpu(hardware, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        env_overrides = None
        if gpu_selection is not None:
            write_text(run_results / "gpu_selection.json", json.dumps(gpu_selection, ensure_ascii=False, indent=2))
            env_overrides = {"CUDA_VISIBLE_DEVICES": str(gpu_selection["cuda_visible_devices"])}
        execute = _run_process_streaming(
            [
                str(_environment_python(environment_prefix)),
                "-m",
                "code_agent.experiments.execute_plan",
                "--plan-file",
                str(plan_file),
                "--workspace-dir",
                str(workspace),
                "--results-dir",
                str(run_results),
                "--implementation-file",
                str(implementation_workspace_file),
            ],
            cwd=PROJECT_ROOT,
            timeout=request.timeout_seconds,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            env_overrides=env_overrides,
        )
        state.stdout_file = str(stdout_file)
        state.stderr_file = str(stderr_file)
        state.metrics_file = str(run_results / "metrics.json")
        state.report_file = str(run_results / "comparison.md")
        if execute.returncode != 0:
            raise RuntimeError(f"Experiment execution failed. See {stderr_file}")
        state.status = "completed"
    except Exception as exc:
        state.status = "failed"
        state.error = str(exc)

    write_text(run_results / "state.json", state.model_dump_json(indent=2))
    return state
