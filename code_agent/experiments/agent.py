from __future__ import annotations

import codecs
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

from code_agent.experiments.implementer import build_implementation_prompt, request_implementation
from code_agent.experiments.models import ExperimentRequest, ExperimentRunState, default_run_id
from code_agent.experiments.planner import prepare_plan_request, request_experiment_plan
from code_agent.tools.file_tools import ensure_dir, write_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ProgressCallback = Callable[[str], None]
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"


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


def _write_environment_file(path: Path, request: ExperimentRequest) -> Path:
    environment = {
        "channels": ["conda-forge"],
        "dependencies": [
            f"python={request.environment_python}",
            "pip",
            "git",
        ],
    }
    return write_text(path, yaml.safe_dump(environment, sort_keys=False))


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


def _detect_hardware() -> HardwareProfile:
    try:
        result = _run_process(["nvidia-smi"], cwd=PROJECT_ROOT, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return HardwareProfile(accelerator="cpu")
    if result.returncode != 0:
        return HardwareProfile(accelerator="cpu")

    details = _run_process(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        cwd=PROJECT_ROOT,
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
    command = [python, "-m", "pip", "install", "torch>=2.4"]
    if hardware.accelerator == "cuda":
        uninstall = _run_process([python, "-m", "pip", "uninstall", "-y", "torch"], cwd=cwd, timeout=timeout)
        command.extend(["--index-url", hardware.torch_index_url or PYTORCH_CUDA_INDEX_URL])
    install = _run_process(command, cwd=cwd, timeout=timeout)
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
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
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
        environment_prefix = workspace / "conda-env"
        state.hardware_file = str(hardware_file)
        state.environment_file = str(environment_file)
        state.environment_prefix = str(environment_prefix)
        state.status = "preparing_environment"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))

        if environment_prefix.exists() and request.reuse_environment:
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
        elif environment_prefix.exists():
            raise FileExistsError(
                f"Experiment environment already exists: {environment_prefix}. Use a new run name or --reuse-environment."
            )
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

        if progress_callback is not None:
            progress_callback("run_experiment")
        state.status = "running"
        write_text(run_results / "state.json", state.model_dump_json(indent=2))
        stdout_file = run_results / "experiment_stdout.txt"
        stderr_file = run_results / "experiment_stderr.txt"
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
