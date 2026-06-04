import json
import subprocess
import sys
import tomllib
from io import StringIO
from pathlib import Path

from code_agent.experiments.agent import (
    PYTORCH_CUDA_INDEX_URL,
    HardwareProfile,
    _environment_cache_key,
    _environment_cache_spec,
    _record_cached_environment,
    _select_cached_environment,
    _select_execution_gpu,
    _select_previous_completed_environment,
    _try_clone_current_environment,
    _configured_conda_url_channels,
    _detect_hardware,
    _install_experiment_dependencies,
    _install_torch_runtime,
    _read_hardware_profile,
    _record_verified_runtime,
    _resolve_hardware_profile,
    _run_process_streaming,
    _select_pytorch_cuda_index_url,
    _write_environment_file,
)
from code_agent.experiments.models import ExperimentRequest


def test_base_environment_does_not_install_experiment_stack_before_torch(monkeypatch, tmp_path):
    monkeypatch.setattr("code_agent.experiments.agent._configured_conda_url_channels", lambda: [])
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="train text classifier",
        api_provider="deepseek",
    )

    content = _write_environment_file(tmp_path / "environment.yml", request).read_text(encoding="utf-8")

    assert "[experiment]" not in content
    assert "torch" not in content
    assert "channels:" not in content


def test_environment_file_uses_configured_url_channels(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "code_agent.experiments.agent._configured_conda_url_channels",
        lambda: ["https://mirrors.example.test/anaconda/cloud/conda-forge"],
    )
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="train text classifier",
        api_provider="deepseek",
    )

    content = _write_environment_file(tmp_path / "environment.yml", request).read_text(encoding="utf-8")

    assert "https://mirrors.example.test/anaconda/cloud/conda-forge" in content
    assert "nodefaults" in content


def test_configured_conda_channels_ignore_named_official_channels(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            '{"channels": ["https://mirrors.example.test/anaconda/cloud/conda-forge", "conda-forge"]}',
            "",
        )

    monkeypatch.setattr("code_agent.experiments.agent.subprocess.run", fake_run)

    assert _configured_conda_url_channels() == ["https://mirrors.example.test/anaconda/cloud/conda-forge"]


def test_environment_cache_records_and_selects_verified_runtime(tmp_path):
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="train text classifier",
        api_provider="deepseek",
        workspace_root=tmp_path / "experiments",
    )
    environment_file = tmp_path / "environment.yml"
    environment_file.write_text("dependencies:\n- python=3.11\n- pip\n- git\n", encoding="utf-8")
    hardware = HardwareProfile(
        accelerator="cuda",
        torch_index_url="https://download.pytorch.org/whl/cu128",
        torch_version="2.11.0+cu128",
    )
    runtime = {"torch_version": "2.11.0+cu128", "cuda_available": True}
    spec = _environment_cache_spec(request, hardware, environment_file, runtime=runtime)
    cache_key = _environment_cache_key(spec)
    cache_root = tmp_path / "experiments" / "asset_cache" / "environments"
    cache_root.mkdir(parents=True)
    environment_prefix = cache_root / "env-existing"
    environment_prefix.joinpath("bin").mkdir(parents=True)
    environment_prefix.joinpath("bin", "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")

    metadata = _record_cached_environment(
        cache_root,
        environment_prefix,
        cache_key=cache_key,
        requested_cache_key="requested",
        spec=spec,
        run_id="run-1",
        environment_file=environment_file,
        hardware=hardware,
        runtime=runtime,
        reused=False,
    )

    selected = _select_cached_environment(cache_root, cache_key)
    assert selected is not None
    assert selected[0] == environment_prefix
    assert selected[1]["run_id"] == "run-1"
    assert metadata["shared_metadata_file"].endswith(".code_agent_environment.json")


def test_environment_cache_ignores_missing_interpreter(tmp_path):
    cache_root = tmp_path / "environments"
    cache_root.mkdir()
    registry = {
        "schema_version": 1,
        "latest": {
            "abc": {
                "prefix": str(cache_root / "missing-python"),
                "run_id": "run-1",
            }
        },
    }
    cache_root.joinpath("registry.json").write_text(json.dumps(registry), encoding="utf-8")

    assert _select_cached_environment(cache_root, "abc") is None


def test_try_clone_current_environment_reuses_verified_torch(monkeypatch, tmp_path):
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="train",
        api_provider="deepseek",
        environment_python=f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    hardware = HardwareProfile(
        accelerator="cuda",
        torch_version="2.11.0+cu128",
        torch_cuda_version="12.8",
    )
    cache_root = tmp_path / "asset_cache" / "environments"
    environment_prefix = cache_root / "env-abc123"
    environment_prefix.mkdir(parents=True)
    environment_prefix.joinpath("stale.txt").write_text("partial", encoding="utf-8")
    seed_prefix = tmp_path / "code-agent-env"
    seed_prefix.mkdir()
    commands: list[list[str]] = []

    monkeypatch.setattr("code_agent.experiments.agent.sys.prefix", str(seed_prefix))

    def fake_run_process(command, *, cwd, timeout):
        commands.append(command)
        if command[0] == sys.executable:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "torch_version": "2.11.0+cu128",
                        "cuda_available": True,
                        "torch_cuda_version": "12.8",
                        "device_name": "GPU",
                    }
                ),
                "",
            )
        if command[:3] == ["conda", "create", "--yes"]:
            environment_prefix.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0, "cloned", "")
        raise AssertionError(command)

    monkeypatch.setattr("code_agent.experiments.agent._run_process", fake_run_process)

    cloned = _try_clone_current_environment(
        request,
        hardware,
        environment_prefix=environment_prefix,
        cache_root=cache_root,
        cwd=tmp_path,
        timeout=60,
    )

    assert cloned is not None
    setup, runtime = cloned
    assert setup.stdout == "cloned"
    assert runtime["torch_version"] == "2.11.0+cu128"
    assert not environment_prefix.joinpath("stale.txt").exists()
    assert any(command[:3] == ["conda", "create", "--yes"] and "--clone" in command for command in commands)


def test_try_clone_current_environment_skips_mismatched_python(monkeypatch, tmp_path):
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="train",
        api_provider="deepseek",
        environment_python="9.99",
    )
    cache_root = tmp_path / "asset_cache" / "environments"
    environment_prefix = cache_root / "env-abc123"
    calls: list[list[str]] = []

    def fake_run_process(command, *, cwd, timeout):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr("code_agent.experiments.agent._run_process", fake_run_process)

    assert (
        _try_clone_current_environment(
            request,
            HardwareProfile(accelerator="cuda"),
            environment_prefix=environment_prefix,
            cache_root=cache_root,
            cwd=tmp_path,
            timeout=60,
        )
        is None
    )
    assert calls == []


def test_environment_cache_can_reuse_previous_completed_run(tmp_path):
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="train text classifier",
        api_provider="deepseek",
        workspace_root=tmp_path / "experiments",
        results_root=tmp_path / "results",
    )
    run_dir = tmp_path / "results" / "run-1"
    run_dir.mkdir(parents=True)
    workspace = tmp_path / "experiments" / "run-1"
    environment_prefix = workspace / "conda-env"
    environment_prefix.joinpath("bin").mkdir(parents=True)
    environment_prefix.joinpath("bin", "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
    environment_file = workspace / "environment.yml"
    environment_file.parent.mkdir(parents=True, exist_ok=True)
    environment_file.write_text("dependencies:\n- python=3.11\n- pip\n- git\n", encoding="utf-8")
    hardware_file = run_dir / "hardware.json"
    hardware_file.write_text(
        json.dumps(
            {
                "accelerator": "cuda",
                "torch_index_url": "https://download.pytorch.org/whl/cu128",
                "torch_version": "2.11.0+cu128",
            }
        ),
        encoding="utf-8",
    )
    runtime_file = run_dir / "torch_runtime.json"
    runtime_file.write_text(json.dumps({"torch_version": "2.11.0+cu128"}), encoding="utf-8")
    run_dir.joinpath("state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "run-1",
                "environment_prefix": str(environment_prefix),
                "environment_file": str(environment_file),
                "hardware_file": str(hardware_file),
                "torch_runtime_file": str(runtime_file),
            }
        ),
        encoding="utf-8",
    )
    spec = _environment_cache_spec(
        request,
        HardwareProfile(
            accelerator="cuda",
            torch_index_url="https://download.pytorch.org/whl/cu128",
            torch_version="2.11.0+cu128",
        ),
        environment_file,
        runtime={"torch_version": "2.11.0+cu128"},
    )

    selected = _select_previous_completed_environment(request, _environment_cache_key(spec))

    assert selected is not None
    assert selected[0] == environment_prefix
    assert selected[1]["source"] == "previous_completed_run"


def test_experiment_dependencies_are_installed_after_selected_torch(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        "code_agent.experiments.agent._run_process",
        lambda command, **kwargs: commands.append(command) or subprocess.CompletedProcess(command, 0, "", ""),
    )

    _install_experiment_dependencies(tmp_path, cwd=Path.cwd(), timeout=10)

    assert "[experiment]" in commands[0][-1]


def test_torch_install_pins_verified_runtime_and_retries(monkeypatch, tmp_path):
    commands = []
    install_attempts = 0

    def fake_run(command, **kwargs):
        nonlocal install_attempts
        commands.append(command)
        if "install" not in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        install_attempts += 1
        return subprocess.CompletedProcess(command, 0 if install_attempts == 3 else 1, "", "connection reset")

    monkeypatch.setattr("code_agent.experiments.agent._run_process", fake_run)

    _install_torch_runtime(
        tmp_path,
        HardwareProfile(
            accelerator="cuda",
            torch_index_url="https://download.pytorch.org/whl/cu128",
            torch_version="2.11.0+cu128",
        ),
        cwd=Path.cwd(),
        timeout=10,
    )

    install_commands = [command for command in commands if "install" in command]
    assert len(install_commands) == 3
    assert "torch==2.11.0+cu128" in install_commands[0]
    assert "--retries" in install_commands[0]
    assert "--timeout" in install_commands[0]
    assert "--index-url" in install_commands[0]


def test_experiment_dependencies_include_huggingface_xet_download_support():
    configuration = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert any(
        dependency.startswith("hf_xet")
        for dependency in configuration["project"]["optional-dependencies"]["experiment"]
    )


def test_detect_hardware_uses_cuda_wheel_index_when_nvidia_smi_is_available(monkeypatch):
    def fake_run(command, **kwargs):
        if "--query-gpu=name,driver_version" in command:
            return subprocess.CompletedProcess(command, 0, "NVIDIA GeForce RTX 5060 Laptop GPU, 591.74\n", "")
        return subprocess.CompletedProcess(command, 0, "| Driver Version: 591.74 CUDA Version: 13.1 |", "")

    monkeypatch.setattr("code_agent.experiments.agent._run_process", fake_run)

    hardware = _detect_hardware()

    assert hardware.accelerator == "cuda"
    assert hardware.gpu_name == "NVIDIA GeForce RTX 5060 Laptop GPU"
    assert hardware.reported_cuda_version == "13.1"
    assert hardware.torch_index_url == "https://download.pytorch.org/whl/cu130"


def test_select_pytorch_cuda_index_url_matches_driver_capability():
    assert _select_pytorch_cuda_index_url("13.1") == "https://download.pytorch.org/whl/cu130"
    assert _select_pytorch_cuda_index_url("12.8") == "https://download.pytorch.org/whl/cu128"
    assert _select_pytorch_cuda_index_url("12.6") == "https://download.pytorch.org/whl/cu126"
    assert _select_pytorch_cuda_index_url("12.4") == "https://download.pytorch.org/whl/cu124"
    assert _select_pytorch_cuda_index_url("12.1") == "https://download.pytorch.org/whl/cu121"
    assert _select_pytorch_cuda_index_url("11.8") == "https://download.pytorch.org/whl/cu118"
    assert _select_pytorch_cuda_index_url("11.7") is None


def test_select_execution_gpu_prefers_most_free_memory(monkeypatch):
    def fake_run(command, **kwargs):
        assert "--query-gpu=index,memory.free,memory.used,memory.total" in command
        return subprocess.CompletedProcess(
            command,
            0,
            "0, 1024, 23540, 24564\n1, 22000, 2564, 24564\n2, 12000, 12564, 24564\n",
            "",
        )

    monkeypatch.setattr("code_agent.experiments.agent._run_process", fake_run)

    selected = _select_execution_gpu(HardwareProfile(accelerator="cuda"))

    assert selected["selected_gpu_index"] == 1
    assert selected["cuda_visible_devices"] == "1"
    assert selected["free_memory_mb"] == 22000


def test_select_execution_gpu_respects_existing_numeric_visibility(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            "0, 24000, 564, 24564\n4, 10000, 14564, 24564\n5, 18000, 6564, 24564\n",
            "",
        )

    monkeypatch.setattr("code_agent.experiments.agent._run_process", fake_run)

    selected = _select_execution_gpu(HardwareProfile(accelerator="cuda"), cuda_visible_devices="4,5")

    assert selected["selected_gpu_index"] == 5
    assert selected["cuda_visible_devices"] == "5"


def test_cached_hardware_profile_skips_detection(monkeypatch, tmp_path):
    profile_file = tmp_path / "hardware_profile.local.yaml"
    profile_file.write_text(
        "accelerator: cuda\ngpu_name: cached gpu\ntorch_index_url: https://download.pytorch.org/whl/cu128\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "code_agent.experiments.agent._detect_hardware",
        lambda: (_ for _ in ()).throw(AssertionError("should not detect hardware")),
    )

    hardware = _resolve_hardware_profile(profile_file)

    assert hardware.gpu_name == "cached gpu"


def test_refresh_hardware_profile_replaces_cached_value(monkeypatch, tmp_path):
    profile_file = tmp_path / "hardware_profile.local.yaml"
    profile_file.write_text("accelerator: cpu\n", encoding="utf-8")
    monkeypatch.setattr(
        "code_agent.experiments.agent._detect_hardware",
        lambda: HardwareProfile(accelerator="cuda", gpu_name="new gpu", torch_index_url=PYTORCH_CUDA_INDEX_URL),
    )

    hardware = _resolve_hardware_profile(profile_file, refresh=True)

    assert hardware.gpu_name == "new gpu"
    assert _read_hardware_profile(profile_file).accelerator == "cuda"


def test_verified_runtime_is_saved_to_machine_profile(tmp_path):
    profile_file = tmp_path / "hardware_profile.local.yaml"
    hardware = HardwareProfile(accelerator="cuda", torch_index_url=PYTORCH_CUDA_INDEX_URL)

    saved = _record_verified_runtime(
        profile_file,
        hardware,
        {"torch_version": "2.11.0+cu128", "torch_cuda_version": "12.8", "device_name": "GPU"},
    )

    loaded = _read_hardware_profile(profile_file)
    assert saved.verified_at is not None
    assert loaded.torch_version == "2.11.0+cu128"
    assert loaded.torch_cuda_version == "12.8"
    assert loaded.verified_device_name == "GPU"


def test_streaming_process_writes_logs_before_returning(tmp_path):
    output = StringIO()
    stdout_file = tmp_path / "experiment_stdout.txt"
    stderr_file = tmp_path / "experiment_stderr.txt"

    completed = _run_process_streaming(
        [sys.executable, "-c", "import sys; print('train step 1'); print('eval step 1', file=sys.stderr)"],
        cwd=Path.cwd(),
        timeout=10,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        relay_stream=output,
    )

    assert completed.returncode == 0
    assert stdout_file.read_text(encoding="utf-8").strip() == "train step 1"
    assert stderr_file.read_text(encoding="utf-8").strip() == "eval step 1"
    assert "train step 1" in output.getvalue()
    assert "eval step 1" in output.getvalue()


def test_streaming_process_preserves_carriage_return_progress_updates(tmp_path):
    output = StringIO()
    stdout_file = tmp_path / "experiment_stdout.txt"
    stderr_file = tmp_path / "experiment_stderr.txt"

    completed = _run_process_streaming(
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('\\rstep 1/2'); sys.stderr.flush(); "
            "sys.stderr.write('\\rstep 2/2\\n'); sys.stderr.flush()",
        ],
        cwd=Path.cwd(),
        timeout=10,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        relay_stream=output,
    )

    assert completed.returncode == 0
    rendered = output.getvalue()
    assert rendered.startswith("\rstep 1/2\rstep 2/2")
    assert "\nstep 2/2" not in rendered
    with stderr_file.open("r", encoding="utf-8", newline="") as progress_log:
        assert progress_log.read() == rendered


def test_streaming_process_applies_environment_overrides(tmp_path):
    stdout_file = tmp_path / "experiment_stdout.txt"
    stderr_file = tmp_path / "experiment_stderr.txt"

    completed = _run_process_streaming(
        [sys.executable, "-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"],
        cwd=Path.cwd(),
        timeout=10,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        relay_stream=None,
        env_overrides={"CUDA_VISIBLE_DEVICES": "5"},
    )

    assert completed.returncode == 0
    assert stdout_file.read_text(encoding="utf-8").strip() == "5"
