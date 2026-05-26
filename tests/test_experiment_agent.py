import subprocess
import sys
import tomllib
from io import StringIO
from pathlib import Path

from code_agent.experiments.agent import (
    PYTORCH_CUDA_INDEX_URL,
    HardwareProfile,
    _detect_hardware,
    _install_experiment_dependencies,
    _read_hardware_profile,
    _record_verified_runtime,
    _resolve_hardware_profile,
    _run_process_streaming,
    _write_environment_file,
)
from code_agent.experiments.models import ExperimentRequest


def test_base_environment_does_not_install_experiment_stack_before_torch(tmp_path):
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="train text classifier",
        api_provider="deepseek",
    )

    content = _write_environment_file(tmp_path / "environment.yml", request).read_text(encoding="utf-8")

    assert "[experiment]" not in content
    assert "torch" not in content


def test_experiment_dependencies_are_installed_after_selected_torch(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(
        "code_agent.experiments.agent._run_process",
        lambda command, **kwargs: commands.append(command) or subprocess.CompletedProcess(command, 0, "", ""),
    )

    _install_experiment_dependencies(tmp_path, cwd=Path.cwd(), timeout=10)

    assert "[experiment]" in commands[0][-1]


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
    assert hardware.torch_index_url == PYTORCH_CUDA_INDEX_URL


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
