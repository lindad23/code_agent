import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from code_agent.experiments import generic_agent
from code_agent.experiments.agent import GpuMemorySnapshot, HardwareProfile
from code_agent.experiments.generic_agent import (
    MaterializedResource,
    _adapt_torch_pip_install_argv,
    _write_generated_files,
    materialize_generic_resources,
    request_generic_debug_execution_spec,
    request_generic_execution_spec,
    request_generic_experiment_plan,
    run_generic_experiment_agent,
    validate_real_resource_execution_spec,
)
from code_agent.experiments.models import ExperimentRequest


def _request(tmp_path):
    return ExperimentRequest(
        baseline_url="DLinear",
        benchmark_url="ETTm1",
        task="实现频率感知残差校正。",
        api_provider="deepseek",
        baseline_resources={"DLinear": ""},
        benchmark_resources={"ETTm1": ""},
        workspace_root=tmp_path / "workspaces",
        results_root=tmp_path / "results",
        hardware_profile_file=tmp_path / "hardware.yaml",
    )


def _explicit_request(tmp_path):
    return ExperimentRequest(
        baseline_url="https://github.com/example/DLinear",
        benchmark_url="https://github.com/example/ETDataset",
        task="实现频率感知残差校正。",
        api_provider="deepseek",
        baseline_resources={"DLinear": "https://github.com/example/DLinear"},
        benchmark_resources={"ETTm1": "https://github.com/example/ETDataset"},
        workspace_root=tmp_path / "workspaces",
        results_root=tmp_path / "results",
        hardware_profile_file=tmp_path / "hardware.yaml",
    )


@pytest.fixture(autouse=True)
def _fake_generic_environment(monkeypatch):
    def fake_prepare_environment(request, *, workspace, results_dir, run_id, hardware):
        environment_file = workspace / "environment.yml"
        environment_file.write_text("dependencies:\n- python\n", encoding="utf-8")
        runtime_file = results_dir / "torch_runtime.json"
        runtime_file.write_text(json.dumps({"cuda_available": hardware.accelerator == "cuda"}), encoding="utf-8")
        cache_file = results_dir / "environment_cache.json"
        cache_file.write_text(json.dumps({"reused": False}), encoding="utf-8")
        return Path(sys.prefix), hardware, {
            "environment_file": str(environment_file),
            "environment_prefix": sys.prefix,
            "environment_cache_file": str(cache_file),
            "torch_runtime_file": str(runtime_file),
        }

    monkeypatch.setattr(generic_agent, "_prepare_generic_environment", fake_prepare_environment)


def test_request_generic_execution_spec_parses_llm_json(monkeypatch, tmp_path):
    def fake_call_llm(*args, **kwargs):
        return json.dumps(
            {
                "files": [{"path": "scripts/run.py", "content": "print('ok')"}],
                "commands": [{"name": "run", "cwd": ".", "argv": ["python", "scripts/run.py"]}],
                "expected_outputs": ["results.json"],
                "notes": "smoke test",
            }
        )

    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)

    spec, prompt, response = request_generic_execution_spec(
        _request(tmp_path),
        plan={"task_type": "time_series_forecasting"},
        resources=[],
        workspace=tmp_path / "workspace",
        results_dir=tmp_path / "results",
    )

    assert spec["commands"][0]["argv"] == ["python", "scripts/run.py"]
    assert "time_series_forecasting" in prompt
    assert "smoke test" in response


def test_request_generic_execution_spec_accepts_experiment_table(monkeypatch, tmp_path):
    def fake_call_llm(*args, **kwargs):
        return json.dumps(
            {
                "files": [{"path": "my_main.py", "content": "print('ok')"}],
                "commands": [],
                "experiment_table": {
                    "entrypoint": {
                        "cwd": ".",
                        "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                        "timeout_seconds": 30,
                    },
                    "records": [
                        {
                            "id": "baseline_subset",
                            "target": {"code_path": "resources/baseline_DLinear"},
                            "params": {"seed": 13},
                            "expected_metrics": ["mse", "mae"],
                        }
                    ],
                },
                "expected_outputs": ["generic_metrics.json"],
                "notes": "table execution",
            }
        )

    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)

    spec, prompt, response = request_generic_execution_spec(
        _request(tmp_path),
        plan={"task_type": "time_series_forecasting"},
        resources=[],
        workspace=tmp_path / "workspace",
        results_dir=tmp_path / "results",
    )

    assert spec["experiment_table"]["records"][0]["id"] == "baseline_subset"
    assert "experiment_table" in prompt
    assert "table execution" in response


def test_request_generic_execution_spec_writes_template(monkeypatch, tmp_path):
    def fake_call_llm(*args, **kwargs):
        return json.dumps(
            {
                "files": [{"path": "my_main.py", "content": "print('ok')"}],
                "commands": [],
                "experiment_table": {
                    "entrypoint": {
                        "cwd": ".",
                        "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                        "timeout_seconds": 30,
                    },
                    "resource_aliases": [
                        {"role": "baseline", "name": "DLinear", "path": "resources/baseline_DLinear"}
                    ],
                    "records": [
                        {
                            "id": "baseline_subset",
                            "target": {"code_path": "my_main.py", "resources": ["resources/baseline_DLinear"]},
                            "params": {"seed": 13},
                            "expected_metrics": ["mse", "mae"],
                        }
                    ],
                },
                "expected_outputs": ["generic_metrics.json"],
                "notes": "template execution",
            }
        )

    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)
    workspace = tmp_path / "workspace"
    results_dir = tmp_path / "results"

    spec, prompt, response = request_generic_execution_spec(
        _request(tmp_path),
        plan={"task_type": "time_series_forecasting"},
        resources=[
            MaterializedResource(
                role="baseline",
                name="DLinear",
                location="https://github.com/example/DLinear",
                status="cloned_git",
                local_path=str(workspace / "resources" / "baseline_DLinear"),
            )
        ],
        workspace=workspace,
        results_dir=results_dir,
    )

    template = json.loads((results_dir / "generic_execution_template.json").read_text(encoding="utf-8"))
    assert template["experiment_table"]["resource_aliases"][0]["path"] == "resources/baseline_DLinear"
    assert "Local execution spec template JSON" in prompt
    assert spec["experiment_table"]["records"][0]["target"]["resources"] == ["resources/baseline_DLinear"]
    assert "template execution" in response


def test_resource_python_interface_summary_extracts_config_constructor(tmp_path):
    repo = tmp_path / "resources" / "baseline_repo"
    model_file = repo / "models" / "DLinear.py"
    model_file.parent.mkdir(parents=True)
    model_file.write_text(
        """
class Model:
    def __init__(self, configs):
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

    def forward(self, x):
        return x
""",
        encoding="utf-8",
    )

    summary = generic_agent._summarize_resource_python_interfaces(
        [
            MaterializedResource(
                role="baseline",
                name="repo",
                location="repo",
                status="copied_local",
                local_path=str(repo),
            )
        ]
    )

    assert "class Model.__init__(self, configs)" in summary
    assert "configs.seq_len" in summary
    assert "configs.pred_len" in summary
    assert "configs.enc_in" in summary
    assert "class Model.forward(self, x)" in summary


def test_request_generic_execution_spec_rejects_oversized_table(monkeypatch, tmp_path):
    def fake_call_llm(*args, **kwargs):
        return json.dumps(
            {
                "files": [{"path": "my_main.py", "content": "print('ok')"}],
                "commands": [],
                "experiment_table": {
                    "entrypoint": {
                        "cwd": ".",
                        "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                        "timeout_seconds": 30,
                    },
                    "records": [
                        {"id": f"cell_{index}", "target": {"code_path": "my_main.py"}, "params": {}, "expected_metrics": []}
                        for index in range(generic_agent.MAX_EXPERIMENT_TABLE_RECORDS + 1)
                    ],
                },
                "expected_outputs": ["generic_metrics.json"],
                "notes": "too many records",
            }
        )

    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)

    with pytest.raises(ValueError, match="at most"):
        request_generic_execution_spec(
            _request(tmp_path),
            plan={"task_type": "time_series_forecasting"},
            resources=[],
            workspace=tmp_path / "workspace",
            results_dir=tmp_path / "results",
        )


def test_request_generic_execution_spec_rejects_absolute_paths_in_records(monkeypatch, tmp_path):
    def fake_call_llm(*args, **kwargs):
        return json.dumps(
            {
                "files": [{"path": "my_main.py", "content": "print('ok')"}],
                "commands": [],
                "experiment_table": {
                    "entrypoint": {
                        "cwd": ".",
                        "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                        "timeout_seconds": 30,
                    },
                    "records": [
                        {
                            "id": "bad_abs",
                            "target": {"code_path": "my_main.py", "dataset": "/hard_data/user/x/resources/benchmark_ETTm1"},
                            "params": {},
                            "expected_metrics": [],
                        }
                    ],
                },
                "expected_outputs": ["generic_metrics.json"],
                "notes": "absolute path in record",
            }
        )

    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)

    with pytest.raises(ValueError, match="absolute paths"):
        request_generic_execution_spec(
            _request(tmp_path),
            plan={"task_type": "time_series_forecasting"},
            resources=[],
            workspace=tmp_path / "workspace",
            results_dir=tmp_path / "results",
        )


def test_request_generic_experiment_plan_repairs_malformed_json(monkeypatch, tmp_path):
    responses = iter(
        [
            '{"task_type": "time_series_forecasting" "resources": []}',
            json.dumps(
                {
                    "task_type": "time_series_forecasting",
                    "resources": [],
                    "environment": {},
                    "implementation_plan": {},
                    "experiment_matrix": {},
                    "execution_plan": {},
                    "metrics": [],
                    "result_artifacts": [],
                    "risks": [],
                }
            ),
        ]
    )

    def fake_call_llm(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)

    plan, prompt, response = request_generic_experiment_plan(_request(tmp_path))

    assert plan["task_type"] == "time_series_forecasting"
    assert "===== repaired JSON =====" in response
    assert "DistilBERT" not in prompt


def test_request_generic_debug_execution_spec_parses_llm_json(monkeypatch, tmp_path):
    def fake_call_llm(*args, **kwargs):
        return json.dumps(
            {
                "files": [{"path": "scripts/fixed.py", "content": "print('fixed')"}],
                "commands": [{"name": "fixed", "cwd": ".", "argv": ["python", "scripts/fixed.py"]}],
                "expected_outputs": ["generic_metrics.json"],
                "notes": "fixed smoke test",
            }
        )

    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)

    spec, prompt, response = request_generic_debug_execution_spec(
        _request(tmp_path),
        plan={"task_type": "time_series_forecasting"},
        resources=[],
        previous_spec={"files": [], "commands": []},
        command_results=[
            {
                "name": "broken",
                "returncode": 1,
                "stdout_file": str(tmp_path / "missing_stdout.txt"),
                "stderr_file": str(tmp_path / "missing_stderr.txt"),
            }
        ],
        failure_summary="script failed",
        workspace=tmp_path / "workspace",
        results_dir=tmp_path / "results",
        attempt=1,
    )

    assert spec["files"][0]["path"] == "scripts/fixed.py"
    assert "script failed" in prompt
    assert "fixed smoke test" in response


def test_run_generic_experiment_agent_executes_generated_python(monkeypatch, tmp_path):
    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    def fake_materialize(request, workspace):
        return []

    def fake_execution_spec(request, *, plan, resources, workspace, results_dir):
        script = f"""
import json
from pathlib import Path

Path({str(results_dir / "generic_metrics.json")!r}).write_text(
    json.dumps({{"mse": 0.1, "mae": 0.2}}),
    encoding="utf-8",
)
"""
        return (
            {
                "files": [{"path": "scripts/run_generic.py", "content": script}],
                "commands": [
                    {
                        "name": "smoke",
                        "cwd": ".",
                        "argv": ["python", "scripts/run_generic.py"],
                        "timeout_seconds": 30,
                    }
                ],
                "expected_outputs": ["generic_metrics.json"],
                "notes": "smoke execution",
            },
            "execution prompt",
            "{}",
        )

    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", fake_materialize)
    monkeypatch.setattr(generic_agent, "request_generic_execution_spec", fake_execution_spec)
    monkeypatch.setattr(generic_agent, "_resolve_hardware_profile", lambda *args, **kwargs: HardwareProfile(accelerator="cpu"))

    state = run_generic_experiment_agent(_request(tmp_path), execute=True, progress_callback=lambda step: None)

    assert state.status == "generic_completed"
    assert state.metrics_file is not None
    assert json.loads(Path(state.metrics_file).read_text(encoding="utf-8"))["mse"] == 0.1


def test_run_generic_experiment_agent_saves_malformed_execution_json(monkeypatch, tmp_path):
    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    responses = iter(["{bad", "{still bad", "{still bad again"])

    def fake_call_llm(*args, **kwargs):
        return next(responses)

    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", lambda request, workspace: [])
    monkeypatch.setattr(generic_agent, "call_llm", fake_call_llm)

    state = run_generic_experiment_agent(_request(tmp_path), execute=True, progress_callback=lambda step: None)

    assert state.status == "failed"
    assert "malformed JSON" in state.error
    assert state.implementation_prompt_file is not None
    assert state.implementation_response_file is not None
    assert state.implementation_error_file is not None
    run_results = Path(state.implementation_error_file).parent
    assert Path(state.implementation_response_file).read_text(encoding="utf-8") == "{bad"
    assert (run_results / "generic_execution_repair_response_attempt_01.txt").read_text(encoding="utf-8") == "{still bad"
    assert (run_results / "generic_execution_repair_response_attempt_02.txt").read_text(encoding="utf-8") == "{still bad again"


def test_run_generic_experiment_agent_executes_experiment_table(monkeypatch, tmp_path):
    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    def fake_execution_spec(request, *, plan, resources, workspace, results_dir):
        script = """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cell-json", required=True)
parser.add_argument("--output-json", required=True)
args = parser.parse_args()

cell = json.loads(Path(args.cell_json).read_text(encoding="utf-8"))
params = cell["params"]
Path(args.output_json).write_text(
    json.dumps(
        {
            "status": "completed",
            "params": params,
            "metrics": {"score": params["value"]},
            "artifacts": [],
        }
    ),
    encoding="utf-8",
)
"""
        return (
            {
                "files": [{"path": "my_main.py", "content": script}],
                "commands": [],
                "experiment_table": {
                    "entrypoint": {
                        "cwd": ".",
                        "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                        "timeout_seconds": 30,
                    },
                    "records": [
                        {
                            "id": "cell_a",
                            "target": {"code_path": "my_main.py"},
                            "params": {"value": 0.1},
                            "expected_metrics": ["score"],
                        },
                        {
                            "id": "cell_b",
                            "target": {"code_path": "my_main.py"},
                            "params": {"value": 0.2},
                            "expected_metrics": ["score"],
                        },
                    ],
                },
                "expected_outputs": ["generic_metrics.json"],
                "notes": "table execution",
            },
            "execution prompt",
            "{}",
        )

    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", lambda request, workspace: [])
    monkeypatch.setattr(generic_agent, "request_generic_execution_spec", fake_execution_spec)
    monkeypatch.setattr(generic_agent, "_resolve_hardware_profile", lambda *args, **kwargs: HardwareProfile(accelerator="cpu"))

    state = run_generic_experiment_agent(_request(tmp_path), execute=True, progress_callback=lambda step: None)

    assert state.status == "generic_completed"
    assert state.metrics_file is not None
    metrics = json.loads(Path(state.metrics_file).read_text(encoding="utf-8"))
    assert metrics["execution_mode"] == "experiment_table"
    assert metrics["num_records"] == 2
    assert metrics["completed_records"] == 2
    assert [record["output"]["metrics"]["score"] for record in metrics["records"]] == [0.1, 0.2]
    summary = json.loads(Path(state.report_file).read_text(encoding="utf-8"))
    assert [command["kind"] for command in summary["commands"]] == ["experiment_table_record", "experiment_table_record"]


def test_experiment_table_greedily_runs_on_multiple_idle_gpus(monkeypatch, tmp_path):
    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    def fake_execution_spec(request, *, plan, resources, workspace, results_dir):
        script = """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cell-json", required=True)
parser.add_argument("--output-json", required=True)
args = parser.parse_args()
cell = json.loads(Path(args.cell_json).read_text(encoding="utf-8"))
Path(args.output_json).write_text(json.dumps({"status": "completed", "metrics": {"index": cell["index"]}}), encoding="utf-8")
"""
        return (
            {
                "files": [{"path": "my_main.py", "content": script}],
                "commands": [],
                "experiment_table": {
                    "entrypoint": {
                        "cwd": ".",
                        "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                        "timeout_seconds": 30,
                    },
                    "records": [
                        {"id": "cell_a", "target": {"code_path": "my_main.py"}, "params": {}, "expected_metrics": ["index"]},
                        {"id": "cell_b", "target": {"code_path": "my_main.py"}, "params": {}, "expected_metrics": ["index"]},
                        {"id": "cell_c", "target": {"code_path": "my_main.py"}, "params": {}, "expected_metrics": ["index"]},
                    ],
                },
                "expected_outputs": ["generic_metrics.json"],
                "notes": "greedy table execution",
            },
            "execution prompt",
            "{}",
        )

    lock = threading.Lock()
    active = 0
    max_active = 0
    used_gpus: list[str] = []

    def fake_run_process_streaming(command, *, cwd, timeout, stdout_file, stderr_file, relay_stream=None, env_overrides=None):
        nonlocal active, max_active
        output_file = Path(command[command.index("--output-json") + 1])
        cell_file = Path(command[command.index("--cell-json") + 1])
        with lock:
            active += 1
            max_active = max(max_active, active)
            used_gpus.append((env_overrides or {}).get("CUDA_VISIBLE_DEVICES", ""))
        time.sleep(0.05)
        cell = json.loads(cell_file.read_text(encoding="utf-8"))
        output_file.write_text(json.dumps({"status": "completed", "metrics": {"index": cell["index"]}}), encoding="utf-8")
        Path(stdout_file).write_text("", encoding="utf-8")
        Path(stderr_file).write_text("", encoding="utf-8")
        with lock:
            active -= 1
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", lambda request, workspace: [])
    monkeypatch.setattr(generic_agent, "request_generic_execution_spec", fake_execution_spec)
    monkeypatch.setattr(generic_agent, "_resolve_hardware_profile", lambda *args, **kwargs: HardwareProfile(accelerator="cuda"))
    monkeypatch.setattr(generic_agent, "_select_execution_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        generic_agent,
        "_query_gpu_memory_snapshots",
        lambda: [
            GpuMemorySnapshot(index=0, free_memory_mb=20000, used_memory_mb=1000, total_memory_mb=24000),
            GpuMemorySnapshot(index=1, free_memory_mb=21000, used_memory_mb=500, total_memory_mb=24000),
        ],
    )
    monkeypatch.setattr(generic_agent, "_run_process_streaming", fake_run_process_streaming)

    state = run_generic_experiment_agent(_request(tmp_path), execute=True, progress_callback=lambda step: None)

    assert state.status == "generic_completed"
    assert max_active == 2
    assert set(used_gpus[:2]) == {"0", "1"}
    schedule_file = Path(state.report_file).parent / "generic_gpu_schedule_attempt_01.json"
    schedule = json.loads(schedule_file.read_text(encoding="utf-8"))
    assert schedule["strategy"] == "greedy_gpu_worker_pool"
    assert len(schedule["slots"]) == 2


def test_run_generic_experiment_agent_self_debugs_failed_command(monkeypatch, tmp_path):
    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    def fake_execution_spec(request, *, plan, resources, workspace, results_dir):
        return (
            {
                "files": [{"path": "scripts/run_generic.py", "content": "raise RuntimeError('first attempt failed')\n"}],
                "commands": [
                    {
                        "name": "smoke",
                        "cwd": ".",
                        "argv": ["python", "scripts/run_generic.py"],
                        "timeout_seconds": 30,
                    }
                ],
                "expected_outputs": ["generic_metrics.json"],
                "notes": "first attempt",
            },
            "execution prompt",
            "{}",
        )

    def fake_debug_spec(request, *, plan, resources, previous_spec, command_results, failure_summary, workspace, results_dir, attempt):
        script = f"""
import json
from pathlib import Path

Path({str(results_dir / "generic_metrics.json")!r}).write_text(
    json.dumps({{"debug_fixed": True}}),
    encoding="utf-8",
)
"""
        return (
            {
                "files": [{"path": "scripts/run_generic.py", "content": script}],
                "commands": [
                    {
                        "name": "smoke_fixed",
                        "cwd": ".",
                        "argv": ["python", "scripts/run_generic.py"],
                        "timeout_seconds": 30,
                    }
                ],
                "expected_outputs": ["generic_metrics.json"],
                "notes": "debug fixed",
            },
            "debug prompt",
            "{}",
        )

    progress = []
    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", lambda request, workspace: [])
    monkeypatch.setattr(generic_agent, "request_generic_execution_spec", fake_execution_spec)
    monkeypatch.setattr(generic_agent, "request_generic_debug_execution_spec", fake_debug_spec)
    monkeypatch.setattr(generic_agent, "_resolve_hardware_profile", lambda *args, **kwargs: HardwareProfile(accelerator="cpu"))

    state = run_generic_experiment_agent(_request(tmp_path), execute=True, progress_callback=progress.append)

    assert state.status == "generic_completed"
    assert "debug_generic_execution" in progress
    assert state.metrics_file is not None
    assert json.loads(Path(state.metrics_file).read_text(encoding="utf-8"))["debug_fixed"] is True
    summary = json.loads(Path(state.report_file).read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in summary["attempts"]] == ["failed", "completed"]


def test_real_resource_spec_rejects_dummy_workflow(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="real resource-backed"):
        validate_real_resource_execution_spec(
            {
                "files": [
                    {
                        "path": "train_eval.py",
                        "content": "class DummyBaseline: pass\ntrain_x = torch.randn(10, 96, 7)\n",
                    }
                ],
                "commands": [{"name": "smoke", "cwd": ".", "argv": ["python", "train_eval.py", "--smoke"]}],
                "expected_outputs": [],
                "notes": "user must replace dummy baseline later",
            },
            [
                MaterializedResource(
                    role="baseline",
                    name="DLinear",
                    location="https://github.com/example/DLinear",
                    status="cloned_git",
                    local_path=str(baseline_dir),
                ),
                MaterializedResource(
                    role="benchmark",
                    name="ETTm1",
                    location="https://github.com/example/ETDataset",
                    status="cloned_git",
                    local_path=str(dataset_dir),
                ),
            ],
        )


def test_real_resource_spec_allows_seed_and_relative_resource_paths(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    validate_real_resource_execution_spec(
        {
            "files": [
                {
                    "path": "train_eval.py",
                    "content": (
                        "import numpy as np\n"
                        "np.random.seed(42)\n"
                        "start = np.random.randint(0, 10)\n"
                        "baseline_path = 'resources/baseline_DLinear'\n"
                        "dataset_path = 'resources/benchmark_ETTm1/ETT-small/ETTm1.csv'\n"
                    ),
                }
            ],
            "commands": [{"name": "real", "cwd": ".", "argv": ["python", "train_eval.py"]}],
            "expected_outputs": ["generic_metrics.json"],
            "notes": "Runs DLinear on ETTm1 using cloned resources.",
        },
        [
            MaterializedResource(
                role="baseline",
                name="DLinear",
                location="https://github.com/example/DLinear",
                status="cloned_git",
                local_path=str(baseline_dir),
            ),
            MaterializedResource(
                role="benchmark",
                name="ETTm1",
                location="https://github.com/example/ETDataset",
                status="cloned_git",
                local_path=str(dataset_dir),
            ),
        ],
    )


def test_real_resource_spec_allows_random_parameter_initialization(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    validate_real_resource_execution_spec(
        {
            "files": [
                {
                    "path": "frc_module.py",
                    "content": (
                        "import torch\n"
                        "import torch.nn as nn\n"
                        "class FFTResidualPredictor(nn.Module):\n"
                        "    def __init__(self):\n"
                        "        super().__init__()\n"
                        "        self.amplitude = nn.Parameter(torch.randn(5, 7))\n"
                        "        self.phase = nn.Parameter(torch.randn(5, 7))\n"
                        "baseline_path = 'resources/baseline_DLinear'\n"
                        "dataset_path = 'resources/benchmark_ETTm1/ETT-small/ETTm1.csv'\n"
                    ),
                }
            ],
            "commands": [{"name": "real", "cwd": ".", "argv": ["python", "frc_module.py"]}],
            "expected_outputs": ["generic_metrics.json"],
            "notes": "Runs DLinear on ETTm1 using cloned resources.",
        },
        [
            MaterializedResource(
                role="baseline",
                name="DLinear",
                location="https://github.com/example/DLinear",
                status="cloned_git",
                local_path=str(baseline_dir),
            ),
            MaterializedResource(
                role="benchmark",
                name="ETTm1",
                location="https://github.com/example/ETDataset",
                status="cloned_git",
                local_path=str(dataset_dir),
            ),
        ],
    )


def test_real_resource_spec_allows_random_latency_probe_when_real_data_is_used(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    validate_real_resource_execution_spec(
        {
            "files": [
                {
                    "path": "my_main.py",
                    "content": (
                        "import pandas as pd\n"
                        "import torch\n"
                        "baseline_path = 'resources/baseline_DLinear'\n"
                        "dataset_path = 'resources/benchmark_ETTm1/ETT-small/ETTm1.csv'\n"
                        "df = pd.read_csv(dataset_path)\n"
                        "sample = torch.randn(1, input_len, df.shape[1] - 1)\n"
                    ),
                }
            ],
            "commands": [{"name": "real", "cwd": ".", "argv": ["python", "my_main.py"]}],
            "expected_outputs": ["generic_metrics.json"],
            "notes": "Runs DLinear on ETTm1 using cloned resources.",
        },
        [
            MaterializedResource(
                role="baseline",
                name="DLinear",
                location="https://github.com/example/DLinear",
                status="cloned_git",
                local_path=str(baseline_dir),
            ),
            MaterializedResource(
                role="benchmark",
                name="ETTm1",
                location="https://github.com/example/ETDataset",
                status="cloned_git",
                local_path=str(dataset_dir),
            ),
        ],
    )


def test_real_resource_spec_still_rejects_synthetic_random_training_data(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="synthetic random data assignment"):
        validate_real_resource_execution_spec(
            {
                "files": [
                    {
                        "path": "my_main.py",
                        "content": (
                            "import torch\n"
                            "baseline_path = 'resources/baseline_DLinear'\n"
                            "dataset_path = 'resources/benchmark_ETTm1/ETT-small/ETTm1.csv'\n"
                            "train_x = torch.randn(10, 96, 7)\n"
                        ),
                    }
                ],
                "commands": [{"name": "bad", "cwd": ".", "argv": ["python", "my_main.py"]}],
                "expected_outputs": ["generic_metrics.json"],
                "notes": "bad synthetic training data",
            },
            [
                MaterializedResource(
                    role="baseline",
                    name="DLinear",
                    location="https://github.com/example/DLinear",
                    status="cloned_git",
                    local_path=str(baseline_dir),
                ),
                MaterializedResource(
                    role="benchmark",
                    name="ETTm1",
                    location="https://github.com/example/ETDataset",
                    status="cloned_git",
                    local_path=str(dataset_dir),
                ),
            ],
        )


def test_real_resource_spec_rejects_baseline_named_standin_that_ignores_repo(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="baseline|target.resources"):
        validate_real_resource_execution_spec(
            {
                "files": [
                    {
                        "path": "my_main.py",
                        "content": (
                            "import json\n"
                            "import torch.nn as nn\n"
                            "class DLinearWrapper(nn.Module):\n"
                            "    def __init__(self):\n"
                            "        super().__init__()\n"
                            "        self.linear = nn.Linear(96, 96)\n"
                            "    def forward(self, x):\n"
                            "        return self.linear(x)\n"
                            "cell = json.load(open('cell.json'))\n"
                            "params = cell['params']\n"
                        ),
                    }
                ],
                "commands": [],
                "experiment_table": {
                    "entrypoint": {
                        "cwd": ".",
                        "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                        "timeout_seconds": 30,
                    },
                    "records": [
                        {
                            "id": "fake",
                            "target": {
                                "code_path": "my_main.py",
                                "resources": ["resources/baseline_DLinear", "resources/benchmark_ETTm1"],
                            },
                            "params": {"seed": 42},
                            "expected_metrics": ["mse"],
                        }
                    ],
                },
                "expected_outputs": ["generic_metrics.json"],
                "notes": "The script defines lightweight wrappers for the baselines.",
            },
            [
                MaterializedResource(
                    role="baseline",
                    name="DLinear",
                    location="https://github.com/example/DLinear",
                    status="cloned_git",
                    local_path=str(baseline_dir),
                ),
                MaterializedResource(
                    role="benchmark",
                    name="ETTm1",
                    location="https://github.com/example/ETDataset",
                    status="cloned_git",
                    local_path=str(dataset_dir),
                ),
            ],
        )


def test_real_resource_spec_allows_repo_adapter_that_reads_target_resources(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    validate_real_resource_execution_spec(
        {
            "files": [
                {
                    "path": "my_main.py",
                    "content": (
                        "import importlib\n"
                        "import json\n"
                        "import sys\n"
                        "from pathlib import Path\n"
                        "cell = json.loads(Path(args.cell_json).read_text())\n"
                        "target = cell['target']\n"
                        "target_resources = target['resources']\n"
                        "baseline_path = next(p for p in target_resources if 'baseline_DLinear' in p)\n"
                        "dataset_path = next(p for p in target_resources if 'benchmark_ETTm1' in p)\n"
                        "sys.path.insert(0, baseline_path)\n"
                        "repo_model = importlib.import_module('models.DLinear')\n"
                        "csv_path = Path(dataset_path) / 'ETT-small' / 'ETTm1.csv'\n"
                    ),
                }
            ],
            "commands": [],
            "experiment_table": {
                "entrypoint": {
                    "cwd": ".",
                    "argv": ["python", "my_main.py", "--cell-json", "{cell_file}", "--output-json", "{output_file}"],
                    "timeout_seconds": 30,
                },
                "records": [
                    {
                        "id": "real",
                        "target": {
                            "code_path": "my_main.py",
                            "resources": ["resources/baseline_DLinear", "resources/benchmark_ETTm1"],
                        },
                        "params": {"seed": 42},
                        "expected_metrics": ["mse"],
                    }
                ],
            },
            "expected_outputs": ["generic_metrics.json"],
            "notes": "Runs through an adapter that imports the materialized baseline repository.",
        },
        [
            MaterializedResource(
                role="baseline",
                name="DLinear",
                location="https://github.com/example/DLinear",
                status="cloned_git",
                local_path=str(baseline_dir),
            ),
            MaterializedResource(
                role="benchmark",
                name="ETTm1",
                location="https://github.com/example/ETDataset",
                status="cloned_git",
                local_path=str(dataset_dir),
            ),
        ],
    )


def test_real_resource_spec_allows_notes_that_describe_removed_synthetic_data(tmp_path):
    baseline_dir = tmp_path / "resources" / "baseline_DLinear"
    dataset_dir = tmp_path / "resources" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    validate_real_resource_execution_spec(
        {
            "files": [
                {
                    "path": "my_main.py",
                    "content": (
                        "import pandas as pd\n"
                        "baseline_path = 'resources/baseline_DLinear'\n"
                        "dataset_path = 'resources/benchmark_ETTm1/ETT-small/ETTm1.csv'\n"
                        "df = pd.read_csv(dataset_path)\n"
                    ),
                }
            ],
            "commands": [{"name": "real", "cwd": ".", "argv": ["python", "my_main.py"]}],
            "expected_outputs": ["generic_metrics.json"],
            "notes": "The previous failure was due to synthetic data; this version uses actual CSV resources.",
        },
        [
            MaterializedResource(
                role="baseline",
                name="DLinear",
                location="https://github.com/example/DLinear",
                status="cloned_git",
                local_path=str(baseline_dir),
            ),
            MaterializedResource(
                role="benchmark",
                name="ETTm1",
                location="https://github.com/example/ETDataset",
                status="cloned_git",
                local_path=str(dataset_dir),
            ),
        ],
    )


def test_generic_pip_torch_install_uses_detected_cuda_index():
    argv = ["/env/bin/python", "-m", "pip", "install", "torch"]
    hardware = HardwareProfile(accelerator="cuda", torch_index_url="https://download.pytorch.org/whl/cu128")

    adapted = _adapt_torch_pip_install_argv(argv, hardware)

    assert adapted == [
        "/env/bin/python",
        "-m",
        "pip",
        "install",
        "torch",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ]


def test_generic_pip_torch_install_with_other_packages_is_not_rewritten():
    argv = ["/env/bin/python", "-m", "pip", "install", "torch", "numpy"]
    hardware = HardwareProfile(accelerator="cuda", torch_index_url="https://download.pytorch.org/whl/cu128")

    assert _adapt_torch_pip_install_argv(argv, hardware) == argv


def test_generic_pip_torch_install_keeps_existing_index():
    argv = ["/env/bin/python", "-m", "pip", "install", "torch", "--index-url", "https://example.test/simple"]
    hardware = HardwareProfile(accelerator="cuda", torch_index_url="https://download.pytorch.org/whl/cu128")

    assert _adapt_torch_pip_install_argv(argv, hardware) == argv


def test_materialize_generic_resources_reuses_local_resource_cache(tmp_path):
    source = tmp_path / "source_data"
    source.mkdir()
    (source / "data.csv").write_text("x\n1\n", encoding="utf-8")
    request = _explicit_request(tmp_path)
    request.baseline_resources = {"LocalData": str(source)}
    request.benchmark_resources = {"unused": ""}

    first = materialize_generic_resources(request, tmp_path / "workspaces" / "run1")
    (Path(first[0].local_path) / "data.csv").write_text("mutated\n", encoding="utf-8")
    second = materialize_generic_resources(request, tmp_path / "workspaces" / "run2")

    assert first[0].cache_status == "miss"
    assert second[0].cache_status == "hit"
    assert first[0].cache_path == second[0].cache_path
    assert first[0].local_path != second[0].local_path
    assert (Path(second[0].local_path) / "data.csv").read_text(encoding="utf-8") == "x\n1\n"


def test_materialize_generic_resources_reuses_git_resource_cache(tmp_path):
    source = tmp_path / "source_repo"
    source.mkdir()
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=source, check=True)
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=source, check=True, capture_output=True)
    bare = tmp_path / "source.git"
    subprocess.run(["git", "clone", "--bare", str(source), str(bare)], check=True, capture_output=True)

    request = _explicit_request(tmp_path)
    request.baseline_resources = {"Repo": f"file://{bare}"}
    request.benchmark_resources = {"unused": ""}

    first = materialize_generic_resources(request, tmp_path / "workspaces" / "run1")
    second = materialize_generic_resources(request, tmp_path / "workspaces" / "run2")

    assert first[0].status == "cloned_git"
    assert first[0].cache_status == "miss"
    assert second[0].cache_status == "hit"
    assert first[0].cache_path == second[0].cache_path
    assert (Path(second[0].local_path) / "README.md").read_text(encoding="utf-8") == "hello\n"


def test_git_transport_failure_is_not_reported_as_missing_url(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            128,
            "",
            (
                "Cloning into 'repo'...\n"
                "error: RPC failed; curl 56 GnuTLS recv error (-9): Error decoding the received TLS packet.\n"
                "fatal: the remote end hung up unexpectedly\n"
                "fatal: early EOF\n"
                "fatal: index-pack failed\n"
            ),
        )

    monkeypatch.setattr(generic_agent.subprocess, "run", fake_run)

    shared, cache_key, status, error = generic_agent._stage_git_resource_in_cache(
        role="baseline",
        name="DLinear",
        location="https://github.com/cure-lab/LTSF-Linear",
        resource_cache_root=tmp_path / "cache",
        timeout=30,
    )

    assert shared is None
    assert cache_key
    assert status == "failed"
    assert error is not None
    assert error.startswith("Git 资源下载失败")
    assert "不等于网址不存在" in error
    assert "GnuTLS recv error" in error


def test_git_missing_repository_is_reported_as_missing_url(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            128,
            "",
            "remote: Repository not found.\nfatal: repository 'https://github.com/example/missing/' not found\n",
        )

    monkeypatch.setattr(generic_agent.subprocess, "run", fake_run)

    shared, cache_key, status, error = generic_agent._stage_git_resource_in_cache(
        role="baseline",
        name="Missing",
        location="https://github.com/example/missing",
        resource_cache_root=tmp_path / "cache",
        timeout=30,
    )

    assert shared is None
    assert cache_key
    assert status == "failed"
    assert error is not None
    assert error.startswith("指定网址不存在或无权限访问")


def test_materialize_generic_resources_reuses_http_resource_cache(monkeypatch, tmp_path):
    def fake_verify(location, *, timeout=20):
        assert location == "https://example.test/data.csv"

    def fake_download(location, target, *, timeout):
        target.write_text("x\n1\n", encoding="utf-8")

    monkeypatch.setattr(generic_agent, "_verify_http_resource", fake_verify)
    monkeypatch.setattr(generic_agent, "_download_http_resource", fake_download)
    request = _explicit_request(tmp_path)
    request.baseline_resources = {"unused": ""}
    request.benchmark_resources = {"RemoteData": "https://example.test/data.csv"}

    first = materialize_generic_resources(request, tmp_path / "workspaces" / "run1")
    second = materialize_generic_resources(request, tmp_path / "workspaces" / "run2")
    first_remote = next(resource for resource in first if resource.name == "RemoteData")
    second_remote = next(resource for resource in second if resource.name == "RemoteData")

    assert first_remote.status == "downloaded_http"
    assert first_remote.cache_status == "miss"
    assert second_remote.cache_status == "hit"
    assert (Path(second_remote.local_path) / "data.csv").read_text(encoding="utf-8") == "x\n1\n"


def test_ltsf_shaped_resources_still_use_ai_execution_spec(monkeypatch, tmp_path):
    resource_root = tmp_path / "resources"
    dlinear = resource_root / "baseline_DLinear_LTSF_Linear"
    patchtst = resource_root / "baseline_PatchTST_supervised"
    ettm1 = resource_root / "benchmark_ETTm1"
    etth1 = resource_root / "benchmark_ETTh1"
    weather = resource_root / "benchmark_Weather"
    for path in (dlinear, patchtst, ettm1, etth1, weather):
        path.mkdir(parents=True)
    (ettm1 / "ETT-small").mkdir()
    (etth1 / "ETT-small").mkdir()
    (ettm1 / "ETT-small" / "ETTm1.csv").write_text("date,a\n2020-01-01,1\n", encoding="utf-8")
    (etth1 / "ETT-small" / "ETTh1.csv").write_text("date,a\n2020-01-01,1\n", encoding="utf-8")
    (weather / "weather.csv").write_text("date,a\n2020-01-01,1\n", encoding="utf-8")

    resources = [
        MaterializedResource("baseline", "DLinear_LTSF_Linear", "https://github.com/cure-lab/LTSF-Linear", "cloned_git", str(dlinear)),
        MaterializedResource("baseline", "PatchTST_supervised", "https://github.com/yuqinie98/PatchTST", "cloned_git", str(patchtst)),
        MaterializedResource("benchmark", "ETTm1", "https://github.com/zhouhaoyi/ETDataset", "cloned_git", str(ettm1)),
        MaterializedResource("benchmark", "ETTh1", "https://github.com/zhouhaoyi/ETDataset", "cloned_git", str(etth1)),
        MaterializedResource("benchmark", "Weather", "https://example.test/weather", "copied_local", str(weather)),
    ]

    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    def fake_execution_spec(request, *, plan, resources, workspace, results_dir):
        script = f"""
import json
from pathlib import Path

Path({str(results_dir / "generic_metrics.json")!r}).write_text(
    json.dumps({{"ai_spec_used": True, "resources": {len(resources)}}}),
    encoding="utf-8",
)
"""
        return (
            {
                "files": [{"path": "ai_generated_runner.py", "content": script}],
                "commands": [{"name": "ai_runner", "cwd": ".", "argv": ["python", "ai_generated_runner.py"], "timeout_seconds": 30}],
                "expected_outputs": ["generic_metrics.json"],
                "notes": f"AI-generated spec uses {dlinear}, {patchtst}, {ettm1}, {etth1}, and {weather}.",
            },
            "ai execution prompt",
            "{}",
        )

    progress = []
    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", lambda request, workspace: resources)
    monkeypatch.setattr(generic_agent, "request_generic_execution_spec", fake_execution_spec)
    monkeypatch.setattr(generic_agent, "_resolve_hardware_profile", lambda *args, **kwargs: HardwareProfile(accelerator="cpu"))

    state = run_generic_experiment_agent(_explicit_request(tmp_path), execute=True, progress_callback=progress.append)

    assert state.status == "generic_completed"
    assert state.metrics_file is not None
    metrics = json.loads(Path(state.metrics_file).read_text(encoding="utf-8"))
    assert metrics["ai_spec_used"] is True
    assert "request_generic_execution" in progress
    assert "deterministic_real_ltsf_runner" not in Path(state.implementation_prompt_file).read_text(encoding="utf-8")


def test_real_resource_validation_self_debugs_before_execution(monkeypatch, tmp_path):
    baseline_dir = tmp_path / "materialized" / "baseline_DLinear"
    dataset_dir = tmp_path / "materialized" / "benchmark_ETTm1"
    baseline_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    resources = [
        MaterializedResource(
            role="baseline",
            name="DLinear",
            location="https://github.com/example/DLinear",
            status="cloned_git",
            local_path=str(baseline_dir),
        ),
        MaterializedResource(
            role="benchmark",
            name="ETTm1",
            location="https://github.com/example/ETDataset",
            status="cloned_git",
            local_path=str(dataset_dir),
        ),
    ]

    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    def fake_execution_spec(request, *, plan, resources, workspace, results_dir):
        return (
            {
                "files": [
                    {
                        "path": "train_eval.py",
                        "content": "class DummyBaseline: pass\ntrain_x = torch.randn(10, 96, 7)\n",
                    }
                ],
                "commands": [{"name": "smoke", "cwd": ".", "argv": ["python", "train_eval.py", "--smoke"]}],
                "expected_outputs": ["generic_metrics.json"],
                "notes": "user must replace dummy baseline later",
            },
            "execution prompt",
            "{}",
        )

    def fake_debug_spec(request, *, plan, resources, previous_spec, command_results, failure_summary, workspace, results_dir, attempt):
        script = f"""
import json
from pathlib import Path

BASELINE_RESOURCE = {str(baseline_dir)!r}
DATASET_RESOURCE = {str(dataset_dir)!r}

Path({str(results_dir / "generic_metrics.json")!r}).write_text(
    json.dumps({{"resource_backed": True, "baseline": BASELINE_RESOURCE, "dataset": DATASET_RESOURCE}}),
    encoding="utf-8",
)
"""
        return (
            {
                "files": [{"path": "run_real_subset.py", "content": script}],
                "commands": [
                    {
                        "name": "real_subset",
                        "cwd": ".",
                        "argv": ["python", "run_real_subset.py"],
                        "timeout_seconds": 30,
                    }
                ],
                "expected_outputs": ["generic_metrics.json"],
                "notes": f"Uses {baseline_dir} and {dataset_dir}.",
            },
            "debug prompt",
            "{}",
        )

    progress = []
    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", lambda request, workspace: resources)
    monkeypatch.setattr(generic_agent, "request_generic_execution_spec", fake_execution_spec)
    monkeypatch.setattr(generic_agent, "request_generic_debug_execution_spec", fake_debug_spec)
    monkeypatch.setattr(generic_agent, "_resolve_hardware_profile", lambda *args, **kwargs: HardwareProfile(accelerator="cpu"))

    state = run_generic_experiment_agent(_explicit_request(tmp_path), execute=True, progress_callback=progress.append)

    assert state.status == "generic_completed"
    assert "debug_generic_execution" in progress
    assert state.metrics_file is not None
    assert json.loads(Path(state.metrics_file).read_text(encoding="utf-8"))["resource_backed"] is True
    summary = json.loads(Path(state.report_file).read_text(encoding="utf-8"))
    assert [attempt["status"] for attempt in summary["attempts"]] == ["failed_validation", "completed"]


def test_generated_file_paths_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError, match="escapes root"):
        _write_generated_files(
            {"files": [{"path": "../escape.py", "content": "print('bad')"}]},
            tmp_path / "workspace",
        )


def test_unsafe_generic_command_is_rejected(monkeypatch, tmp_path):
    def fake_plan(request):
        return {"task_type": "time_series_forecasting"}, "plan prompt", "{}"

    def fake_execution_spec(request, *, plan, resources, workspace, results_dir):
        return (
            {
                "files": [],
                "commands": [{"name": "bad", "cwd": ".", "argv": ["rm", "-rf", "."]}],
                "expected_outputs": [],
                "notes": "bad command",
            },
            "execution prompt",
            "{}",
        )

    monkeypatch.setattr(generic_agent, "request_generic_experiment_plan", fake_plan)
    monkeypatch.setattr(generic_agent, "materialize_generic_resources", lambda request, workspace: [])
    monkeypatch.setattr(generic_agent, "request_generic_execution_spec", fake_execution_spec)
    monkeypatch.setattr(generic_agent, "_resolve_hardware_profile", lambda *args, **kwargs: HardwareProfile(accelerator="cpu"))

    state = run_generic_experiment_agent(_request(tmp_path), execute=True)

    assert state.status == "failed"
    assert "not allowed" in state.error
