import json

from code_agent import experiment_main
from code_agent.experiments.models import ExperimentRunState


def test_main_input_defaults_to_full_study_when_ablation_enabled(monkeypatch, tmp_path, capsys):
    input_file = tmp_path / "inputs.json"
    input_file.write_text(
        json.dumps(
            {
                "Experience name": "fusion",
                "Improved idea": "实现 hidden states 融合分类头。",
                "Baselines url": {"DistilBERT": ""},
                "Benchmarks url": {"GLUE": ""},
                "Evaluation indexs": "",
                "Ablation": "True",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_study_experiment_agent(request, *, mode_name, progress_callback=None):
        captured["request"] = request
        captured["mode_name"] = mode_name
        return ExperimentRunState(status="completed", run_id="run-1", request_file="request.json")

    monkeypatch.setattr(experiment_main, "run_study_experiment_agent", fake_run_study_experiment_agent)

    code = experiment_main.main(["--input", str(input_file), "--no-progress"])

    assert code == 0
    assert captured["mode_name"] == "full"
    assert captured["request"].api_provider == "deepseek"
    assert captured["request"].run_name == "fusion"
    assert captured["request"].baseline_url_explicit is False
    assert "hidden states" in captured["request"].task
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_main_requires_legacy_arguments_without_input():
    try:
        experiment_main.main(["--no-progress"])
    except SystemExit as exc:
        assert "--baseline-url" in str(exc)
    else:
        raise AssertionError("main should reject missing legacy arguments without --input")


def test_main_routes_github_input_to_generic_backend(monkeypatch, tmp_path):
    input_file = tmp_path / "inputs.json"
    input_file.write_text(
        json.dumps(
            {
                "Experience name": "ltsf",
                "Improved idea": "实现频率感知残差校正。",
                "Baselines url": {"DLinear": "https://github.com/cure-lab/LTSF-Linear"},
                "Benchmarks url": {"ETTm1": "https://github.com/zhouhaoyi/ETDataset"},
                "Evaluation indexs": "MSE, MAE",
                "Ablation": "True",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_generic_agent(request, *, execute=True, progress_callback=None):
        captured["request"] = request
        captured["execute"] = execute
        return ExperimentRunState(status="generic_planned", run_id="run-1", request_file="request.json")

    def fail_hf_agent(*args, **kwargs):
        raise AssertionError("GitHub inputs should not use the Hugging Face study backend.")

    monkeypatch.setattr(experiment_main, "run_generic_experiment_agent", fake_generic_agent)
    monkeypatch.setattr(experiment_main, "run_study_experiment_agent", fail_hf_agent)

    code = experiment_main.main(["--input", str(input_file), "--no-progress"])

    assert code == 0
    assert captured["execute"] is True
    assert captured["request"].baseline_url == "https://github.com/cure-lab/LTSF-Linear"
    assert "频率感知" in captured["request"].task


def test_main_plan_only_keeps_generic_backend_in_planning_mode(monkeypatch, tmp_path):
    input_file = tmp_path / "inputs.json"
    input_file.write_text(
        json.dumps(
            {
                "Experience name": "ltsf",
                "Improved idea": "实现频率感知残差校正。",
                "Baselines url": {"DLinear": "https://github.com/cure-lab/LTSF-Linear"},
                "Benchmarks url": {"ETTm1": "https://github.com/zhouhaoyi/ETDataset"},
                "Evaluation indexs": "MSE, MAE",
                "Ablation": "True",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_generic_agent(request, *, execute=True, progress_callback=None):
        captured["execute"] = execute
        return ExperimentRunState(status="generic_planned", run_id="run-1", request_file="request.json")

    monkeypatch.setattr(experiment_main, "run_generic_experiment_agent", fake_generic_agent)

    code = experiment_main.main(["--input", str(input_file), "--plan-only", "--no-progress"])

    assert code == 0
    assert captured["execute"] is False
