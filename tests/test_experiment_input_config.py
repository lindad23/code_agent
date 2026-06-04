import json

import pytest

from code_agent.experiments.input_config import load_input_run_config
from code_agent.experiments.models import unique_run_id
from code_agent.experiments.planner import prepare_study_plan_request


def _write_input(tmp_path, payload):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_input_config_builds_full_study_request_from_resource_names(tmp_path):
    path = _write_input(
        tmp_path,
        {
            "Experience name": "fusion check",
            "Improved idea": "实现 hidden states 融合分类头。",
            "Baselines url": {"DistilBERT": ""},
            "Benchmarks url": {"GLUE": ""},
            "Evaluation indexs": "",
            "Ablation": "True",
        },
    )

    config = load_input_run_config(path)

    assert config.use_study is True
    assert config.study_mode == "full"
    assert config.request.run_name == "fusion check"
    assert config.request.baseline_url == "DistilBERT"
    assert config.request.benchmark_url == "GLUE"
    assert config.request.baseline_url_explicit is False
    assert config.request.benchmark_url_explicit is False
    assert "AI should resolve" in config.request.resource_context
    assert "choose suitable metrics" in config.request.task

    model_id, dataset_id, prompt = prepare_study_plan_request(config.request)
    assert model_id is None
    assert dataset_id is None
    assert "Resolve model_id" in prompt
    assert "Resolve dataset_id" in prompt


def test_input_config_preserves_user_specified_urls(tmp_path):
    path = _write_input(
        tmp_path,
        {
            "Experience name": "",
            "Improved idea": "实现 focal loss。",
            "Baselines url": {"distilbert": "https://huggingface.co/distilbert/distilbert-base-uncased"},
            "Benchmarks url": {"glue": "https://huggingface.co/datasets/nyu-mll/glue"},
            "Evaluation indexs": ["accuracy", "runtime"],
            "Ablation": "False",
        },
    )

    config = load_input_run_config(path)

    assert config.use_study is False
    assert config.request.baseline_url_explicit is True
    assert config.request.benchmark_url_explicit is True
    assert "Required evaluation indexes: accuracy, runtime" in config.request.task


def test_input_config_routes_non_huggingface_resources_to_generic_backend(tmp_path):
    path = _write_input(
        tmp_path,
        {
            "Experience name": "ltsf",
            "Improved idea": "实现频率感知残差校正。",
            "Baselines url": {"DLinear": "https://github.com/cure-lab/LTSF-Linear"},
            "Benchmarks url": {"ETTm1": "https://github.com/zhouhaoyi/ETDataset"},
            "Evaluation indexs": "MSE, MAE",
            "Ablation": "True",
        },
    )

    config = load_input_run_config(path)

    assert config.backend == "generic_ai_experiment"
    assert config.request.baseline_url_explicit is True
    assert config.request.benchmark_url_explicit is True


def test_input_config_rejects_missing_improved_idea(tmp_path):
    path = _write_input(
        tmp_path,
        {
            "Experience name": "",
            "Improved idea": " ",
            "Baselines url": {"baseline": ""},
            "Benchmarks url": {"benchmark": ""},
            "Evaluation indexs": "",
            "Ablation": "True",
        },
    )

    with pytest.raises(ValueError, match="Improved idea"):
        load_input_run_config(path)


def test_input_config_rejects_empty_resource_keys(tmp_path):
    path = _write_input(
        tmp_path,
        {
            "Experience name": "",
            "Improved idea": "实现 focal loss。",
            "Baselines url": {"": "https://huggingface.co/distilbert/distilbert-base-uncased"},
            "Benchmarks url": {"glue": ""},
            "Evaluation indexs": "",
            "Ablation": "True",
        },
    )

    with pytest.raises(ValueError, match="empty resource name"):
        load_input_run_config(path)


def test_unique_run_id_uses_experience_name_as_prefix():
    run_id = unique_run_id("my experiment")

    assert run_id.startswith("my_experiment-experiment-")
