import pytest

from code_agent.experiments.models import ComparisonPlan, ExperimentRequest
from code_agent.experiments.planner import create_experiment_plan, parse_huggingface_id, prepare_plan_request


def _implementation() -> dict:
    return {
        "name": "Focal classification loss",
        "implementation_instructions": "Subclass Trainer and replace cross entropy with focal loss using gamma 2.0.",
    }


def test_parse_huggingface_model_and_dataset_urls():
    assert (
        parse_huggingface_id(
            "https://huggingface.co/distilbert/distilbert-base-uncased",
            repo_type="model",
        )
        == "distilbert/distilbert-base-uncased"
    )
    assert (
        parse_huggingface_id("https://huggingface.co/datasets/nyu-mll/glue", repo_type="dataset")
        == "nyu-mll/glue"
    )


def test_create_experiment_plan_requests_generated_code_improvement(monkeypatch):
    response = """{
      "model_id": "distilbert/distilbert-base-uncased",
      "dataset_id": "nyu-mll/glue",
      "dataset_config": "sst2",
      "task_type": "sequence_classification",
      "text_columns": ["sentence"],
      "label_column": "label",
      "train_split": "train",
      "eval_split": "validation",
      "metric_name": "accuracy",
      "num_labels": 2,
      "max_length": 128,
      "seed": 42,
      "use_cpu": true,
      "max_train_samples": 32,
      "max_eval_samples": 32,
      "baseline": {
        "method": "baseline",
        "main_change": "none",
        "learning_rate": 0.00002,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "num_train_epochs": 1,
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "label_smoothing_factor": 0.0,
        "classifier_dropout": null,
        "early_stopping_patience": null
      },
      "improved": {
        "method": "improved",
        "main_change": "focal loss",
        "learning_rate": 0.00002,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "num_train_epochs": 1,
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "label_smoothing_factor": 0.0,
        "classifier_dropout": null,
        "early_stopping_patience": null
      },
      "implementation": {
        "name": "Focal classification loss",
        "implementation_instructions": "Subclass Trainer and replace cross entropy with focal loss using gamma 2.0."
      },
      "rationale": "SST-2 is binary sentiment classification."
    }"""
    captured = {}

    def fake_call_llm(*args, **kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("code_agent.experiments.planner.call_llm", fake_call_llm)
    request = ExperimentRequest(
        baseline_url="https://huggingface.co/distilbert/distilbert-base-uncased",
        benchmark_url="https://huggingface.co/datasets/nyu-mll/glue",
        task="Implement focal loss with gamma 2.0 for improved and compare it with baseline on SST-2.",
        api_provider="deepseek",
        plan_timeout_seconds=17,
    )

    plan, prompt, _ = create_experiment_plan(request)

    assert plan.dataset_config == "sst2"
    assert plan.text_columns == ["sentence"]
    assert plan.improved.main_change == "focal loss"
    assert plan.implementation.name == "Focal classification loss"
    assert "Do not invent an improvement" in prompt
    assert captured["timeout"] == 17


def test_prepare_plan_request_does_not_call_api():
    request = ExperimentRequest(
        baseline_url="https://huggingface.co/distilbert/distilbert-base-uncased",
        benchmark_url="https://huggingface.co/datasets/nyu-mll/glue",
        task="Fine tune a text classifier on SST-2.",
        api_provider="deepseek",
    )

    model_id, dataset_id, prompt = prepare_plan_request(request)

    assert model_id == "distilbert/distilbert-base-uncased"
    assert dataset_id == "nyu-mll/glue"
    assert "Fine tune a text classifier on SST-2." in prompt


def test_comparison_plan_rejects_parameter_change_instead_of_generated_code():
    payload = {
        "model_id": "model/id",
        "dataset_id": "dataset/id",
        "task_type": "sequence_classification",
        "text_columns": ["sentence"],
        "baseline": {"method": "baseline", "main_change": "none"},
        "improved": {"method": "improved", "main_change": "warmup only", "warmup_ratio": 0.1},
        "implementation": _implementation(),
    }

    with pytest.raises(ValueError, match="Generated-code comparisons"):
        ComparisonPlan.model_validate(payload)


def test_comparison_plan_treats_zero_early_stopping_as_disabled():
    plan = ComparisonPlan.model_validate(
        {
            "model_id": "model/id",
            "dataset_id": "dataset/id",
            "task_type": "sequence_classification",
            "text_columns": ["sentence"],
            "baseline": {"method": "baseline", "main_change": "none", "early_stopping_patience": 0},
            "improved": {"method": "improved", "main_change": "focal loss", "early_stopping_patience": 0},
            "implementation": _implementation(),
        }
    )

    assert plan.baseline.early_stopping_patience is None
    assert plan.improved.early_stopping_patience is None
