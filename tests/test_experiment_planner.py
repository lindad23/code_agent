from code_agent.experiments.models import ExperimentRequest
from code_agent.experiments.planner import create_experiment_plan, parse_huggingface_id, prepare_plan_request


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


def test_create_experiment_plan_uses_user_repositories(monkeypatch):
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
      "num_train_epochs": 1,
      "learning_rate": 0.00002,
      "train_batch_size": 8,
      "eval_batch_size": 16,
      "max_length": 128,
      "weight_decay": 0.01,
      "seed": 42,
      "use_cpu": true,
      "max_train_samples": 32,
      "max_eval_samples": 32,
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
        task="在 SST-2 上微调文本分类模型",
        api_provider="deepseek",
        plan_timeout_seconds=17,
    )

    plan, prompt, _ = create_experiment_plan(request)

    assert plan.dataset_config == "sst2"
    assert plan.text_columns == ["sentence"]
    assert "SST-2" in prompt
    assert captured["timeout"] == 17


def test_prepare_plan_request_does_not_call_api():
    request = ExperimentRequest(
        baseline_url="https://huggingface.co/distilbert/distilbert-base-uncased",
        benchmark_url="https://huggingface.co/datasets/nyu-mll/glue",
        task="使用 sst2 做文本分类",
        api_provider="deepseek",
    )

    model_id, dataset_id, prompt = prepare_plan_request(request)

    assert model_id == "distilbert/distilbert-base-uncased"
    assert dataset_id == "nyu-mll/glue"
    assert "使用 sst2 做文本分类" in prompt
