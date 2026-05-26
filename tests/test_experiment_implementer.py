import pytest

from code_agent.experiments.implementer import (
    build_implementation_prompt,
    request_implementation,
    validate_implementation_source,
)
from code_agent.experiments.models import ComparisonPlan, ExperimentRequest


def _plan() -> ComparisonPlan:
    return ComparisonPlan.model_validate(
        {
            "model_id": "model/id",
            "dataset_id": "dataset/id",
            "task_type": "sequence_classification",
            "text_columns": ["sentence"],
            "baseline": {"method": "baseline", "main_change": "none"},
            "improved": {"method": "improved", "main_change": "focal loss"},
            "implementation": {
                "name": "Focal loss",
                "implementation_instructions": "Subclass Trainer and implement focal cross entropy with gamma two.",
            },
        }
    )


def test_implementation_prompt_exposes_hook_contract():
    prompt = build_implementation_prompt(_plan(), "Implement focal loss with gamma=2.")

    assert "configure_model_config" in prompt
    assert "build_trainer_class" in prompt
    assert "Focal loss" in prompt
    assert "Implement focal loss with gamma=2." in prompt
    assert "baseline remains unchanged" in prompt


def test_request_implementation_returns_valid_generated_module(monkeypatch):
    source = """
import torch.nn.functional as F

CHANGE_SUMMARY = "focal loss"

def configure_model_config(config):
    return config

def build_trainer_class(base_trainer):
    class FocalTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            return probabilities.sum() * 0
    return FocalTrainer
"""
    monkeypatch.setattr("code_agent.experiments.implementer.call_llm", lambda *args, **kwargs: source)
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="compare an improvement",
        api_provider="deepseek",
    )

    generated, response = request_implementation(request, _plan(), prompt="prompt")

    assert generated == source.strip() + "\n"
    assert response == source


def test_generated_implementation_rejects_filesystem_import():
    source = """
import os
CHANGE_SUMMARY = "unsafe"
def configure_model_config(config):
    return config
def build_trainer_class(base_trainer):
    return base_trainer
"""

    with pytest.raises(ValueError, match="disallowed module"):
        validate_implementation_source(source)
