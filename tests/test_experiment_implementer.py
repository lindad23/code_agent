import pytest

from code_agent.experiments.implementer import (
    ImplementationGenerationError,
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


def _fusion_plan() -> ComparisonPlan:
    return ComparisonPlan.model_validate(
        {
            "model_id": "model/id",
            "dataset_id": "dataset/id",
            "task_type": "sequence_classification",
            "text_columns": ["sentence"],
            "baseline": {"method": "baseline", "main_change": "none"},
            "improved": {"method": "improved", "main_change": "hidden states fusion head"},
            "implementation": {
                "name": "Multi layer hidden states fusion",
                "implementation_instructions": "Use output_hidden_states and a custom fusion head.",
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
    assert "same device as the existing model parameters" in prompt
    assert "DataParallel" in prompt
    assert "output_hidden_states=True" in prompt
    assert "unused custom head" in prompt
    assert "SequenceClassifierOutput" in prompt
    assert "Never return `(loss, logits)`" in prompt
    assert "Do not include `hidden_states`, `attentions`" in prompt
    assert "SequenceClassifierOutput(loss=loss, logits=fused_logits)" in prompt


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


def test_request_implementation_uses_builtin_hidden_state_fusion(monkeypatch):
    def fail_call_llm(*args, **kwargs):
        raise AssertionError("hidden-state fusion should use the built-in implementation")

    monkeypatch.setattr("code_agent.experiments.implementer.call_llm", fail_call_llm)
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="compare hidden states fusion heads",
        api_provider="deepseek",
    )

    generated, response = request_implementation(request, _fusion_plan(), prompt="prompt")

    assert generated == response
    assert "custom_fusion_head" in generated
    assert "fusion_strategy" in generated
    assert "SequenceClassifierOutput(loss=loss, logits=logits)" in generated


def test_request_implementation_repairs_validation_failure(monkeypatch):
    bad_source = """
import os
CHANGE_SUMMARY = "bad focal loss"
def configure_model_config(config):
    return config
def build_trainer_class(base_trainer):
    return base_trainer
"""
    fixed_source = """
import torch.nn.functional as F
CHANGE_SUMMARY = "fixed focal loss"
def configure_model_config(config):
    return config
def build_trainer_class(base_trainer):
    class FocalTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            return F.cross_entropy(outputs.logits, labels)
    return FocalTrainer
"""
    prompts = []

    def fake_call_llm(prompt, *args, **kwargs):
        prompts.append(prompt)
        return bad_source if len(prompts) == 1 else fixed_source

    monkeypatch.setattr("code_agent.experiments.implementer.call_llm", fake_call_llm)
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="compare an improvement",
        api_provider="deepseek",
    )

    generated, response = request_implementation(request, _plan(), prompt="original prompt")

    assert generated == fixed_source.strip() + "\n"
    assert response == fixed_source
    assert len(prompts) == 2
    assert "failed validation" in prompts[1]
    assert "disallowed module" in prompts[1]


def test_request_implementation_error_carries_failed_attempts(monkeypatch):
    bad_source = """
import os
CHANGE_SUMMARY = "bad focal loss"
def configure_model_config(config):
    return config
def build_trainer_class(base_trainer):
    return base_trainer
"""
    monkeypatch.setattr("code_agent.experiments.implementer.call_llm", lambda *args, **kwargs: bad_source)
    request = ExperimentRequest(
        baseline_url="model/id",
        benchmark_url="dataset/id",
        task="compare an improvement",
        api_provider="deepseek",
    )

    with pytest.raises(ImplementationGenerationError) as exc_info:
        request_implementation(request, _plan(), prompt="prompt", max_attempts=1)

    assert exc_info.value.responses == [bad_source]
    assert exc_info.value.sources == [bad_source.strip() + "\n"]
    assert "disallowed module" in exc_info.value.errors[0]


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


def test_hidden_state_fusion_rejects_default_logits_loss_path():
    source = """
import torch.nn.functional as F
CHANGE_SUMMARY = "bad fusion"
def configure_model_config(config):
    return config
def build_trainer_class(base_trainer):
    class FusionTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            return F.cross_entropy(outputs.logits, labels)
    return FusionTrainer
"""

    with pytest.raises(ValueError, match="fused hidden-state logits"):
        validate_implementation_source(source, plan=_fusion_plan())


def test_hidden_state_fusion_rejects_unused_model_subclass():
    source = """
import torch
CHANGE_SUMMARY = "bad fusion"
def configure_model_config(config):
    return config
class DistilBertForSequenceClassificationWithFusion:
    pass
def build_trainer_class(base_trainer):
    class FusionTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            return hidden_states[-1].sum() * 0
    return FusionTrainer
"""

    with pytest.raises(ValueError, match="executor will not instantiate"):
        validate_implementation_source(source, plan=_fusion_plan())


def test_hidden_state_fusion_rejects_bare_logits_return_outputs():
    source = """
import torch
import torch.nn.functional as F
CHANGE_SUMMARY = "bad fusion outputs"
def configure_model_config(config):
    return config
def build_trainer_class(base_trainer):
    class FusionTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            fused_logits = hidden_states[-1].mean(dim=1)
            loss = F.cross_entropy(fused_logits, labels)
            if return_outputs:
                return (loss, fused_logits)
            return loss
    return FusionTrainer
"""

    with pytest.raises(ValueError, match="bare logits tensor"):
        validate_implementation_source(source, plan=_fusion_plan())


def test_hidden_state_fusion_rejects_auxiliary_prediction_outputs():
    source = """
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
CHANGE_SUMMARY = "bad fusion outputs"
def configure_model_config(config):
    return config
def build_trainer_class(base_trainer):
    class FusionTrainer(base_trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            fused_logits = hidden_states[-1].mean(dim=1)
            loss = F.cross_entropy(fused_logits, labels)
            wrapped_outputs = SequenceClassifierOutput(
                loss=loss,
                logits=fused_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            if return_outputs:
                return (loss, wrapped_outputs)
            return loss
    return FusionTrainer
"""

    with pytest.raises(ValueError, match="only loss/logits"):
        validate_implementation_source(source, plan=_fusion_plan())
