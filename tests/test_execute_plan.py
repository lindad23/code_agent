import pytest

from code_agent.experiments.execute_plan import (
    _asset_cache_dirs,
    _assert_custom_head_gradients_flow,
    _dataset_repository_patterns,
    _infer_fusion_strategy,
    _is_cuda_out_of_memory_error,
    _move_trainer_model_to_device,
    _plan_with_oom_retry_batches,
    _retry_variant_after_oom,
    _select_metric_logits,
    _summarize_custom_head_gradients,
)
from code_agent.experiments.models import ComparisonPlan, TrainingVariant


def test_asset_caches_are_shared_outside_long_run_id(tmp_path):
    workspace = tmp_path / "experiments" / ("experiment-" + "x" * 80)

    code_cache, data_repo_cache, dataset_cache = _asset_cache_dirs(workspace)

    assert code_cache == (tmp_path / "experiments" / "asset_cache" / "code").resolve()
    assert data_repo_cache == (tmp_path / "experiments" / "asset_cache" / "data" / "repositories").resolve()
    assert dataset_cache == (tmp_path / "experiments" / "asset_cache" / "data" / "datasets").resolve()
    assert "experiment-" not in str(dataset_cache)


def test_existing_dataset_cache_is_imported_into_shared_data_folder(tmp_path):
    workspace = tmp_path / "experiments" / "new-run"
    legacy_cache = tmp_path / "experiments" / ".dataset_cache"
    legacy_cache.mkdir(parents=True)
    legacy_cache.joinpath("cached.arrow").write_text("already prepared", encoding="utf-8")

    _, _, dataset_cache = _asset_cache_dirs(workspace)

    assert dataset_cache.joinpath("cached.arrow").read_text(encoding="utf-8") == "already prepared"


def test_dataset_repository_patterns_select_config_folder():
    assert _dataset_repository_patterns("sst2") == [
        ".gitattributes",
        "README*",
        "dataset_infos.json",
        "*.py",
        "sst2/**",
    ]
    assert _dataset_repository_patterns(None) is None


class _FakeModel:
    def __init__(self):
        self.devices = []
        self.params = []

    def to(self, device):
        self.devices.append(device)
        return self

    def named_parameters(self):
        return iter(self.params)


class _FakeParameter:
    def __init__(self, grad):
        self.grad = grad


class _FakeArgs:
    device = "cuda:0"


class _FakeTrainer:
    def __init__(self):
        self.args = _FakeArgs()
        self.model = _FakeModel()
        self.model_wrapped = _FakeModel()


def test_move_trainer_model_to_device_moves_model_and_wrapper():
    trainer = _FakeTrainer()

    _move_trainer_model_to_device(trainer)

    assert trainer.model.devices == ["cuda:0"]
    assert trainer.model_wrapped.devices == ["cuda:0"]


def test_move_trainer_model_to_device_noops_without_device():
    trainer = _FakeTrainer()
    trainer.args = object()

    _move_trainer_model_to_device(trainer)

    assert trainer.model.devices == []


def test_summarize_custom_head_gradients_watches_custom_parameters():
    model = _FakeModel()
    model.params = [
        ("distilbert.weight", _FakeParameter(100)),
        ("custom_head.classifier.weight", _FakeParameter(3)),
        ("fusion_head.weight", _FakeParameter(2)),
    ]

    summary = _summarize_custom_head_gradients(model)

    assert summary["watched_parameter_count"] == 2
    assert summary["watched_parameter_names"] == ["custom_head.classifier.weight", "fusion_head.weight"]
    assert summary["watched_gradient_abs_sum"] == 5.0


def test_assert_custom_head_gradients_flow_rejects_unused_head():
    with pytest.raises(RuntimeError, match="received no gradients"):
        _assert_custom_head_gradients_flow(
            {
                "watched_parameter_count": 1,
                "watched_parameter_names": ["custom_head.classifier.weight"],
                "watched_gradient_abs_sum": 0.0,
            }
        )


def test_select_metric_logits_unwraps_trainer_auxiliary_outputs():
    logits = [[0.1, 0.9], [0.8, 0.2]]
    hidden_states = ("large auxiliary tensor",)

    assert _select_metric_logits((logits, hidden_states)) == logits
    assert _select_metric_logits({"logits": logits}) == logits


def test_select_metric_logits_rejects_missing_logits_dict():
    with pytest.raises(ValueError, match="without a logits key"):
        _select_metric_logits({"hidden_states": []})


def test_infer_fusion_strategy_prefers_structured_variant_fields():
    variant = TrainingVariant(
        method="improved",
        main_change="custom fusion head",
        strategy="learned_weighted_sum",
        k=4,
    )

    assert _infer_fusion_strategy(variant) == ("learned_weighted_sum", 4)


def test_infer_fusion_strategy_defaults_generic_hidden_fusion_to_mean_last_k():
    variant = TrainingVariant(
        method="improved",
        main_change="multi-layer hidden states fusion classification head",
    )

    assert _infer_fusion_strategy(variant) == ("mean_last_k", 4)


def test_cuda_oom_detection_matches_wrapped_torch_error_text():
    try:
        raise RuntimeError("CUDA out of memory. Tried to allocate 96.00 MiB.")
    except RuntimeError as exc:
        wrapped = RuntimeError("training failed")
        wrapped.__cause__ = exc

    assert _is_cuda_out_of_memory_error(wrapped)


def test_retry_variant_halves_batches_and_preserves_effective_train_batch():
    variant = TrainingVariant(
        method="improved",
        main_change="custom head",
        train_batch_size=32,
        eval_batch_size=64,
    )

    retry = _retry_variant_after_oom(variant)

    assert retry.train_batch_size == 16
    assert retry.eval_batch_size == 32
    assert retry.gradient_accumulation_steps == 2


def test_retry_plan_adjusts_baseline_and_improved_together():
    plan = ComparisonPlan.model_validate(
        {
            "model_id": "distilbert/distilbert-base-uncased",
            "dataset_id": "nyu-mll/glue",
            "dataset_config": "rte",
            "task_type": "sequence_classification",
            "text_columns": ["sentence1", "sentence2"],
            "label_column": "label",
            "baseline": {
                "method": "baseline",
                "main_change": "none",
                "train_batch_size": 32,
                "eval_batch_size": 64,
            },
            "improved": {
                "method": "improved",
                "main_change": "custom head",
                "train_batch_size": 32,
                "eval_batch_size": 64,
            },
            "implementation": {
                "name": "custom_head",
                "implementation_instructions": "Install a custom classification head.",
            },
        }
    )

    retry = _plan_with_oom_retry_batches(plan)

    assert retry.baseline.train_batch_size == 16
    assert retry.improved.train_batch_size == 16
    assert retry.baseline.gradient_accumulation_steps == 2
    assert retry.improved.gradient_accumulation_steps == 2
