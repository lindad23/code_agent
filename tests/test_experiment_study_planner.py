from code_agent.experiments.models import ExperimentRequest, ExperimentStudyPlan
from code_agent.experiments.planner import PlanValidationError, create_experiment_study_plan, prepare_study_plan_request
from code_agent.experiments.study_agent import run_study_planning_agent
from code_agent.experiments.study import (
    QUICK_MODE_DEFAULT_EPOCHS,
    QUICK_MODE_DEFAULT_EVAL_SAMPLES,
    QUICK_MODE_DEFAULT_TRAIN_SAMPLES,
    comparison_plan_for_variant,
    expand_study_plan,
    glue_benchmark,
    normalize_study_plan,
)


def _study_payload() -> dict:
    return {
        "model_id": "distilbert/distilbert-base-uncased",
        "dataset_id": "nyu-mll/glue",
        "task_type": "sequence_classification",
        "modes": {
            "quick": {
                "seeds": [13],
                "max_train_samples": 512,
                "max_eval_samples": 512,
                "num_train_epochs": 1,
                "train_batch_size": 16,
                "eval_batch_size": 32,
            },
            "full": {
                "seeds": [13, 42, 3407],
                "max_train_samples": None,
                "max_eval_samples": None,
                "num_train_epochs": 3,
                "train_batch_size": 32,
                "eval_batch_size": 64,
            },
        },
        "benchmarks": [
            glue_benchmark("sst2").model_dump(mode="json"),
            glue_benchmark("mrpc").model_dump(mode="json"),
            glue_benchmark("rte").model_dump(mode="json"),
            glue_benchmark("qnli").model_dump(mode="json"),
        ],
        "variants": [
            {
                "name": "baseline_last_cls",
                "family": "baseline",
                "main_change": "none",
                "strategy": "last_layer",
            },
            {
                "name": "fusion_last_layer",
                "family": "improved",
                "main_change": "multi-layer fusion head using last hidden layer only",
                "strategy": "last_layer",
            },
            {
                "name": "fusion_mean_last_4",
                "family": "improved",
                "main_change": "multi-layer fusion head averaging the last 4 hidden layers",
                "strategy": "mean_last_k",
                "k": 4,
            },
            {
                "name": "fusion_learned_last_4",
                "family": "improved",
                "main_change": "multi-layer fusion head with learned weights over the last 4 hidden layers",
                "strategy": "learned_weighted_sum",
                "k": 4,
            },
        ],
        "implementation": {
            "name": "multi_layer_hidden_states_fusion",
            "implementation_instructions": "Implement last_layer, mean_last_k, and learned_weighted_sum fusion heads.",
        },
        "resource_logging": {
            "record_wall_time": True,
            "record_train_runtime": True,
            "record_eval_runtime": True,
            "record_samples_per_second": True,
            "record_max_gpu_memory": True,
            "record_gpu_name": True,
        },
        "launch": {
            "strategy": "gpu_worker_pool",
            "preferred_parallelism": "one_run_per_gpu",
            "max_concurrent_runs": 4,
        },
        "failure_policy": {
            "preflight_tiny_run": True,
            "retry_once": True,
            "on_oom": "halve_batch_size_and_use_gradient_accumulation",
            "write_failure_signature": True,
        },
        "result_table_columns": [
            "mode",
            "benchmark",
            "variant",
            "strategy",
            "k",
            "seeds",
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "delta_vs_baseline",
            "avg_train_seconds",
            "max_gpu_memory_mb",
            "status",
        ],
        "rationale": "Full mode covers 4 GLUE tasks, 3 seeds, and all requested ablations.",
    }


def test_study_plan_expands_requested_glue_matrix():
    study = ExperimentStudyPlan.model_validate(_study_payload())

    full_cells = expand_study_plan(study, mode_name="full")
    quick_cells = expand_study_plan(study, mode_name="quick")

    assert len(full_cells) == 4 * 3 * 4
    assert len(quick_cells) == 4 * 1 * 4
    assert {cell.benchmark for cell in full_cells} == {"sst2", "mrpc", "rte", "qnli"}
    assert {cell.seed for cell in full_cells} == {13, 42, 3407}
    assert {cell.variant for cell in full_cells} == {
        "baseline_last_cls",
        "fusion_last_layer",
        "fusion_mean_last_4",
        "fusion_learned_last_4",
    }

    mrpc = next(cell for cell in full_cells if cell.benchmark == "mrpc")
    assert mrpc.text_columns == ["sentence1", "sentence2"]
    assert mrpc.metrics == ["accuracy", "f1"]


def test_study_plan_can_expand_improved_variant_to_current_comparison_plan():
    study = ExperimentStudyPlan.model_validate(_study_payload())

    plan = comparison_plan_for_variant(
        study,
        mode_name="quick",
        benchmark_name="qnli",
        seed=13,
        variant_name="fusion_mean_last_4",
    )

    assert plan.dataset_config == "qnli"
    assert plan.text_columns == ["question", "sentence"]
    assert plan.seed == 13
    assert plan.max_train_samples == 512
    assert plan.baseline.main_change == "none"
    assert plan.improved.main_change == "multi-layer fusion head averaging the last 4 hidden layers"
    assert plan.improved.strategy == "mean_last_k"
    assert plan.improved.k == 4


def test_study_plan_normalizes_ablation_family_to_improved():
    payload = _study_payload()
    payload["variants"][1]["family"] = "ablation"
    payload["variants"][2]["family"] = "ablation"
    payload["variants"][3]["family"] = "ablation"

    study = ExperimentStudyPlan.model_validate(payload)

    assert [variant.family for variant in study.variants] == ["baseline", "improved", "improved", "improved"]
    assert len(expand_study_plan(study, mode_name="full")) == 4 * 3 * 4


def test_normalize_study_plan_adds_quick_sample_caps_and_short_epoch():
    payload = _study_payload()
    payload["modes"]["quick"]["max_train_samples"] = None
    payload["modes"]["quick"]["max_eval_samples"] = None
    payload["modes"]["quick"]["num_train_epochs"] = 3
    study = ExperimentStudyPlan.model_validate(payload)

    normalized = normalize_study_plan(study)
    quick_cells = expand_study_plan(normalized, mode_name="quick")

    assert normalized.modes["quick"].max_train_samples == QUICK_MODE_DEFAULT_TRAIN_SAMPLES
    assert normalized.modes["quick"].max_eval_samples == QUICK_MODE_DEFAULT_EVAL_SAMPLES
    assert normalized.modes["quick"].num_train_epochs == QUICK_MODE_DEFAULT_EPOCHS
    assert {(cell.max_train_samples, cell.max_eval_samples) for cell in quick_cells} == {
        (QUICK_MODE_DEFAULT_TRAIN_SAMPLES, QUICK_MODE_DEFAULT_EVAL_SAMPLES)
    }
    assert {cell.training.num_train_epochs for cell in quick_cells} == {QUICK_MODE_DEFAULT_EPOCHS}
    assert normalized.modes["full"].max_train_samples is None
    assert normalized.modes["full"].num_train_epochs == 3


def test_normalize_study_plan_pins_hidden_state_fusion_variants():
    payload = _study_payload()
    payload["variants"] = [
        {
            "name": "last_layer_cls",
            "family": "baseline",
            "main_change": "none",
            "strategy": "last_layer",
        },
        {
            "name": "mean_last_k",
            "family": "improved",
            "main_change": "mean_pooling_last_k",
            "strategy": "mean_last_k",
            "k": 4,
        },
        {
            "name": "learned_weighted_sum",
            "family": "improved",
            "main_change": "learned_weighted_sum",
            "strategy": "learned_weighted_sum",
        },
    ]
    study = ExperimentStudyPlan.model_validate(payload)

    normalized = normalize_study_plan(
        study,
        user_task="Compare hidden states fusion heads: last_layer, mean_last_k, learned_weighted_sum.",
    )
    full_cells = expand_study_plan(normalized, mode_name="full")

    assert [variant.name for variant in normalized.variants] == [
        "baseline_last_cls",
        "fusion_last_layer",
        "fusion_mean_last_4",
        "fusion_learned_last_4",
    ]
    assert normalized.variants[3].k == 4
    assert "Do not hard-code a 12-layer BERT shape" in normalized.implementation.implementation_instructions
    assert len(full_cells) == 4 * 3 * 4


def test_create_experiment_study_plan_requests_matrix_without_collapsing(monkeypatch):
    payload = _study_payload()
    payload["modes"]["quick"]["max_train_samples"] = None
    payload["modes"]["quick"]["max_eval_samples"] = None
    payload["modes"]["quick"]["num_train_epochs"] = 3
    response = ExperimentStudyPlan.model_validate(payload).model_dump_json()
    captured = {}

    def fake_call_llm(*args, **kwargs):
        captured["prompt"] = args[0]
        captured.update(kwargs)
        return response

    monkeypatch.setattr("code_agent.experiments.planner.call_llm", fake_call_llm)
    request = ExperimentRequest(
        baseline_url="https://huggingface.co/distilbert/distilbert-base-uncased",
        benchmark_url="https://huggingface.co/datasets/nyu-mll/glue",
        task=(
            "Plan DistilBERT + GLUE on SST-2, MRPC, RTE, QNLI with seeds 13, 42, 3407 "
            "and last_layer, mean_last_k, learned_weighted_sum ablations."
        ),
        api_provider="deepseek",
    )

    plan, prompt, _ = create_experiment_study_plan(request)

    assert set(plan.modes["full"].seeds) == {13, 42, 3407}
    assert plan.modes["quick"].max_train_samples == QUICK_MODE_DEFAULT_TRAIN_SAMPLES
    assert plan.modes["quick"].max_eval_samples == QUICK_MODE_DEFAULT_EVAL_SAMPLES
    assert plan.modes["quick"].num_train_epochs == QUICK_MODE_DEFAULT_EPOCHS
    assert [variant.name for variant in plan.variants] == [
        "baseline_last_cls",
        "fusion_last_layer",
        "fusion_mean_last_4",
        "fusion_learned_last_4",
    ]
    assert [benchmark.name for benchmark in plan.benchmarks] == ["sst2", "mrpc", "rte", "qnli"]
    assert "Do not collapse multiple benchmarks" in prompt
    assert "max_train_samples 200" in prompt
    assert "SST-2" in prompt and "QNLI" in prompt
    assert captured["max_tokens"] == 4096


def test_study_agent_persists_invalid_llm_response(monkeypatch, tmp_path):
    def fake_request(*args, **kwargs):
        raise PlanValidationError("bad study", response='{"variants": "bad"}')

    monkeypatch.setattr("code_agent.experiments.study_agent.request_experiment_study_plan", fake_request)
    request = ExperimentRequest(
        baseline_url="distilbert/distilbert-base-uncased",
        benchmark_url="nyu-mll/glue",
        task="Plan a multi-seed GLUE study.",
        api_provider="deepseek",
        workspace_root=tmp_path / "workspaces",
        results_root=tmp_path / "results",
        run_name="bad-study",
    )

    state = run_study_planning_agent(request)

    assert state.status == "failed"
    assert state.plan_response_file is not None
    assert state.plan_error_file is not None
    assert '{"variants": "bad"}' in (tmp_path / "results" / "bad-study" / "study_plan_response.txt").read_text(
        encoding="utf-8"
    )
    assert "bad study" in (tmp_path / "results" / "bad-study" / "study_plan_validation_error.txt").read_text(
        encoding="utf-8"
    )


def test_prepare_study_plan_request_does_not_call_api():
    request = ExperimentRequest(
        baseline_url="distilbert/distilbert-base-uncased",
        benchmark_url="nyu-mll/glue",
        task="Plan a multi-seed GLUE study.",
        api_provider="deepseek",
    )

    model_id, dataset_id, prompt = prepare_study_plan_request(request)

    assert model_id == "distilbert/distilbert-base-uncased"
    assert dataset_id == "nyu-mll/glue"
    assert "experiment matrix" in prompt
