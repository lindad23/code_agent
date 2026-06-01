from __future__ import annotations

from code_agent.experiments.models import (
    BenchmarkSpec,
    ComparisonPlan,
    ExecutionCell,
    ExperimentStudyPlan,
    ImprovementSpec,
    StudyMode,
    TrainingVariant,
    VariantSpec,
)


GLUE_TASKS: dict[str, dict] = {
    "sst2": {
        "name": "sst2",
        "dataset_config": "sst2",
        "text_columns": ["sentence"],
        "metrics": ["accuracy"],
        "num_labels": 2,
    },
    "mrpc": {
        "name": "mrpc",
        "dataset_config": "mrpc",
        "text_columns": ["sentence1", "sentence2"],
        "metrics": ["accuracy", "f1"],
        "num_labels": 2,
    },
    "rte": {
        "name": "rte",
        "dataset_config": "rte",
        "text_columns": ["sentence1", "sentence2"],
        "metrics": ["accuracy"],
        "num_labels": 2,
    },
    "qnli": {
        "name": "qnli",
        "dataset_config": "qnli",
        "text_columns": ["question", "sentence"],
        "metrics": ["accuracy"],
        "num_labels": 2,
    },
}


DEFAULT_RESULT_TABLE_COLUMNS = [
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
]

QUICK_MODE_DEFAULT_TRAIN_SAMPLES = 200
QUICK_MODE_DEFAULT_EVAL_SAMPLES = 50
QUICK_MODE_DEFAULT_EPOCHS = 1.0
HIDDEN_STATE_FUSION_IMPLEMENTATION = ImprovementSpec(
    name="multi_layer_hidden_states_fusion",
    implementation_instructions=(
        "Implement DistilBERT sequence classification with output_hidden_states enabled. "
        "Keep baseline_last_cls on the default last-layer [CLS] representation. For improved variants, "
        "use a custom fusion classification head: fusion_last_layer uses the final hidden layer [CLS] "
        "through the same custom-head code path as a sanity check, fusion_mean_last_4 averages the [CLS] "
        "representations from the last 4 hidden layers, and fusion_learned_last_4 learns softmax-normalized "
        "weights over the last 4 hidden-layer [CLS] representations. Do not hard-code a 12-layer BERT shape; "
        "derive available hidden states from the model output."
    ),
)


def glue_benchmark(name: str) -> BenchmarkSpec:
    key = name.strip().lower().replace("-", "")
    if key == "sst2":
        key = "sst2"
    if key not in GLUE_TASKS:
        raise ValueError(f"Unsupported GLUE task for built-in metadata: {name}")
    return BenchmarkSpec.model_validate(GLUE_TASKS[key])


def default_result_table_columns() -> list[str]:
    return list(DEFAULT_RESULT_TABLE_COLUMNS)


def is_quick_mode_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in {"quick", "smoke", "dry_run", "debug", "preflight"}


def is_hidden_state_fusion_task(user_task: str | None, study: ExperimentStudyPlan) -> bool:
    haystack = " ".join(
        [
            user_task or "",
            study.implementation.name,
            study.implementation.implementation_instructions,
            " ".join(variant.name for variant in study.variants),
            " ".join(str(variant.strategy or "") for variant in study.variants),
            " ".join(variant.main_change for variant in study.variants),
        ]
    ).lower()
    signals = ("hidden", "hidden_states", "mean_last_k", "learned_weighted_sum", "last_layer")
    return ("fusion" in haystack and "hidden" in haystack) or any(signal in haystack for signal in signals[1:])


def hidden_state_fusion_variants() -> list[VariantSpec]:
    return [
        VariantSpec(
            name="baseline_last_cls",
            family="baseline",
            main_change="none",
            strategy="last_layer",
            implementation_notes="Default DistilBERT sequence classification head using final-layer [CLS].",
        ),
        VariantSpec(
            name="fusion_last_layer",
            family="improved",
            main_change="custom fusion head using final hidden-layer [CLS] only",
            strategy="last_layer",
            implementation_notes="Sanity-check custom-head variant that should track the baseline closely.",
        ),
        VariantSpec(
            name="fusion_mean_last_4",
            family="improved",
            main_change="custom fusion head averaging the last 4 hidden-layer [CLS] representations",
            strategy="mean_last_k",
            k=4,
        ),
        VariantSpec(
            name="fusion_learned_last_4",
            family="improved",
            main_change="custom fusion head with learned weights over the last 4 hidden-layer [CLS] representations",
            strategy="learned_weighted_sum",
            k=4,
        ),
    ]


def normalize_study_plan(study: ExperimentStudyPlan, *, user_task: str | None = None) -> ExperimentStudyPlan:
    modes: dict[str, StudyMode] = {}
    for name, mode in study.modes.items():
        data = mode.model_dump()
        if is_quick_mode_name(name):
            data["max_train_samples"] = data["max_train_samples"] or QUICK_MODE_DEFAULT_TRAIN_SAMPLES
            data["max_eval_samples"] = data["max_eval_samples"] or QUICK_MODE_DEFAULT_EVAL_SAMPLES
            if data["num_train_epochs"] > QUICK_MODE_DEFAULT_EPOCHS:
                data["num_train_epochs"] = QUICK_MODE_DEFAULT_EPOCHS
        modes[name] = StudyMode.model_validate(data)
    updates: dict[str, object] = {"modes": modes}
    if is_hidden_state_fusion_task(user_task, study):
        updates["variants"] = hidden_state_fusion_variants()
        updates["implementation"] = HIDDEN_STATE_FUSION_IMPLEMENTATION
    return study.model_copy(update=updates)


def variant_to_training(mode: StudyMode, variant: VariantSpec) -> TrainingVariant:
    return TrainingVariant(
        method=variant.family,
        main_change=variant.main_change,
        strategy=variant.strategy,
        k=variant.k,
        learning_rate=mode.learning_rate,
        train_batch_size=mode.train_batch_size,
        eval_batch_size=mode.eval_batch_size,
        num_train_epochs=mode.num_train_epochs,
        weight_decay=mode.weight_decay,
        warmup_ratio=mode.warmup_ratio,
        label_smoothing_factor=mode.label_smoothing_factor,
    )


def expand_study_plan(study: ExperimentStudyPlan, *, mode_name: str | None = None) -> list[ExecutionCell]:
    modes = {mode_name: study.modes[mode_name]} if mode_name is not None else study.modes
    cells: list[ExecutionCell] = []
    for current_mode_name, mode in modes.items():
        for benchmark in study.benchmarks:
            for seed in mode.seeds:
                for variant in study.variants:
                    cells.append(
                        ExecutionCell(
                            mode=current_mode_name,
                            benchmark=benchmark.name,
                            dataset_config=benchmark.dataset_config,
                            text_columns=benchmark.text_columns,
                            label_column=benchmark.label_column,
                            train_split=benchmark.train_split,
                            eval_split=benchmark.eval_split,
                            metrics=benchmark.metrics,
                            num_labels=benchmark.num_labels,
                            max_length=benchmark.max_length,
                            seed=seed,
                            variant=variant.name,
                            family=variant.family,
                            strategy=variant.strategy,
                            k=variant.k,
                            max_train_samples=mode.max_train_samples,
                            max_eval_samples=mode.max_eval_samples,
                            training=variant_to_training(mode, variant),
                        )
                    )
    return cells


def comparison_plan_for_variant(
    study: ExperimentStudyPlan,
    *,
    mode_name: str,
    benchmark_name: str,
    seed: int,
    variant_name: str,
) -> ComparisonPlan:
    mode = study.modes[mode_name]
    benchmark = next((item for item in study.benchmarks if item.name == benchmark_name), None)
    if benchmark is None:
        raise ValueError(f"Unknown benchmark in study plan: {benchmark_name}")
    baseline = next((item for item in study.variants if item.family == "baseline"), None)
    variant = next((item for item in study.variants if item.name == variant_name), None)
    if baseline is None:
        raise ValueError("Study plan does not contain a baseline variant.")
    if variant is None:
        raise ValueError(f"Unknown variant in study plan: {variant_name}")
    if variant.family != "improved":
        raise ValueError("ComparisonPlan conversion requires an improved variant.")

    return ComparisonPlan(
        model_id=study.model_id,
        dataset_id=study.dataset_id,
        dataset_config=benchmark.dataset_config,
        task_type=study.task_type,
        text_columns=benchmark.text_columns,
        label_column=benchmark.label_column,
        train_split=benchmark.train_split,
        eval_split=benchmark.eval_split,
        metric_name="accuracy",
        num_labels=benchmark.num_labels,
        max_length=benchmark.max_length,
        seed=seed,
        max_train_samples=mode.max_train_samples,
        max_eval_samples=mode.max_eval_samples,
        baseline=variant_to_training(mode, baseline),
        improved=variant_to_training(mode, variant),
        implementation=study.implementation,
        rationale=(
            f"Expanded from study plan mode={mode_name}, benchmark={benchmark.name}, "
            f"seed={seed}, variant={variant.name}."
        ),
    )
