from __future__ import annotations

from code_agent.experiments.models import ComparisonPlan, TrainingVariant
from code_agent.tools.file_tools import write_text


def _accuracy(metrics: dict) -> float:
    return float(metrics["eval_metrics"]["eval_accuracy"])


def summarize_comparison(plan: ComparisonPlan, runs: dict[str, dict]) -> dict:
    baseline_accuracy = _accuracy(runs["baseline"])
    improved_accuracy = _accuracy(runs["improved"])
    return {
        "metric_name": plan.metric_name,
        "baseline_accuracy": baseline_accuracy,
        "improved_accuracy": improved_accuracy,
        "accuracy_delta": improved_accuracy - baseline_accuracy,
        "improved_wins": improved_accuracy > baseline_accuracy,
    }


def _row(variant: TrainingVariant, metrics: dict, seed: int) -> str:
    accuracy = _accuracy(metrics)
    return (
        f"| {variant.method} | {variant.main_change} | {variant.learning_rate:g} | "
        f"{variant.train_batch_size} | {variant.num_train_epochs:g} | {seed} | {accuracy:.6f} |"
    )


def write_comparison_report(path, plan: ComparisonPlan, runs: dict[str, dict]) -> str:
    summary = summarize_comparison(plan, runs)
    implementation_summary = runs["improved"].get("implementation_summary", plan.implementation.name)
    implementation_file = runs["improved"].get("implementation_file", "improvement.py")
    report = f"""# Experiment Comparison

Fixed conditions: model `{plan.model_id}`, dataset `{plan.dataset_id}` / `{plan.dataset_config or ""}`, \
evaluation split `{plan.eval_split}`, metric `{plan.metric_name}`, seed `{plan.seed}`.

Implemented user-requested change: `{implementation_summary}`

Implementation file: `{implementation_file}`

| Method | Main Change | LR | Batch Size | Epochs | Seed | Validation Accuracy |
|---|---|---:|---:|---:|---:|---:|
{_row(plan.baseline, runs["baseline"], plan.seed)}
{_row(plan.improved, runs["improved"], plan.seed)}

Accuracy delta (`improved - baseline`): {summary["accuracy_delta"]:+.6f}
"""
    write_text(path, report)
    return report
