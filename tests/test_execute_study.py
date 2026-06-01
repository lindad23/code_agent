import csv
import json

from code_agent.experiments import execute_study
from code_agent.experiments.models import ExperimentStudyPlan


def _study_payload() -> dict:
    return {
        "model_id": "distilbert/distilbert-base-uncased",
        "dataset_id": "nyu-mll/glue",
        "task_type": "sequence_classification",
        "modes": {
            "quick": {
                "seeds": [13],
                "max_train_samples": 8,
                "max_eval_samples": 4,
                "num_train_epochs": 1,
                "train_batch_size": 4,
                "eval_batch_size": 4,
            }
        },
        "benchmarks": [
            {
                "name": "rte",
                "dataset_config": "rte",
                "text_columns": ["sentence1", "sentence2"],
                "label_column": "label",
                "train_split": "train",
                "eval_split": "validation",
                "metrics": ["accuracy"],
                "num_labels": 2,
                "max_length": 128,
            }
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
                "main_change": "custom head using final layer",
                "strategy": "last_layer",
            },
            {
                "name": "fusion_mean_last_4",
                "family": "improved",
                "main_change": "custom head averaging last layers",
                "strategy": "mean_last_k",
                "k": 4,
            },
        ],
        "implementation": {
            "name": "fusion_head",
            "implementation_instructions": "Install a custom fusion head.",
        },
    }


def test_run_study_records_failed_cell_and_continues(monkeypatch, tmp_path):
    study = ExperimentStudyPlan.model_validate(_study_payload())
    calls = []

    def fake_run_plan(plan, workspace_dir, results_dir, implementation_file):
        calls.append(plan.improved.strategy)
        if plan.improved.strategy == "last_layer":
            raise RuntimeError("CUDA out of memory")
        return {
            "status": "completed",
            "dataset_config": plan.dataset_config,
            "seed": plan.seed,
            "retry": {"attempt": 0},
            "runs": {
                "baseline": {
                    "train_metrics": {"train_runtime": 1.0},
                    "eval_metrics": {"eval_runtime": 0.1},
                },
                "improved": {
                    "main_change": plan.improved.main_change,
                    "strategy": plan.improved.strategy,
                    "k": plan.improved.k,
                    "train_metrics": {"train_runtime": 1.2},
                    "eval_metrics": {"eval_runtime": 0.2},
                    "preflight": {
                        "watched_parameter_count": 2,
                        "watched_gradient_abs_sum": 3.0,
                    },
                },
            },
            "comparison": {
                "baseline_accuracy": 0.5,
                "improved_accuracy": 0.6,
                "accuracy_delta": 0.1,
                "improved_wins": True,
            },
        }

    monkeypatch.setattr(execute_study, "run_plan", fake_run_plan)

    summary = execute_study.run_study(
        study,
        mode_name="quick",
        workspace_dir=tmp_path / "workspace",
        results_dir=tmp_path / "results",
        implementation_file=tmp_path / "improvement.py",
    )

    assert calls == ["last_layer", "mean_last_k"]
    assert summary["status"] == "completed_with_failures"
    assert summary["num_completed"] == 1
    assert summary["num_failed"] == 1

    rows = list(csv.DictReader((tmp_path / "results" / "study_results.csv").open(encoding="utf-8")))
    assert rows[0]["status"] == "failed"
    assert "CUDA out of memory" in rows[0]["error"]
    assert rows[1]["status"] == "completed"

    failures = json.loads((tmp_path / "results" / "cells" / "quick__rte__seed13__fusion_last_layer" / "failure.json").read_text())
    assert failures["status"] == "failed"
