import pytest

from code_agent.experiments.evaluator import summarize_comparison, write_comparison_report
from code_agent.experiments.models import ComparisonPlan


def _plan() -> ComparisonPlan:
    return ComparisonPlan.model_validate(
        {
            "model_id": "distilbert/distilbert-base-uncased",
            "dataset_id": "nyu-mll/glue",
            "dataset_config": "sst2",
            "task_type": "sequence_classification",
            "text_columns": ["sentence"],
            "seed": 42,
            "baseline": {"method": "baseline", "main_change": "none"},
            "improved": {"method": "improved", "main_change": "focal loss"},
            "implementation": {
                "name": "Focal loss",
                "implementation_instructions": "Override Trainer compute_loss with focal loss.",
            },
        }
    )


def test_comparison_summary_and_markdown_report(tmp_path):
    runs = {
        "baseline": {"eval_metrics": {"eval_accuracy": 0.90}},
        "improved": {
            "eval_metrics": {"eval_accuracy": 0.91},
            "implementation_summary": "Focal loss with gamma=2",
            "implementation_file": "results/improvement.py",
        },
    }

    summary = summarize_comparison(_plan(), runs)
    report = write_comparison_report(tmp_path / "comparison.md", _plan(), runs)

    assert summary["accuracy_delta"] == pytest.approx(0.01)
    assert summary["improved_wins"] is True
    assert "Focal loss with gamma=2" in report
    assert "results/improvement.py" in report
    assert "Implemented user-requested change" in report
    assert "| Method | Main Change | LR | Batch Size | Epochs | Seed | Validation Accuracy |" in report
    assert "| improved | focal loss" in report
