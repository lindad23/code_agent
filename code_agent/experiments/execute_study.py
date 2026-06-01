from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from code_agent.experiments.execute_plan import run_plan
from code_agent.experiments.models import ExperimentStudyPlan
from code_agent.experiments.study import comparison_plan_for_variant, expand_study_plan
from code_agent.tools.file_tools import ensure_dir, write_text


def _cell_id(mode: str, benchmark: str, seed: int, variant: str) -> str:
    return f"{mode}__{benchmark}__seed{seed}__{variant}"


def _result_row(cell_id: str, metrics: dict, cell=None) -> dict:
    if metrics.get("status") != "completed":
        return {
            "cell_id": cell_id,
            "dataset_config": getattr(cell, "dataset_config", None),
            "seed": getattr(cell, "seed", None),
            "variant": getattr(getattr(cell, "training", None), "main_change", None),
            "strategy": getattr(cell, "strategy", None),
            "k": getattr(cell, "k", None),
            "baseline_accuracy": None,
            "improved_accuracy": None,
            "accuracy_delta": None,
            "improved_wins": None,
            "baseline_train_runtime": None,
            "improved_train_runtime": None,
            "baseline_eval_runtime": None,
            "improved_eval_runtime": None,
            "preflight_parameter_count": None,
            "preflight_gradient_abs_sum": None,
            "retry_attempt": None,
            "status": metrics.get("status", "failed"),
            "error": metrics.get("error"),
        }
    comparison = metrics["comparison"]
    improved = metrics["runs"]["improved"]
    baseline = metrics["runs"]["baseline"]
    retry = metrics.get("retry") or {}
    return {
        "cell_id": cell_id,
        "dataset_config": metrics.get("dataset_config"),
        "seed": metrics.get("seed"),
        "variant": improved.get("main_change"),
        "strategy": improved.get("strategy"),
        "k": improved.get("k"),
        "baseline_accuracy": comparison.get("baseline_accuracy"),
        "improved_accuracy": comparison.get("improved_accuracy"),
        "accuracy_delta": comparison.get("accuracy_delta"),
        "improved_wins": comparison.get("improved_wins"),
        "baseline_train_runtime": baseline.get("train_metrics", {}).get("train_runtime"),
        "improved_train_runtime": improved.get("train_metrics", {}).get("train_runtime"),
        "baseline_eval_runtime": baseline.get("eval_metrics", {}).get("eval_runtime"),
        "improved_eval_runtime": improved.get("eval_metrics", {}).get("eval_runtime"),
        "preflight_parameter_count": improved.get("preflight", {}).get("watched_parameter_count"),
        "preflight_gradient_abs_sum": improved.get("preflight", {}).get("watched_gradient_abs_sum"),
        "retry_attempt": retry.get("attempt"),
        "status": metrics.get("status"),
        "error": None,
    }


def _cell_workspace(study_workspace: Path, cell_id: str) -> Path:
    # run_plan stores reusable assets under workspace_dir.parent / "asset_cache".
    # Keep each cell workspace as a direct child of the shared experiments root so
    # study execution reuses the same model/dataset cache as normal single runs.
    return study_workspace.parent / f"{study_workspace.name}__{cell_id}"


def _write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        return write_text(path, "")
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_study(
    study: ExperimentStudyPlan,
    *,
    mode_name: str,
    workspace_dir: Path,
    results_dir: Path,
    implementation_file: Path,
) -> dict:
    cells = [
        cell
        for cell in expand_study_plan(study, mode_name=mode_name)
        if cell.family == "improved"
    ]
    rows: list[dict] = []
    cell_results: list[dict] = []
    for index, cell in enumerate(cells, start=1):
        cell_id = _cell_id(cell.mode, cell.benchmark, cell.seed, cell.variant)
        print(f"[{index}/{len(cells)}] starting {cell_id}", flush=True)
        plan = comparison_plan_for_variant(
            study,
            mode_name=cell.mode,
            benchmark_name=cell.benchmark,
            seed=cell.seed,
            variant_name=cell.variant,
        )
        cell_workspace = ensure_dir(_cell_workspace(workspace_dir, cell_id))
        cell_results_dir = ensure_dir(results_dir / "cells" / cell_id)
        write_text(cell_results_dir / "plan.json", plan.model_dump_json(indent=2))
        try:
            metrics = run_plan(plan, cell_workspace, cell_results_dir, implementation_file)
        except Exception as exc:
            metrics = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_text(cell_results_dir / "failure.json", json.dumps(metrics, ensure_ascii=False, indent=2))
            print(f"[{index}/{len(cells)}] failed {cell_id}: {metrics['error']}", flush=True)
        row = _result_row(cell_id, metrics, cell)
        rows.append(row)
        cell_results.append({"cell": cell.model_dump(mode="json"), "metrics": metrics})
        write_text(results_dir / "study_results.json", json.dumps(cell_results, ensure_ascii=False, indent=2))
        _write_csv(results_dir / "study_results.csv", rows)
        if row["status"] == "completed":
            print(
                f"[{index}/{len(cells)}] finished {cell_id}: "
                f"delta={row['accuracy_delta']}",
                flush=True,
            )

    num_completed = sum(1 for row in rows if row["status"] == "completed")
    num_failed = len(rows) - num_completed
    summary = {
        "status": "completed" if num_failed == 0 else "completed_with_failures",
        "mode": mode_name,
        "num_comparisons": len(cells),
        "num_completed": num_completed,
        "num_failed": num_failed,
        "results_file": str(results_dir / "study_results.json"),
        "results_csv_file": str(results_dir / "study_results.csv"),
        "rows": rows,
    }
    write_text(results_dir / "study_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an expanded experiment study matrix.")
    parser.add_argument("--study-plan-file", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--implementation-file", required=True)
    args = parser.parse_args(argv)

    study = ExperimentStudyPlan.model_validate_json(Path(args.study_plan_file).read_text(encoding="utf-8"))
    summary = run_study(
        study,
        mode_name=args.mode,
        workspace_dir=Path(args.workspace_dir),
        results_dir=Path(args.results_dir),
        implementation_file=Path(args.implementation_file),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
