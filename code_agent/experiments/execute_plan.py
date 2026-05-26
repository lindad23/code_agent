from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from types import ModuleType
from pathlib import Path

from code_agent.experiments.evaluator import summarize_comparison, write_comparison_report
from code_agent.experiments.models import ComparisonPlan, TrainingVariant
from code_agent.tools.dataset_tools import download_huggingface_dataset, stage_huggingface_repository
from code_agent.tools.file_tools import ensure_dir, write_text


def _asset_cache_dirs(workspace_dir: Path) -> tuple[Path, Path, Path]:
    assets = ensure_dir(workspace_dir.parent / "asset_cache")
    dataset_cache = ensure_dir(assets / "data" / "datasets")
    legacy_dataset_cache = workspace_dir.parent / ".dataset_cache"
    if legacy_dataset_cache.exists() and not any(dataset_cache.iterdir()):
        shutil.copytree(legacy_dataset_cache, dataset_cache, dirs_exist_ok=True)
    return (
        ensure_dir(assets / "code"),
        ensure_dir(assets / "data" / "repositories"),
        dataset_cache,
    )


def _load_improvement_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Generated improvement module does not exist: {path}")
    spec = importlib.util.spec_from_file_location("generated_improvement", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated improvement module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("CHANGE_SUMMARY", "configure_model_config", "build_trainer_class"):
        if not hasattr(module, name):
            raise ValueError(f"Generated improvement module does not provide {name}.")
    return module


def run_plan(plan: ComparisonPlan, workspace_dir: Path, results_dir: Path, implementation_file: Path) -> dict:
    try:
        import numpy as np
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Training dependencies are not installed in the experiment environment.") from exc

    improvement_module = _load_improvement_module(implementation_file)
    downloads = ensure_dir(workspace_dir / "downloads")
    code_cache, data_repo_cache, dataset_cache = _asset_cache_dirs(workspace_dir)
    model_repo, shared_model_repo = stage_huggingface_repository(
        plan.model_id,
        downloads / "baseline_repository",
        code_cache,
        repo_type="model",
    )
    dataset_repo, shared_dataset_repo = stage_huggingface_repository(
        plan.dataset_id,
        downloads / "benchmark_repository",
        data_repo_cache,
        repo_type="dataset",
    )
    dataset = download_huggingface_dataset(
        plan.dataset_id,
        plan.dataset_config or "",
        dataset_cache,
    )
    if plan.train_split not in dataset or plan.eval_split not in dataset:
        raise ValueError(f"Dataset does not provide planned splits: {plan.train_split}, {plan.eval_split}")

    train_dataset = dataset[plan.train_split]
    eval_dataset = dataset[plan.eval_split]
    for column in [*plan.text_columns, plan.label_column]:
        if column not in train_dataset.column_names:
            raise ValueError(f"Planned dataset column does not exist: {column}")

    if plan.max_train_samples:
        train_dataset = train_dataset.select(range(min(plan.max_train_samples, len(train_dataset))))
    if plan.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(plan.max_eval_samples, len(eval_dataset))))

    label_feature = train_dataset.features.get(plan.label_column)
    num_labels = plan.num_labels
    if num_labels is None and label_feature is not None and getattr(label_feature, "num_classes", None):
        num_labels = label_feature.num_classes
    if num_labels is None:
        num_labels = len(set(train_dataset[plan.label_column]))

    tokenizer = AutoTokenizer.from_pretrained(str(model_repo))

    def tokenize(batch):
        texts = [batch[column] for column in plan.text_columns]
        return tokenizer(*texts, truncation=True, max_length=plan.max_length)

    drop_columns = [column for column in train_dataset.column_names if column != plan.label_column]
    tokenized_train = train_dataset.map(tokenize, batched=True, remove_columns=drop_columns)
    tokenized_eval = eval_dataset.map(tokenize, batched=True, remove_columns=drop_columns)
    if plan.label_column != "labels":
        tokenized_train = tokenized_train.rename_column(plan.label_column, "labels")
        tokenized_eval = tokenized_eval.rename_column(plan.label_column, "labels")

    def compute_metrics(prediction):
        logits, labels = prediction
        predicted = np.argmax(logits, axis=-1)
        return {"accuracy": float((predicted == labels).mean())}

    def train_variant(variant: TrainingVariant) -> dict:
        print(f"Starting {variant.method}: {variant.main_change}", flush=True)
        set_seed(plan.seed)
        model_config = AutoConfig.from_pretrained(str(model_repo), num_labels=num_labels)
        if variant.classifier_dropout is not None:
            model_config.classifier_dropout = variant.classifier_dropout
            model_config.seq_classif_dropout = variant.classifier_dropout
            model_config.hidden_dropout_prob = variant.classifier_dropout
        trainer_class = Trainer
        if variant.method == "improved":
            model_config = improvement_module.configure_model_config(model_config)
            trainer_class = improvement_module.build_trainer_class(Trainer)
            if not isinstance(trainer_class, type) or not issubclass(trainer_class, Trainer):
                raise TypeError("Generated build_trainer_class must return a transformers.Trainer subclass.")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_repo), config=model_config)
        variant_dir = ensure_dir(results_dir / variant.method)
        model_dir = ensure_dir(variant_dir / "model")
        training_arguments = TrainingArguments(
            output_dir=str(model_dir / "checkpoints"),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=variant.learning_rate,
            per_device_train_batch_size=variant.train_batch_size,
            per_device_eval_batch_size=variant.eval_batch_size,
            num_train_epochs=variant.num_train_epochs,
            weight_decay=variant.weight_decay,
            warmup_ratio=variant.warmup_ratio,
            label_smoothing_factor=variant.label_smoothing_factor,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=50,
            logging_first_step=True,
            report_to=[],
            seed=plan.seed,
            use_cpu=plan.use_cpu,
            fp16=bool(torch.cuda.is_available() and not plan.use_cpu),
        )
        callbacks = []
        if variant.early_stopping_patience is not None:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=variant.early_stopping_patience))
        trainer = trainer_class(
            model=model,
            args=training_arguments,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        train_metrics = trainer.train().metrics
        eval_metrics = trainer.evaluate()
        best_model = model_dir / "best"
        trainer.save_model(str(best_model))
        tokenizer.save_pretrained(str(best_model))
        metrics = {
            "method": variant.method,
            "main_change": variant.main_change,
            "config": variant.model_dump(mode="json"),
            "implementation_file": str(implementation_file) if variant.method == "improved" else None,
            "implementation_summary": (
                str(improvement_module.CHANGE_SUMMARY) if variant.method == "improved" else None
            ),
            "best_model": str(best_model),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        }
        write_text(variant_dir / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Finished {variant.method}: validation accuracy={eval_metrics['eval_accuracy']:.6f}", flush=True)
        return metrics

    runs = {
        "baseline": train_variant(plan.baseline),
        "improved": train_variant(plan.improved),
    }
    metrics = {
        "status": "completed",
        "model_id": plan.model_id,
        "dataset_id": plan.dataset_id,
        "dataset_config": plan.dataset_config,
        "model_repository": str(model_repo),
        "dataset_repository": str(dataset_repo),
        "shared_model_repository": str(shared_model_repo),
        "shared_dataset_repository": str(shared_dataset_repo),
        "shared_dataset_cache": str(dataset_cache),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "seed": plan.seed,
        "implementation_file": str(implementation_file),
        "implementation_summary": str(improvement_module.CHANGE_SUMMARY),
        "runs": runs,
        "comparison": summarize_comparison(plan, runs),
    }
    write_text(results_dir / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
    write_comparison_report(results_dir / "comparison.md", plan, runs)
    write_text(
        results_dir / "dataset_summary.json",
        json.dumps({"splits": {name: len(value) for name, value in dataset.items()}}, ensure_ascii=False, indent=2),
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute a validated Hugging Face experiment plan.")
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--implementation-file", required=True)
    args = parser.parse_args(argv)

    plan = ComparisonPlan.model_validate_json(Path(args.plan_file).read_text(encoding="utf-8"))
    metrics = run_plan(plan, Path(args.workspace_dir), Path(args.results_dir), Path(args.implementation_file))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
