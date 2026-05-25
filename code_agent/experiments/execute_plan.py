from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from code_agent.experiments.models import ExperimentPlan
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


def run_plan(plan: ExperimentPlan, workspace_dir: Path, results_dir: Path) -> dict:
    try:
        import numpy as np
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError("Training dependencies are not installed in the experiment environment.") from exc

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

    set_seed(plan.seed)
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

    model = AutoModelForSequenceClassification.from_pretrained(str(model_repo), num_labels=num_labels)
    model_dir = ensure_dir(results_dir / "model")
    training_arguments = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=plan.learning_rate,
        per_device_train_batch_size=plan.train_batch_size,
        per_device_eval_batch_size=plan.eval_batch_size,
        num_train_epochs=plan.num_train_epochs,
        weight_decay=plan.weight_decay,
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

    def compute_metrics(prediction):
        logits, labels = prediction
        predicted = np.argmax(logits, axis=-1)
        return {"accuracy": float((predicted == labels).mean())}

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    train_metrics = trainer.train().metrics
    eval_metrics = trainer.evaluate()
    best_model = model_dir / "best"
    trainer.save_model(str(best_model))
    tokenizer.save_pretrained(str(best_model))

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
        "best_model": str(best_model),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    write_text(results_dir / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
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
    args = parser.parse_args(argv)

    plan = ExperimentPlan.model_validate_json(Path(args.plan_file).read_text(encoding="utf-8"))
    metrics = run_plan(plan, Path(args.workspace_dir), Path(args.results_dir))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
