from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
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


def _dataset_repository_patterns(dataset_config: str | None) -> list[str] | None:
    if not dataset_config:
        return None
    return [
        ".gitattributes",
        "README*",
        "dataset_infos.json",
        "*.py",
        f"{dataset_config}/**",
    ]


def _move_trainer_model_to_device(trainer) -> None:
    args = getattr(trainer, "args", None)
    device = getattr(args, "device", None)
    if device is None:
        return
    model = getattr(trainer, "model", None)
    if model is not None and callable(getattr(model, "to", None)):
        model.to(device)
    wrapped = getattr(trainer, "model_wrapped", None)
    if wrapped is not None and wrapped is not model and callable(getattr(wrapped, "to", None)):
        wrapped.to(device)


def _clear_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_cuda_out_of_memory_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        error_name = type(current).__name__.lower()
        if "outofmemoryerror" in error_name:
            return True
        if "cuda" in message and "out of memory" in message:
            return True
        if "cublas_status_alloc_failed" in message or "cuda error: out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _retry_variant_after_oom(variant: TrainingVariant) -> TrainingVariant | None:
    if variant.train_batch_size <= 1 and variant.eval_batch_size <= 1:
        return None
    train_batch_size = max(1, variant.train_batch_size // 2)
    eval_batch_size = max(1, variant.eval_batch_size // 2)
    accumulation_multiplier = max(1, math.ceil(variant.train_batch_size / train_batch_size))
    return variant.model_copy(
        update={
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": variant.gradient_accumulation_steps * accumulation_multiplier,
        }
    )


def _plan_with_oom_retry_batches(plan: ComparisonPlan) -> ComparisonPlan | None:
    baseline = _retry_variant_after_oom(plan.baseline)
    improved = _retry_variant_after_oom(plan.improved)
    if baseline is None or improved is None:
        return None
    return plan.model_copy(
        update={
            "baseline": baseline,
            "improved": improved,
            "rationale": (
                f"{plan.rationale}\nOOM retry: halved per-device batch sizes and increased "
                "gradient_accumulation_steps to preserve the approximate effective train batch size."
            ).strip(),
        }
    )


def _target_model(model):
    return model.module if hasattr(model, "module") else model


def _watched_custom_head_parameters(model) -> list[tuple[str, object]]:
    target = _target_model(model)
    named_parameters = getattr(target, "named_parameters", None)
    if not callable(named_parameters):
        return []
    markers = ("custom_head", "fusion")
    return [
        (name, parameter)
        for name, parameter in named_parameters()
        if any(marker in name.lower() for marker in markers)
    ]


def _gradient_abs_sum(parameter) -> float:
    grad = getattr(parameter, "grad", None)
    if grad is None:
        return 0.0
    if hasattr(grad, "detach"):
        grad = grad.detach()
    if hasattr(grad, "abs"):
        grad = grad.abs()
    else:
        grad = abs(grad)
    if hasattr(grad, "sum"):
        grad = grad.sum()
    if hasattr(grad, "item"):
        grad = grad.item()
    return float(grad)


def _summarize_custom_head_gradients(model) -> dict:
    watched = _watched_custom_head_parameters(model)
    return {
        "watched_parameter_count": len(watched),
        "watched_parameter_names": [name for name, _ in watched],
        "watched_gradient_abs_sum": sum(_gradient_abs_sum(parameter) for _, parameter in watched),
    }


def _assert_custom_head_gradients_flow(summary: dict) -> None:
    if summary["watched_parameter_count"] > 0 and summary["watched_gradient_abs_sum"] <= 0:
        raise RuntimeError(
            "Improved preflight failed: custom fusion/head parameters exist but received no gradients. "
            "The generated implementation may be training on the default model logits instead of fused logits."
        )


def _run_improved_preflight(trainer) -> dict:
    model = trainer.model
    was_training = bool(getattr(model, "training", False))
    model.train()
    if callable(getattr(model, "zero_grad", None)):
        model.zero_grad(set_to_none=True)
    batch = next(iter(trainer.get_train_dataloader()))
    prepare_inputs = getattr(trainer, "_prepare_inputs", None)
    if callable(prepare_inputs):
        batch = prepare_inputs(batch)
    loss = trainer.compute_loss(model, batch)
    if isinstance(loss, tuple):
        loss = loss[0]
    loss.backward()
    summary = _summarize_custom_head_gradients(model)
    _assert_custom_head_gradients_flow(summary)
    if callable(getattr(model, "zero_grad", None)):
        model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return summary


def _select_metric_logits(logits):
    if isinstance(logits, dict):
        if "logits" not in logits:
            raise ValueError("Metric computation received a prediction dictionary without a logits key.")
        return logits["logits"]
    if isinstance(logits, tuple):
        if not logits:
            raise ValueError("Metric computation received an empty prediction tuple.")
        logits = logits[0]
    return logits


def _infer_fusion_strategy(variant: TrainingVariant) -> tuple[str | None, int | None]:
    if variant.strategy:
        return variant.strategy, variant.k
    text = variant.main_change.lower()
    if "learned" in text or "weighted" in text:
        return "learned_weighted_sum", variant.k or 4
    if "mean" in text or "averag" in text or "last 4" in text:
        return "mean_last_k", variant.k or 4
    if "hidden" in text and "fusion" in text:
        return "mean_last_k", variant.k or 4
    if "last_layer" in text or "last layer" in text or "final hidden" in text:
        return "last_layer", variant.k
    return None, variant.k


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
    dataset_repo = downloads / "benchmark_repository"
    shared_dataset_repo: Path | None = None
    dataset_repository_warning: str | None = None
    try:
        dataset_repo, shared_dataset_repo = stage_huggingface_repository(
            plan.dataset_id,
            dataset_repo,
            data_repo_cache,
            repo_type="dataset",
            allow_patterns=_dataset_repository_patterns(plan.dataset_config),
        )
    except Exception as exc:
        dataset_repository_warning = (
            "Dataset repository cache staging failed; continuing with datasets.load_dataset cache only.\n"
            f"{type(exc).__name__}: {exc}"
        )
        write_text(results_dir / "dataset_repository_warning.txt", dataset_repository_warning)
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
        logits = _select_metric_logits(logits)
        logits_array = np.asarray(logits)
        labels_array = np.asarray(labels)
        if logits_array.ndim < 2:
            raise ValueError(
                f"Metric computation expected logits shaped [num_examples, num_labels], got {logits_array.shape}."
            )
        if logits_array.shape[0] != labels_array.shape[0]:
            raise ValueError(
                "Metric computation received mismatched prediction/label counts: "
                f"logits shape {logits_array.shape}, labels shape {labels_array.shape}. "
                "Generated Trainer outputs should expose logits only, without hidden_states or attentions."
            )
        predicted = np.argmax(logits_array, axis=-1)
        return {"accuracy": float((predicted == labels_array).mean())}

    def train_variant(variant: TrainingVariant) -> dict:
        print(f"Starting {variant.method}: {variant.main_change}", flush=True)
        trainer = None
        model = None
        try:
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
                overwrite_output_dir=True,
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=variant.learning_rate,
                per_device_train_batch_size=variant.train_batch_size,
                per_device_eval_batch_size=variant.eval_batch_size,
                gradient_accumulation_steps=variant.gradient_accumulation_steps,
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
            fusion_strategy, fusion_k = _infer_fusion_strategy(variant)
            if variant.method == "improved" and fusion_strategy is not None:
                setattr(training_arguments, "fusion_strategy", fusion_strategy)
                setattr(training_arguments, "fusion_k", fusion_k or 4)
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
            _move_trainer_model_to_device(trainer)
            preflight = _run_improved_preflight(trainer) if variant.method == "improved" else None
            train_metrics = trainer.train().metrics
            eval_metrics = trainer.evaluate()
            best_model = model_dir / "best"
            trainer.save_model(str(best_model))
            tokenizer.save_pretrained(str(best_model))
            metrics = {
                "method": variant.method,
                "main_change": variant.main_change,
                "config": variant.model_dump(mode="json"),
                "strategy": fusion_strategy,
                "k": fusion_k,
                "implementation_file": str(implementation_file) if variant.method == "improved" else None,
                "implementation_summary": (
                    str(improvement_module.CHANGE_SUMMARY) if variant.method == "improved" else None
                ),
                "preflight": preflight,
                "best_model": str(best_model),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            }
            write_text(variant_dir / "metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
            print(f"Finished {variant.method}: validation accuracy={eval_metrics['eval_accuracy']:.6f}", flush=True)
            return metrics
        finally:
            if trainer is not None and callable(getattr(getattr(trainer, "accelerator", None), "free_memory", None)):
                trainer.accelerator.free_memory()
            del trainer
            del model
            _clear_accelerator_memory()

    active_plan = plan
    retry_info = {"attempt": 0, "oom_retry": False}
    while True:
        try:
            runs = {
                "baseline": train_variant(active_plan.baseline),
                "improved": train_variant(active_plan.improved),
            }
            break
        except Exception as exc:
            _clear_accelerator_memory()
            retry_plan = _plan_with_oom_retry_batches(active_plan)
            if retry_info["attempt"] >= 1 or not _is_cuda_out_of_memory_error(exc) or retry_plan is None:
                raise
            retry_info = {
                "attempt": 1,
                "oom_retry": True,
                "original_train_batch_size": active_plan.baseline.train_batch_size,
                "retry_train_batch_size": retry_plan.baseline.train_batch_size,
                "original_eval_batch_size": active_plan.baseline.eval_batch_size,
                "retry_eval_batch_size": retry_plan.baseline.eval_batch_size,
                "retry_gradient_accumulation_steps": retry_plan.baseline.gradient_accumulation_steps,
                "trigger_error": f"{type(exc).__name__}: {exc}",
            }
            write_text(results_dir / "oom_retry.json", json.dumps(retry_info, ensure_ascii=False, indent=2))
            print(
                "CUDA OOM detected; retrying this comparison with "
                f"train_batch_size={retry_plan.baseline.train_batch_size}, "
                f"eval_batch_size={retry_plan.baseline.eval_batch_size}, "
                f"gradient_accumulation_steps={retry_plan.baseline.gradient_accumulation_steps}",
                flush=True,
            )
            active_plan = retry_plan

    metrics = {
        "status": "completed",
        "model_id": plan.model_id,
        "dataset_id": plan.dataset_id,
        "dataset_config": plan.dataset_config,
        "model_repository": str(model_repo),
        "dataset_repository": str(dataset_repo),
        "shared_model_repository": str(shared_model_repo),
        "shared_dataset_repository": str(shared_dataset_repo) if shared_dataset_repo is not None else None,
        "shared_dataset_cache": str(dataset_cache),
        "dataset_repository_warning": dataset_repository_warning,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "seed": plan.seed,
        "retry": retry_info,
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
