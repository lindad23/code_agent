from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator


class ExperimentRequest(BaseModel):
    baseline_url: str
    benchmark_url: str
    task: str = Field(min_length=3)
    api_provider: Literal["deepseek", "openai"]
    llm_model: str | None = None
    baseline_url_explicit: bool = True
    benchmark_url_explicit: bool = True
    baseline_resources: dict[str, str] = Field(default_factory=dict)
    benchmark_resources: dict[str, str] = Field(default_factory=dict)
    resource_context: str | None = None
    workspace_root: Path = Path("./workspaces/experiments")
    results_root: Path = Path("./results/experiments")
    run_name: str | None = None
    run_name_is_prefix: bool = False
    environment_python: str = "3.11"
    timeout_seconds: int = Field(default=86400, gt=0)
    plan_timeout_seconds: int = Field(default=60, gt=0, le=600)
    reuse_environment: bool = False
    hardware_profile_file: Path = Path("./configs/hardware_profile.local.yaml")
    refresh_hardware_profile: bool = False


class TrainingVariant(BaseModel):
    method: Literal["baseline", "improved"]
    main_change: str
    strategy: str | None = None
    k: int | None = Field(default=None, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0, le=1)
    train_batch_size: int = Field(default=16, gt=0, le=1024)
    eval_batch_size: int = Field(default=32, gt=0, le=1024)
    gradient_accumulation_steps: int = Field(default=1, gt=0, le=1024)
    num_train_epochs: float = Field(default=3.0, gt=0, le=100)
    weight_decay: float = Field(default=0.01, ge=0, le=10)
    warmup_ratio: float = Field(default=0.0, ge=0, le=1)
    label_smoothing_factor: float = Field(default=0.0, ge=0, lt=1)
    classifier_dropout: float | None = Field(default=None, ge=0, lt=1)
    early_stopping_patience: int | None = Field(default=None, ge=1, le=20)

    @field_validator("early_stopping_patience", mode="before")
    @classmethod
    def normalize_disabled_early_stopping(cls, value):
        if value == 0:
            return None
        return value


class ImprovementSpec(BaseModel):
    name: str = Field(min_length=3)
    implementation_instructions: str = Field(min_length=10)


class ComparisonPlan(BaseModel):
    model_id: str
    dataset_id: str
    dataset_config: str | None = None
    task_type: Literal["sequence_classification"]
    text_columns: list[str] = Field(min_length=1, max_length=2)
    label_column: str = "label"
    train_split: str = "train"
    eval_split: str = "validation"
    metric_name: Literal["accuracy"] = "accuracy"
    num_labels: int | None = Field(default=None, gt=1)
    max_length: int = Field(default=128, gt=0, le=4096)
    seed: int = 42
    use_cpu: bool = False
    max_train_samples: int | None = Field(default=None, gt=0)
    max_eval_samples: int | None = Field(default=None, gt=0)
    baseline: TrainingVariant
    improved: TrainingVariant
    implementation: ImprovementSpec
    rationale: str = ""

    @model_validator(mode="after")
    def validate_comparison(self) -> "ComparisonPlan":
        if len(set(self.text_columns)) != len(self.text_columns):
            raise ValueError("text_columns cannot contain duplicates.")
        if self.baseline.method != "baseline" or self.improved.method != "improved":
            raise ValueError("Plans must provide baseline and improved variants.")
        changed_fields = [
            field
            for field in (
                "learning_rate",
                "train_batch_size",
                "eval_batch_size",
                "gradient_accumulation_steps",
                "num_train_epochs",
                "weight_decay",
                "warmup_ratio",
                "label_smoothing_factor",
                "classifier_dropout",
                "early_stopping_patience",
            )
            if getattr(self.baseline, field) != getattr(self.improved, field)
        ]
        if changed_fields:
            raise ValueError(
                "Generated-code comparisons must keep planned training settings identical; "
                "put the improvement in implementation code."
            )
        if self.baseline.main_change.strip().lower() != "none":
            raise ValueError("The baseline main_change must be none.")
        if self.improved.main_change.strip().lower() == "none":
            raise ValueError("The improved method must name its generated-code change.")
        return self


class ExperimentRunState(BaseModel):
    status: str
    run_id: str
    request_file: str
    plan_file: str | None = None
    plan_prompt_file: str | None = None
    plan_response_file: str | None = None
    plan_error_file: str | None = None
    study_plan_file: str | None = None
    expanded_cells_file: str | None = None
    implementation_prompt_file: str | None = None
    implementation_response_file: str | None = None
    implementation_file: str | None = None
    implementation_workspace_file: str | None = None
    implementation_error_file: str | None = None
    environment_file: str | None = None
    environment_prefix: str | None = None
    environment_cache_file: str | None = None
    hardware_file: str | None = None
    torch_runtime_file: str | None = None
    stdout_file: str | None = None
    stderr_file: str | None = None
    metrics_file: str | None = None
    report_file: str | None = None
    error: str | None = None


def default_run_id() -> str:
    return "experiment-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def unique_run_id(prefix: str | None = None) -> str:
    generated = f"{default_run_id()}-{uuid.uuid4().hex[:8]}"
    if prefix is None or not prefix.strip():
        return generated
    cleaned = re.sub(r"[\s/\\:]+", "_", prefix.strip()).strip("._-")
    return f"{cleaned}-{generated}" if cleaned else generated


class StudyMode(BaseModel):
    seeds: list[int] = Field(min_length=1)
    max_train_samples: int | None = Field(default=None, gt=0)
    max_eval_samples: int | None = Field(default=None, gt=0)
    num_train_epochs: float = Field(default=3.0, gt=0, le=100)
    train_batch_size: int = Field(default=32, gt=0, le=1024)
    eval_batch_size: int = Field(default=64, gt=0, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0, le=10)
    warmup_ratio: float = Field(default=0.0, ge=0, le=1)
    label_smoothing_factor: float = Field(default=0.0, ge=0, lt=1)

    @field_validator("seeds")
    @classmethod
    def validate_unique_seeds(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("Study mode seeds cannot contain duplicates.")
        return value


class BenchmarkSpec(BaseModel):
    name: str = Field(min_length=1)
    dataset_config: str = Field(min_length=1)
    text_columns: list[str] = Field(min_length=1, max_length=2)
    label_column: str = "label"
    train_split: str = "train"
    eval_split: str = "validation"
    metrics: list[Literal["accuracy", "f1"]] = Field(default_factory=lambda: ["accuracy"], min_length=1)
    num_labels: int | None = Field(default=None, gt=1)
    max_length: int = Field(default=128, gt=0, le=4096)

    @field_validator("text_columns")
    @classmethod
    def validate_unique_text_columns(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Benchmark text_columns cannot contain duplicates.")
        return value

    @field_validator("metrics")
    @classmethod
    def validate_unique_metrics(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Benchmark metrics cannot contain duplicates.")
        return value


class VariantSpec(BaseModel):
    name: str = Field(min_length=1)
    family: Literal["baseline", "improved"]
    main_change: str
    strategy: str | None = None
    k: int | None = Field(default=None, gt=0)
    implementation_notes: str | None = None

    @field_validator("family", mode="before")
    @classmethod
    def normalize_family(cls, value):
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        if normalized in {"ablation", "variant", "fusion", "treatment", "experiment"}:
            return "improved"
        if normalized in {"control", "baseline"}:
            return "baseline"
        return normalized

    @model_validator(mode="after")
    def validate_variant(self) -> "VariantSpec":
        if self.family == "baseline" and self.main_change.strip().lower() != "none":
            raise ValueError("Baseline study variants must use main_change='none'.")
        if self.family == "improved" and self.main_change.strip().lower() == "none":
            raise ValueError("Improved study variants must name their code change.")
        return self


class ResourceLoggingSpec(BaseModel):
    record_wall_time: bool = True
    record_train_runtime: bool = True
    record_eval_runtime: bool = True
    record_samples_per_second: bool = True
    record_max_gpu_memory: bool = True
    record_gpu_name: bool = True


class LaunchSpec(BaseModel):
    strategy: Literal["serial", "gpu_worker_pool", "torchrun", "accelerate"] = "serial"
    preferred_parallelism: str | None = None
    max_concurrent_runs: int | None = Field(default=None, gt=0)


class FailurePolicySpec(BaseModel):
    preflight_tiny_run: bool = True
    retry_once: bool = True
    on_oom: Literal["fail", "halve_batch_size", "halve_batch_size_and_use_gradient_accumulation"] = (
        "halve_batch_size_and_use_gradient_accumulation"
    )
    write_failure_signature: bool = True


class ExperimentStudyPlan(BaseModel):
    model_id: str
    dataset_id: str
    task_type: Literal["sequence_classification"]
    modes: dict[str, StudyMode] = Field(min_length=1)
    benchmarks: list[BenchmarkSpec] = Field(min_length=1)
    variants: list[VariantSpec] = Field(min_length=2)
    implementation: ImprovementSpec
    resource_logging: ResourceLoggingSpec = Field(default_factory=ResourceLoggingSpec)
    launch: LaunchSpec = Field(default_factory=LaunchSpec)
    failure_policy: FailurePolicySpec = Field(default_factory=FailurePolicySpec)
    result_table_columns: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_study(self) -> "ExperimentStudyPlan":
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("Study mode names cannot contain duplicates.")
        benchmark_names = [benchmark.name for benchmark in self.benchmarks]
        if len(set(benchmark_names)) != len(benchmark_names):
            raise ValueError("Study benchmark names cannot contain duplicates.")
        variant_names = [variant.name for variant in self.variants]
        if len(set(variant_names)) != len(variant_names):
            raise ValueError("Study variant names cannot contain duplicates.")
        if not any(variant.family == "baseline" for variant in self.variants):
            raise ValueError("Study plans must include at least one baseline variant.")
        if not any(variant.family == "improved" for variant in self.variants):
            raise ValueError("Study plans must include at least one improved variant.")
        return self


class ExecutionCell(BaseModel):
    mode: str
    benchmark: str
    dataset_config: str
    text_columns: list[str] = Field(min_length=1, max_length=2)
    label_column: str = "label"
    train_split: str = "train"
    eval_split: str = "validation"
    metrics: list[Literal["accuracy", "f1"]] = Field(default_factory=lambda: ["accuracy"], min_length=1)
    num_labels: int | None = Field(default=None, gt=1)
    max_length: int = Field(default=128, gt=0, le=4096)
    seed: int
    variant: str
    family: Literal["baseline", "improved"]
    strategy: str | None = None
    k: int | None = Field(default=None, gt=0)
    max_train_samples: int | None = Field(default=None, gt=0)
    max_eval_samples: int | None = Field(default=None, gt=0)
    training: TrainingVariant
