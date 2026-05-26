from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ExperimentRequest(BaseModel):
    baseline_url: str
    benchmark_url: str
    task: str = Field(min_length=3)
    api_provider: Literal["deepseek", "openai"]
    llm_model: str | None = None
    workspace_root: Path = Path("./workspaces/experiments")
    results_root: Path = Path("./results/experiments")
    run_name: str | None = None
    environment_python: str = "3.11"
    timeout_seconds: int = Field(default=86400, gt=0)
    plan_timeout_seconds: int = Field(default=60, gt=0, le=600)
    reuse_environment: bool = False
    hardware_profile_file: Path = Path("./configs/hardware_profile.local.yaml")
    refresh_hardware_profile: bool = False


class TrainingVariant(BaseModel):
    method: Literal["baseline", "improved"]
    main_change: str
    learning_rate: float = Field(default=2e-5, gt=0, le=1)
    train_batch_size: int = Field(default=16, gt=0, le=1024)
    eval_batch_size: int = Field(default=32, gt=0, le=1024)
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
    implementation_prompt_file: str | None = None
    implementation_response_file: str | None = None
    implementation_file: str | None = None
    implementation_workspace_file: str | None = None
    implementation_error_file: str | None = None
    environment_file: str | None = None
    environment_prefix: str | None = None
    hardware_file: str | None = None
    torch_runtime_file: str | None = None
    stdout_file: str | None = None
    stderr_file: str | None = None
    metrics_file: str | None = None
    report_file: str | None = None
    error: str | None = None


def default_run_id() -> str:
    return "experiment-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
