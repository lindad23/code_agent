from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from code_agent.experiments.models import ComparisonPlan, ExperimentRequest
from code_agent.tools.llm_tools import call_llm


def parse_huggingface_id(value: str, *, repo_type: str) -> str:
    cleaned = value.strip().rstrip("/")
    if not cleaned:
        raise ValueError("Hugging Face URL cannot be empty.")
    if "://" not in cleaned:
        return cleaned.removeprefix("datasets/") if repo_type == "dataset" else cleaned

    parsed = urlparse(cleaned)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"Only Hugging Face URLs are supported: {value}")
    parts = [part for part in parsed.path.split("/") if part]
    if repo_type == "dataset":
        if parts and parts[0] == "datasets":
            parts = parts[1:]
    elif parts and parts[0] == "datasets":
        raise ValueError("A dataset URL cannot be used as the baseline URL.")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse a repository id from URL: {value}")
    return "/".join(parts[:2])


def build_experiment_plan_prompt(request: ExperimentRequest, model_id: str, dataset_id: str) -> str:
    return f"""Prepare a controlled baseline-versus-user-requested-code-change Hugging Face text-classification experiment plan.

Baseline model URL:
{request.baseline_url}
Resolved model id:
{model_id}

Benchmark dataset URL:
{request.benchmark_url}
Resolved dataset id:
{dataset_id}

User task:
{request.task}

The executor supports only Hugging Face sequence classification fine-tuning. Infer the dataset configuration/subtask,
input text column names, label column, train/evaluation split names, and reasonable training hyperparameters.
For GLUE SST-2, use dataset_config "sst2", text_columns ["sentence"], label_column "label",
train_split "train", eval_split "validation", and accuracy.

The user task specifies the algorithm or code change to implement. Do not invent an improvement,
choose among alternatives, broaden the requested change, or optimize hyperparameters. Your job is
only to resolve the experiment setup and restate the user's requested code change precisely enough
for an implementation agent to implement it.

The model, dataset, splits, metric, max_length, seed, sample limits, and all listed training
hyperparameters are fixed for both methods. The requested change must be implemented in Python
code used only by improved.

Return only one JSON object with exactly these top-level keys:
model_id, dataset_id, dataset_config, task_type, text_columns, label_column, train_split, eval_split,
metric_name, num_labels, max_length, seed, use_cpu, max_train_samples, max_eval_samples,
baseline, improved, implementation, rationale.

baseline and improved must each contain exactly:
method, main_change, learning_rate, train_batch_size, eval_batch_size, num_train_epochs,
weight_decay, warmup_ratio, label_smoothing_factor, classifier_dropout, early_stopping_patience.

implementation must contain exactly:
name, implementation_instructions.

Requirements:
- model_id must be "{model_id}".
- dataset_id must be "{dataset_id}".
- task_type must be "sequence_classification".
- metric_name must be "accuracy".
- baseline.method must be "baseline"; improved.method must be "improved".
- baseline.main_change should be "none".
- improved.main_change should briefly name the code-level algorithm change explicitly requested by the user.
- All numeric and nullable training settings in baseline and improved must be identical.
- Set classifier_dropout and early_stopping_patience to null when unused.
- Set unused numeric ratio fields such as warmup_ratio and label_smoothing_factor to 0.
- implementation_instructions must translate only the code behavior explicitly requested by the
  user into implementation steps through a custom Transformers Trainer or model configuration hook.
- If the user does not specify a concrete algorithm/code change, do not invent one; return an
  implementation name and instructions stating that an explicit change is required.
- Do not include markdown fences or any shell commands.
"""


def extract_json_object(text: str) -> dict:
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The planning model did not return a JSON object.")
        candidate = candidate[start : end + 1]
    result = json.loads(candidate)
    if not isinstance(result, dict):
        raise ValueError("The experiment plan must be a JSON object.")
    return result


def prepare_plan_request(request: ExperimentRequest) -> tuple[str, str, str]:
    model_id = parse_huggingface_id(request.baseline_url, repo_type="model")
    dataset_id = parse_huggingface_id(request.benchmark_url, repo_type="dataset")
    return model_id, dataset_id, build_experiment_plan_prompt(request, model_id, dataset_id)


def request_experiment_plan(
    request: ExperimentRequest,
    *,
    model_id: str,
    dataset_id: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> tuple[ComparisonPlan, str]:
    response = call_llm(
        prompt,
        provider=request.api_provider,
        model=request.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=(
            "You prepare ML experiment configuration from a user-specified change. "
            "Do not design new algorithms. Return only strict JSON for the supported executor."
        ),
        timeout=request.plan_timeout_seconds,
    )
    plan = ComparisonPlan.model_validate(extract_json_object(response))
    if plan.model_id != model_id or plan.dataset_id != dataset_id:
        raise ValueError("The planned repositories do not match the user-provided URLs.")
    return plan, response


def create_experiment_plan(
    request: ExperimentRequest,
    *,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> tuple[ComparisonPlan, str, str]:
    model_id, dataset_id, prompt = prepare_plan_request(request)
    plan, response = request_experiment_plan(
        request,
        model_id=model_id,
        dataset_id=dataset_id,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return plan, prompt, response
