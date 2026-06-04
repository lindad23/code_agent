from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_agent.experiments.models import ExperimentRequest


@dataclass(frozen=True)
class InputRunConfig:
    request: ExperimentRequest
    use_study: bool
    study_mode: str
    backend: str


def _string_value(data: dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "需要", "是"}:
            return True
        if normalized in {"false", "0", "no", "n", "不需要", "否"}:
            return False
    return bool(value)


def _resource_mapping(data: dict[str, Any], key: str) -> dict[str, str]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object whose keys are resource names and values are URLs or resource ids.")
    resources: dict[str, str] = {}
    for raw_name, raw_location in value.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{key} contains an empty resource name; resource keys cannot be empty.")
        location = "" if raw_location is None else str(raw_location).strip()
        resources[name] = location
    if not resources:
        raise ValueError(f"{key} must contain at least one resource.")
    return resources


def _primary_resource(resources: dict[str, str]) -> tuple[str, bool]:
    name, location = next(iter(resources.items()))
    if location:
        return location, True
    return name, False


def _is_huggingface_resource(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    if normalized.startswith(("https://huggingface.co/", "http://huggingface.co/")):
        return True
    if normalized.startswith(("https://www.huggingface.co/", "http://www.huggingface.co/")):
        return True
    return "://" not in normalized and "/" in normalized


def _select_backend(baselines: dict[str, str], benchmarks: dict[str, str]) -> str:
    explicit_resources = [value for value in [*baselines.values(), *benchmarks.values()] if value.strip()]
    if explicit_resources and not all(_is_huggingface_resource(value) for value in explicit_resources):
        return "generic_ai_experiment"
    return "hf_sequence_classification"


def _format_resources(label: str, resources: dict[str, str]) -> str:
    lines = [f"{label}:"]
    for name, location in resources.items():
        if location:
            lines.append(f"- {name}: {location} (user-specified; use this exact resource)")
        else:
            lines.append(f"- {name}: <AI should resolve a feasible Hugging Face resource>")
    return "\n".join(lines)


def _format_evaluation_indexes(value: Any) -> str:
    if value is None:
        return "Evaluation indexes are unspecified; choose suitable metrics and resource/runtime measurements."
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            return "Evaluation indexes are unspecified; choose suitable metrics and resource/runtime measurements."
        return "Required evaluation indexes: " + ", ".join(items)
    text = str(value).strip()
    if not text:
        return "Evaluation indexes are unspecified; choose suitable metrics and resource/runtime measurements."
    return f"Required evaluation indexes: {text}"


def load_input_run_config(path: str | Path) -> InputRunConfig:
    input_file = Path(path).expanduser()
    data = json.loads(input_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Input file must contain one JSON object.")

    experience_name = _string_value(data, "Experience name")
    improved_idea = _string_value(data, "Improved idea")
    if not improved_idea:
        raise ValueError("Improved idea cannot be empty.")

    baselines = _resource_mapping(data, "Baselines url")
    benchmarks = _resource_mapping(data, "Benchmarks url")
    backend = _select_backend(baselines, benchmarks)
    baseline_url, baseline_explicit = _primary_resource(baselines)
    benchmark_url, benchmark_explicit = _primary_resource(benchmarks)
    ablation = _bool_value(data.get("Ablation"), default=False)
    study_mode = _string_value(data, "Study mode", default="full") or "full"
    api_provider = _string_value(data, "API", default="deepseek") or "deepseek"
    llm_model = _string_value(data, "Model") or None

    evaluation_note = _format_evaluation_indexes(data.get("Evaluation indexs"))
    ablation_note = (
        "Ablation planning is required; include ablation variants when designing the experiment matrix."
        if ablation
        else "Ablation planning is not required; focus on the main baseline-vs-improved comparison."
    )
    resource_context = "\n".join(
        [
            _format_resources("Baseline resources", baselines),
            _format_resources("Benchmark resources", benchmarks),
            evaluation_note,
            ablation_note,
        ]
    )
    task = "\n\n".join(
        [
            "Improved idea:",
            improved_idea,
            "Structured input requirements:",
            resource_context,
        ]
    )

    request = ExperimentRequest(
        baseline_url=baseline_url,
        benchmark_url=benchmark_url,
        task=task,
        api_provider=api_provider,  # type: ignore[arg-type]
        llm_model=llm_model,
        baseline_url_explicit=baseline_explicit,
        benchmark_url_explicit=benchmark_explicit,
        baseline_resources=baselines,
        benchmark_resources=benchmarks,
        resource_context=resource_context,
        run_name=experience_name or None,
        run_name_is_prefix=bool(experience_name),
    )
    return InputRunConfig(request=request, use_study=ablation, study_mode=study_mode, backend=backend)
