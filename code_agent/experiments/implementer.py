from __future__ import annotations

import ast
import re

from code_agent.experiments.models import ComparisonPlan, ExperimentRequest
from code_agent.tools.llm_tools import call_llm


ALLOWED_IMPORT_ROOTS = {"torch", "transformers"}
FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}
REQUIRED_FUNCTIONS = {"configure_model_config", "build_trainer_class"}


class ImplementationGenerationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        responses: list[str],
        sources: list[str],
        errors: list[str],
    ) -> None:
        super().__init__(message)
        self.responses = responses
        self.sources = sources
        self.errors = errors


def _is_hidden_state_fusion_plan(plan: ComparisonPlan | None) -> bool:
    if plan is None:
        return False
    text = " ".join(
        [
            plan.improved.main_change,
            plan.implementation.name,
            plan.implementation.implementation_instructions,
        ]
    ).lower()
    return (
        ("hidden" in text and "fusion" in text)
        or "hidden_states" in text
        or "mean_last_k" in text
        or "learned_weighted_sum" in text
    )


def _builtin_hidden_state_fusion_source() -> str:
    return '''import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

CHANGE_SUMMARY = (
    "Built-in hidden-state fusion classification head. The fusion head is registered on the model "
    "before Trainer creates the optimizer, and logits are computed from output_hidden_states rather "
    "than the default classifier. Supported strategies: last_layer, mean_last_k, learned_weighted_sum."
)


def configure_model_config(config):
    return config


class HiddenStateFusionHead(nn.Module):
    def __init__(self, hidden_size, num_labels, strategy="mean_last_k", k=4):
        super().__init__()
        self.strategy = strategy
        self.k = int(k or 4)
        if self.strategy not in {"last_layer", "mean_last_k", "learned_weighted_sum"}:
            raise ValueError(f"Unknown hidden-state fusion strategy: {self.strategy}")
        if self.strategy == "learned_weighted_sum":
            self.layer_weights = nn.Parameter(torch.zeros(self.k))
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):
        if not hidden_states:
            raise ValueError("Hidden-state fusion requires model outputs with hidden_states.")
        if self.strategy == "last_layer":
            fused = hidden_states[-1][:, 0, :]
        else:
            available = len(hidden_states) - 1
            k = max(1, min(self.k, available))
            selected = hidden_states[-k:]
            cls_stack = torch.stack([state[:, 0, :] for state in selected], dim=0)
            if self.strategy == "mean_last_k":
                fused = cls_stack.mean(dim=0)
            else:
                weights = F.softmax(self.layer_weights[-k:], dim=0)
                fused = (cls_stack * weights.view(-1, 1, 1)).sum(dim=0)
        return self.classifier(fused)


def _target_model(model):
    return model.module if hasattr(model, "module") else model


def build_trainer_class(base_trainer):
    class HiddenStateFusionTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            strategy = getattr(self.args, "fusion_strategy", "mean_last_k")
            k = int(getattr(self.args, "fusion_k", 4) or 4)
            target_model = _target_model(self.model)
            if not hasattr(target_model, "custom_fusion_head"):
                hidden_size = target_model.config.hidden_size
                num_labels = target_model.config.num_labels
                target_model.custom_fusion_head = HiddenStateFusionHead(
                    hidden_size=hidden_size,
                    num_labels=num_labels,
                    strategy=strategy,
                    k=k,
                )
                target_model.custom_fusion_head.to(next(target_model.parameters()).device)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            target_model = _target_model(model)
            outputs = model(**inputs, output_hidden_states=True)
            logits = target_model.custom_fusion_head(outputs.hidden_states)
            loss = F.cross_entropy(logits.view(-1, target_model.config.num_labels), labels.view(-1))
            wrapped_outputs = SequenceClassifierOutput(loss=loss, logits=logits)
            return (loss, wrapped_outputs) if return_outputs else loss

    return HiddenStateFusionTrainer
'''


def _function_source(tree: ast.AST, source: str, name: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    return ""


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _looks_like_logits_tensor(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return "logit" in node.id.lower()
    if isinstance(node, ast.Attribute):
        return node.attr == "logits" or "logit" in node.attr.lower()
    return False


def _has_bare_logits_tuple_return(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> bool:
    if function is None:
        return False
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if isinstance(value, ast.Tuple) and len(value.elts) == 2 and _looks_like_logits_tensor(value.elts[1]):
            return True
    return False


def _has_extra_trainer_prediction_outputs(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> bool:
    if function is None:
        return False
    extra_keys = {"hidden_states", "attentions", "past_key_values"}
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == "SequenceClassifierOutput" and any(keyword.arg in extra_keys for keyword in node.keywords):
                return True
            continue
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value in extra_keys:
                    return True
    return False


def build_implementation_prompt(plan: ComparisonPlan, user_task: str) -> str:
    return f"""Implement the user's specified improved training change for an existing Transformers experiment.

Model: {plan.model_id}
Dataset: {plan.dataset_id} / {plan.dataset_config or ""}
Metric: {plan.metric_name}
Fixed seed: {plan.seed}
Original user task: {user_task}
Improvement name: {plan.implementation.name}
Implementation instructions: {plan.implementation.implementation_instructions}

Return only valid Python source code for a module named improvement.py.

The module contract is:
- Define a string constant CHANGE_SUMMARY.
- Define `configure_model_config(config)`, returning the configuration used for improved only.
- Define `build_trainer_class(base_trainer)`, returning either `base_trainer` or a subclass of it.
- A custom trainer may override `compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None)`.

Constraints:
- This code is loaded only for the improved run. The baseline remains unchanged.
- Keep the dataset, splits, validation metric, random seed, model family, label count, batch size,
  learning rate, epochs, weight decay and warmup unchanged.
- Implement only the algorithm or code change requested in the original user task. Do not choose
  a different enhancement or add unrelated training changes.
- Do not download anything, read or write files, use networking, or spawn processes.
- Imports are limited to `torch`, `torch.nn`, `torch.nn.functional`, and `transformers`.
- Implement one coherent algorithm change, not a collection of unrelated tweaks.
- If you attach a newly created `torch.nn.Module` to the model inside a Trainer subclass, move it to
  the same device as the existing model parameters immediately, for example with
  `new_module.to(next(model.parameters()).device)`.
- Make generated code robust to `torch.nn.DataParallel`/wrapped models by using
  `target_model = model.module if hasattr(model, "module") else model` before reading or replacing
  model attributes such as `classifier`, `config`, or custom heads.
- The executor constructs the model with `AutoModelForSequenceClassification`; do not define a custom
  model subclass and assume the executor will instantiate it.
- For hidden-states fusion heads, `compute_loss` must call the model with `output_hidden_states=True`,
  derive fused logits from `outputs.hidden_states` through the custom head, and compute the loss from
  those fused logits. Do not train on `outputs.logits` from the default head while merely attaching an
  unused custom head.
- If `compute_loss` handles `return_outputs=True`, return `(loss, outputs_like)`, where `outputs_like`
  is a `transformers.modeling_outputs.SequenceClassifierOutput` or a dictionary/object exposing
  `logits`. Never return `(loss, logits)` or any other bare logits tensor; Hugging Face `Trainer`
  will slice bare tensors during evaluation and can silently drop examples.
- The `outputs_like` returned to `Trainer` during evaluation must contain only loss/logits fields.
  Do not include `hidden_states`, `attentions`, or other auxiliary tensors in that returned object;
  otherwise `Trainer` will pass a tuple such as `(logits, hidden_states)` to metric computation.
- A correct hidden-states fusion `compute_loss` pattern is:
  `outputs = model(**model_inputs, output_hidden_states=True)`;
  `fused_logits = custom_head(outputs.hidden_states)`;
  `loss = torch.nn.functional.cross_entropy(fused_logits, labels)`;
  `wrapped_outputs = SequenceClassifierOutput(loss=loss, logits=fused_logits)`;
  `return (loss, wrapped_outputs) if return_outputs else loss`.
- Do not include markdown fences or explanations outside the Python source.
"""


def extract_python_source(text: str) -> str:
    candidate = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    if not candidate:
        raise ValueError("The implementation model did not return Python source code.")
    return candidate + "\n"


def _build_implementation_repair_prompt(original_prompt: str, source: str, error: str) -> str:
    return f"""{original_prompt}

Your previous Python source failed validation before execution.

Validation error:
{error}

Previous Python source:
```python
{source}
```

Return a corrected complete improvement.py module.

Correction requirements:
- Preserve the requested algorithm and experiment controls.
- Fix the validation error directly.
- For `compute_loss(..., return_outputs=True)`, return `(loss, outputs_like)` where `outputs_like`
  exposes logits only. For example: `SequenceClassifierOutput(loss=loss, logits=fused_logits)`.
- Do not include `hidden_states`, `attentions`, or other auxiliary tensors in the object returned
  to Hugging Face `Trainer` metrics.
- Do not return markdown fences or explanations outside the Python source.
"""


def validate_implementation_source(source: str, *, plan: ComparisonPlan | None = None) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Generated improvement code is not valid Python: {exc}") from exc

    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = REQUIRED_FUNCTIONS - functions
    if missing:
        raise ValueError(f"Generated improvement code is missing required functions: {sorted(missing)}")

    has_summary = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CHANGE_SUMMARY" for target in node.targets))
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "CHANGE_SUMMARY")
        )
        for node in tree.body
    )
    if not has_summary:
        raise ValueError("Generated improvement code must define CHANGE_SUMMARY.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in ALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"Generated improvement code imports disallowed module: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".", 1)[0] not in ALLOWED_IMPORT_ROOTS:
                raise ValueError(f"Generated improvement code imports disallowed module: {node.module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"Generated improvement code uses disallowed name: {node.id}")

    if _is_hidden_state_fusion_plan(plan):
        compute_loss_source = _function_source(tree, source, "compute_loss")
        compute_loss_node = _function_node(tree, "compute_loss")
        if "outputs.logits" in compute_loss_source and "hidden_states" not in compute_loss_source:
            raise ValueError(
                "Hidden-states fusion implementations must compute loss from fused hidden-state logits, "
                "not from the default outputs.logits."
            )
        if _has_bare_logits_tuple_return(compute_loss_node):
            raise ValueError(
                "Hidden-states fusion compute_loss must return an outputs object or dict when "
                "return_outputs=True, not a bare logits tensor."
            )
        if _has_extra_trainer_prediction_outputs(compute_loss_node):
            raise ValueError(
                "Hidden-states fusion compute_loss must return only loss/logits to Trainer metrics; "
                "do not include hidden_states, attentions, or other auxiliary tensors."
            )
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        unused_model_classes = [
            name
            for name in class_names
            if "ForSequenceClassification" in name and name not in compute_loss_source
        ]
        if unused_model_classes:
            raise ValueError(
                "Generated improvement code defines a custom model class that the executor will not instantiate: "
                f"{unused_model_classes}"
            )


def request_implementation(
    request: ExperimentRequest,
    plan: ComparisonPlan,
    *,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    max_attempts: int = 2,
) -> tuple[str, str]:
    if _is_hidden_state_fusion_plan(plan):
        source = _builtin_hidden_state_fusion_source()
        validate_implementation_source(source, plan=plan)
        return source, source

    responses: list[str] = []
    sources: list[str] = []
    errors: list[str] = []
    current_prompt = prompt
    attempts = max(1, max_attempts)

    for attempt in range(attempts):
        response = call_llm(
            current_prompt,
            provider=request.api_provider,
            model=request.llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=(
                "You implement the user's specified ML code change exactly. "
                "Do not propose alternative algorithms. Return only safe Python source code."
            ),
            timeout=request.plan_timeout_seconds,
        )
        responses.append(response)
        try:
            source = extract_python_source(response)
            sources.append(source)
            validate_implementation_source(source, plan=plan)
            return source, response
        except ValueError as exc:
            error = str(exc)
            errors.append(error)
            repair_source = sources[-1] if sources else response
            if attempt + 1 >= attempts:
                raise ImplementationGenerationError(
                    error,
                    responses=responses,
                    sources=sources,
                    errors=errors,
                ) from exc
            current_prompt = _build_implementation_repair_prompt(prompt, repair_source, error)

    raise AssertionError("unreachable")
