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


def validate_implementation_source(source: str) -> None:
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


def request_implementation(
    request: ExperimentRequest,
    plan: ComparisonPlan,
    *,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> tuple[str, str]:
    response = call_llm(
        prompt,
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
    source = extract_python_source(response)
    validate_implementation_source(source)
    return source, response
