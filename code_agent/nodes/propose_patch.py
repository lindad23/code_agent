from __future__ import annotations

from code_agent.state import CodeAgentState
from code_agent.tools.file_tools import write_text
from code_agent.tools.llm_tools import build_patch_prompt, build_task_prompt, call_llm_for_patch, extract_unified_diff
from code_agent.tools.repo_context import collect_repo_context


def propose_patch(state: CodeAgentState) -> dict:
    repo_dir = state.get("repo_dir")
    if not repo_dir:
        raise ValueError("repo_dir is required before proposing a patch")

    if state.get("user_task"):
        repo_context = collect_repo_context(repo_dir)
        repo_context_file = write_text(f"{state.get('results_root', './results')}/repo_context.txt", repo_context)
        prompt = build_task_prompt(
            repo_dir=repo_dir,
            user_task=state["user_task"] or "",
            repo_context=repo_context,
            test_command=state["test_command"],
        )
    else:
        repo_context_file = None
        prompt = build_patch_prompt(
            repo_dir=repo_dir,
            failure_summary=state.get("failure_summary") or "No failure summary available.",
            test_command=state["test_command"],
        )
    results_root = state.get("results_root", "./results")
    prompt_file = write_text(f"{results_root}/patch_prompt.md", prompt)

    suggestion = call_llm_for_patch(
        prompt,
        provider=state.get("api_provider"),
        model=state.get("llm_model"),
        temperature=state.get("llm_temperature", 0.2),
        max_tokens=state.get("llm_max_tokens", 4096),
    )
    diff = extract_unified_diff(suggestion)
    patch_file = write_text(f"{results_root}/generated.patch", diff or suggestion)

    return {
        "patch_suggestion": suggestion,
        "patch_file": str(patch_file),
        "patch_prompt_file": str(prompt_file),
        "repo_context_file": str(repo_context_file) if repo_context_file else None,
    }
