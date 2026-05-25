from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from code_agent.graph import invoke_code_agent
from code_agent.state import initial_state
from code_agent.utils.progress import CliProgress


STEP_LABELS = {
    "clone_repo": "clone repository",
    "run_tests": "run tests",
    "analyze_failure": "analyze test failure",
    "propose_patch": "generate patch",
    "apply_patch": "apply patch",
    "evaluate_result": "write result",
}


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML config files.") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML object.")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行最小版 Code Agent Demo。")
    parser.add_argument("--repo-url", required=True, help="要 clone 的 Git 仓库地址。")
    parser.add_argument("--task", default=None, help="需求驱动模式：让 Agent 根据这段需求生成并应用代码修改。")
    parser.add_argument("--fresh-clone", action="store_true", help="运行前删除 workspaces 中已有的同名仓库并重新 clone。")
    parser.add_argument("--workspace-root", default=None, help="clone 仓库的根目录。")
    parser.add_argument("--results-root", default=None, help="日志、prompt、patch 和最终状态的输出目录。")
    parser.add_argument(
        "--test-command",
        default=None,
        help='在目标仓库中执行的测试命令，例如："python -m pytest -q --tb=short"。',
    )
    parser.add_argument("--allow-apply-patch", action="store_true", help="允许自动执行 git apply。")
    parser.add_argument(
        "-api",
        "--api",
        dest="api_provider",
        choices=["none", "deepseek", "openai"],
        default=None,
        help="用于生成 patch 的 LLM 服务商。",
    )
    parser.add_argument("--model", dest="llm_model", default=None, help="覆盖当前服务商的默认模型。")
    parser.add_argument("--temperature", dest="llm_temperature", type=float, default=None, help="LLM 请求的采样温度。")
    parser.add_argument("--max-tokens", dest="llm_max_tokens", type=int, default=None, help="LLM 返回内容的最大 token 数。")
    parser.add_argument("--max-debug-attempts", type=int, default=None, help="测试失败后的最大 debug 循环次数。")
    parser.add_argument("--max-patch-repair-attempts", type=int, default=None, help="git apply 失败后请求模型修复 patch 的最大次数。")
    parser.add_argument("--command-timeout-seconds", type=int, default=None, help="shell 命令超时时间，单位为秒。")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML 配置文件路径。")
    parser.add_argument(
        "--no-langgraph",
        action="store_true",
        help="使用内置顺序执行 fallback，而不是 LangGraph。",
    )
    parser.add_argument("--no-progress", action="store_true", help="不在终端显示执行步骤进度。")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config)

    state = initial_state(
        repo_url=args.repo_url,
        user_task=args.task,
        fresh_clone=args.fresh_clone,
        workspace_root=args.workspace_root or config.get("workspace_root", "./workspaces"),
        results_root=args.results_root or config.get("results_root", "./results"),
        test_command=args.test_command or config.get("test_command", "python -m pytest -q --tb=short"),
        api_provider=args.api_provider or config.get("api_provider"),
        llm_model=args.llm_model or config.get("llm_model"),
        llm_temperature=(
            args.llm_temperature
            if args.llm_temperature is not None
            else float(config.get("llm_temperature", 0.2))
        ),
        llm_max_tokens=(
            args.llm_max_tokens
            if args.llm_max_tokens is not None
            else int(config.get("llm_max_tokens", 4096))
        ),
        allow_apply_patch=args.allow_apply_patch or bool(config.get("allow_apply_patch", False)),
        max_debug_attempts=(
            args.max_debug_attempts
            if args.max_debug_attempts is not None
            else int(config.get("max_debug_attempts", 1))
        ),
        max_patch_repair_attempts=(
            args.max_patch_repair_attempts
            if args.max_patch_repair_attempts is not None
            else int(config.get("max_patch_repair_attempts", 1))
        ),
        command_timeout_seconds=(
            args.command_timeout_seconds
            if args.command_timeout_seconds is not None
            else int(config.get("command_timeout_seconds", 120))
        ),
    )

    total_steps = 5 if state.get("user_task") else 3
    progress = CliProgress(total_steps, enabled=not args.no_progress)

    def report_progress(step: str) -> None:
        if not state.get("user_task") and step == "analyze_failure":
            progress.add_steps(4)
        progress.update(STEP_LABELS.get(step, step))

    try:
        final_state = invoke_code_agent(
            state,
            prefer_langgraph=not args.no_langgraph,
            progress_callback=report_progress,
        )
    except Exception:
        progress.fail()
        raise
    progress.finish()
    print(json.dumps(final_state, indent=2, ensure_ascii=False))
    return 0 if final_state.get("test_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
