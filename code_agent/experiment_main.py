from __future__ import annotations

import argparse
import json

from code_agent.experiments.agent import run_experiment_agent
from code_agent.experiments.models import ExperimentRequest
from code_agent.utils.progress import CliProgress


STEP_LABELS = {
    "initialize": "initialize run",
    "request_plan": "request experiment plan",
    "prepare_environment": "prepare conda environment",
    "run_experiment": "execute experiment",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="根据模型网址、数据集网址和任务描述自主运行 Hugging Face 实验。")
    parser.add_argument("--baseline-url", required=True, help="Baseline 的 Hugging Face 模型仓库网址或 repo id。")
    parser.add_argument("--benchmark-url", required=True, help="Benchmark 的 Hugging Face 数据集网址或 repo id。")
    parser.add_argument("--task", required=True, help="自然语言实验任务，例如：使用 DistilBERT 微调 SST-2 文本分类。")
    parser.add_argument("-api", "--api", dest="api_provider", required=True, choices=["deepseek", "openai"], help="规划实验所用 API。")
    parser.add_argument("--model", dest="llm_model", default=None, help="覆盖规划 API 的默认模型。")
    parser.add_argument("--workspace-root", default="./workspaces/experiments", help="实验 workspace 根目录。")
    parser.add_argument("--results-root", default="./results/experiments", help="实验结果根目录。")
    parser.add_argument("--run-name", default=None, help="本次实验名称；不指定时自动生成唯一名称。")
    parser.add_argument("--python-version", default="3.11", help="新 Conda 环境使用的 Python 版本。")
    parser.add_argument("--timeout-seconds", type=int, default=86400, help="环境安装与实验执行超时秒数。")
    parser.add_argument("--plan-timeout-seconds", type=int, default=60, help="调用 API 生成实验计划的超时秒数。")
    parser.add_argument("--reuse-environment", action="store_true", help="同名运行目录已存在时更新并复用其中的环境。")
    parser.add_argument(
        "--hardware-profile",
        default="./configs/hardware_profile.local.yaml",
        help="保存并复用本机 GPU 与 PyTorch CUDA 选择的 YAML 文件路径。",
    )
    parser.add_argument(
        "--refresh-hardware-profile",
        action="store_true",
        help="忽略已有硬件配置，重新检测显卡并更新 PyTorch CUDA 选择。",
    )
    parser.add_argument("--plan-only", action="store_true", help="仅调用 API 生成并校验实验计划，不创建环境或运行实验。")
    parser.add_argument("--no-progress", action="store_true", help="不在终端显示执行步骤进度。")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = ExperimentRequest(
        baseline_url=args.baseline_url,
        benchmark_url=args.benchmark_url,
        task=args.task,
        api_provider=args.api_provider,
        llm_model=args.llm_model,
        workspace_root=args.workspace_root,
        results_root=args.results_root,
        run_name=args.run_name,
        environment_python=args.python_version,
        timeout_seconds=args.timeout_seconds,
        plan_timeout_seconds=args.plan_timeout_seconds,
        reuse_environment=args.reuse_environment,
        hardware_profile_file=args.hardware_profile,
        refresh_hardware_profile=args.refresh_hardware_profile,
    )
    progress = CliProgress(2 if args.plan_only else 4, enabled=not args.no_progress)
    state = run_experiment_agent(
        request,
        plan_only=args.plan_only,
        progress_callback=lambda step: progress.update(STEP_LABELS.get(step, step)),
    )
    if state.status == "failed":
        progress.fail()
    else:
        progress.finish()
    print(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0 if state.status in {"planned", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
