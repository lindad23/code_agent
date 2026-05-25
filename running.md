# `code_agent.experiment_main` 使用说明

该命令用于根据 Hugging Face 模型、数据集和自然语言任务生成实验计划，并运行文本分类微调实验。

## 基本命令

PowerShell 一行运行：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "使用 DistilBERT 在 GLUE 的 SST-2 子任务上微调文本分类模型，输出验证集 accuracy" -api deepseek
```

使用 OpenAI：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "使用 DistilBERT 在 GLUE 的 SST-2 子任务上微调文本分类模型，输出验证集 accuracy" -api openai
```

## 仅生成计划

只调用 API 生成并校验实验计划，不创建环境、不下载资源、不训练：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "使用 DistilBERT 在 GLUE 的 SST-2 子任务上微调文本分类模型，输出验证集 accuracy" -api deepseek --plan-only
```

## 硬件配置

程序默认复用本机硬件配置文件：

```text
configs/hardware_profile.local.yaml
```

需要重新检测显卡并更新 PyTorch CUDA 选择时：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "使用 DistilBERT 在 GLUE 的 SST-2 子任务上微调文本分类模型，输出验证集 accuracy" -api deepseek --refresh-hardware-profile
```

## 常用参数

```text
--baseline-url URL          Hugging Face 模型仓库 URL 或 repo id，必填
--benchmark-url URL         Hugging Face 数据集 URL 或 repo id，必填
--task TEXT                 自然语言实验任务，必填
-api, --api PROVIDER       规划 API：deepseek 或 openai，必填
--model MODEL              覆盖规划 API 使用的模型
--run-name NAME            指定本次实验名称
--workspace-root PATH      实验 workspace 根目录
--results-root PATH        实验结果根目录
--python-version VERSION   实验 Conda 环境 Python 版本，默认 3.11
--timeout-seconds SECONDS  环境安装与实验运行超时
--plan-timeout-seconds N   生成实验计划的 API 超时
--reuse-environment        复用同名运行的实验环境
--hardware-profile PATH    指定本机硬件 profile 文件
--refresh-hardware-profile 重新检测 GPU 与 CUDA 配置
--plan-only                仅生成实验计划
--no-progress              关闭终端步骤进度显示
```

查看完整帮助：

```powershell
python -m code_agent.experiment_main --help
```

## 运行输出

每次运行结果位于：

```text
results/experiments/<run-id>/
```

常用文件：

```text
plan.json                 实验计划
metrics.json              最终训练和评估指标
experiment_stdout.txt     实时训练标准输出记录
experiment_stderr.txt     实时训练错误输出记录
hardware.json             本次使用的硬件配置快照
torch_runtime.json        实际验证的 PyTorch/CUDA 信息
model/best/               保存的最佳模型
```

可复用的模型和数据缓存位于：

```text
workspaces/experiments/asset_cache/
```
