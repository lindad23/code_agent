# Code Agent

一个面向 Hugging Face 文本分类实验的轻量 Code Agent。

你在 `--task` 中指定需要验证的算法修改，Agent 会：

1. 解析模型、数据集和对比设置。
2. 生成仅用于 `improved` 分支的实现代码。
3. 在相同数据集、指标、训练参数和 seed 下运行 `baseline` 与 `improved`。
4. 保存生成代码、训练日志、指标与对比报告。

当前实验执行器支持 Hugging Face sequence classification，例如 DistilBERT 在 GLUE/SST-2 上的微调对比。

## 安装

推荐使用 Python 3.11 和 Conda：

```powershell
conda create -n code-agent python=3.11 -y
conda activate code-agent
pip install -r requirements.txt
```

在 `.env` 中配置 API Key：

```env
DEEPSEEK_API_KEY=your_key_here
# 或
OPENAI_API_KEY=your_key_here
```

## 运行实验

以下示例要求 AI 在 `improved` 中实现 focal loss，并与原始 baseline 对比：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "在相同 DistilBERT、GLUE/SST-2、validation accuracy、训练设置和 seed 下，保留 baseline；在 improved 中实现 focal loss（gamma=2.0）替换普通 cross entropy loss；分别训练并输出 accuracy 对比结果。" -api deepseek
```

只生成实验计划，不生成代码或启动训练：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "在 improved 中实现 focal loss（gamma=2.0），并与 baseline 对比 validation accuracy。" -api deepseek --plan-only
```

常用参数：

```text
--model MODEL                指定规划与实现使用的 LLM
--run-name NAME              指定运行名称
--reuse-environment          复用同名实验环境
--refresh-hardware-profile   重新检测本机 GPU/CUDA 配置
--no-progress                关闭 CLI 进度显示
```

## 输出与缓存

每次运行的结果写入：

```text
results/experiments/<run-id>/
  plan.json
  improvement.py
  comparison.md
  metrics.json
  experiment_stdout.txt
  experiment_stderr.txt
```

实际执行的 AI 实现代码位于：

```text
workspaces/experiments/<run-id>/generated/improvement.py
```

模型与数据仓库会缓存到 `workspaces/experiments/asset_cache/`，实验环境会复用本机 GPU 对应的 PyTorch/CUDA 配置。实验依赖包含 `hf_xet`，以提升 Hugging Face Xet Storage 下载性能。

## 其他入口

项目仍保留通用代码仓库修补入口 `python -m code_agent.main`，用于 clone 仓库、运行测试、生成并可选应用 patch；当前主要开发方向为 `code_agent.experiment_main` 实验流程。

## 测试

```powershell
conda activate code-agent
python -m pytest -q
```
