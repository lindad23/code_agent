# `code_agent.experiment_main`

该命令要求在 `--task` 中写明要验证的算法或代码改动。AI 只负责将指定改动实现为 `improved` 代码，然后在相同模型、数据集、验证指标和训练设置下运行 `baseline` 与 `improved` 对比。

例如，明确要求实现 focal loss：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "在相同 DistilBERT、GLUE/SST-2、validation accuracy、训练设置和 seed 下，保留 baseline；在 improved 中实现 focal loss（gamma=2.0）替换普通 cross entropy loss；分别训练并输出 accuracy 对比结果。" -api deepseek
```

仅生成结构化实验计划，不生成实现代码或启动训练：

```powershell
python -m code_agent.experiment_main --baseline-url https://huggingface.co/distilbert/distilbert-base-uncased --benchmark-url https://huggingface.co/datasets/nyu-mll/glue --task "在相同 DistilBERT、GLUE/SST-2、validation accuracy、训练设置和 seed 下，保留 baseline；在 improved 中实现 focal loss（gamma=2.0）替换普通 cross entropy loss；分别训练并输出 accuracy 对比结果。" -api deepseek --plan-only
```

每次完整运行的关键输出位于 `results/experiments/<run-id>/`：

```text
plan.json                    提取出的固定条件与用户指定改动
implementation_prompt.md     生成代码所使用的 prompt
implementation_response.txt  AI 原始代码响应
improvement.py               improved 代码的审计副本
comparison.md                baseline/improved 对比报告
metrics.json                 汇总指标
experiment_stdout.txt        训练实时输出
experiment_stderr.txt        训练错误输出
```

实际执行的 `improvement.py` 副本保存在对应的 `workspaces/experiments/<run-id>/generated/` 中，baseline 不加载此文件。

实验依赖包含 `hf_xet`，用于加速 Hugging Face Xet Storage 仓库的下载，避免回退到普通 HTTP 的性能提示。
