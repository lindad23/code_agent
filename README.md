# Code Agent

Code Agent 是一个由 AI 参与实验设计、代码实现和自 debug 的通用实验执行框架。当前推荐入口是从 `inputs.json` 读取任务，而不是在 CLI 中堆很多参数。

它的核心边界是：

- AI 负责理解任务、规划实验、生成实验入口代码、生成实验执行表，并在失败后修复代码或执行 spec。
- 本地 pipeline 负责资源准备、缓存、环境创建/复用、CUDA/PyTorch 验证、GPU 调度、真实性校验、执行 cell 和汇总结果。

也就是说，它不是固定写死某一个任务的执行器；但也不是把所有事情完全交给 API 裸跑。AI 生成的内容会被本地模板、校验器和执行器约束。

## 当前架构

标准流程如下：

```text
inputs.json
  -> 解析任务配置
  -> 创建唯一 run_id、results 目录和 workspace
  -> 检查/初始化资源 cache
  -> materialize baseline / benchmark 资源
  -> API 生成实验规划
  -> API 生成执行 spec、my_main.py 和 experiment_table
  -> 本地校验是否真实资源驱动，拒绝 dummy/synthetic/stand-in baseline
  -> 创建或复用 conda 实验环境
  -> 验证 PyTorch/CUDA runtime
  -> 按 experiment_table 执行实验 cell
  -> 如失败，调用 API self-debug 并重试
  -> 汇总 metrics、日志和 report
```

API 参与的阶段：

- `request_generic_experiment_plan`: 生成实验规划。
- `request_generic_execution_spec`: 生成 `my_main.py`、依赖安装命令和 `experiment_table`。
- `request_generic_debug_execution_spec`: 根据 stderr/stdout、失败摘要和上一次 spec 生成修复版。

本地 pipeline 固定负责：

- `inputs.json` 解析和 run 命名。
- 资源 clone/download/copy 与 `asset_cache` 复用。
- execution template 生成。
- dummy/synthetic/preflight/baseline stand-in 校验。
- conda 环境创建或复用。
- PyTorch CUDA 自动适配与验证。
- GPU 贪心调度。
- cell 执行、output JSON contract 检查和 metrics 汇总。

## 安装

推荐使用 Python 3.11 和 Conda：

```bash
conda create -n code-agent python=3.11 -y
conda activate code-agent
pip install -r requirements.txt
```

配置 API Key：

```bash
export DEEPSEEK_API_KEY=your_key_here
# 或
export OPENAI_API_KEY=your_key_here
```

## inputs.json

推荐使用仓库根目录下的 `inputs.json` 作为唯一任务输入。当前支持字段如下：

```json
{
  "Experience name": "optional experiment name prefix",
  "Improved idea": "required detailed description of the algorithm idea",
  "Baselines url": {
    "baseline_name": "https://example.com/baseline-repo-or-resource"
  },
  "Benchmarks url": {
    "benchmark_name": "https://example.com/benchmark-repo-or-resource"
  },
  "Evaluation indexs": "optional metrics and reporting requirements",
  "Ablation": "True"
}
```

字段说明：

- `Experience name`: 可以为空。非空时会作为自动 run name 的前缀，保证不同实验命名更容易识别。
- `Improved idea`: 不可为空。写清楚要验证的算法改进、关键模块、训练方式和对比目标。
- `Baselines url`: 对象。每个 key 是 baseline 名称，不可为空；value 是 URL、repo id 或本地资源路径。value 为空时由 AI 在规划阶段提出可行资源。
- `Benchmarks url`: 对象。每个 key 是 benchmark/dataset 名称，不可为空；value 是 URL、repo id 或本地资源路径。value 为空时由 AI 在规划阶段提出可行资源。
- `Evaluation indexs`: 可以为空。非空时表示必须报告的指标，例如 accuracy、MSE、MAE、运行时间、显存、延迟等。为空时由 AI 选择适合任务的指标。
- `Ablation`: 是否要求 AI 在实验规划中加入消融实验。可写 `True/False`、`yes/no`、`是/否`。
- `API`: 可选，默认 `deepseek`。
- `Model`: 可选，用来覆盖默认 LLM model。
- `Study mode`: 可选，默认 `full`。

注意：

- 如果资源 value 是用户明确写入的网址或路径，系统会优先使用该资源；失败时会报错，不会静默替换成别的资源。
- 如果资源 value 为空，AI 可以在规划阶段建议可行资源。
- `Evaluation indexs` 这个字段名当前按代码保留了原拼写。

## 启动方式

标准运行：

```bash
python -m code_agent.experiment_main --input inputs.json
```

只生成规划和执行 spec，不真正运行实验：

```bash
python -m code_agent.experiment_main --input inputs.json --plan-only
```

常用参数：

```text
--model MODEL                覆盖规划/实现/debug 使用的 LLM model
--workspace-root PATH        workspace 根目录，默认 ./workspaces/experiments
--results-root PATH          结果根目录，默认 ./results/experiments
--python-version VERSION     实验 conda 环境 Python 版本，默认 3.11
--timeout-seconds SECONDS    环境安装和实验执行总超时
--plan-timeout-seconds SEC   单次 API 规划/代码生成超时
--hardware-profile PATH      本机 GPU/CUDA 配置缓存文件
--refresh-hardware-profile   重新检测 GPU/CUDA 并更新硬件配置
--no-progress                关闭 CLI 进度条
```

## 输出与缓存

每次运行会写入：

```text
results/experiments/<run-id>/
  request.json
  generic_plan.json
  generic_plan_prompt.md
  generic_plan_response.txt
  generic_execution_template.json
  generic_execution_spec.json
  generic_execution_prompt.md
  generic_execution_response.txt
  generic_debug_*                  # 如触发 self-debug
  generic_generated_files_*.json
  generic_experiment_table_*.json
  generic_experiment_cells_*/      # 每个 cell 的 input/output JSON
  generic_metrics.json
  generic_execution_summary.json
  environment_cache.json
  torch_runtime.json
  state.json
```

运行 workspace 会写入：

```text
workspaces/experiments/<run-id>/
  resources/       # 本次实验使用的 materialized resources
  my_main.py       # API 生成的实验入口，或其他生成文件
  environment.yml
```

共享缓存：

```text
workspaces/experiments/asset_cache/resources      # repo/dataset/resource cache
workspaces/experiments/asset_cache/environments  # conda env cache
```

如果只想清理历史运行产物但保留资源 cache，可以删除 `results/experiments/*`，并删除 `workspaces/experiments/` 下除 `asset_cache` 外的目录。

## 真实性校验

generic backend 会在执行前检查 API 生成的 spec，拒绝常见不可信实现：

- dummy baseline。
- synthetic/random training data。
- placeholder metrics。
- preflight/smoke-only workflow。
- 用手写同名 wrapper 替代 materialized baseline repo。
- experiment table record 中反复写绝对路径或超大矩阵。

如果校验或执行失败，系统会把失败摘要、stdout/stderr tail、上一版 spec、资源信息和接口摘要交给 API 进行 self-debug。

## 测试

```bash
conda activate code-agent
python -m pytest -q
```

`tests/` 不是正式实验运行依赖，但它保存了输入解析、资源 cache、环境复用、GPU 调度、dummy 校验、JSON 修复和 self-debug 的回归测试，建议保留。
