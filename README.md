# Code Agent 最小 Demo：项目代码架构说明

> 当前版本目标：先做一个可以闭环的最小 Demo：**输入一个 GitHub 仓库地址，Agent 自动 clone，运行 pytest，读取失败日志，生成 patch 建议，必要时应用 patch 后重跑测试**。  
> 暂不做完整科研实验系统；实验规划、数据管理、MLflow、Docker、报告生成等模块先保留占位文件和接口说明。

---

## 1. 当前项目定位

这个项目不是一个普通聊天机器人，而是一个面向代码仓库的 **实验执行型 Code Agent**。

第一版不追求“自动完成复杂科研创新”，而是先把最小闭环跑通：

```text
输入 repo_url
  ↓
clone 仓库到 workspaces/
  ↓
运行 pytest
  ↓
测试是否通过？
  ├── 通过：记录结果，结束
  └── 失败：读取失败日志
          ↓
        生成 patch 建议
          ↓
        是否允许自动应用 patch？
          ├── 否：保存 patch 建议，结束
          └── 是：git apply patch
                    ↓
                  重新运行 pytest
```

第一版的工程原则：

1. **安全优先**：默认不自动修改目标仓库，只生成 patch 建议。
2. **状态显式**：所有节点都围绕 `CodeAgentState` 读写状态。
3. **节点单一职责**：clone、run test、analyze failure、propose patch、apply patch 分开写。
4. **先本地可跑**：第一阶段不引入 Kubernetes、Ray、复杂前端。
5. **方便迁移到完整科研 Agent**：预留 planner、experiment_designer、evaluator、reporter 等模块。

---

## 2. 技术栈选择

当前最小 Demo 使用：

```text
Python
LangGraph
Pydantic
pytest
Git CLI
YAML
subprocess
pathlib
```

后续预留但暂不启用：

```text
Docker
MLflow
DVC
Hydra
SQLite
Web UI
多 Agent 协作
```

---

## 3. 项目目录结构

```text
code_agent_minimal_demo/
├── README.md
├── pyproject.toml
├── .env.example
├── configs/
│   └── default.yaml
├── code_agent/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   ├── main.py
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── clone_repo.py
│   │   ├── run_tests.py
│   │   ├── analyze_failure.py
│   │   ├── propose_patch.py
│   │   ├── apply_patch.py
│   │   ├── evaluate_result.py
│   │   ├── planner.py
│   │   ├── repo_analyzer.py
│   │   ├── experiment_designer.py
│   │   └── reporter.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── command_tools.py
│   │   ├── git_tools.py
│   │   ├── file_tools.py
│   │   ├── llm_tools.py
│   │   ├── docker_tools.py
│   │   ├── mlflow_tools.py
│   │   ├── dataset_tools.py
│   │   └── safety.py
│   ├── prompts/
│   │   └── patch_prompt.md
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── paths.py
├── scripts/
│   └── run_minimal_demo.py
├── tests/
│   └── test_state.py
├── workspaces/
├── logs/
└── results/
```

---

## 4. 核心文件说明

### 4.1 `code_agent/state.py`

定义 Agent 在整张图中流动的状态。

它相当于 Agent 的“工作台记录表”：

```python
class CodeAgentState(TypedDict):
    repo_url: str
    workspace_root: str
    repo_dir: str | None

    test_command: str
    test_passed: bool | None
    test_stdout: str | None
    test_stderr: str | None
    test_returncode: int | None

    failure_summary: str | None
    patch_suggestion: str | None
    patch_file: str | None

    allow_apply_patch: bool
    patch_applied: bool
    debug_attempts: int
    max_debug_attempts: int

    final_status: str | None
```

为什么要把状态集中定义？

因为 Code Agent 后面会越来越复杂，如果每个函数都随意传参，会很快失控。集中状态的好处是：

```text
所有节点输入输出统一
可以保存 checkpoint
可以打印、审计、回放
可以清楚知道当前执行到哪一步
```

---

### 4.2 `code_agent/graph.py`

定义 LangGraph 工作流。

当前图结构：

```text
START
  ↓
clone_repo
  ↓
run_tests
  ↓
route_after_tests
  ├── passed → evaluate_result → END
  ├── failed_and_can_debug → analyze_failure → propose_patch → apply_patch → run_tests
  └── failed_and_stop → evaluate_result → END
```

注意：`apply_patch` 节点默认不会真的修改代码，除非：

```text
allow_apply_patch = true
并且 patch_suggestion 中包含合法 unified diff
```

---

### 4.3 `code_agent/main.py`

命令行入口。

目标调用方式：

```bash
python -m code_agent.main \
  --repo-url https://github.com/xxx/yyy.git \
  --workspace-root ./workspaces \
  --test-command "python -m pytest -q --tb=short" \
  --max-debug-attempts 1
```

如果要允许自动应用 patch：

```bash
python -m code_agent.main \
  --repo-url https://github.com/xxx/yyy.git \
  --allow-apply-patch
```

第一版建议先不要开启自动 patch，先检查 patch 建议质量。

---

## 5. Nodes 层说明

`nodes/` 目录存放 LangGraph 中的节点。每个节点本质上是一个函数：

```python
def node_name(state: CodeAgentState) -> dict:
    ...
    return {"some_field": new_value}
```

### 5.1 `clone_repo.py`

职责：

```text
根据 repo_url clone 仓库到 workspaces/
如果目录已存在，则默认复用
记录 repo_dir
```

后续可以增强：

```text
指定 branch / commit
自动清理旧 workspace
clone 前检查磁盘空间
clone 后记录 git commit hash
```

---

### 5.2 `run_tests.py`

职责：

```text
在 repo_dir 下执行测试命令
捕获 stdout / stderr / returncode
判断测试是否通过
```

默认命令：

```bash
python -m pytest -q --tb=short
```

这里使用 `python -m pytest` 是为了让 Python 当前解释器路径更明确。

后续可以增强：

```text
支持 tox / nox
支持 npm test
支持 benchmark 脚本
支持 timeout
支持 Docker 内运行
```

---

### 5.3 `analyze_failure.py`

职责：

```text
读取测试失败日志
压缩成简短 failure_summary
供 LLM 或规则模块生成 patch 建议
```

第一版只做简单截断和摘要，不做复杂 traceback 分析。

后续可以增强：

```text
提取失败测试名
提取 AssertionError
提取 ImportError / ModuleNotFoundError
定位相关文件
结合 git diff 和代码搜索分析上下文
```

---

### 5.4 `propose_patch.py`

职责：

```text
根据 failure_summary 和仓库信息生成 patch 建议
把 prompt 保存到 results/patch_prompt.md
把 patch_suggestion 写入 state
```

第一版默认是“占位 LLM 接口”：

```text
如果你接入真实 LLM：返回 unified diff
如果没有接入真实 LLM：保存 prompt，提示用户手动补 patch
```

后续可以增强：

```text
接 OpenAI / DeepSeek / Claude
结构化输出 patch
多轮读取文件上下文
自动验证 patch 格式
```

---

### 5.5 `apply_patch.py`

职责：

```text
检查是否允许自动应用 patch
从 patch_suggestion 中提取 unified diff
执行 git apply
记录 patch_applied
```

默认行为：

```text
allow_apply_patch = False 时不应用 patch
```

这是为了避免 Agent 初期误改目标代码。

---

### 5.6 `evaluate_result.py`

职责：

```text
根据最终 test_passed 判断 final_status
保存最终状态摘要
```

后续科研 Agent 中，这个节点会扩展为：

```text
评估 benchmark 指标
判断结果是否可信
检查是否比 baseline 提升
生成实验结论
```

---

## 6. Tools 层说明

`tools/` 目录不是 LangGraph 节点，而是节点内部调用的基础能力。

### 6.1 `command_tools.py`

封装 subprocess：

```text
run_command(cmd, cwd, timeout)
```

返回：

```text
stdout
stderr
returncode
elapsed_time
```

这样所有命令执行都有统一日志和超时控制。

---

### 6.2 `git_tools.py`

封装 Git 操作：

```text
git clone
git diff
git apply
git rev-parse HEAD
```

第一版先用 Git CLI，不急着引入 GitPython。

---

### 6.3 `file_tools.py`

封装安全文件读写：

```text
read_text
write_text
safe_resolve_path
```

后续要加路径沙箱，防止 Agent 写出 workspace 外部。

---

### 6.4 `llm_tools.py`

LLM 接口占位层。

第一版不绑定具体模型厂商，只定义：

```text
build_patch_prompt
call_llm_for_patch
extract_unified_diff
```

后面你可以接：

```text
OpenAI
DeepSeek
Claude
本地模型
```

---

### 6.5 `docker_tools.py`

暂时只放注释。

后续用于：

```text
构建容器
在容器中运行测试
隔离外部仓库依赖
限制文件系统和网络权限
```

---

### 6.6 `mlflow_tools.py`

暂时只放注释。

后续用于：

```text
记录实验参数
记录测试结果
记录指标
记录 patch diff
记录 artifacts
```

---

### 6.7 `dataset_tools.py`

暂时只放注释。

后续用于：

```text
下载 benchmark 数据
校验数据 hash
缓存数据集
管理数据版本
```

---

### 6.8 `safety.py`

安全策略占位。

第一版至少要控制：

```text
禁止 rm -rf /
禁止访问 workspace 外部路径
禁止默认自动 push
禁止默认自动删除文件
限制命令 timeout
```

---

## 7. 运行流程详解

### Step 1：初始化状态

`main.py` 根据命令行参数构造初始状态：

```python
state = {
    "repo_url": args.repo_url,
    "workspace_root": args.workspace_root,
    "repo_dir": None,
    "test_command": args.test_command,
    "test_passed": None,
    "allow_apply_patch": args.allow_apply_patch,
    "debug_attempts": 0,
    "max_debug_attempts": args.max_debug_attempts,
}
```

---

### Step 2：clone 仓库

`clone_repo` 节点：

```text
输入 repo_url
输出 repo_dir
```

---

### Step 3：运行测试

`run_tests` 节点：

```text
执行 test_command
捕获 stdout / stderr
根据 returncode 判断 test_passed
```

---

### Step 4：条件分支

如果测试通过：

```text
run_tests → evaluate_result → END
```

如果测试失败且还能 debug：

```text
run_tests → analyze_failure → propose_patch → apply_patch → run_tests
```

如果测试失败但达到最大 debug 次数：

```text
run_tests → evaluate_result → END
```

---

### Step 5：生成 patch 建议

`propose_patch` 节点会生成一个 prompt：

```text
当前失败日志是什么？
仓库路径是什么？
请输出 unified diff 格式的 patch。
```

第一版中，如果没有接入真实 LLM，它会把 prompt 保存到：

```text
results/patch_prompt.md
```

你可以手动把这个 prompt 给模型，让模型输出 patch。

---

### Step 6：可选应用 patch

如果启用：

```text
allow_apply_patch = true
```

并且模型输出了合法 unified diff，则执行：

```bash
git apply generated.patch
```

然后重新运行测试。

---

## 8. 为什么不是直接写一个大 while 循环？

第一版确实可以用 while 循环写。

但是我们这里故意提前使用 LangGraph，是为了后面扩展：

```text
checkpoint：每一步保存状态
interrupt：修改代码前人工确认
conditional edge：根据结果走不同分支
debug loop：自动重试
time travel：回到历史状态重新跑
```

对于一个长期运行的实验 Agent，这些能力会比普通 if-else 更重要。

---

## 9. 当前最小 Demo 的限制

当前版本刻意不做以下事情：

```text
不自动下载数据集
不自动选择 baseline
不自动跑大型训练
不自动写论文式报告
不默认自动修改仓库
不默认自动 push 到远程
不默认在 Docker 中执行
```

原因是：

```text
第一阶段目标不是强，而是稳。
```

先把这个闭环跑通：

```text
clone → test → failure log → patch proposal → optional apply → retest
```

再往科研实验 Agent 扩展。

---

## 10. 后续扩展路线

### 阶段 1：当前最小 Demo

```text
clone repo
run pytest
read failure
propose patch
optional apply patch
rerun pytest
```

---

### 阶段 2：增强代码理解

增加：

```text
repo_analyzer
代码搜索
AST 分析
相关文件定位
测试失败定位
```

---

### 阶段 3：增强安全执行

增加：

```text
Docker sandbox
命令白名单
路径沙箱
资源限制
网络限制
```

---

### 阶段 4：实验管理

增加：

```text
experiment_designer
配置文件生成
Hydra / YAML
MLflow tracking
结果表格
```

---

### 阶段 5：科研 Agent

增加：

```text
baseline 选择
benchmark 数据集管理
自动 ablation
多次随机种子
结果可信度检查
最终报告生成
```

---

## 11. 推荐开发顺序

建议你不要一次性写完整 Agent，而是按下面顺序推进：

```text
1. 跑通 clone_repo 节点
2. 跑通 run_tests 节点
3. 把失败日志保存到文件
4. 让 LLM 根据失败日志生成 patch
5. 手动检查 patch
6. 再开放 allow_apply_patch
7. 接入 LangGraph checkpoint
8. 加 Docker sandbox
9. 加实验管理模块
```

---

## 12. 第一版验收标准

第一版完成的标准不是“能修所有 bug”，而是：

```text
给一个 Python 仓库地址
Agent 能 clone
Agent 能运行 pytest
Agent 能保存失败日志
Agent 能生成修复 prompt
Agent 能接收 patch 并尝试应用
Agent 能重新运行测试
Agent 能输出最终状态
```

只要这个闭环稳定，后面就可以逐步增强智能程度。

---

## 13. 当前文件中的 TODO 约定

代码里会出现几类 TODO：

```text
TODO(minimal)：最小 Demo 很快要实现
TODO(next)：下一阶段实现
TODO(future)：科研 Agent 完整版再实现
```

优先只处理 `TODO(minimal)`。

---

## 14. 一句话总结

当前项目架构的核心不是“让 LLM 直接乱改代码”，而是：

> 用 LangGraph 管理 Code Agent 的执行流程，用 State 保存每一步证据，用 Git 和 pytest 提供可验证反馈，用 patch 机制控制代码修改风险。
