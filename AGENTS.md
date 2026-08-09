# AGENTS.md

## 项目是什么

my-hermes 是 hermes-agent 的精简克隆（阉割版），用于个人学习 / 扩展用途。

### 目录结构

- `agent/`：对话循环、运行时、错误分类等核心逻辑
- `tools/`：工具注册表 + 各工具实现
- `hermes_cli/`：交互式 CLI
- 顶层：`run_agent.py`（入口）、`model_tools.py`（装配层）、`utils.py`、`hermes_constants.py`

## 环境

- Python 3.11.15，uv 管理，虚拟环境在 `.venv`
- 运行：`uv run hermes-agent`；测试：`uv run pytest`

## 代码约定

- 注释和 docstring 一律用中文
- 保持与原版 hermes-agent 结构对齐：模块同名、函数签名逐字一致
- 移植时能干净删减就删，删不干净的跳过并在交付报告里说明，不要硬塞

## 铁律

- `/home/yang/code/hermes-agent` 是只读参考，严禁修改
