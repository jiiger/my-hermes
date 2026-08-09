"""工具包命名空间（精简移植版）。

对应原版 hermes-agent 的 tools/__init__.py（751 字节）。保持包导入
副作用最小：import tools 不会主动导入任何工具模块；具体工具由
model_tools._TOOL_MODULES 显式导入以触发模块级自注册。
"""

