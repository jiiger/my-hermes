"""可配置的工具结果持久化预算常量（精简移植版）。

对应原版 hermes-agent 的 tools/budget_config.py（114 行）。
按工具解析阈值：pinned > 配置覆盖 > registry > 默认。
"""

from dataclasses import dataclass, field
from typing import Dict

# 阈值永不允许被覆盖的工具。
# read_file=inf 防止无限「持久化 → 读取 → 再持久化」循环。
PINNED_THRESHOLDS: Dict[str, float] = {
    "read_file": float("inf"),
}

# 与 tool_result_storage.py 当前硬编码值一致的默认值。
# 这里作为唯一事实来源；tool_result_storage.py 从中导入。
DEFAULT_RESULT_SIZE_CHARS: int = 100_000
DEFAULT_TURN_BUDGET_CHARS: int = 200_000
DEFAULT_PREVIEW_SIZE_CHARS: int = 1_500


@dataclass(frozen=True)
class BudgetConfig:
    """三层工具结果持久化系统的不可变预算常量。

    Layer 2（单结果）：resolve_threshold(tool_name) -> 字符数阈值。
    Layer 3（单轮）：turn_budget -> 单个 assistant 回合内所有工具结果的
                    累计字符预算。
    Preview：preview_size -> 持久化后内联摘要的字符数。
    """

    default_result_size: int = DEFAULT_RESULT_SIZE_CHARS
    turn_budget: int = DEFAULT_TURN_BUDGET_CHARS
    preview_size: int = DEFAULT_PREVIEW_SIZE_CHARS
    tool_overrides: Dict[str, int] = field(default_factory=dict)

    def resolve_threshold(self, tool_name: str) -> int | float:
        """解析某个工具的持久化阈值。

        优先级：pinned -> tool_overrides -> registry 按工具 -> 默认。

        registry 的按工具值会被 default_result_size 封顶，这样缩小后的
        预算（小模型）能真正约束那些注册了大固定 max_result_size_chars
        的工具（web/terminal/x_search 都注册 100K）。默认预算下这是
        无操作（两者都等于 100K）；缩小预算后可以防止按工具注册值把
        上限重新撑到超过模型窗口（对应原版 #23767）。
        """
        if tool_name in PINNED_THRESHOLDS:
            return PINNED_THRESHOLDS[tool_name]
        if tool_name in self.tool_overrides:
            return self.tool_overrides[tool_name]
        # 函数内 import：tools/registry.py 不能 import 任何工具实现文件，
        # 避免循环导入（原版 resolve_threshold 也是这么做的）。
        from tools.registry import registry
        registry_value = registry.get_max_result_size(tool_name, default=self.default_result_size)
        if registry_value == float("inf"):
            return registry_value
        return min(registry_value, self.default_result_size)


# 默认配置 —— 与当前硬编码行为完全一致。
DEFAULT_BUDGET = BudgetConfig()


# 把预算缩放到模型上下文窗口时使用的 token<->字符换算系数。
# 刻意保守（除数越小 = 每 token 字符越多 = 字符预算越大）会低估对
# 小模型的保护，所以沿用估算器使用的约 4 字符/token 比例
# （原版对应 agent/model_metadata.py）。
_CHARS_PER_TOKEN: int = 4

# 单个工具结果在持久化/截断前允许占模型上下文窗口的比例，
# 以及整轮工具输出允许占的比例。工具输出不是窗口里唯一的内容
# （系统提示、工具 schema、对话历史、模型自身回复都在竞争），
# 所以这些值都远低于 1.0。
_PER_RESULT_WINDOW_FRACTION: float = 0.15
_PER_TURN_WINDOW_FRACTION: float = 0.30

# 下限：即便模型极小，仍保留可用的预览/结果，而不是 0 字符预算。
_MIN_RESULT_SIZE_CHARS: int = 8_000
_MIN_TURN_BUDGET_CHARS: int = 16_000


def budget_for_context_window(context_length: int | None) -> BudgetConfig:
    """返回按当前模型上下文窗口缩放后的 BudgetConfig。

    固定默认值（单结果 100K / 单轮 200K 字符）对大模型（200K+ token）
    是正确的，但会忽略小模型：65K token 模型上，按 100K 字符阈值持久化
    单个工具结果，或 200K 字符单轮预算（约 50K token），本身就可能接近
    或超过整个窗口，从而强制产生超大请求（对应原版 #23767）。

    缩放保持大模型与今天逐字节一致（按比例的值被钳制到现有默认值作为
    上限），同时按窗口比例缩小小模型的预算，并以地板保底让可用预览
    始终存在。
    """
    if not context_length or context_length <= 0:
        return DEFAULT_BUDGET

    window_chars = context_length * _CHARS_PER_TOKEN
    per_result = int(window_chars * _PER_RESULT_WINDOW_FRACTION)
    per_turn = int(window_chars * _PER_TURN_WINDOW_FRACTION)

    # 钳制：绝不超历史默认（大模型保持不变），绝不低于地板（小模型保持可用）。
    per_result = max(_MIN_RESULT_SIZE_CHARS, min(per_result, DEFAULT_RESULT_SIZE_CHARS))
    per_turn = max(_MIN_TURN_BUDGET_CHARS, min(per_turn, DEFAULT_TURN_BUDGET_CHARS))

    return BudgetConfig(
        default_result_size=per_result,
        turn_budget=per_turn,
        preview_size=DEFAULT_PREVIEW_SIZE_CHARS,
    )
