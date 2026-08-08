"""LLM 执行中间件（精简版，对应原版 hermes_cli/middleware.py）。

原版依赖 hermes_cli 插件系统的中间件注册表（LLM_EXECUTION_MIDDLEWARE +
invoke_middleware）；精简版用模块级列表模拟同一语义：

- ``register_llm_execution_middleware(cb)`` 注册回调；
- ``run_llm_execution_middleware(request, next_call, **context)`` 按洋葱模型
  执行整条链；没有注册任何回调时就是 ``next_call(request)`` 一行直通。

中间件回调签名：``cb(request: dict, next_call: Callable, **context) -> Any``
"""

from typing import Any, Callable, Dict, List

# 注册的 LLM 执行中间件回调（按注册顺序执行；洋葱模型：后注册的最先包）
_LLM_EXECUTION_MIDDLEWARE: List[Callable] = []


def register_llm_execution_middleware(callback: Callable) -> None:
    """注册一个 LLM 执行中间件。

    Args:
        callback: ``(request, next_call, **context) -> Any``。
            next_call 是链上下一层（最内层是真正的 API 调用）。
    """
    if callback not in _LLM_EXECUTION_MIDDLEWARE:
        _LLM_EXECUTION_MIDDLEWARE.append(callback)


def clear_llm_execution_middleware() -> None:
    """清空所有已注册的 LLM 执行中间件（测试/重置用）。"""
    _LLM_EXECUTION_MIDDLEWARE.clear()


def run_llm_execution_middleware(
    request: Dict[str, Any],
    next_call: Callable[[Dict[str, Any]], Any],
    **context: Any,
) -> Any:
    """按注册顺序执行 LLM 执行中间件链（对应原版 hermes_cli/middleware.py:187）。

    - 没有注册回调 → 直接 ``next_call(request)``（零开销直通）；
    - 有回调 → 洋葱模型：最后注册的回调最先拿到 next_call，
      逐层包裹，最内层是真正的 API 调用。

    Args:
        request: API 请求参数（api_kwargs）。
        next_call: 链上下一层调用（最内层是 _perform_api_call）。
        **context: 透传给每个中间件的上下文（agent、api_call_count 等）。

    Returns:
        最内层调用（API 响应）的返回值。
    """
    if not _LLM_EXECUTION_MIDDLEWARE:
        return next_call(request)

    # 洋葱包裹：从链尾开始，每个回调包住前一个
    handler: Callable = next_call
    for callback in reversed(_LLM_EXECUTION_MIDDLEWARE):
        previous = handler
        handler = (
            lambda cb, nxt: (
                lambda req, **ctx: cb(req, nxt, **ctx)
            )
        )(callback, previous)
    return handler(request, **context)
