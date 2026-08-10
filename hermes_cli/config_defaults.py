"""精简版 DEFAULT_CONFIG：只保留 my-hermes 当前被引用的键及其默认值。

原版 hermes_cli/config_defaults.py 是 3000+ 行的巨型默认配置，这里只抄被
引用的键（值照原版不改）：
- 顶层 ``context_file_max_chars``：原版默认 ``None``（表示动态缩放，见
  prompt_builder 的 _get_context_file_max_chars——对 None 会 fallback 到
  20000 下限），不是 20000；
- ``agent.api_max_retries`` / ``agent.environment_hint``；
- 顶层 ``timezone``（hermes_time 使用）。

TODO compression 段（threshold_percent 等）留给下一步"上下文压缩"模块实现
时再从原版抄。
"""

DEFAULT_CONFIG = {
    "context_file_max_chars": None,
    "agent": {
        "api_max_retries": 3,
        "environment_hint": "",
    },
    "timezone": "",
}
