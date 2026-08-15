"""Web 搜索 Provider 抽象基类（精简移植版）。

对应原版 hermes-agent 的 agent/web_search_provider.py（211 行）。
定义可插拔后端的统一接口：插件经 ``PluginContext.register_web_search_provider()``
注册实例，活动后端（由 ``web.search_backend`` / ``web.extract_backend`` /
``web.backend`` 配置项选定）服务每次 ``web_search`` / ``web_extract`` 工具调用。

Provider 放在 ``<项目根>/plugins/web/<name>/``（内置，自动加载）或
``~/.hermes/plugins/web/<name>/``（用户，经 plugins.enabled 启用）。

响应形状（保留原版 legacy 契约，工具包装层无需翻译）：

搜索结果::

    {
        "success": True,
        "data": {
            "web": [
                {"title": str, "url": str, "description": str, "position": int},
                ...
            ]
        }
    }

提取结果::

    {"success": True, "data": [{"url", "title", "content", "raw_content", "metadata"}, ...]}

失败（任一能力）::

    {"success": False, "error": str}

裁剪项（相对原版）：``get_setup_schema``（hermes tools 配置向导用的，
my-hermes 没有该向导）；``get_provider_env`` 简化成只读 os.environ
（原版经 hermes_cli.config.get_env_value 读 config 层，my-hermes 未移植）。
"""

from __future__ import annotations

import abc
import os
from typing import Any, Dict, List


def get_provider_env(name: str) -> str:
    """读取 web provider 的环境变量（简化为直接 os.getenv）。

    原版经 ``hermes_cli.config.get_env_value`` 先查 os.environ 再查
    ``~/.hermes/.env``；my-hermes 的 config 层没有 get_env_value，故
    精简为裸 os.getenv——凭据仍需预先 export 进进程环境。

    返回去空白后的值，未设置时返回 ``""``。
    """
    return (os.getenv(name, "") or "").strip()


class WebSearchProvider(abc.ABC):
    """web 搜索/提取后端抽象基类。

    子类必须实现 :meth:`is_available` 与 :meth:`search` / :meth:`extract`
    至少其一。:meth:`supports_search` / :meth:`supports_extract` 能力标志
    让注册表把每次工具调用路由到正确的 provider，也让多能力 provider
    （Firecrawl、Tavily、Exa 等）从单个类宣告多种能力。
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """稳定短标识，用于 ``web.search_backend`` / ``web.extract_backend`` /
        ``web.backend`` 配置键。小写、无空格，允许连字符。例：``ddgs``。"""

    @property
    def display_name(self) -> str:
        """人类可读标签。默认返回 ``name``。"""
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """返回本 provider 当前是否可服务调用。

        通常是廉价检查（环境变量存在、可选 Python 依赖可导入、实例 URL
        已设置）。**禁止**做网络调用——它在工具注册期和每次工具枚举时运行。
        """

    def supports_search(self) -> bool:
        """返回本 provider 是否实现了 :meth:`search`。"""
        return True

    def supports_extract(self) -> bool:
        """返回本 provider 是否实现了 :meth:`extract`。

        ``extract`` 的同步与异步实现都合法——派发方用
        :func:`inspect.iscoroutinefunction` 检测并按需 await。
        """
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """执行一次 web 搜索。

        :meth:`supports_search` 返回 True 时覆写。默认抛
        NotImplementedError；调用方应先按 :meth:`supports_search` 门控。
        """
        raise NotImplementedError(
            f"{self.name} does not support search (override supports_search)"
        )

    def extract(self, urls: List[str], **kwargs: Any) -> Any:
        """从一个或多个 URL 提取内容。

        :meth:`supports_extract` 返回 True 时覆写。默认抛
        NotImplementedError；调用方应先按 :meth:`supports_extract` 门控。

        返回形状：结果 dict 列表，与 legacy ``web_extract_tool`` 后处理
        管线兼容：``[{"url", "title", "content", "raw_content",
        "metadata"(可选), "error"(可选)}, ...]``。
        实现可以是 ``async def``——派发方检测协程并按需 await。
        """
        raise NotImplementedError(
            f"{self.name} does not support extract (override supports_extract)"
        )
