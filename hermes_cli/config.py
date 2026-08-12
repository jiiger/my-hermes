"""配置读取模块（hermes-cli 阉割版，约 150 行）。

只提供 my-hermes 需要的三个读取函数：
- ``load_config``：完整读取（默认值合并 + deepcopy，调用方可安全修改结果）；
- ``load_config_readonly``：无 deepcopy 的快路径（调用方只读，勿改返回值）；
- ``read_raw_config``：原始 YAML 读取（不合并默认值、不迁移）。

与原版 hermes_cli/config.py（5434 行）的阉割项：
- 砍掉配置迁移（config_migrations）、managed_scope / 托管检测、安装方式检测、
  ${VAR} 环境变量展开、last-known-good 回退、save_config 写路径；
- YAML 加载直接用 PyYAML 的 ``yaml.safe_load``（不追求 libyaml C 加速）；
- 缓存策略照原版：按配置文件 (mtime_ns, size) 键控，文件变化自动失效。

本模块是纯读取层，只依赖 hermes_constants + yaml，禁止 import run_agent /
agent.*（避免循环依赖）。
"""

import copy
import logging
import threading
from typing import Any, Dict, Optional, Tuple

import yaml

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_constants import get_config_path

logger = logging.getLogger(__name__)

# 缓存：str(配置路径) → (mtime_ns, size, 值)。键控策略与原版一致
# （原版 _LOAD_CONFIG_CACHE / _RAW_CONFIG_CACHE），配置文件被编辑后
# stat() 能看到新的 mtime_ns/size，缓存自动失效，无需显式失效钩子。
_LOAD_CONFIG_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
_RAW_CONFIG_CACHE: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}

# 串行化配置读取（并发调用时保护模块级缓存 dict；RLock 与原版一致）。
_CONFIG_LOCK = threading.RLock()


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 *override* 到 *base*，保留嵌套默认值（原版 :2435）。

    用户只覆盖某棵子树里的一个键时，其它默认键保持原样；override 中值为
    None 且 base 同键为 dict 时视为"未设置"跳过（对应 YAML 空段 ``agent:``）。
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif key in result and isinstance(result[key], dict) and value is None:
            continue
        else:
            result[key] = value
    return result


def read_raw_config() -> Dict[str, Any]:
    """读取 config.yaml 原文，不合并默认值、不迁移（原版 :2933）。

    文件缺失 / 解析失败返回 ``{}``（原版 :2950-2951 明确 FileNotFoundError
    → {}）。缓存按 (mtime_ns, size) 键控；每次调用返回 deepcopy（调用方可能
    修改结果后再写回，语义与原版一致）。
    """
    with _CONFIG_LOCK:
        try:
            config_path = get_config_path()
            st = config_path.stat()
            cache_key = (st.st_mtime_ns, st.st_size)
        except (FileNotFoundError, OSError):
            return {}

        path_key = str(config_path)
        cached = _RAW_CONFIG_CACHE.get(path_key)
        if cached is not None and cached[:2] == cache_key:
            return copy.deepcopy(cached[2])

        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.debug("Could not parse config %s: %s", config_path, e)
            return {}

        if not isinstance(data, dict):
            data = {}
        _RAW_CONFIG_CACHE[path_key] = (cache_key[0], cache_key[1], copy.deepcopy(data))
        return data


def _load_config_impl(*, want_deepcopy: bool) -> Dict[str, Any]:
    """load_config / load_config_readonly 的公共实现（原版 :3283 的阉割版）。

    - 文件缺失：返回 DEFAULT_CONFIG 的深拷贝（"用默认值"，与
      read_raw_config 的 {} 语义区分）；
    - 文件存在：读 YAML 后与 DEFAULT_CONFIG 深合并；
    - 缓存命中 + want_deepcopy：返回缓存值的 deepcopy（调用方可随意修改）；
    - 缓存命中 + not want_deepcopy：直接返回缓存 dict（调用方只读）。
    """
    with _CONFIG_LOCK:
        config_path = get_config_path()
        path_key = str(config_path)

        try:
            st = config_path.stat()
            cache_sig = (st.st_mtime_ns, st.st_size)
        except (FileNotFoundError, OSError):
            cache_sig = None

        cached = _LOAD_CONFIG_CACHE.get(path_key)
        if cached is not None and cache_sig is not None and cached[:2] == cache_sig:
            return copy.deepcopy(cached[2]) if want_deepcopy else cached[2]

        config = copy.deepcopy(DEFAULT_CONFIG)
        if cache_sig is not None:
            try:
                with open(config_path, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                if isinstance(user_config, dict):
                    config = _deep_merge(config, user_config)
            except Exception as e:
                logger.debug("Could not read config %s: %s", config_path, e)

        if cache_sig is not None:
            # 缓存一份独立 deepcopy：load_config() 的调用方随便改，readonly
            # 调用方始终看到同一份稳定对象。
            cached_copy = copy.deepcopy(config)
            _LOAD_CONFIG_CACHE[path_key] = (*cache_sig, cached_copy)
            if not want_deepcopy:
                return cached_copy
        else:
            _LOAD_CONFIG_CACHE.pop(path_key, None)
        return config


def load_config() -> Dict[str, Any]:
    """完整读取配置（默认值合并 + deepcopy）。签名与原版 :3115 一致。

    每次返回 deepcopy——调用方通常会修改结果（如
    ``cfg["agent"]["..."] = ...``），修改不会污染进程内缓存。
    """
    return _load_config_impl(want_deepcopy=True)


def load_config_readonly() -> Dict[str, Any]:
    """无 deepcopy 的快路径，供只读调用方使用。签名与原版 :3132 一致。

    直接返回缓存 dict，**修改返回值会污染进程内缓存**——只读场景才用它
    （prompt 构建、超时读取等热路径）。
    """
    return _load_config_impl(want_deepcopy=False)


def cfg_get(cfg: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    """安全遍历嵌套 dict 键，任何 miss 都返回 ``default``。

    对应原版 hermes_cli/config.py:2886 cfg_get。统一处理三个常见坑：

      1. 缺中间键（返回 default，无 KeyError）；
      2. 中间值不是 dict（如用户把段写成了字符串）——返回 default，
         而不是在 .get() 上 AttributeError；
      3. cfg 为 None（调用方有时传 ``load_config() or None``）。

    显式 None 值原样返回（与 ``dict.get(key, default)`` 语义一致——
    default 只在键**缺失**时返回，键存在但值为 None 时返回 None）。

    例子：:
        >>> cfg_get({"agent": {"reasoning_effort": "high"}}, "agent", "reasoning_effort")
        'high'
        >>> cfg_get({}, "agent", "reasoning_effort", default="medium")
        'medium'
        >>> cfg_get({"agent": "oops_a_string"}, "agent", "reasoning_effort", default="low")
        'low'
        >>> cfg_get({"a": {"b": None}}, "a", "b", default="def")  # 显式 None 保留
        >>> cfg_get({"a": {"b": False}}, "a", "b", default=True)  # 假值保留
        False
    """
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict):
            return default
        if key not in node:
            return default
        node = node[key]
    return node
