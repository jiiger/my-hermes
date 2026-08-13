"""Responses API（/v1/responses）格式转换与归一化适配器。

纯格式转换与归一化逻辑，供 OpenAI Responses API（Codex / xAI / GitHub
Models 等兼容端点）使用。从 run_agent.py 抽出，隔离 Responses 特有逻辑
与核心 agent 循环；所有函数无状态，只对传入数据做转换并返回结果。"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from agent.message_sanitization import deterministic_call_id
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

logger = logging.getLogger(__name__)


def _classify_responses_issuer(
    *,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
    is_codex_backend: bool = False,
    base_url: Optional[str] = None,
) -> str:
    """给 Responses 端点生成稳定标识（issuer），供推理项盖章与重放过滤。"""
    if is_xai_responses:
        return "xai_responses"
    if is_github_responses:
        return "github_responses"
    if is_codex_backend:
        return "codex_backend"
    if base_url:
        return f"other:{base_url}"
    return "other"


# 限制跨签发方跳过的每进程告警：长历史含多个过期 issuer 推理块时避免刷日志。
_CROSS_ISSUER_WARN_EMITTED = False


# 匹配 Codex/Harmony 工具调用序列化偶尔泄漏进 assistant 内容的情况
# （模型没发结构化 function_call 项时）。常见形式：
#
#   to=functions.exec_command
#   assistant to=functions.exec_command
#   <|channel|>commentary to=functions.exec_command
#
# ``to=functions.<name>`` 是稳定标记——可选的 ``assistant`` 或 Harmony 通道
# 前缀因退化模式而异。大小写不敏感，以覆盖大小写变体。
_TOOL_CALL_LEAK_PATTERN = re.compile(
    r"(?:^|[\s>|])to=functions\.[A-Za-z_][\w.]*",
    re.IGNORECASE,
)


# ChatGPT Codex 后端保留这些 Harmony wire token。任何地方重放字面拼写，
# 后端会在推理前拒绝（``invalid_prompt: Request blocked.``）。
# Category-Cf 处理覆盖早期 U+200B 弱化持久化会话；全角竖线在格式字符
# 剥离后存活，同时保持源码可读。
_HARMONY_CONTROL_TOKEN_RE = re.compile(
    r"<\|(start|end|channel|message|constrain|return|call)\|>"
)
_FULLWIDTH_PIPE = "\uff5c"


def _neutralize_harmony_tokens(text: str) -> str:
    """中和 Codex/Harmony 保留字面量：把 <|start|end|...|> 替换为全角竖线变体，
    保持源码可读又不触发后端保留 token 拦截（invalid_prompt: Request blocked.）。"""
    if not text or "<" not in text or "|" not in text:
        return text

    replacement = rf"<{_FULLWIDTH_PIPE}\1{_FULLWIDTH_PIPE}>"
    if not any(unicodedata.category(char) == "Cf" for char in text):
        return _HARMONY_CONTROL_TOKEN_RE.sub(replacement, text)

# 已确认 Codex 后端在其保留 token 检查前剥离 U+200B。把所有 Unicode 格式
# 控制符等价处理，避免把字符移到 token 其他位置（或换成别的 Cf）
# 重新构造同款隐藏形态。
    visible_chars: List[str] = []
    original_positions: List[int] = []
    for index, char in enumerate(text):
        if unicodedata.category(char) == "Cf":
            continue
        visible_chars.append(char)
        original_positions.append(index)

    visible_text = "".join(visible_chars)
    matches = list(_HARMONY_CONTROL_TOKEN_RE.finditer(visible_text))
    if not matches:
        return text

    result: List[str] = []
    original_cursor = 0
    for match in matches:
        original_start = original_positions[match.start()]
        original_end = original_positions[match.end() - 1] + 1
        result.append(text[original_cursor:original_start])
        result.append(f"<{_FULLWIDTH_PIPE}{match.group(1)}{_FULLWIDTH_PIPE}>")
        original_cursor = original_end
    result.append(text[original_cursor:])
    return "".join(result)


def _neutralize_harmony_structure(value: Any) -> Any:
    """递归中和 JSON 值里的 Harmony 保留 token；对象键含保留 token 时直接报错
    （改写键会破坏工具 schema 与执行器的契约，不能静默改）。"""
    if isinstance(value, str):
        return _neutralize_harmony_tokens(value)
    if isinstance(value, (list, tuple)):
        return [_neutralize_harmony_structure(item) for item in value]
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if isinstance(key, str) and _neutralize_harmony_tokens(key) != key:
                raise ValueError(
                    "Reserved Harmony tokens in a JSON object key cannot be "
                    "neutralized without changing its contract."
                )
            normalized[key] = _neutralize_harmony_structure(item)
        return normalized
    return value


# ---------------------------------------------------------------------------
# 多模态内容辅助
# ---------------------------------------------------------------------------

def _chat_content_to_responses_parts(content: Any, *, role: str = "user") -> List[Dict[str, Any]]:
    """把 chat 多模态 content 转成 Responses 的 input parts。
    
    输入 [{"type":"text"|"image_url",...}]（OpenAI Chat 格式），输出
    [{"type":"input_text"|"output_text"|"input_image",...}]（Responses 格式）。
    role 决定文本类型：user→input_text，assistant→output_text（Responses 拒绝
    assistant 里的 input_text 和 user 里的 output_text，调用方必须传对 role）。
    无识别 part 时返回空列表，调用方回退到字符串路径。"""
    text_type = "output_text" if role == "assistant" else "input_text"
    if not isinstance(content, list):
        return []
    converted: List[Dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                converted.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = str(part.get("type") or "").strip().lower()
        if ptype in {"text", "input_text", "output_text"}:
            text = part.get("text")
            if isinstance(text, str) and text:
                converted.append({"type": text_type, "text": text})
            continue
        if ptype in {"image_url", "input_image"}:
            image_ref = part.get("image_url")
            detail = part.get("detail")
            if isinstance(image_ref, dict):
                url = image_ref.get("url")
                detail = image_ref.get("detail", detail)
            else:
                url = image_ref
            if not isinstance(url, str) or not url:
                continue
            image_part: Dict[str, Any] = {"type": "input_image", "image_url": url}
            if isinstance(detail, str) and detail.strip():
                image_part["detail"] = detail.strip()
            converted.append(image_part)
    return converted


def _summarize_user_message_for_log(content: Any, *, sep: str = " ") -> str:
    """把消息 content 压成纯文本摘要。
    
    多模态消息是 [{type:text|image_url,...}] 列表，多个消费方要纯字符串：
    - 日志/预览/轨迹文件（默认 sep=" "）；
    - 外部记忆 provider（传给正则/文本 API，列表会崩，用 sep="\n"）。
    文本 part 用 sep 拼接；图片变成 [N image(s)] 占位。空列表返回空串，
    意外标量返回 str(content)。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_bits: List[str] = []
        image_count = 0
        for part in content:
            if isinstance(part, str):
                if part:
                    text_bits.append(part)
                continue
            if not isinstance(part, dict):
                continue
            ptype = str(part.get("type") or "").strip().lower()
            if ptype in {"text", "input_text", "output_text"}:
                text = part.get("text")
                if isinstance(text, str) and text:
                    text_bits.append(text)
            elif ptype in {"image_url", "input_image"}:
                image_count += 1
        summary = sep.join(text_bits).strip()
        if image_count:
            note = f"[{image_count} image{'s' if image_count != 1 else ''}]"
            summary = f"{note} {summary}" if summary else note
        return summary
    try:
        return str(content)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# ID 辅助
# ---------------------------------------------------------------------------

def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """根据工具调用内容生成确定性 call_id（策略唯一归属
    agent.message_sanitization.deterministic_call_id，本模块保留名字供
    run_agent/测试从这里导入）。确定性 ID 防止缓存失效——随机 UUID 会让
    每次 API 调用前缀唯一，破坏 OpenAI 提示缓存。"""
    return deterministic_call_id(fn_name, arguments, index)


def _clamp_responses_call_id(call_id: str) -> str:
    """把 call_id 压进 Responses API 的 64 字符上限（#73492）。
    
    codex app-server 给 MCP 工具 id 加命名空间前缀后容易超长，Responses API
    以不可重试的 400 拒绝整个载荷，重放每轮都炸、永久卡死会话。替代值是
    原值的纯确定性函数，function_call 与对应的 function_call_output 带同一
    原始 id，映射到同一替代值保持配对。短 id 原样通过，保留提示缓存前缀。"""
    if len(call_id) <= _MAX_RESPONSES_ITEM_ID_LENGTH:
        return call_id
    digest = hashlib.sha256(call_id.encode("utf-8", errors="replace")).hexdigest()[:32]
    return f"call_{digest}"


def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
    """把存储的工具 id 拆成 (call_id, response_item_id)。"""
    if not isinstance(raw_id, str):
        return None, None
    value = raw_id.strip()
    if not value:
        return None, None
    if "|" in value:
        call_id, response_item_id = value.split("|", 1)
        call_id = call_id.strip() or None
        response_item_id = response_item_id.strip() or None
        return call_id, response_item_id
    if value.startswith("fc_"):
        return None, value
    return value, None


def _derive_responses_function_call_id(
    call_id: str,
    response_item_id: Optional[str] = None,
) -> str:
    """构造合法的 Responses function_call.id（必须以 fc_ 开头）。"""
    if isinstance(response_item_id, str):
        candidate = response_item_id.strip()
        if candidate.startswith("fc_"):
            return candidate

    source = (call_id or "").strip()
    if source.startswith("fc_"):
        return source
    if source.startswith("call_") and len(source) > len("call_"):
        return f"fc_{source[len('call_'):]}"

    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
    if sanitized.startswith("fc_"):
        return sanitized
    if sanitized.startswith("call_") and len(sanitized) > len("call_"):
        return f"fc_{sanitized[len('call_'):]}"
    if sanitized:
        return f"fc_{sanitized[:48]}"

    seed = source or str(response_item_id or "") or uuid.uuid4().hex
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
    return f"fc_{digest}"


# ---------------------------------------------------------------------------
# Schema 转换
# ---------------------------------------------------------------------------

def _responses_tools(tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
    """把 chat-completions 工具 schema 转成 Responses function 工具 schema。"""
    if not tools:
        return None

    converted: List[Dict[str, Any]] = []
    for item in tools:
        fn = item.get("function", {}) if isinstance(item, dict) else {}
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description", ""),
            "strict": False,
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted or None


# provider 执行的"内置工具声明"类型，Responses tools 数组接受。只用 type
# 声明（无客户端 name/parameters schema），服务端运行——provider 拥有实现
# 并通过匹配的 ``*_call`` 输出项报告进度。Hermes 为 xAI 传输注入 xAI 原生
# web_search（见 agent/transports/codex.py）；其余列出以便预检放行而不是
# 当 "unsupported type" 拒绝。镜像 _normalize_codex_response 里的
# ``*_call`` 项类型集合。
_RESPONSES_BUILTIN_TOOL_TYPES = {
    "web_search",
    "web_search_preview",
    "file_search",
    "code_interpreter",
    "image_generation",
    "computer_use_preview",
    "local_shell",
}


# ---------------------------------------------------------------------------
# 消息格式转换
# ---------------------------------------------------------------------------

_RESPONSE_MESSAGE_STATUSES = {"completed", "incomplete", "in_progress"}

# Responses API 拒绝超过此长度的 input[].id（不可重试 400 "string too long"）。
# Codex 签发的 assistant 消息 id 是服务端分配的 base64 blob，可超 400 字符；
# Hermes 自造 id（msg_...）远低于上限、值得保留以获得前缀缓存命中。
# 重放时只丢弃超长的。
_MAX_RESPONSES_ITEM_ID_LENGTH = 64


def _normalize_responses_message_status(value: Any, *, default: str = "completed") -> str:
    """归一化 Responses assistant 消息的 status 用于重放。
    
    API 接受 completed/incomplete/in_progress；原样保留这些值（忽略大小写/
    连字符），避免未完成的 Codex continuation 回合被误标为完成。"""
    if isinstance(value, str):
        status = value.strip().lower().replace("-", "_").replace(" ", "_")
        if status in _RESPONSE_MESSAGE_STATUSES:
            return status
    return default


def _chat_messages_to_responses_input(
    messages: List[Dict[str, Any]],
    *,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
    replay_encrypted_reasoning: bool = True,
    current_issuer_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """把内部 chat 消息列表转成 Responses input items。
    
    - system 消息跳过（指令走 instructions）；user/assistant 转 content parts
      （复用 _chat_content_to_responses_parts）；assistant 的 tool_calls 转
      function_call 项；tool 消息转 function_call_output 项。
    - is_xai_responses：保留签名兼容，不再抑制加密推理重放（xAI 依赖跨轮
      回传加密推理保持思考连贯）。
    - is_github_responses：重放 message item 时丢弃 id（Copilot 后端把 id
      绑定到连接，轮换/重启后旧 id 会 401）。
    - replay_encrypted_reasoning：会话级重放加密推理总开关；False 时不携带
      任何加密推理项（兼容中继可能拒收旧 blob → 400）。
    - current_issuer_kind：按签发方过滤——加密推理 blob 只能被签发端点
      解密，重放历史里 issuer 与当前端点不一致的推理项直接丢弃
      （见 _classify_responses_issuer）。"""
    items: List[Dict[str, Any]] = []
    seen_item_ids: set = set()

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            continue

        if role in {"user", "assistant"}:
            content = msg.get("content", "")
            if isinstance(content, list):
                content_parts = _chat_content_to_responses_parts(content, role=role)
                text_type = "output_text" if role == "assistant" else "input_text"
                content_text = "".join(
                    p.get("text", "") for p in content_parts if p.get("type") == text_type
                )
            else:
                content_parts = []
                content_text = str(content) if content is not None else ""

            if role == "assistant":
# 从先前回合重放加密推理项，让 API 保持连贯推理链。
# 适用于所有 Responses 传输，包括 xAI——见 _chat_messages_to_responses_input
# docstring 里 2026 年 5 月对早期 xAI 门控的逆转说明。
                codex_reasoning = (
                    msg.get("codex_reasoning_items")
                    if replay_encrypted_reasoning
                    else None
                )
                has_codex_reasoning = False
                if isinstance(codex_reasoning, list):
                    for ri in codex_reasoning:
                        if isinstance(ri, dict) and ri.get("encrypted_content"):
                            item_id = ri.get("id")
                            if item_id and item_id in seen_item_ids:
                                continue
# 跨签发方保护：丢弃由不同 Responses 端点签发的推理块。
# 当前端点无法解密外来 encrypted_content，会以 400 invalid_encrypted_content
# 拒绝整个请求。无印章（旧）项放行。
                            item_issuer = ri.get("_issuer_kind")
                            if (
                                current_issuer_kind is not None
                                and item_issuer is not None
                                and item_issuer != current_issuer_kind
                            ):
                                global _CROSS_ISSUER_WARN_EMITTED
                                if not _CROSS_ISSUER_WARN_EMITTED:
                                    logger.warning(
                                        "Dropping reasoning item minted by %s while "
                                        "calling %s — encrypted_content is sealed to "
                                        "its issuer. This happens when a session "
                                        "switches model providers mid-conversation.",
                                        item_issuer, current_issuer_kind,
                                    )
                                    _CROSS_ISSUER_WARN_EMITTED = True
                                continue
# 去掉 "id" 字段——store=False 时 Responses API 无法按 ID 查项会返回 404。
# encrypted_content blob 自包含，足以维持推理链。
# 也去掉内部 "_issuer_kind" 印章；它是 Hermes 侧元数据键，
# 不属于 Responses API schema。
                            replay_item = {
                                k: v for k, v in ri.items()
                                if k not in ("id", "_issuer_kind")
                            }
                            items.append(replay_item)
                            if item_id:
                                seen_item_ids.add(item_id)
                            has_codex_reasoning = True

# 从先前回合重放精确的 assistant 消息项（带 id/phase）以维持前缀缓存命中。
# OpenAI 文档："保留并在所有 assistant 消息上重发 phase——丢弃会降低性能。"
                codex_message_items = msg.get("codex_message_items")
                replayed_message_items = 0
                if isinstance(codex_message_items, list):
                    for raw_item in codex_message_items:
                        if not isinstance(raw_item, dict):
                            continue
                        if raw_item.get("type") != "message" or raw_item.get("role") != "assistant":
                            continue
                        raw_content_parts = raw_item.get("content")
                        if not isinstance(raw_content_parts, list):
                            continue

                        normalized_content_parts = []
                        for part in raw_content_parts:
                            if not isinstance(part, dict):
                                continue
                            part_type = str(part.get("type") or "").strip()
                            if part_type not in {"output_text", "text"}:
                                continue
                            text = part.get("text", "")
                            if text is None:
                                text = ""
                            if not isinstance(text, str):
                                text = str(text)
                            normalized_content_parts.append({"type": "output_text", "text": text})

                        if not normalized_content_parts:
                            continue

                        replay_item = {
                            "type": "message",
                            "role": "assistant",
                            "status": _normalize_responses_message_status(raw_item.get("status")),
                            "content": normalized_content_parts,
                        }
                        item_id = raw_item.get("id")
                        if (
                            not is_github_responses
                            and isinstance(item_id, str)
                            and item_id.strip()
                        ):
                            stripped_id = item_id.strip()
                            if len(stripped_id) <= _MAX_RESPONSES_ITEM_ID_LENGTH:
                                replay_item["id"] = stripped_id
                        phase = raw_item.get("phase")
                        if isinstance(phase, str) and phase.strip():
                            replay_item["phase"] = phase.strip()
                        items.append(replay_item)
                        replayed_message_items += 1

                if replayed_message_items > 0:
                    pass
                elif content_parts:
                    items.append({"role": "assistant", "content": content_parts})
                elif content_text.strip():
                    items.append({"role": "assistant", "content": content_text})
                elif has_codex_reasoning:
# Responses API 要求每个 reasoning 项后有后续项（否则 missing_following_item 错误）。
# 当 assistant 只产出推理、没有可见内容时，发出空 assistant 消息作为必需的后续项。
                    items.append({"role": "assistant", "content": ""})

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function", {})
                        fn_name = fn.get("name")
                        if not isinstance(fn_name, str) or not fn_name.strip():
                            continue

                        embedded_call_id, embedded_response_item_id = _split_responses_tool_id(
                            tc.get("id")
                        )
                        call_id = tc.get("call_id")
                        if not isinstance(call_id, str) or not call_id.strip():
                            call_id = embedded_call_id
                        if not isinstance(call_id, str) or not call_id.strip():
                            if (
                                isinstance(embedded_response_item_id, str)
                                and embedded_response_item_id.startswith("fc_")
                                and len(embedded_response_item_id) > len("fc_")
                            ):
                                call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                            else:
                                _raw_args = str(fn.get("arguments", "{}"))
                                call_id = _deterministic_call_id(fn_name, _raw_args, len(items))
                        call_id = call_id.strip()

                        arguments = fn.get("arguments", "{}")
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments, ensure_ascii=False)
                        elif not isinstance(arguments, str):
                            arguments = str(arguments)
                        arguments = arguments.strip() or "{}"

                        items.append({
                            "type": "function_call",
                            "call_id": _clamp_responses_call_id(call_id),
                            "name": fn_name,
                            "arguments": arguments,
                        })
                continue

# 非 assistant（user）角色：有多模态 parts 时发出，否则回退文本载荷。
            if content_parts:
                items.append({"role": role, "content": content_parts})
            else:
                items.append({"role": role, "content": content_text})
            continue

        if role == "tool":
            raw_tool_call_id = msg.get("tool_call_id")
            call_id, _ = _split_responses_tool_id(raw_tool_call_id)
            if not isinstance(call_id, str) or not call_id.strip():
                if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                    call_id = raw_tool_call_id.strip()
            if not isinstance(call_id, str) or not call_id.strip():
                continue

# 多模态工具结果：把 OpenAI 风格 content 列表转成 Responses
# function_call_output.output 数组。Responses API 接受 output 为字符串
# 或 input_text/input_image 项数组。见
# https://developers.openai.com/api/reference/python/resources/responses/
            tool_content = msg.get("content")
            output_value: Any
            if isinstance(tool_content, list):
                converted = _chat_content_to_responses_parts(
                    tool_content, role="user",
                )
                if converted:
                    output_value = converted
                else:
                    output_value = ""
            else:
                output_value = str(tool_content or "")

            items.append({
                "type": "function_call_output",
                "call_id": _clamp_responses_call_id(call_id),
                "output": output_value,
            })

    return items


# ---------------------------------------------------------------------------
# 输入预检/校验
# ---------------------------------------------------------------------------

def _preflight_codex_input_items(
    raw_items: Any,
    *,
    is_github_responses: bool = False,
    sanitize_harmony_tokens: bool = False,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError("Codex Responses input must be a list of input items.")

    sanitize_text = (
        _neutralize_harmony_tokens
        if sanitize_harmony_tokens
        else lambda text: text
    )
    normalized: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"Codex Responses input[{idx}] must be an object.")

        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

            arguments = item.get("arguments", "{}")
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = str(arguments)
            arguments = sanitize_text(arguments.strip() or "{}")

            normalized.append(
                {
                    "type": "function_call",
                    "call_id": call_id.strip(),
                    "name": name.strip(),
                    "arguments": arguments,
                }
            )
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
            output = item.get("output", "")
            if output is None:
                output = ""
# output 可能是字符串或结构化内容项数组（input_text/input_image，
# 多模态工具结果）。两种形态 Responses API 都接受；有数组时保留数组形态。
            if isinstance(output, list):
# 校验每项是已识别内容形态；丢弃其他以避免 API 4xx。
                cleaned: List[Dict[str, Any]] = []
                for part in output:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "input_text":
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            cleaned.append({"type": "input_text", "text": sanitize_text(text)})
                    elif ptype == "input_image":
                        url = part.get("image_url")
                        if isinstance(url, str) and url:
                            entry: Dict[str, Any] = {"type": "input_image", "image_url": url}
                            detail = part.get("detail")
                            if isinstance(detail, str) and detail.strip():
                                entry["detail"] = detail.strip()
                            cleaned.append(entry)
                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": cleaned if cleaned else "",
                    }
                )
                continue
            if not isinstance(output, str):
                output = str(output)

            normalized.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id.strip(),
                    "output": sanitize_text(output),
                }
            )
            continue

        if item_type == "reasoning":
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id:
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)
                reasoning_item: Dict[str, Any] = {
                    "type": "reasoning",
                    "encrypted_content": encrypted,
                }
# 出站项不要包含 "id"——store=False（默认）时 API 尝试服务端解析 id 返回 404。
# id 仍用于上面 seen_ids 的本地去重。
                summary = item.get("summary")
                if isinstance(summary, list):
                    reasoning_item["summary"] = (
                        _neutralize_harmony_structure(summary)
                        if sanitize_harmony_tokens
                        else summary
                    )
                else:
                    reasoning_item["summary"] = []
                normalized.append(reasoning_item)
            continue

        if item_type == "compaction":
# 重放的原生服务端压缩检查点（gpt-5.6，直连 OpenAI/Codex 路由）。
# 不透明、签发方密封；只转发 API 定义的字段。
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                normalized.append(
                    {"type": "compaction", "encrypted_content": encrypted}
                )
            continue

        if item_type == "message":
            role = item.get("role")
            if role != "assistant":
                raise ValueError(f"Codex Responses input[{idx}] message items must have role='assistant'.")
            content = item.get("content")
            if not isinstance(content, list):
                raise ValueError(f"Codex Responses input[{idx}] message item must have content list.")
            normalized_content = []
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    raise ValueError(
                        f"Codex Responses input[{idx}] message content[{part_idx}] must be an object."
                    )
                part_type = part.get("type")
                if part_type not in {"output_text", "text"}:
                    raise ValueError(
                        f"Codex Responses input[{idx}] message content[{part_idx}] has unsupported type {part_type!r}."
                    )
                text = part.get("text", "")
                if text is None:
                    text = ""
                if not isinstance(text, str):
                    text = str(text)
                normalized_content.append({"type": "output_text", "text": sanitize_text(text)})
            if not normalized_content:
                raise ValueError(f"Codex Responses input[{idx}] message item must contain at least one text part.")
            normalized_item: Dict[str, Any] = {
                "type": "message",
                "role": "assistant",
                "status": _normalize_responses_message_status(item.get("status")),
                "content": normalized_content,
            }
            item_id = item.get("id")
            if (
                not is_github_responses
                and isinstance(item_id, str)
                and item_id.strip()
            ):
                stripped_id = item_id.strip()
                if len(stripped_id) <= _MAX_RESPONSES_ITEM_ID_LENGTH:
                    normalized_item["id"] = stripped_id
            phase = item.get("phase")
            if isinstance(phase, str) and phase.strip():
                normalized_item["phase"] = phase.strip()
            normalized.append(normalized_item)
            continue

        role = item.get("role")
        if role in {"user", "assistant"}:
            content = item.get("content", "")
            if content is None:
                content = ""
            if isinstance(content, list):
# _chat_messages_to_responses_input 产生的多模态 content 已是 Responses 格式
# （input_text/output_text/input_image）。校验每个 part 并透传。
# 用与角色匹配的正确文本类型——assistant 消息用 output_text，user 消息用 input_text。
                text_type = "output_text" if role == "assistant" else "input_text"
                validated: List[Dict[str, Any]] = []
                for part_idx, part in enumerate(content):
                    if isinstance(part, str):
                        if part:
                            validated.append({"type": text_type, "text": sanitize_text(part)})
                        continue
                    if not isinstance(part, dict):
                        raise ValueError(
                            f"Codex Responses input[{idx}].content[{part_idx}] must be an object or string."
                        )
                    ptype = str(part.get("type") or "").strip().lower()
                    if ptype in {"input_text", "text", "output_text"}:
                        text = part.get("text", "")
                        if not isinstance(text, str):
                            text = str(text or "")
                        validated.append({"type": text_type, "text": sanitize_text(text)})
                    elif ptype in {"input_image", "image_url"}:
                        image_ref = part.get("image_url", "")
                        detail = part.get("detail")
                        if isinstance(image_ref, dict):
                            url = image_ref.get("url", "")
                            detail = image_ref.get("detail", detail)
                        else:
                            url = image_ref
                        if not isinstance(url, str):
                            url = str(url or "")
                        image_part: Dict[str, Any] = {"type": "input_image", "image_url": url}
                        if isinstance(detail, str) and detail.strip():
                            image_part["detail"] = detail.strip()
                        validated.append(image_part)
                    else:
                        raise ValueError(
                            f"Codex Responses input[{idx}].content[{part_idx}] has unsupported type {part.get('type')!r}."
                        )
                normalized.append({"role": role, "content": validated})
                continue
            if not isinstance(content, str):
                content = str(content)

            normalized.append({"role": role, "content": sanitize_text(content)})
            continue

        raise ValueError(
            f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
        )

    return normalized


def _preflight_codex_api_kwargs(
    api_kwargs: Any,
    *,
    allow_stream: bool = False,
    is_github_responses: bool = False,
    sanitize_harmony_tokens: bool = False,
) -> Dict[str, Any]:
    if not isinstance(api_kwargs, dict):
        raise ValueError("Codex Responses request must be a dict.")

    required = {"model", "instructions", "input"}
    missing = [key for key in required if key not in api_kwargs]
    if missing:
        raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

    model = api_kwargs.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Codex Responses request 'model' must be a non-empty string.")
    model = model.strip()

    instructions = api_kwargs.get("instructions")
    if instructions is None:
        instructions = ""
    if not isinstance(instructions, str):
        instructions = str(instructions)
    instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY
    if sanitize_harmony_tokens:
        instructions = _neutralize_harmony_tokens(instructions)

    normalized_input = _preflight_codex_input_items(
        api_kwargs.get("input"),
        is_github_responses=is_github_responses,
        sanitize_harmony_tokens=sanitize_harmony_tokens,
    )

    tools = api_kwargs.get("tools")
    normalized_tools = None
    if tools is not None:
        if not isinstance(tools, list):
            raise ValueError("Codex Responses request 'tools' must be a list when provided.")
        normalized_tools = []
        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"Codex Responses tools[{idx}] must be an object.")

            tool_type = tool.get("type")

# provider 执行的内置工具（xAI 原生 web_search、code interpreter 等）只用 type
# 声明，无 name/parameters schema——provider 拥有实现。原样透传而不是强制走
# 下面的函数工具校验（否则被 "unsupported type" 拒绝）。
# xAI 原生 web_search 的注入点见 agent/transports/codex.py。
            if tool_type in _RESPONSES_BUILTIN_TOOL_TYPES:
                normalized_tools.append(dict(tool))
                continue

            if tool_type != "function":
                raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

            name = tool.get("name")
            parameters = tool.get("parameters")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
            if not isinstance(parameters, dict):
                raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

            description = tool.get("description", "")
            if description is None:
                description = ""
            if not isinstance(description, str):
                description = str(description)

            strict = tool.get("strict", False)
            if not isinstance(strict, bool):
                strict = bool(strict)

            normalized_tools.append(
                {
                    "type": "function",
                    "name": name.strip(),
                    "description": description,
                    "strict": strict,
                    "parameters": parameters,
                }
            )

    if sanitize_harmony_tokens and normalized_tools is not None:
        normalized_tools = _neutralize_harmony_structure(normalized_tools)

    store = api_kwargs.get("store", False)
    if store is not False:
        raise ValueError("Codex Responses contract requires 'store' to be false.")

    allowed_keys = {
        "model", "instructions", "input", "tools", "store",
        "reasoning", "include", "max_output_tokens", "temperature",
        "tool_choice", "parallel_tool_calls", "prompt_cache_key",
        "prompt_cache_retention", "service_tier", "context_management",
        "extra_headers", "extra_body", "timeout",
    }
    normalized: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": normalized_input,
        "store": False,
    }
    if normalized_tools is not None:
        normalized["tools"] = normalized_tools

# 透传 reasoning 配置
    reasoning = api_kwargs.get("reasoning")
    if isinstance(reasoning, dict):
        normalized["reasoning"] = reasoning
    include = api_kwargs.get("include")
    if isinstance(include, list):
        normalized["include"] = include
    service_tier = api_kwargs.get("service_tier")
    if isinstance(service_tier, str) and service_tier.strip():
        normalized["service_tier"] = service_tier.strip()

# 透传 max_output_tokens 和 temperature
    max_output_tokens = api_kwargs.get("max_output_tokens")
    if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
        normalized["max_output_tokens"] = int(max_output_tokens)
    timeout = api_kwargs.get("timeout")
    if (
        isinstance(timeout, (int, float))
        and not isinstance(timeout, bool)
        and 0 < float(timeout) < float("inf")
    ):
        normalized["timeout"] = float(timeout)
    temperature = api_kwargs.get("temperature")
    if isinstance(temperature, (int, float)):
        normalized["temperature"] = float(temperature)

# 透传缓存路由/保留与工具派发提示。
    for passthrough_key in (
        "tool_choice",
        "parallel_tool_calls",
        "prompt_cache_key",
        "prompt_cache_retention",
    ):
        val = api_kwargs.get(passthrough_key)
        if val is not None:
            normalized[passthrough_key] = val

# 原生服务端压缩指令（gpt-5.6 直连 OpenAI/Codex 路由——资格已在上游
# agent/native_compaction.py 解析；预检只保留形态）。
    context_management = api_kwargs.get("context_management")
    if isinstance(context_management, list) and context_management:
        normalized["context_management"] = context_management

    extra_headers = api_kwargs.get("extra_headers")
    if extra_headers is not None:
        if not isinstance(extra_headers, dict):
            raise ValueError("Codex Responses request 'extra_headers' must be an object.")
        normalized_headers: Dict[str, str] = {}
        for key, value in extra_headers.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Codex Responses request 'extra_headers' keys must be non-empty strings.")
            if value is None:
                continue
            normalized_headers[key.strip()] = str(value)
        if normalized_headers:
            normalized["extra_headers"] = normalized_headers

    extra_body = api_kwargs.get("extra_body")
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise ValueError("Codex Responses request 'extra_body' must be an object.")
# 原样透传 extra_body——xAI Responses 用它承载 body 级 prompt_cache_key 字段
# （/v1/responses 文档化的缓存路由表面）。openai SDK 把 extra_body 序列化进
# JSON body 且不做逐字段类型检查，所以它能挺过 Responses.stream() kwarg
# 签名变化（否则上线路前就 TypeError）。
        if extra_body:
            normalized["extra_body"] = dict(extra_body)

    if allow_stream:
        stream = api_kwargs.get("stream")
        if stream is not None and stream is not True:
            raise ValueError("Codex Responses 'stream' must be true when set.")
        if stream is True:
            normalized["stream"] = True
        allowed_keys.add("stream")
    elif "stream" in api_kwargs:
        raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

# xAI Responses 安全网净化（#28490）：对 chat_completion_helpers 和
# auxiliary_client 在请求构建时执行的同一斜杠枚举剥离做纵深防御。
# 未来代码路径忘了先净化再调用，这里兜住绕过，避免 xAI 以
# "Invalid arguments passed to the model" 400
# （MCP 工具 schema 里的 HuggingFace id 如 Qwen/Qwen3.5-0.8B）。
    #
# 按模型名模式门控，因为原生 Codex（OpenAI）接受含斜杠枚举值——
# 在那里剥离会静默弱化工具 schema 约束。xAI 是唯一拒绝该形态的
# Responses API 表面。
    model_name_for_provider_check = str(api_kwargs.get("model") or "").lower()
    is_xai_model = model_name_for_provider_check.startswith(("grok-", "x-ai/grok-"))
    if is_xai_model and normalized.get("tools"):
        try:
            from tools.schema_sanitizer import strip_slash_enum
            normalized["tools"], _ = strip_slash_enum(normalized["tools"])
        except Exception:
            pass  # Best-effort — the caller-level sanitization should have handled it

    unexpected = sorted(key for key in api_kwargs if key not in allowed_keys)
    if unexpected:
        raise ValueError(
            f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
        )

    return normalized


# ---------------------------------------------------------------------------
# 响应提取辅助
# ---------------------------------------------------------------------------

def _extract_responses_message_text(item: Any) -> str:
    """从 Responses 的 message 输出项里提取 assistant 文本。
    
    只收集 output_text/text 类型 part 的文本并拼接；其他类型（图片/工具
    等）忽略。"""
    content = getattr(item, "content", None)
    if not isinstance(content, list):
        return ""

    chunks: List[str] = []
    for part in content:
        ptype = getattr(part, "type", None)
        if ptype not in {"output_text", "text"}:
            continue
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)
    return "".join(chunks).strip()


def _extract_responses_reasoning_text(item: Any) -> str:
    """从 Responses 的 reasoning 输出项里提取推理文本（summary part）。"""
    summary = getattr(item, "summary", None)
    if isinstance(summary, list):
        chunks: List[str] = []
        for part in summary:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()
    text = getattr(item, "text", None)
    if isinstance(text, str) and text:
        return text.strip()
    return ""


def _format_responses_error(error_obj: Any, response_status: str) -> str:
    """把 Responses 错误对象格式化成可读错误串（供上层日志/报错）。"""
# 从 dict 或属性风格载荷取 code 和 message。
    code: Any = None
    message: Any = None
    if isinstance(error_obj, dict):
        code = error_obj.get("code")
        message = error_obj.get("message")
    elif error_obj is not None:
        code = getattr(error_obj, "code", None)
        message = getattr(error_obj, "message", None)

    code_str = str(code).strip() if isinstance(code, str) else (str(code).strip() if code else "")
    message_str = str(message).strip() if isinstance(message, str) else (str(message).strip() if message else "")

    if code_str and message_str:
        return f"{code_str}: {message_str}"
    if message_str:
        return message_str
    if code_str:
        return code_str
    if error_obj:
# 兜底：把 provider 发的任何东西字符串化，至少日志/UI 可见，而不是静默吞掉。
        return str(error_obj)
    return f"Responses API returned status '{response_status}'"


# ---------------------------------------------------------------------------
# 完整响应归一化
# ---------------------------------------------------------------------------

def _normalize_codex_response(
    response: Any,
    *,
    issuer_kind: Optional[str] = None,
) -> tuple[Any, str]:
    """把 Responses API 响应对象归一化成 assistant_message-like 对象 + finish_reason。
    
    返回 (assistant_message, finish_reason)：assistant_message 带 content /
    tool_calls / reasoning / codex_reasoning_items 等字段，主循环直接消费。
    处理 message/reasoning/function_call/custom_tool_call/compaction 各类
    输出项、状态机判定 finish_reason（stop/tool_calls/incomplete/...）、
    工具调用泄漏恢复与 xAI reasoning 通道答案抢救。"""
    response_status = getattr(response, "status", None)
    if isinstance(response_status, str):
        response_status = response_status.strip().lower()
    else:
        response_status = None

    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = ""
    if isinstance(incomplete_details, dict):
        incomplete_reason = str(incomplete_details.get("reason") or "").strip().lower()
    elif incomplete_details is not None:
        incomplete_reason = str(getattr(incomplete_details, "reason", "") or "").strip().lower()
    response_incomplete_content_filter = (
        response_status == "incomplete" and incomplete_reason == "content_filter"
    )

    output = getattr(response, "output", None)
    if not isinstance(output, list) or not output:
# Codex 后端可能在答案完全经流式事件交付时返回空 output。
# 发 RuntimeError 前把 output_text 作为最后兜底检查。
        out_text = getattr(response, "output_text", None)
        if isinstance(out_text, str) and out_text.strip():
            logger.debug(
                "Codex response has empty output but output_text is present (%d chars); "
                "synthesizing output item.", len(out_text.strip()),
            )
            output = [SimpleNamespace(
                type="message", role="assistant", status="completed",
                content=[SimpleNamespace(type="output_text", text=out_text.strip())],
            )]
            response.output = output
        elif response_incomplete_content_filter:
# 这是确定性的 provider 安全拦截，不是部分答案。合成空消息让下方
# finish_reason 变为 content_filter，对话循环能回退/展示它，
# 而不是烧掉三次续跑尝试。
            output = [SimpleNamespace(
                type="message", role="assistant", status="completed", content=[]
            )]
            response.output = output
        else:
            raise RuntimeError("Responses API returned no output items")

    if response_status in {"failed", "cancelled"}:
        error_obj = getattr(response, "error", None)
        error_msg = _format_responses_error(error_obj, response_status)
        raise RuntimeError(error_msg)

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    reasoning_items_raw: List[Dict[str, Any]] = []
    message_items_raw: List[Dict[str, Any]] = []
    tool_calls: List[Any] = []
    has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
    saw_streaming_or_item_incomplete = response_status in {"queued", "in_progress"}
    saw_commentary_phase = False
    saw_final_answer_phase = False
    saw_reasoning_item = False

# 服务端内置工具调用（xAI 原生 web_search、code interpreter 等）由 provider 执行，
# 报告为离散 ``*_call`` 输出项。xAI /v1/responses 表面（如 SuperGrok OAuth 的
# grok-composer-2.5-fast）经常把这些项留在 status="in_progress"，即使整体
# response.status == "completed"——搜索在服务端跑完了，只是单项状态没同步。
# 这些不是模型回合未完成的信号，绝不能翻转 has_incomplete_items。
# 只有响应级状态和真正的模型输出项（message/reasoning/function_call）
# 决定 incomplete 判定。没有这个保护，任何 grok-composer 调用服务端搜索的
# 回合都会被误判 finish_reason="incomplete"，烧掉 3 次徒劳续跑后报
# "Codex response remained incomplete after 3 continuation attempts"。
# 客户端函数/自定义工具调用保留自己的 in_progress 处理（下面跳过，不等待）。
    _SERVER_SIDE_TOOL_CALL_TYPES = {
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
        "image_generation_call",
        "computer_call",
        "local_shell_call",
        "mcp_call",
    }

    for item in output:
        item_type = getattr(item, "type", None)
        item_status = getattr(item, "status", None)
        if isinstance(item_status, str):
            item_status = item_status.strip().lower()
        else:
            item_status = None

        if (
            item_status in {"queued", "in_progress", "incomplete"}
            and item_type not in _SERVER_SIDE_TOOL_CALL_TYPES
        ):
            has_incomplete_items = True
            saw_streaming_or_item_incomplete = True

        if item_type == "message":
            item_phase = getattr(item, "phase", None)
            normalized_phase = None
            is_commentary_phase = False
            if isinstance(item_phase, str):
                normalized_phase = item_phase.strip().lower()
                if normalized_phase in {"commentary", "analysis"}:
                    saw_commentary_phase = True
                    is_commentary_phase = True
                elif normalized_phase in {"final_answer", "final"}:
                    saw_final_answer_phase = True
            message_text = _extract_responses_message_text(item)
            if message_text:
# Responses 的 commentary/analysis 阶段文本是回合中的序言/进度叙述，
# 绝不是回合最终答案（Codex CLI 从最后消息提取中排除它；issues #24933/#41293）。
# 别放进 assistant content 以免被拼进或泄漏成最终回复，
# 但通过 reasoning 通道展示，让 CLI/gateway 像思考文本一样显示。
# 精确 message 项仍保留用于重放/缓存连贯。
                if is_commentary_phase:
                    reasoning_parts.append(message_text)
                else:
                    content_parts.append(message_text)
                raw_message_item: Dict[str, Any] = {
                    "type": "message",
                    "role": "assistant",
                    "status": _normalize_responses_message_status(item_status),
                    "content": [{"type": "output_text", "text": message_text}],
                }
                item_id = getattr(item, "id", None)
                if isinstance(item_id, str) and item_id:
                    raw_message_item["id"] = item_id
                if normalized_phase:
                    raw_message_item["phase"] = normalized_phase
                message_items_raw.append(raw_message_item)
        elif item_type == "reasoning":
            saw_reasoning_item = True
            reasoning_text = _extract_responses_reasoning_text(item)
            if reasoning_text:
                reasoning_parts.append(reasoning_text)
# 捕获完整 reasoning 项以维持多轮连贯。encrypted_content 是不透明 blob，
# 后续回合 API 需要它回来维持连贯推理链。
            encrypted = getattr(item, "encrypted_content", None)
            if isinstance(encrypted, str) and encrypted:
                raw_item = {"type": "reasoning", "encrypted_content": encrypted}
# 盖签发方印章，让未来回合能检测模型切换是否把会话移到无法解密此 blob 的端点
# ——见 _chat_messages_to_responses_input 的跨签发方保护。
                if issuer_kind:
                    raw_item["_issuer_kind"] = issuer_kind
                item_id = getattr(item, "id", None)
                if isinstance(item_id, str) and item_id.startswith("rs_tmp_"):
                    logger.debug(
                        "Skipping transient Codex reasoning item during normalization: %s",
                        item_id,
                    )
                    continue
                if isinstance(item_id, str) and item_id:
                    raw_item["id"] = item_id
# 捕获 summary——重放 reasoning 项时 API 需要
                summary = getattr(item, "summary", None)
                if isinstance(summary, list):
                    raw_summary = []
                    for part in summary:
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            raw_summary.append({"type": "summary_text", "text": text})
                    raw_item["summary"] = raw_summary
                reasoning_items_raw.append(raw_item)
        elif item_type == "compaction":
# 原生服务端压缩检查点（gpt-5.6 直连 OpenAI/Codex 路由）。加密 blob 在
# 后续请求中代表被剪枝的旧上下文。它搭 codex_reasoning_items sidecar
# 顺带获得持久化（state.db）、会话重放、跨签发方保护与
# invalid-encrypted-content 总开关，无需新状态。
            encrypted = getattr(item, "encrypted_content", None)
            if isinstance(encrypted, str) and encrypted:
                raw_item = {"type": "compaction", "encrypted_content": encrypted}
                if issuer_kind:
                    raw_item["_issuer_kind"] = issuer_kind
                reasoning_items_raw.append(raw_item)
                logger.info(
                    "Native Responses compaction item captured (%d chars encrypted).",
                    len(encrypted),
                )
        elif item_type == "function_call":
            if item_status in {"queued", "in_progress", "incomplete"}:
                continue
            fn_name = getattr(item, "name", "") or ""
            arguments = getattr(item, "arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            raw_call_id = getattr(item, "call_id", None)
            raw_item_id = getattr(item, "id", None)
            embedded_call_id, _ = _split_responses_tool_id(raw_item_id)
            call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = _deterministic_call_id(fn_name, arguments, len(tool_calls))
            call_id = call_id.strip()
            response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
            response_item_id = _derive_responses_function_call_id(call_id, response_item_id)
            tool_calls.append(SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=response_item_id,
                type="function",
                function=SimpleNamespace(name=fn_name, arguments=arguments),
            ))
        elif item_type == "custom_tool_call":
            fn_name = getattr(item, "name", "") or ""
            arguments = getattr(item, "input", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            raw_call_id = getattr(item, "call_id", None)
            raw_item_id = getattr(item, "id", None)
            embedded_call_id, _ = _split_responses_tool_id(raw_item_id)
            call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
            if not isinstance(call_id, str) or not call_id.strip():
                call_id = _deterministic_call_id(fn_name, arguments, len(tool_calls))
            call_id = call_id.strip()
            response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
            response_item_id = _derive_responses_function_call_id(call_id, response_item_id)
            tool_calls.append(SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=response_item_id,
                type="function",
                function=SimpleNamespace(name=fn_name, arguments=arguments),
            ))

    final_text = "\n".join([p for p in content_parts if p]).strip()
    if (
        not final_text
        and hasattr(response, "output_text")
        and not (saw_commentary_phase and not saw_final_answer_phase)
    ):
        out_text = getattr(response, "output_text", "")
        if isinstance(out_text, str):
            final_text = out_text.strip()

# ── 工具调用泄漏恢复 ──────────────────────────────────
# gpt-5.x 在 Codex Responses API 上偶尔退化，把本该是结构化 function_call
# 项的内容以纯 assistant 文本发出（Harmony/Codex 序列化 ``to=functions.foo
# {json}`` 或 ``assistant to=functions.foo {json}``）。模型本意调用工具，
# 但意图从未进入 ``response.output`` 的 function_call 项，所以这里
# tool_calls 为空。直接透传的话，父层看到自信的摘要却无审计轨迹
# （空 tool_trace）、实际没跑任何工具——台湾使馆邮件事件。
    #
# 检测：泄漏 token 总含 ``to=functions.<name>`` 且 assistant 消息无真实工具调用。
# 视为 incomplete，让现有 Codex-incomplete 续跑路径（3 次重试，
# run_agent.py 处理）有机会重新引出正确的 ``function_call`` 项。
# 现有循环已处理消息追加、去重与重试预算。
    leaked_tool_call_text = False
    if final_text and not tool_calls and _TOOL_CALL_LEAK_PATTERN.search(final_text):
        leaked_tool_call_text = True
        logger.warning(
            "Codex response contains leaked tool-call text in assistant content "
            "(no structured function_call items). Treating as incomplete so the "
            "continuation path can re-elicit a proper tool call. Leaked snippet: %r",
            final_text[:300],
        )
# 清空文本，避免下游把垃圾当摘要展示。加密推理项（如有）保留，
# 让模型在重试时保持思维链。
        final_text = ""

# ── reasoning 通道答案抢救（xAI grok）──────────────────
# grok-4.x 在 xAI /v1/responses 表面有时把最终答案发在 reasoning 项里而不是
# message 输出项，用 grok 内部 ``<response>`` 分隔符标记答案起点。没有抢救，
# 下方 reasoning-only 规则会把回合判为 incomplete——而且该表面 reasoning 项
# 不带 encrypted_content，中间消息重放为空，所以每次续跑请求与刚失败的
# 逐字节相同。回合烧掉 3 次重试，报 "Codex response remained incomplete
# after 3 continuation attempts"，尽管答案第一次就产出了。
# 2026-07-13 在 xai-oauth 的 grok-4.20 上实测。把分隔符后的尾部提升为
# assistant 内容，未标记前缀保留为思考文本。
    if (
        issuer_kind == "xai_responses"
        and not final_text
        and not tool_calls
        and reasoning_parts
    ):
        joined_reasoning = "\n\n".join(reasoning_parts)
        marker = joined_reasoning.rfind("<response>")
        if marker != -1:
            salvaged = joined_reasoning[marker + len("<response>"):]
            closing = salvaged.find("</response>")
            if closing != -1:
                salvaged = salvaged[:closing]
            salvaged = salvaged.strip()
            if salvaged:
                logger.warning(
                    "xAI response delivered its final answer inside the "
                    "reasoning channel (<response> delimiter); promoting "
                    "%d chars to assistant content.",
                    len(salvaged),
                )
                final_text = salvaged
                reasoning_prefix = joined_reasoning[:marker].strip()
                reasoning_parts = [reasoning_prefix] if reasoning_prefix else []

    assistant_message = SimpleNamespace(
        content=final_text,
        tool_calls=tool_calls,
        reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
        reasoning_content=None,
        reasoning_details=None,
        codex_reasoning_items=reasoning_items_raw or None,
        codex_message_items=message_items_raw or None,
    )

    if tool_calls:
        finish_reason = "tool_calls"
    elif response_incomplete_content_filter:
        finish_reason = "content_filter"
    elif leaked_tool_call_text:
        finish_reason = "incomplete"
    elif saw_streaming_or_item_incomplete:
        finish_reason = "incomplete"
    elif (has_incomplete_items or saw_commentary_phase) and not saw_final_answer_phase:
        finish_reason = "incomplete"
    elif (reasoning_items_raw or reasoning_parts or saw_reasoning_item) and not final_text:
# 响应只含推理（加密思考态和/或人类可读摘要），无可见内容或工具调用。
        #
# 对特化后端（Codex、xAI、GitHub/Copilot），reasoning-only + status="completed"
# 意味着"模型还在思考、需要再一轮"——视为 incomplete，让 Codex 续跑路径
# 重试而不是掉进空内容重试循环。
        #
# 对所有其他后端（other:<base_url> 等），信任 provider 自己的 response.status
# 信号。status == "completed" 且无 queued/in_progress/incomplete 项时，
# 仅推理是合法终态——强制 "incomplete" 会造成数分钟停顿
# （续跑路径重发 3 次 × 每次最多 240s）。见 issue #64434。
        if response_status == "completed" and issuer_kind not in (
            "codex_backend",
            "xai_responses",
            "github_responses",
        ):
            finish_reason = "stop"
        else:
            finish_reason = "incomplete"
    else:
        finish_reason = "stop"
    return assistant_message, finish_reason
