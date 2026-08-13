"""消息净化：API 重放前的推理字段策略 + 确定性 call_id 生成。

对应原版 hermes-agent 的 agent/message_sanitization.py（865 行），
my-hermes 精简版只保留实际用到的两个函数。
"""

import hashlib


def apply_reasoning_content_policy(
    source_msg: dict, api_msg: dict, needs_thinking_pad: bool
) -> None:
    """把 provider 侧需要的推理字段复制到 API 重放消息上。

    ``needs_thinking_pad`` 是 require-side 标志（见 needs_reasoning_echo /
    agent 缓存的 ``_needs_thinking_reasoning_pad``）。就地修改 ``api_msg``。
    """
    if source_msg.get("role") != "assistant":
        return

    # 1. 已有显式 reasoning_content。
    #
    # 当当前 provider 强制思考模式回传（DeepSeek / Kimi / MiMo）时原样保留
    # ——包括它们创建时写的空格占位和同一 provider 产生的合法推理内容。
    # #17341 之前持久化的会话把空字符串占位钉在创建时刻；DeepSeek V4 Pro
    # 拒绝空字符串（HTTP 400），所以重放时把 "" 升级为 " "。
    #
    # 当当前 provider 不强制回传时，直接删掉该字段。严格的 OpenAI 兼容
    # provider（Mistral、Cerebras、Groq、SambaNova…）拒绝输入消息里出现
    # 任何 reasoning_content 键（连空串/单空格都拒，HTTP 400/422）。这是
    # 跨 provider 回退场景：推理主 provider 用 " " 填充历史，再回退到严格
    # provider 重放该填充会 422。这里剥离覆盖重建路径；已构建的 api_messages
    # 由 reapply_reasoning_echo 覆盖。参见 #45655。
    existing = source_msg.get("reasoning_content")
    if isinstance(existing, str):
        if not needs_thinking_pad:
            api_msg.pop("reasoning_content", None)
        elif existing == "":
            api_msg["reasoning_content"] = " "
        else:
            api_msg["reasoning_content"] = existing
        return

    # 2. 跨 provider 毒化历史（#15748）：DeepSeek/Kimi 上，如果源回合有
    # tool_calls 且有 'reasoning' 字段但没有 'reasoning_content' 键，说明
    # 'reasoning' 文本是先前 provider（如 MiniMax）写的——修复后同一
    # provider 的 DeepSeek 历史不会出现这种形态。注入单空格满足 API 而
    # 不把别的 provider 的思考链泄漏给 DeepSeek/Kimi。用空格（非 ""）
    # 因为 DeepSeek V4 Pro 思考模式拒绝空字符串（refs #17341）。
    normalized_reasoning = source_msg.get("reasoning")
    if (
        needs_thinking_pad
        and source_msg.get("tool_calls")
        and isinstance(normalized_reasoning, str)
        and normalized_reasoning
    ):
        api_msg["reasoning_content"] = " "
        return

    # 3. 健康会话：对使用内部 'reasoning' 键的 provider，把 'reasoning'
    # 提升为 'reasoning_content'。必须在无条件空串兜底之前做，避免真实
    # 推理内容被覆盖（#15812 回归）。只对强制回传的 provider 提升——
    # 严格 provider 拒绝该字段（refs #45655）。
    if isinstance(normalized_reasoning, str) and normalized_reasoning:
        if needs_thinking_pad:
            api_msg["reasoning_content"] = normalized_reasoning
        else:
            api_msg.pop("reasoning_content", None)
        return

    # 4. DeepSeek / Kimi 思考模式：所有 assistant 消息都需要
    # reasoning_content。没有显式推理内容时注入单空格满足 provider 要求。
    # 覆盖工具调用回合（无任何推理的已毒化历史）和纯文本回合。用空格
    # （非 ""）因为 DeepSeek V4 Pro 收紧校验，拒绝空字符串（HTTP 400）。
    # 参见 #17341。
    if needs_thinking_pad:
        api_msg["reasoning_content"] = " "
        return

    # 5. reasoning_content 存在但不是字符串（如上下文压缩后为 None）。
    # 不要把 null 传给 API。
    api_msg.pop("reasoning_content", None)


def deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
    """根据工具调用内容生成确定性的 call_id（对应原版 message_sanitization.py:525）。

    当 API 未提供 call_id 时用作兜底。确定性 ID 防止提示缓存失效——
    随机 UUID 会让每次 API 调用的前缀都不同，破坏 OpenAI 的提示缓存。
    """
    seed = f"{fn_name}:{arguments}:{index}"
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"call_{digest}"
