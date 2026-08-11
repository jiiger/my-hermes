"""模型元数据精简版（对应原版 agent/model_metadata.py）。

my-hermes 只保留上下文长度解析的「硬编码表 + 最终兜底」两级（原版
机制 8/9）：显式配置（compression.context_length / model.context_length）
命中时由 context_compressor 提前返回，不走这里；这里负责按模型名模糊
匹配硬编码表，未命中时回退 DEFAULT_FALLBACK_CONTEXT 并只警告一次。
探测 / 持久缓存 / provider 分支（原版机制 0c、1-7）全部砍掉。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 运行 hermes-agent 所需的最小上下文长度（压缩阈值下限，保留原值）。
MINIMUM_CONTEXT_LENGTH = 64_000

# 最终兜底：任何机制都没命中时的默认上下文长度。
# 对齐原版 CONTEXT_PROBE_TIERS[0]（256K），my-hermes 无探测阶梯。
DEFAULT_FALLBACK_CONTEXT = 256_000

# (model, base_url) 已发出过兜底警告的集合：兜底结果本身不缓存，
# 没有这个集合同一 model 会在每次解析时重复警告。
_FALLBACK_WARNED: set = set()


def _warn_context_length_fallback(model: str, base_url: str) -> None:
    """每个 model+endpoint 只警告一次：未命中硬编码表，使用默认兜底。"""
    key = (model, base_url or "")
    if key in _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED.add(key)
    logger.warning(
        "Could not determine context length for model %r (base_url=%s) "
        "— falling back to %s tokens. Set model.context_length in "
        "config.yaml to override.",
        model, base_url or "default", f"{DEFAULT_FALLBACK_CONTEXT:,}",
    )


# 硬编码默认表：只覆盖常见模型族（数据对齐原版 agent/model_metadata.py）。
# 键按「最长优先」子串匹配，保证 "deepseek-v4-flash" 命中 1M 而
# 裸 "deepseek" 兜 128K。
DEFAULT_CONTEXT_LENGTHS = {
    # Anthropic Claude 4.6 (1M context) — bare IDs only to avoid
    # fuzzy-match collisions (e.g. "anthropic/claude-sonnet-4" is a
    # substring of "anthropic/claude-sonnet-4.6").
    # OpenRouter-prefixed models resolve via OpenRouter live API or models.dev.
    "claude-fable-5": 1000000,
    "claude-fable": 1000000,
    "claude-opus-5": 1000000,
    "claude-sonnet-5": 1000000,
    "claude-opus-4-8": 1000000,
    "claude-opus-4.8": 1000000,
    "claude-opus-4-7": 1000000,
    "claude-opus-4.7": 1000000,
    "claude-opus-4-6": 1000000,
    "claude-sonnet-4-6": 1000000,
    "claude-opus-4.6": 1000000,
    "claude-sonnet-4.6": 1000000,
    # Catch-all for older Claude models (must sort after specific entries)
    "claude": 200000,
    # OpenAI — GPT-5 family (most have 400k; specific overrides first)
    # Source: https://developers.openai.com/api/docs/models
    # GPT-5.5 (launched Apr 23 2026) is 1.05M on the direct OpenAI API and
    # ChatGPT Codex OAuth caps it at 272K; both paths resolve via their own
    # provider-aware branches (_resolve_codex_oauth_context_length + models.dev).
    # This hardcoded value is only reached when every probe misses.
    # GPT-5.6 series (Sol/Terra/Luna, GA 2026-07-09) — 1.05M on the direct
    # OpenAI API (same as gpt-5.5). Codex OAuth caps these at 272K.
    # (Lookups length-sort keys at match time, so dict order is cosmetic.)
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-terra": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.5": 1050000,
    "gpt-5.4-nano": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4-mini": 400000,           # 400k (not 1.05M like full 5.4)
    "gpt-5.4": 1050000,               # GPT-5.4, GPT-5.4 Pro (1.05M context)
    # gpt-5.3-codex-spark is Codex-OAuth-only (ChatGPT Pro entitlement) and
    # uses a smaller 128k window than other gpt-5.x slugs. Listed here as
    # a defensive override so the longest-substring fallback doesn't match
    # the generic "gpt-5" entry below (400k) and report the wrong limit if
    # Spark's context ever needs to be resolved through this path. Real
    # usage flows through _CODEX_OAUTH_CONTEXT_FALLBACK at line ~1113.
    "gpt-5.3-codex-spark": 128000,
    "gpt-5.1-chat": 128000,           # Chat variant has 128k context
    "gpt-5": 400000,                  # GPT-5.x base, mini, codex variants (400k)
    "gpt-4.1": 1047576,
    "gpt-4": 128000,
    # Google
    "gemini": 1048576,
    # Gemma (open models served via AI Studio)
    "gemma-4": 256000,  # Gemma 4 family
    "gemma4": 256000,  # Ollama-style naming (e.g. gemma4:31b-cloud)
    "gemma-4-31b": 256000,
    "gemma-3": 131072,
    "gemma": 8192,  # fallback for older gemma models
    # DeepSeek — V4 family ships with a 1M context window. The legacy
    # aliases ``deepseek-chat`` / ``deepseek-reasoner`` are server-side
    # mapped to the non-thinking / thinking modes of ``deepseek-v4-flash``
    # and inherit the same 1M window. The ``deepseek`` substring entry
    # below remains as a 128K fallback for older / unknown DeepSeek model
    # ids (e.g. via custom endpoints).
    # https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-chat": 1_000_000,
    "deepseek-reasoner": 1_000_000,
    "deepseek": 128000,
    # Meta
    "llama": 131072,
    # Qwen — specific model families before the catch-all.
    # Official docs: https://help.aliyun.com/zh/model-studio/developer-reference/
    "qwen3.8-max": 1_000_000,     # 1M context (OpenRouter & Nous portal, verified 2026-08-03)
    "qwen3.6-plus": 1048576,      # 1M context (DashScope/Alibaba & OpenRouter)
    "qwen3.7-plus": 1048576,      # 1M context (DashScope/Alibaba)
    "qwen3-coder-plus": 1000000,  # 1M context
    "qwen3-coder": 262144,        # 256K context
    "qwen3-max": 262144,          # 256K context (qwen3-max-2026-01-23 snapshot, Coding Plan)
    "qwen": 131072,
    # MiniMax — M3 is 1M context (max output 512K); M2.x series is 204,800.
    # Keys use substring matching (longest-first), so "minimax-m3" wins over
    # the generic "minimax" catch-all for the M3 slug on every surface
    # (native MiniMax-M3, OpenRouter/Nous minimax/minimax-m3).
    # https://platform.minimax.io/docs/api-reference/text-chat-openai
    "minimax-m3": 1000000,
    "minimax": 204800,
    # GLM — GLM-5.2 ships with a 1M context window (verified empirically:
    # needle-in-a-haystack retrieval at 789K prompt tokens succeeded with
    # zero errors on api.z.ai/api/coding/paas/v4).  Older GLM models
    # (5, 5.1, 5-turbo) are ~202K.  Longest-key-first substring matching
    # ensures "glm-5.2" resolves to 1M while older variants still hit the
    # generic 202K fallback.
    "glm-5.2": 1_048_576,
    "glm": 202752,
    # xAI Grok — xAI /v1/models does not return context_length metadata,
    # so these hardcoded fallbacks prevent Hermes from probing-down to
    # the default 128k when the user points at https://api.x.ai/v1
    # via a custom provider. Values sourced from models.dev (2026-04).
    # Keys use substring matching (longest-first), so e.g. "grok-4.20"
    # matches "grok-4.20-0309-reasoning" / "-non-reasoning" / "-multi-agent-0309".
    # OAuth-only slug; absent from GET /v1/models. xAI publishes a 200k
    # usable context window for Composer 2.5 on Grok Build (SuperGrok /
    # Premium+); /v1/responses additionally enforces a ~262144 input+output
    # budget, but the usable context (what we track here) is 200k.
    "grok-composer": 200000,    # grok-composer-2.5-fast (Grok Build CLI)
    "grok-build-latest": 500000,  # alias of grok-4.5 (early access)
    "grok-build": 256000,       # grok-build-0.1
    "grok-code-fast": 256000,   # grok-code-fast-1
    "grok-2-vision": 8192,      # grok-2-vision, -1212, -latest
    "grok-4-fast": 2000000,     # grok-4-fast-(non-)reasoning, also matches -reasoning
    "grok-4.20": 2000000,       # grok-4.20-0309-(non-)reasoning, -multi-agent-0309
    "grok-4.5": 500000,         # grok-4.5, grok-4.5-latest — 500K context per docs.x.ai
    "grok-4.3": 1000000,        # grok-4.3, grok-4.3-latest — 1M context per docs.x.ai
    "grok-4": 256000,           # grok-4, grok-4-0709
    "grok-3": 131072,           # grok-3, grok-3-mini, grok-3-fast, grok-3-mini-fast
    "grok-2": 131072,           # grok-2, grok-2-1212, grok-2-latest
    "grok": 131072,             # catch-all (grok-beta, unknown grok-*)
    # Kimi — K3 ships with a 1 Mi context window (1,048,576; verified against
    # models.dev and OpenRouter live metadata, matching the endpoint-scoped
    # override in _endpoint_scoped_context_length). Longest-key-first substring
    # matching ensures "kimi-k3" resolves to 1M while older/unknown Kimi models
    # still hit the generic 256K fallback.
    "kimi-k3": 1_048_576,
    "kimi": 262144,
    # Upstage Solar — api.upstage.ai/v1/models does not return context_length,
    # so these fallbacks keep token budgeting / compression from probing down
    # to the 128k default. Ids are matched longest-first, so dated variants
    # (e.g. solar-pro3-250127) resolve via their family prefix.
    # Sources: Solar Pro 3 = 128K, Solar Pro 2 = 64K, Solar Mini = 32K,
    # Solar Open 2 = 256K.
    "solar-open2": 262144,  # 256K
    "solar-pro3": 131072,
    "solar-pro2": 65536,
    "solar-mini": 32768,
    # Tencent — Hy3 Preview (Hunyuan) with 256K context window.
    # OpenRouter live metadata reports 262144 (256 × 1024); align the
    # static fallback so cache and offline both agree (issue #22268).
    "hy3-preview": 262144,
    # Tencent — Hy3 (GA successor to Hy3 Preview), same 256K window.
    "hy3": 262144,
    # Nemotron — NVIDIA's open-weights series (128K context across all sizes)
    "nemotron": 131072,
    # Arcee
    "trinity": 262144,
    # OpenRouter
    "elephant": 262144,
    # Hugging Face Inference Providers — model IDs use org/name format
    "Qwen/Qwen3.5-397B-A17B": 131072,
    "Qwen/Qwen3.5-35B-A3B": 131072,
    "deepseek-ai/DeepSeek-V3.2": 65536,
    "moonshotai/Kimi-K2.5": 262144,
    "moonshotai/Kimi-K2.6": 262144,
    "moonshotai/Kimi-K2-Thinking": 262144,
    "MiniMaxAI/MiniMax-M2.5": 204800,
    "XiaomiMiMo/MiMo-V2-Flash": 262144,
    "mimo-v2-pro": 1048576,
    "mimo-v2.5-pro": 1048576,
    "mimo-v2.5": 1048576,
    "mimo-v2-omni": 262144,
    "mimo-v2-flash": 262144,
    "zai-org/GLM-5": 202752,
}


def get_default_context_length(model: str, base_url: str = "") -> int:
    """按模型名解析上下文长度：硬编码表（最长键优先子串匹配）> 256K 兜底。

    对齐原版 agent/model_metadata.py 机制 8/9；my-hermes 无探测/缓存，
    显式配置命中时由调用方（context_compressor）提前返回，不走这里。
    """
    model_lower = (model or "").lower()
    for default_model, length in sorted(
        DEFAULT_CONTEXT_LENGTHS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if default_model in model_lower:
            return length
    _warn_context_length_fallback(model, base_url)
    return DEFAULT_FALLBACK_CONTEXT
