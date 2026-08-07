def apply_reasoning_content_policy(
    source_msg: dict, api_msg: dict, needs_thinking_pad: bool
) -> None:
    """Copy provider-facing reasoning fields onto an API replay message.

    ``needs_thinking_pad`` is the require-side flag (see
    ``needs_reasoning_echo`` / the agent's cached
    ``_needs_thinking_reasoning_pad``). Mutates ``api_msg`` in place.
    """
    if source_msg.get("role") != "assistant":
        return

    # 1. Explicit reasoning_content already set.
    #
    # When the active provider enforces the thinking-mode echo-back
    # (DeepSeek / Kimi / MiMo), preserve it verbatim — that includes their
    # own space-placeholder written at creation time and any valid reasoning
    # from the same provider. Sessions persisted BEFORE #17341 have
    # empty-string placeholders pinned at creation time; DeepSeek V4 Pro
    # rejects those with HTTP 400, so upgrade "" → " " on replay.
    #
    # When the active provider does NOT enforce echo-back, strip the field
    # entirely. Strict OpenAI-compatible providers (Mistral, Cerebras, Groq,
    # SambaNova, …) reject ANY reasoning_content key in input messages with
    # HTTP 400/422 ("Extra inputs are not permitted"), even an empty string
    # or a single-space pad. This is the cross-provider fallback case: a
    # reasoning primary (DeepSeek/Kimi/MiMo) pads history with " ", then a
    # fallback to a strict provider replays that pad and 422s. Stripping
    # here covers the rebuild path; ``reapply_reasoning_echo`` covers the
    # already-built api_messages path. Refs #45655.
    existing = source_msg.get("reasoning_content")
    if isinstance(existing, str):
        if not needs_thinking_pad:
            api_msg.pop("reasoning_content", None)
        elif existing == "":
            api_msg["reasoning_content"] = " "
        else:
            api_msg["reasoning_content"] = existing
        return

    # 2. Cross-provider poisoned history (#15748): on DeepSeek/Kimi,
    # if the source turn has tool_calls AND a 'reasoning' field but no
    # 'reasoning_content' key, the 'reasoning' text was written by a
    # prior provider (e.g. MiniMax) — DeepSeek's own _build_assistant_message
    # pins reasoning_content at creation time for tool-call turns, so the
    # shape (reasoning set, reasoning_content absent, tool_calls present)
    # is unreachable from same-provider DeepSeek history after this fix.
    # Inject a single space to satisfy the API without leaking another
    # provider's chain of thought to DeepSeek/Kimi. Space (not "")
    # because DeepSeek V4 Pro rejects empty-string reasoning_content
    # in thinking mode (refs #17341).
    normalized_reasoning = source_msg.get("reasoning")
    if (
        needs_thinking_pad
        and source_msg.get("tool_calls")
        and isinstance(normalized_reasoning, str)
        and normalized_reasoning
    ):
        api_msg["reasoning_content"] = " "
        return

    # 3. Healthy session: promote 'reasoning' field to 'reasoning_content'
    # for providers that use the internal 'reasoning' key.
    # This must happen before the unconditional empty-string fallback so
    # genuine reasoning content is not overwritten (#15812 regression in
    # PR #15478). Only promote for providers that enforce echo-back —
    # strict providers reject the field (refs #45655).
    if isinstance(normalized_reasoning, str) and normalized_reasoning:
        if needs_thinking_pad:
            api_msg["reasoning_content"] = normalized_reasoning
        else:
            api_msg.pop("reasoning_content", None)
        return

    # 4. DeepSeek / Kimi thinking mode: all assistant messages need
    # reasoning_content. Inject a single space to satisfy the provider's
    # requirement when no explicit reasoning content is present. Covers
    # both tool-call turns (already-poisoned history with no reasoning
    # at all) and plain text turns. Space (not "") because DeepSeek V4
    # Pro tightened validation and rejects empty string with HTTP 400
    # ("The reasoning content in the thinking mode must be passed back
    # to the API"). Refs #17341.
    if needs_thinking_pad:
        api_msg["reasoning_content"] = " "
        return

    # 5. reasoning_content was present but not a string (e.g. None after
    # context compaction).  Don't pass null to the API.
    api_msg.pop("reasoning_content", None)