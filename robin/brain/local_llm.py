from __future__ import annotations

import re
from typing import Any, Generator, List, Optional, Tuple

from robin.brain.settings import DEFAULT_MODEL, MAX_NEW_TOKENS, SHOW_THINKING

MODEL_NAME = DEFAULT_MODEL
client = None


def load_brain():
    """Verify that Robin's local model and embedded inference engine are available."""
    from robin.brain.providers import get_provider
    provider = get_provider()
    provider.initialize()
    return provider.default_model, provider


def stream_reply(
    model: str,
    client: Any = None,
    messages: Optional[List[dict]] = None,
    stop_event: Any = None,
    max_new_tokens: Optional[int] = None,
    provider: Any = None,
) -> Generator[Tuple[str, str], None, None]:
    """Yield (kind, text) pairs for thinking and final-answer text.

    Uses Robin's embedded inference provider by default.
    """
    effective_max_tokens = max_new_tokens if max_new_tokens is not None else 1024

    # 1. If provider is supplied or client is an InferenceProvider instance
    from robin.brain.providers.base import InferenceProvider
    active_provider = provider
    if active_provider is None and isinstance(client, InferenceProvider):
        active_provider = client

    if active_provider is not None:
        yield from active_provider.stream_chat(
            model=model,
            messages=messages or [],
            stop_event=stop_event,
            max_new_tokens=effective_max_tokens,
        )
        return

    # 2. If an explicit client with chat method was passed (e.g. in mock tests)
    if client is not None and hasattr(client, "chat"):
        stream = client.chat(
            model=model,
            messages=messages or [],
            stream=True,
            think=SHOW_THINKING,
            options={
                "num_predict": effective_max_tokens,
                "temperature": 0.4,
            },
        )

        pending = ""
        inside_inline_thinking = False

        for part in stream:
            if stop_event is not None and stop_event.is_set():
                close_stream = getattr(stream, "close", None)
                if close_stream:
                    close_stream()
                break

            message = part.message
            thinking = getattr(message, "thinking", None) or ""
            if thinking:
                yield ("thinking", thinking)

            pending += getattr(message, "content", None) or ""

            while pending:
                if inside_inline_thinking:
                    closing = re.search(
                        r"</think\s*>",
                        pending,
                        flags=re.IGNORECASE,
                    )

                    if closing:
                        if closing.start():
                            yield ("thinking", pending[:closing.start()])

                        pending = pending[closing.end():]
                        inside_inline_thinking = False
                        continue

                    keep = len("</think>") - 1
                    if len(pending) > keep:
                        yield ("thinking", pending[:-keep])
                        pending = pending[-keep:]
                    break

                opening = re.search(
                    r"<think\b[^>]*>",
                    pending,
                    flags=re.IGNORECASE,
                )

                if opening:
                    if opening.start():
                        yield ("answer", pending[:opening.start()])

                    pending = pending[opening.end():]
                    inside_inline_thinking = True
                    continue

                partial_start = pending.lower().rfind("<think")
                if partial_start >= 0 and ">" not in pending[partial_start:]:
                    if partial_start:
                        yield ("answer", pending[:partial_start])
                    pending = pending[partial_start:]
                    break

                keep = len("<think") - 1
                if len(pending) > keep:
                    yield ("answer", pending[:-keep])
                    pending = pending[-keep:]
                break

        if pending:
            if inside_inline_thinking:
                yield ("thinking", pending)
            elif not pending.lower().startswith("<think"):
                yield ("answer", pending)
        return

    # 3. Default to Robin's shared embedded provider
    from robin.brain.providers import get_default_provider
    p = get_default_provider()
    yield from p.stream_chat(
        model=model,
        messages=messages or [],
        stop_event=stop_event,
        max_new_tokens=effective_max_tokens,
    )
