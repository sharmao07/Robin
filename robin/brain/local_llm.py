import re

import ollama
from robin.brain.settings import MAX_NEW_TOKENS, SHOW_THINKING


MODEL_NAME = "qwen3:4b"
client = ollama.Client(host="http://127.0.0.1:11434")


def load_brain():
    """Verify that Robin's local model is available."""
    client.show(MODEL_NAME)
    return MODEL_NAME, client


def stream_reply(model, client, messages):
    """Yield (kind, text) pairs for thinking and final-answer text."""

    stream = client.chat(
        model=model,
        messages=messages,
        stream=True,
        think=SHOW_THINKING,
        options={
            "num_predict": MAX_NEW_TOKENS,
            "temperature": 0.4,
        },
    )

    pending = ""
    inside_inline_thinking = False

    for part in stream:
        message = part.message

        # Ollama's native thinking field is separate from its answer.
        thinking = getattr(message, "thinking", None) or ""
        if thinking:
            yield ("thinking", thinking)

        # Some models or templates may still put <think> tags in content.
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

                # Keep the end of the buffer in case a closing tag
                # is split between streaming chunks.
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

            # Wait for the next chunk if a tag begins at the end.
            partial_start = pending.lower().rfind("<think")

            if (
                partial_start >= 0
                and ">" not in pending[partial_start:]
            ):
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
