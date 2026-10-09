import re
from pathlib import Path

from mlx_lm import load, stream_generate


MODEL_DIR = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Robin"
    / "models"
    / "SmolLM3-3B-4bit"
)


def load_brain():
    """Load Robin's language model locally."""

    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {MODEL_DIR}"
        )

    return load(str(MODEL_DIR))


def clean_response(text: str) -> str:
    """Remove visible thinking blocks."""

    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return text.strip()


def stream_reply(model, tokenizer, messages):
    """Stream Robin's answer while suppressing thinking blocks."""

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    pending = ""
    inside_think = False

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=256,
    ):
        pending += response.text

        while pending:
            if inside_think:
                closing = re.search(
                    r"</think\s*>",
                    pending,
                    flags=re.IGNORECASE,
                )

                if closing:
                    pending = pending[closing.end():]
                    inside_think = False
                    continue

                # Retain a suffix in case the closing tag
                # is split between generated chunks.
                pending = pending[-7:]
                break

            opening = re.search(
                r"<think\b",
                pending,
                flags=re.IGNORECASE,
            )

            if opening:
                tag_end = pending.find(">", opening.start())

                if tag_end == -1:
                    if opening.start() > 0:
                        yield pending[:opening.start()]

                    pending = pending[opening.start():]
                    break

                if opening.start() > 0:
                    yield pending[:opening.start()]

                pending = pending[tag_end + 1:]
                inside_think = True
                continue

            # Keep a short suffix to detect a tag that may
            # begin in the next generated chunk.
            safe_length = len(pending) - 5

            if safe_length > 0:
                yield pending[:safe_length]
                pending = pending[safe_length:]

            break

    if not inside_think and pending:
        lower = pending.lower()
        tag_start = lower.rfind("<")

        if tag_start >= 0 and "<think".startswith(lower[tag_start:]):
            pending = pending[:tag_start]

        if pending:
            yield pending
