import re
from pathlib import Path

from mlx_lm import generate, load


MODEL_DIR = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Robin"
    / "models"
    / "SmolLM3-3B-4bit"
)


def load_brain():
    """Load Robin's model from local storage."""

    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {MODEL_DIR}\n"
            "Download the model before starting Robin."
        )

    model, tokenizer = load(str(MODEL_DIR))
    return model, tokenizer


def clean_response(text: str) -> str:
    """Remove visible thinking blocks from the model's response."""

    # Remove complete <think>...</think> blocks.
    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove an unfinished thinking block at the end.
    text = re.sub(
        r"<think\b[^>]*>.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Remove any leftover thinking tags.
    text = re.sub(
        r"</?think\s*>",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return text.strip()


def generate_reply(model, tokenizer, messages):
    """Generate a cleaned response using Robin's local model."""

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=400,
        verbose=False,
    )

    return clean_response(response)
