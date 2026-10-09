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
    """Load Robin's model from a local directory."""

    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {MODEL_DIR}\n"
            "Download the model before starting Robin."
        )

    model, tokenizer = load(str(MODEL_DIR))
    return model, tokenizer


def generate_reply(model, tokenizer, messages):
    """Generate a response using Robin's local model."""

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

    return response.strip()
