from pathlib import Path

APP_DATA_DIR = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Robin"
)

MODEL_DIR = APP_DATA_DIR / "models" / "SmolLM3-3B-4bit"

MAX_NEW_TOKENS = 256
HISTORY_LIMIT = 8
