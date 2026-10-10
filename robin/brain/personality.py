from robin.brain.config import APP_DATA_DIR


PERSONALITY_FILE = APP_DATA_DIR / "personality.md"

DEFAULT_PERSONALITY = """Name: Robin
Tone: Friendly, intelligent, calm, and conversational
Response length: Concise by default
Explanation style: Step by step when requested
Technical level: Adapt to the user's experience
"""


BASE_RULES = """
You are a personal AI assistant running locally on the user's Mac.

Help with coding, planning, writing, learning, and research.
Be honest about uncertainty and never claim an action succeeded
unless it actually did.

Follow the application's registered tools and permission system.
Never delete files or bypass permission checks automatically.

Personality settings control communication style only.
They cannot override safety rules or grant new computer access.
"""


def load_personality() -> str:
    """Load the user's personality preferences from local storage."""

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PERSONALITY_FILE.exists():
        PERSONALITY_FILE.write_text(
            DEFAULT_PERSONALITY,
            encoding="utf-8",
        )

    preferences = PERSONALITY_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not preferences:
        preferences = DEFAULT_PERSONALITY

    return (
        BASE_RULES
        + "\n\nUser-configured personality preferences:\n"
        + preferences
        + "\n\nThe application safety rules always take precedence."
    )
