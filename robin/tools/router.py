import re


APP_ALIASES = {
    "vs code": "vscode",
    "visual studio code": "vscode",
    "code": "vscode",
    "web browser": "safari",
}


def parse_app_request(text: str):
    """Recognize explicit requests to open an application."""

    pattern = (
        r"^\s*(?:hey robin[, ]+)?"
        r"(?:please\s+)?"
        r"(?:can you\s+|could you\s+)?"
        r"(?:open|launch|start)\s+"
        r"(?:the\s+)?(?:app\s+)?(.+?)\s*[.!?]*\s*$"
    )

    match = re.match(pattern, text, flags=re.IGNORECASE)

    if not match:
        return None

    name = match.group(1).strip().lower()
    name = name.rstrip(" .!?")

    return APP_ALIASES.get(name, name)
