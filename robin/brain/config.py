from __future__ import annotations

import os
import platform
from pathlib import Path


def get_app_data_dir() -> Path:
    """Return platform-appropriate application data directory with env override support."""
    env_dir = os.environ.get("ROBIN_APP_DATA_DIR")
    if env_dir and env_dir.strip():
        p = Path(env_dir.strip()).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    system = platform.system()
    home = Path.home()
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        return base / "Robin"
    elif system == "Darwin":
        p = home / "Library" / "Application Support" / "Robin"
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else (home / ".local" / "share")
        p = base / "Robin"
    p.mkdir(parents=True, exist_ok=True)
    return p


APP_DATA_DIR = get_app_data_dir()

MODEL_DIR = APP_DATA_DIR / "models" / "SmolLM3-3B-4bit"


MAX_NEW_TOKENS = 256
HISTORY_LIMIT = 8


def get_or_create_api_token() -> str:
    """Return an active API token from env, or secure local storage, or generate a new one."""
    import os
    import secrets

    env_token = os.environ.get("ROBIN_API_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    token_path = APP_DATA_DIR / ".auth_token"
    if token_path.exists():
        try:
            stored = token_path.read_text(encoding="utf-8").strip()
            if stored:
                return stored
        except Exception:
            pass

    # Generate cryptographically secure 256-bit token
    new_token = secrets.token_urlsafe(32)
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        mode = 0o600  # Accessible only to current OS user
        fd = os.open(str(token_path), flags, mode)
        with open(fd, "w", encoding="utf-8") as f:
            f.write(new_token)
    except OSError:
        pass
    return new_token

