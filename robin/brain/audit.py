import json
from datetime import datetime, timezone

from robin.brain.config import APP_DATA_DIR


AUDIT_PATH = APP_DATA_DIR / "activity.jsonl"


def record_event(action: str, target: str, outcome: str):
    """Append an action event to Robin's local activity log."""

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "target": target,
        "outcome": outcome,
    }

    with AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
