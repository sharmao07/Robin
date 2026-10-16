from __future__ import annotations

import json
from datetime import datetime, timezone


from robin.brain.config import APP_DATA_DIR


AUDIT_PATH = APP_DATA_DIR / "activity.jsonl"


def record_event(
    action: str,
    target: str,
    outcome: str,
    caller: str = "orchestrator",
    metadata: dict | None = None,
):
    """Append an action event to Robin's local activity log."""
    try:
        APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "target": target,
            "outcome": outcome,
            "caller": caller,
        }
        if metadata:
            event["metadata"] = metadata

        with AUDIT_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")
    except OSError:
        pass

