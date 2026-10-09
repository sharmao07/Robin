import json
from datetime import datetime, timezone

from robin.brain.config import APP_DATA_DIR


AUDIT_PATH = APP_DATA_DIR / "activity.jsonl"


def record_event(action: str, target: str, outcome: str):
    """Append one local action event."""

    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "target": target,
        "outcome": outcome,
    }

    with AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event) + "\n")
python - <<'PY'
from pathlib import Path

path = Path("robin/main.py")
text = path.read_text()

marker = "from robin.brain.memory import ("
if "from robin.brain.audit import record_event" not in text:
    text = text.replace(
        marker,
        "from robin.brain.audit import record_event\n" + marker,
        1,
    )

old = '''        if app_name is not None:
            if authorize("open_app", app_name):
                result = open_app(app_name)
                robin_says(result)
            else:
                robin_says("This action is not permitted.")
            continue
'''

new = '''        if app_name is not None:
            if authorize("open_app", app_name):
                result = open_app(app_name)
                record_event("open_app", app_name, result)
                robin_says(result)
            else:
                record_event("open_app", app_name, "blocked")
                robin_says("This action is not permitted.")
            continue
'''

if old not in text:
    raise SystemExit("Could not locate the permission handler.")

path.write_text(text.replace(old, new, 1))
print("Connected the local audit log.")
