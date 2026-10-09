import sqlite3
from datetime import datetime, timezone

from robin.brain.config import APP_DATA_DIR


DB_PATH = APP_DATA_DIR / "memory.sqlite3"


def initialize_memory():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def load_recent_messages(limit=8):
    initialize_memory()

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM messages
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (limit,),
        ).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]


def save_turn(user_text, assistant_text):
    initialize_memory()
    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO messages (role, content, created_at)
            VALUES (?, ?, ?)
            """,
            [
                ("user", user_text, timestamp),
                ("assistant", assistant_text, timestamp),
            ],
        )
