"""
robin/brain/memory.py
─────────────────────
Local SQLite-backed session continuity, conversation memory, project facts,
and task handoffs for Robin.

Tables:
- `sessions`: Distinct conversation threads with titles and timestamps.
- `messages`: Chronological conversation turns mapped to sessions.
- `project_facts`: Verified, inspectable project knowledge distinct from raw chat.
- `task_handoffs`: Explicit, structured checkpoint records (progress, blockers,
  file references, next steps) for session continuity.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from robin.brain.config import APP_DATA_DIR

DB_PATH = APP_DATA_DIR / "memory.sqlite3"


def initialize_memory(db_path: Optional[Path] = None) -> None:
    """Ensure SQLite database and required tables exist with up-to-date schema."""
    target_path = db_path or DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")

        # 1. Sessions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                project_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        # 2. Messages table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL DEFAULT 'default',
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

        # Migrate existing messages table if session_id is missing
        cursor = conn.execute("PRAGMA table_info(messages);")
        columns = [row[1] for row in cursor.fetchall()]
        if "session_id" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default';")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);"
        )

        # 3. Project Facts table (verified knowledge distinct from ephemeral chat)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_path, fact_key)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_project_facts_path ON project_facts(project_path);"
        )

        # 4. Task Handoffs table (continuity checkpoints)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_handoffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT NOT NULL,
                session_id TEXT,
                progress TEXT NOT NULL,
                blockers TEXT,
                file_references TEXT,
                next_step TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_handoffs_path ON task_handoffs(project_path, id DESC);"
        )


def _get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Session & Message Operations
# ─────────────────────────────────────────────────────────────────────────────


def create_session(
    title: str = "New Chat",
    project_path: Optional[str] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Create a new conversation session and return its unique identifier."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH
    sid = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{datetime.now().microsecond:06d}"
    now = _get_utc_now()

    with sqlite3.connect(target_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO sessions (id, title, project_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sid, title.strip() or "New Chat", project_path, now, now),
        )
    return sid


def list_sessions(
    project_path: Optional[str] = None,
    limit: int = 30,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return recent sessions with their message count and last updated timestamp."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH

    query = """
        SELECT s.id, s.title, s.project_path, s.created_at, s.updated_at,
               COUNT(m.id) as message_count
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id
    """
    params: List[Any] = []
    if project_path:
        query += " WHERE s.project_path = ?"
        params.append(project_path)

    query += " GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "project_path": row["project_path"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"],
        }
        for row in rows
    ]


def delete_session(session_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete a session and all its associated conversation turns."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH

    with sqlite3.connect(target_path) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


def search_sessions(
    query: str,
    project_path: Optional[str] = None,
    limit: int = 15,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Search for text across historical conversation turns."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH
    needle = f"%{query.strip()}%"

    sql = """
        SELECT m.id, m.session_id, m.role, m.content, m.created_at, s.title as session_title
        FROM messages m
        LEFT JOIN sessions s ON s.id = m.session_id
        WHERE m.content LIKE ?
    """
    params: List[Any] = [needle]
    if project_path:
        sql += " AND (s.project_path = ? OR s.project_path IS NULL)"
        params.append(project_path)

    sql += " ORDER BY m.id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    return [
        {
            "message_id": row["id"],
            "session_id": row["session_id"],
            "session_title": row["session_title"] or "Conversation",
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def load_recent_messages(
    limit: int = 8,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """
    Load the most recent conversation messages.

    If session_id is provided, loads specifically for that session.
    If session_id is None, defaults to the 'default' session or global tail.
    """
    initialize_memory(db_path)
    target_path = db_path or DB_PATH

    with sqlite3.connect(target_path) as conn:
        if session_id:
            rows = conn.execute(
                """
                SELECT role, content
                FROM (
                    SELECT id, role, content
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
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

    return [{"role": role, "content": content} for role, content in rows]


def save_turn(
    user_text: str,
    assistant_text: str,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Save a user prompt and assistant reply turn to conversation memory."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH
    sid = session_id or "default"
    now = _get_utc_now()

    # Generate a descriptive title from first prompt if new session
    derived_title = user_text.strip()[:40].replace("\n", " ")
    if len(user_text.strip()) > 40:
        derived_title += "…"

    with sqlite3.connect(target_path) as conn:
        # Ensure session record exists
        conn.execute(
            """
            INSERT INTO sessions (id, title, project_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at = excluded.updated_at
            """,
            (sid, derived_title, project_path, now, now),
        )

        conn.executemany(
            """
            INSERT INTO messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (sid, "user", user_text, now),
                (sid, "assistant", assistant_text, now),
            ],
        )


def clear_history(session_id: Optional[str] = None, db_path: Optional[Path] = None) -> None:
    """Delete saved conversation history (for a specific session or completely)."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH

    with sqlite3.connect(target_path) as conn:
        if session_id:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        else:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")


# ─────────────────────────────────────────────────────────────────────────────
# Verified Project Facts
# ─────────────────────────────────────────────────────────────────────────────


def save_project_fact(
    project_path: str,
    fact_key: str,
    fact_value: str,
    source: str = "user",
    db_path: Optional[Path] = None,
) -> int:
    """Store an explicit, verified fact about the project workspace."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH
    now = _get_utc_now()

    with sqlite3.connect(target_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO project_facts (project_path, fact_key, fact_value, source, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_path, fact_key) DO UPDATE SET
                fact_value = excluded.fact_value,
                source = excluded.source,
                updated_at = excluded.updated_at
            RETURNING id
            """,
            (project_path.strip(), fact_key.strip(), fact_value.strip(), source, now, now),
        )
        row = cursor.fetchone()
        return row[0] if row else 0


def get_project_facts(
    project_path: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve verified facts stored for the given project workspace."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH

    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, project_path, fact_key, fact_value, source, created_at, updated_at
            FROM project_facts
            WHERE project_path = ?
            ORDER BY fact_key ASC
            """,
            (project_path.strip(),),
        ).fetchall()

    return [
        {
            "id": r["id"],
            "project_path": r["project_path"],
            "key": r["fact_key"],
            "value": r["fact_value"],
            "source": r["source"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def delete_project_fact(fact_id: int, db_path: Optional[Path] = None) -> bool:
    """Delete a stored project fact by its primary key ID."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH

    with sqlite3.connect(target_path) as conn:
        cur = conn.execute("DELETE FROM project_facts WHERE id = ?", (fact_id,))
        return cur.rowcount > 0


# ─────────────────────────────────────────────────────────────────────────────
# Task Handoff Checkpoints
# ─────────────────────────────────────────────────────────────────────────────


def save_handoff(
    project_path: str,
    progress: str,
    next_step: str,
    blockers: Optional[str] = None,
    file_references: Optional[List[str]] = None,
    session_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Save an explicit verified task handoff record for project continuity.

    Records current progress, known blockers, key file references, and the immediate next step.
    """
    initialize_memory(db_path)
    target_path = db_path or DB_PATH
    now = _get_utc_now()
    files_json = json.dumps(file_references or [], ensure_ascii=False)

    with sqlite3.connect(target_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO task_handoffs (
                project_path, session_id, progress, blockers, file_references, next_step, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_path.strip(),
                session_id,
                progress.strip(),
                (blockers or "").strip(),
                files_json,
                next_step.strip(),
                now,
            ),
        )
        return int(cur.lastrowid)


def get_latest_handoff(
    project_path: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent task handoff checkpoint for a project."""
    initialize_memory(db_path)
    target_path = db_path or DB_PATH

    with sqlite3.connect(target_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, project_path, session_id, progress, blockers, file_references, next_step, created_at
            FROM task_handoffs
            WHERE project_path = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (project_path.strip(),),
        ).fetchone()

    if not row:
        return None

    try:
        files = json.loads(row["file_references"]) if row["file_references"] else []
    except Exception:
        files = []

    return {
        "id": row["id"],
        "project_path": row["project_path"],
        "session_id": row["session_id"],
        "progress": row["progress"],
        "blockers": row["blockers"],
        "file_references": files,
        "next_step": row["next_step"],
        "created_at": row["created_at"],
    }
