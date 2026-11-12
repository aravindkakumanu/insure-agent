"""
sessions.py — Durable session storage (SQLite)
===============================================
Conversation state used to live in process memory and vanished on every restart:
each specialist's MemorySaver, plus the router's "previous intent / previous
question" maps that make a bare follow-up ("what about mold?") work. Both now live
in one SQLite file, so a restart, a crash or a second worker process keeps the
conversation.

    SESSIONS_DB   path of the file (default data/sessions.db); ":memory:" gives a
                  private throwaway database (the test suite sets this)

Two pieces:
- `create_checkpointer()`: a LangGraph `SqliteSaver` for a specialist graph. Every
  specialist has its own saver (its own connection), but they share one file, and
  a saver keys rows by thread_id alone. So `thread_id(agent, session_id)` prefixes
  the agent name: without it the Coverage and Exclusions agents would write to the
  same thread and read each other's history, undoing the per-agent isolation of
  CLAUDE.md lesson 20.
- `SessionMap`: a small dict-like store (get / [] = / in / clear) for the router.

PRIVACY: this file holds the conversation text: the questions as each agent
received them (the router's rewrite; the original for claims) and the models'
answers BEFORE output scrubbing (guardrails.py scrubs what is returned and
logged, not what the graph stores). It is created readable by the owner only, and
is gitignored. Treat it as sensitive; nothing here expires or deletes old sessions.

Concurrency: connections allow use from several threads (the coverage fan-out runs
two specialists at once), SQLite runs in WAL mode (SqliteSaver turns it on) so
readers don't block the writer, and a busy writer waits up to BUSY_TIMEOUT_SECONDS
instead of failing with "database is locked".
"""

import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_DB = "data/sessions.db"
BUSY_TIMEOUT_SECONDS = 30.0


def db_path() -> str:
    """Where sessions are stored; read on every call so a test can redirect it."""
    return os.environ.get("SESSIONS_DB") or DEFAULT_DB


def _connect() -> sqlite3.Connection:
    path = db_path()
    if path != ":memory:":
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        # Create it owner-only BEFORE sqlite does (it would use the default 0644);
        # the -wal and -shm files sqlite adds copy the main file's permissions.
        os.close(os.open(file, os.O_CREAT | os.O_RDWR, 0o600))
    return sqlite3.connect(path, check_same_thread=False, timeout=BUSY_TIMEOUT_SECONDS)


def create_checkpointer() -> SqliteSaver:
    """A SqliteSaver on its own connection to the sessions database."""
    return SqliteSaver(_connect())


def thread_id(agent: str, session_id: str) -> str:
    """The checkpoint thread for one specialist's side of a session."""
    return f"{agent}:{session_id}"


class SessionMap:
    """A persistent str -> str map for one kind of router state (a drop-in for dict)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _db(self) -> sqlite3.Connection:
        # Lazy, so importing the router opens no file (CLAUDE.md lesson 11).
        if self._conn is None:
            self._conn = _connect()
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS router_state ("
                "kind TEXT NOT NULL, session_id TEXT NOT NULL, value TEXT NOT NULL, "
                "PRIMARY KEY (kind, session_id))"
            )
            self._conn.commit()
        return self._conn

    def get(self, session_id: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._db().execute(
                "SELECT value FROM router_state WHERE kind = ? AND session_id = ?",
                (self.kind, session_id),
            ).fetchone()
        return row[0] if row else default

    def __setitem__(self, session_id: str, value: str) -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "INSERT INTO router_state (kind, session_id, value) VALUES (?, ?, ?) "
                "ON CONFLICT (kind, session_id) DO UPDATE SET value = excluded.value",
                (self.kind, session_id, value),
            )
            db.commit()

    def __contains__(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def clear(self) -> None:
        with self._lock:
            db = self._db()
            db.execute("DELETE FROM router_state WHERE kind = ?", (self.kind,))
            db.commit()
