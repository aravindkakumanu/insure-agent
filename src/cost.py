"""
cost.py — What one question costs, and a spending limit
========================================================
Every /ask makes several OpenAI calls (the router, one or two specialists that each loop
through several searches, sometimes a confirmation call). About 3 to 6 cents a question, and
a stretch of repeated live runs once cost about $34 before anyone noticed. This module makes
the cost visible and bounded:

- `track()` collects the token usage of EVERY model call made while it is active, including
  calls made in worker threads (the coverage fan-out), and prices it.
- `SpendLedger` keeps a running total per day in the sessions SQLite file, so the total
  survives a restart.
- `check_budget()` refuses a new question once today's total reaches DAILY_BUDGET_USD.

    DAILY_BUDGET_USD   spend cap per UTC day in dollars; unset or 0 means no cap
    MAX_QUESTION_CHARS read by guardrails.py (a long question costs more on every step)

PRICES ARE A SNAPSHOT (dollars per million tokens, November 2026). A model not listed is
priced as gpt-4o, the expensive one, so an unknown model never looks cheaper than it is.
Update PRICES_PER_MILLION when OpenAI changes them; the numbers are estimates, not a bill.
"""

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

from src.agents.sessions import _connect

logger = logging.getLogger(__name__)

# (input, output) dollars per million tokens. Most specific model name first.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
DEFAULT_PRICE = PRICES_PER_MILLION["gpt-4o"]

# One question costing more than this is logged as a warning (typical: $0.03 to $0.06).
SINGLE_QUESTION_WARN_USD = 0.25


def price_for(model: str) -> tuple[float, float]:
    """(input, output) price per million tokens for a model name like 'gpt-4o-2024-08-06'."""
    for name, price in PRICES_PER_MILLION.items():  # dict order: most specific first
        if model.startswith(name):
            return price
    return DEFAULT_PRICE


@dataclass(frozen=True)
class Usage:
    """What a set of model calls consumed."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class UsageTracker(BaseCallbackHandler):
    """Adds up the token usage of every model call it is told about (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._usage = Usage()

    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001 - langchain's signature
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                meta = getattr(message, "usage_metadata", None)
                if not meta:
                    continue
                model = (getattr(message, "response_metadata", None) or {}).get("model_name", "")
                tokens_in, tokens_out = meta.get("input_tokens", 0), meta.get("output_tokens", 0)
                price_in, price_out = price_for(model)
                cost = (tokens_in * price_in + tokens_out * price_out) / 1e6
                with self._lock:
                    u = self._usage
                    self._usage = Usage(
                        u.calls + 1, u.input_tokens + tokens_in, u.output_tokens + tokens_out,
                        u.cost_usd + cost,
                    )

    def usage(self) -> Usage:
        with self._lock:
            return self._usage


# langchain attaches the handler held in this variable to every model call made in the same
# context. Worker threads inherit it because the router submits them with copy_context()
# (src/tracing.py `submit_in_context`), the same mechanism that carries the trace.
_current: ContextVar[Optional[UsageTracker]] = ContextVar("usage_tracker", default=None)
register_configure_hook(_current, True)

_process_total = 0.0
_process_lock = threading.Lock()


def process_total() -> float:
    """Dollars tracked so far by this process (the live evaluation budgets against it)."""
    return _process_total


@contextmanager
def track() -> Iterator[UsageTracker]:
    """Collect the usage of every model call made inside the block."""
    global _process_total
    tracker = UsageTracker()
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)
        with _process_lock:
            _process_total += tracker.usage().cost_usd


class BudgetExceeded(Exception):
    """Today's spend has reached DAILY_BUDGET_USD. The message is written to be shown."""


def daily_budget() -> Optional[float]:
    """The cap in dollars, or None for no cap. Read on every call so a test can set it."""
    raw = os.environ.get("DAILY_BUDGET_USD", "").strip()
    try:
        value = float(raw) if raw else 0.0
    except ValueError:
        logger.warning("DAILY_BUDGET_USD=%r is not a number; no cap applied", raw)
        return None
    return value if value > 0 else None


class SpendLedger:
    """Dollars spent per UTC day, in the sessions database (survives a restart)."""

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:  # lazy, like SessionMap: importing opens no file
            self._conn = _connect()
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS spend (day TEXT PRIMARY KEY, usd REAL NOT NULL)"
            )
            self._conn.commit()
        return self._conn

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def today(self) -> float:
        with self._lock:
            row = self._db().execute("SELECT usd FROM spend WHERE day = ?", (self._today(),)).fetchone()
        return row[0] if row else 0.0

    def add(self, usd: float) -> float:
        """Add to today's total and return it."""
        with self._lock:
            db = self._db()
            db.execute(
                "INSERT INTO spend (day, usd) VALUES (?, ?) "
                "ON CONFLICT (day) DO UPDATE SET usd = usd + excluded.usd",
                (self._today(), usd),
            )
            db.commit()
        return self.today()


ledger = SpendLedger()


def check_budget() -> None:
    """Raise BudgetExceeded if a daily cap is set and today's spend has reached it."""
    cap = daily_budget()
    if cap is None:
        return
    spent = ledger.today()
    if spent >= cap:
        raise BudgetExceeded(
            f"The daily spending limit (${cap:.2f}) has been reached (${spent:.2f} spent today, UTC). "
            "Try again tomorrow, or raise DAILY_BUDGET_USD."
        )


def record(usage: Usage) -> None:
    """Add a finished question's cost to today's total and log it."""
    day_total = ledger.add(usage.cost_usd) if usage.cost_usd else ledger.today()
    logger.info(
        "Cost: $%.4f (%d model calls, %d input and %d output tokens); today $%.2f",
        usage.cost_usd, usage.calls, usage.input_tokens, usage.output_tokens, day_total,
    )
    if usage.cost_usd > SINGLE_QUESTION_WARN_USD:
        logger.warning("One question cost $%.2f, far above the usual 3-6 cents", usage.cost_usd)
