"""
conftest.py — A spending guard for the live evaluation
======================================================
The end-to-end tiers call GPT-4o several times per question (about 3 to 6 cents each), and a
few hours of repeated runs once cost about $34. So every live run has a budget, and stops when
it is used up:

    LIVE_EVAL_BUDGET_USD   dollars one pytest run may spend (default 1.00)

The whole 31-case end-to-end set costs roughly $1.50, so a full run stops partway by default.
That is deliberate: run the cases a change can affect (`-k`), or raise the budget on purpose.
The retrieval tier makes only embedding calls and is not counted. The total is printed at the
end of every run.
"""

import os

import pytest

from src import cost

DEFAULT_BUDGET_USD = 1.00


def _budget() -> float:
    try:
        return float(os.environ.get("LIVE_EVAL_BUDGET_USD", DEFAULT_BUDGET_USD))
    except ValueError:
        return DEFAULT_BUDGET_USD


@pytest.fixture(autouse=True)
def live_budget_guard(request: pytest.FixtureRequest):
    """After each live test, stop the whole run if its budget is spent."""
    config = request.config
    if not hasattr(config, "_live_spend_start"):  # first test of the run, before any spend
        config._live_spend_start = cost.process_total()
    yield
    if not request.node.get_closest_marker("live"):
        return
    spent = cost.process_total() - config._live_spend_start
    if spent > _budget():
        pytest.exit(
            f"Live evaluation budget used up: ${spent:.4f} spent of ${_budget():.2f} "
            "(LIVE_EVAL_BUDGET_USD). Run fewer cases with -k, or raise the budget on purpose.",
            returncode=3,
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    spent = cost.process_total() - getattr(config, "_live_spend_start", 0.0)
    if spent > 0:
        terminalreporter.write_line(f"Live evaluation spend this run: about ${spent:.3f}")
