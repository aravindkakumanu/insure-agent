"""
conftest.py — Test-suite safety net
====================================
Unit tests must be fully offline (CLAUDE.md lesson 12). A test that forgets to
patch a retrieval path would otherwise reach the real ChromaDB and, because
main.py's load_dotenv() puts OPENAI_API_KEY in the environment, the real OpenAI
embeddings API - silently, and at your expense.

This autouse fixture makes the real vector store raise instead. A test that
needs one patches `get_vector_store` itself (monkeypatch overrides this).
"""

import os

import pytest

# Tracing must never leak out of an offline run. main.py's load_dotenv() will not
# override a variable that is already set, so a developer's .env with
# LANGSMITH_TRACING=true cannot switch it on for the unit tests. Done at import
# time because langsmith caches the variable on first read. Live runs
# (RUN_LIVE_EVAL) are left alone so they can be traced on purpose.
if not os.environ.get("RUN_LIVE_EVAL"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Sessions are durable (src/agents/sessions.py). Unit tests get a private in-memory
# database per connection, so they never read or write data/sessions.db. Set at
# import time: the specialists open their checkpointer when their module is imported.
os.environ["SESSIONS_DB"] = ":memory:"

_MESSAGE = (
    "A test tried to use the REAL vector store (ChromaDB + OpenAI embeddings). "
    "Patch get_vector_store in the module under test."
)


def _forbidden():
    raise AssertionError(_MESSAGE)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: needs OpenAI and the ingested ChromaDB; skipped unless RUN_LIVE_EVAL is set "
        "(tests/integration/)",
    )


@pytest.fixture(autouse=True)
def no_real_vector_store(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return  # live evaluation tests deliberately use the real store
    monkeypatch.setattr("src.agents.tools.get_vector_store", _forbidden)
    monkeypatch.setattr("src.agents.langgraph_agent.get_vector_store", _forbidden)
