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

import pytest

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
