"""
test_live_retrieval.py — Does retrieval surface the wording each question needs?
================================================================================
LIVE: embeds each question with OpenAI and searches the real ChromaDB. It makes
only embedding calls (no LLM), so a full run is a handful of tiny requests.

    RUN_LIVE_EVAL=1 pytest tests/integration/test_live_retrieval.py -v -s

Skipped by default so the normal `pytest` run stays offline (CLAUDE.md lesson 12).
-s prints each case's pass/fail line with the rank of every required snippet.
Needs OPENAI_API_KEY (read from .env) and an ingested policy.
"""

import os

import pytest
from dotenv import load_dotenv

from tests.integration.golden import GOLDEN, evaluate

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("RUN_LIVE_EVAL"), reason="live eval: set RUN_LIVE_EVAL=1"),
]


@pytest.fixture(scope="module", autouse=True)
def live_environment() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set (expected in .env)")


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c.id)
def test_required_wording_is_retrieved(case, request: pytest.FixtureRequest) -> None:
    from src.agents import tools

    if case.known_gap:
        request.applymarker(pytest.mark.xfail(reason=case.known_gap, strict=False))

    tool = getattr(tools, case.tool)
    message = tool.invoke(
        {"name": tool.name, "args": {"query": case.search_text}, "id": "eval", "type": "tool_call"}
    )

    result = evaluate(case, message.artifact)

    print("\n" + result.line())
    assert result.passed, result.explain(message.artifact)
