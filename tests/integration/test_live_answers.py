"""
test_live_answers.py — End-to-end smoke test through the router and the LLMs
==============================================================================
LIVE and more expensive: each case runs the classifier, the specialists and
GPT-4o (several LLM calls per question).

    RUN_LIVE_EVAL=full pytest tests/integration/test_live_answers.py -v -s

It asserts only what must hold every time: the question reaches the right
specialist(s), confidence is not "low", and the answer states the required
facts. It PRINTS (does not assert on) the unverified-quote count, so you can
watch that trend as prompts change.
"""

import os

import pytest
from dotenv import load_dotenv

from tests.integration.golden import END_TO_END, evaluate

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_EVAL") != "full", reason="set RUN_LIVE_EVAL=full"),
]


@pytest.fixture(scope="module", autouse=True)
def live_environment() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set (expected in .env)")


@pytest.mark.parametrize("case", END_TO_END, ids=lambda c: c.id)
def test_answer_reaches_the_right_agent_and_states_the_facts(case) -> None:
    from src.agents.router import route_and_answer

    result = route_and_answer(case.question)

    unverified = sum(len(r.unverified_quotes) for r in result.results)
    evidence = [e for r in result.results for e in r.evidence]
    print(
        f"\n{case.id}: agent={result.agent} confidence={result.confidence} "
        f"unverified_quotes={unverified} citations={len(result.citations)}"
    )
    for r in result.results:
        for query in r.queries:
            print(f"    [{r.agent_name}] searched -> {query}")

    # Was the required wording retrieved at all? Separates "never found" from
    # "found but the answer left it out".
    retrieval = evaluate(case, evidence) if case.must_retrieve else None
    if retrieval:
        print("    " + retrieval.line())

    answer = result.answer.lower()
    if case.expected_agent:
        assert result.agent == case.expected_agent
    assert result.confidence != "low"
    for keyword in case.answer_keywords:
        if keyword.lower() not in answer:
            if retrieval is None:
                why = "retrieval was not scored for this case"
            elif retrieval.passed:
                why = "the required wording WAS retrieved - the LLM left it out"
            else:
                why = f"the required wording was NEVER retrieved: {retrieval.missing}"
            pytest.fail(f"answer is missing {keyword!r}: {why}")
