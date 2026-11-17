"""
test_live_behavior.py — End-to-end tier for behaviour cases (G, H, K) and lay-wording cases (H)
===============================================================================================
LIVE and costs real API calls (router + GPT-4o per question).

    RUN_LIVE_EVAL=full pytest tests/integration/test_live_behavior.py -v -s

Asserts only what can be checked mechanically: a guardrail case is rejected before any
agent runs, a forbidden phrase never appears, every dollar figure in the answer appears
in the text the agents retrieved ("no invented figures"), and the H retrieval cases'
wording was retrieved by whatever queries the agents actually wrote. It PRINTS each
answer in full, because "is this a good answer" is a judgement that a person must make
(CLAUDE.md lesson 33). Confidence is NOT asserted for G cases: it measures grounding,
and an unanswerable question will usually still retrieve text.
"""

import os
import re

import pytest
from dotenv import load_dotenv

from tests.integration.golden import BEHAVIOR, GOLDEN, evaluate

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_EVAL") != "full", reason="set RUN_LIVE_EVAL=full"),
]

_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")


@pytest.fixture(scope="module", autouse=True)
def live_environment() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set (expected in .env)")


def _figures(text: str) -> set[str]:
    return {re.sub(r"[\s,]", "", m) for m in _MONEY.findall(text)}


def _show(case_id: str, result) -> None:
    unverified = sum(len(r.unverified_quotes) for r in result.results)
    print(f"\n=== {case_id}: agent={result.agent} confidence={result.confidence} "
          f"unverified_quotes={unverified}")
    for r in result.results:
        for q in r.queries:
            print(f"    [{r.agent_name}] searched -> {q}")
    print("    ANSWER:\n" + "\n".join("      " + line for line in result.answer.splitlines()))


@pytest.mark.parametrize("case", BEHAVIOR, ids=lambda c: c.id)
def test_behavior_case(case) -> None:
    from src.agents.guardrails import GuardrailViolation
    from src.agents.router import route_and_answer

    if case.blocked_by_guardrail:
        with pytest.raises(GuardrailViolation):
            route_and_answer(case.question)
        print(f"\n=== {case.id}: rejected by the guardrail, as expected")
        return

    result = route_and_answer(case.question)
    _show(case.id, result)
    answer = result.answer.lower()
    for forbidden in case.must_not_contain:
        assert forbidden.lower() not in answer, f"answer contains {forbidden!r}"
    if case.no_invented_figures:
        retrieved = " ".join(e["text"] for r in result.results for e in r.evidence)
        invented = _figures(result.answer) - _figures(retrieved)
        assert not invented, f"dollar figures not in the retrieved text: {sorted(invented)}"


H_RETRIEVAL = [c for c in GOLDEN if c.category == "H"]


@pytest.mark.parametrize("case", H_RETRIEVAL, ids=lambda c: c.id)
def test_lay_wording_question_end_to_end(case) -> None:
    from src.agents.router import route_and_answer

    result = route_and_answer(case.question)
    _show(case.id, result)
    evidence = [e for r in result.results for e in r.evidence]
    retrieval = evaluate(case, evidence)
    print("    " + retrieval.line())
    assert retrieval.passed, retrieval.explain(evidence)
