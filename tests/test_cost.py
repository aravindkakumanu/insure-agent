"""
test_cost.py — Cost tracking, the daily spending cap, and the question-length limit
===================================================================================
Fully offline. A tiny fake chat model reports token usage the way ChatOpenAI does, so the
tracker is exercised through langchain's real callback machinery, including a worker thread.
"""

import threading

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src import cost
from src.agents import router
from src.agents.guardrails import GuardrailViolation, check_question_length
from src.cost import BudgetExceeded, SpendLedger, Usage, check_budget, price_for, track
from src.tracing import submit_in_context
from tests.fakes import FakeRouterLLM, make_result


class UsageChat(BaseChatModel):
    """Answers 'ok' and reports usage like ChatOpenAI: tokens, and the model name."""

    model: str = "gpt-4o"
    tokens_in: int = 1000
    tokens_out: int = 100

    @property
    def _llm_type(self) -> str:
        return "usage-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": self.tokens_in,
                "output_tokens": self.tokens_out,
                "total_tokens": self.tokens_in + self.tokens_out,
            },
            response_metadata={"model_name": f"{self.model}-2024-08-06"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
def db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    path = str(tmp_path / "sessions.db")
    monkeypatch.setenv("SESSIONS_DB", path)
    monkeypatch.setattr(cost, "ledger", SpendLedger())
    return path


# --- pricing and tracking -----------------------------------------------------------


def test_the_cheap_model_is_matched_before_the_expensive_one() -> None:
    assert price_for("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
    assert price_for("gpt-4o-2024-08-06") == (2.50, 10.00)


def test_an_unknown_model_is_priced_as_the_expensive_one() -> None:
    assert price_for("some-new-model") == price_for("gpt-4o")


def test_track_prices_a_model_call() -> None:
    with track() as tracker:
        UsageChat(model="gpt-4o", tokens_in=1000, tokens_out=100).invoke("hi")

    usage = tracker.usage()
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (1, 1000, 100)
    assert usage.cost_usd == pytest.approx((1000 * 2.50 + 100 * 10.00) / 1e6)


def test_track_adds_up_calls_to_different_models() -> None:
    with track() as tracker:
        UsageChat(model="gpt-4o-mini", tokens_in=2000, tokens_out=0).invoke("a")
        UsageChat(model="gpt-4o", tokens_in=2000, tokens_out=0).invoke("b")

    assert tracker.usage().calls == 2
    assert tracker.usage().cost_usd == pytest.approx((2000 * 0.15 + 2000 * 2.50) / 1e6)


def test_calls_made_in_a_worker_thread_are_counted() -> None:
    """The coverage fan-out runs two specialists in threads; their cost must not vanish."""
    from concurrent.futures import ThreadPoolExecutor

    with track() as tracker, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [submit_in_context(pool, UsageChat().invoke, "q") for _ in range(2)]
        for f in futures:
            f.result()

    assert tracker.usage().calls == 2


def test_nothing_is_counted_outside_track() -> None:
    before = cost.process_total()
    UsageChat().invoke("not tracked")
    assert cost.process_total() == before


def test_a_model_that_reports_no_usage_costs_nothing() -> None:
    tracker = cost.UsageTracker()
    tracker.on_llm_end(type("R", (), {"generations": [[ChatGeneration(message=AIMessage(content="x"))]]})())
    assert tracker.usage() == Usage()


# --- the ledger and the cap ---------------------------------------------------------


def test_the_ledger_adds_up_within_a_day_and_survives_a_restart(db: str) -> None:
    SpendLedger().add(0.04)
    assert SpendLedger().add(0.06) == pytest.approx(0.10)  # a new instance: only the file remembers
    assert SpendLedger().today() == pytest.approx(0.10)


def test_a_new_day_starts_at_zero(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = SpendLedger()
    monkeypatch.setattr(SpendLedger, "_today", staticmethod(lambda: "2026-11-17"))
    ledger.add(1.0)
    monkeypatch.setattr(SpendLedger, "_today", staticmethod(lambda: "2026-11-18"))
    assert ledger.today() == 0.0


def test_no_cap_means_no_check(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAILY_BUDGET_USD", raising=False)
    cost.ledger.add(1000.0)
    check_budget()  # does not raise


@pytest.mark.parametrize("value", ["0", "abc", "-5"])
def test_a_zero_negative_or_unreadable_cap_is_no_cap(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_BUDGET_USD", value)
    assert cost.daily_budget() is None


def test_the_cap_refuses_once_it_is_reached(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_BUDGET_USD", "1.00")
    cost.ledger.add(0.99)
    check_budget()  # still under the cap

    cost.ledger.add(0.01)
    with pytest.raises(BudgetExceeded, match=r"\$1\.00"):
        check_budget()


# --- the question-length limit ------------------------------------------------------


def test_a_question_over_the_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_QUESTION_CHARS", "50")
    check_question_length("x" * 50)
    with pytest.raises(GuardrailViolation, match="too long"):
        check_question_length("x" * 51)


def test_the_default_limit_is_a_thousand_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_QUESTION_CHARS", raising=False)
    check_question_length("x" * 1000)
    with pytest.raises(GuardrailViolation):
        check_question_length("x" * 1001)


# --- the whole question -------------------------------------------------------------


@pytest.fixture
def fake_pipeline(monkeypatch: pytest.MonkeyPatch):
    """A routed question whose agents 'spend' a known amount, to check the accounting."""
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims"))

    def claims(question, session_id):
        UsageChat(model="gpt-4o", tokens_in=10_000, tokens_out=0).invoke(question)  # $0.025
        return make_result("claims", session_id=session_id)

    monkeypatch.setattr(router.claims_agent, "answer_question", claims)


def test_a_question_reports_its_cost_and_adds_it_to_todays_total(db: str, fake_pipeline) -> None:
    result = router.route_and_answer("How do I file a claim?", "s1")

    assert result.usage.cost_usd == pytest.approx(0.025)
    assert result.usage.calls == 1
    assert cost.ledger.today() == pytest.approx(0.025)


def test_a_question_over_budget_is_refused_before_any_model_call(
    db: str, fake_pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DAILY_BUDGET_USD", "0.02")
    cost.ledger.add(0.02)
    monkeypatch.setattr(router, "_get_router_llm", lambda: pytest.fail("the classifier must not run"))

    with pytest.raises(BudgetExceeded):
        router.route_and_answer("How do I file a claim?", "s1")


def test_an_overlong_question_is_refused_before_any_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_QUESTION_CHARS", "20")
    monkeypatch.setattr(router, "_get_router_llm", lambda: pytest.fail("the classifier must not run"))

    with pytest.raises(GuardrailViolation):
        router.route_and_answer("How do I file a claim for a very long list of things?", "s1")


# --- the API ------------------------------------------------------------------------


def test_the_api_answers_429_when_the_budget_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    def refuse(question, session_id=None):
        raise BudgetExceeded("The daily spending limit ($1.00) has been reached.")

    monkeypatch.setattr(router, "route_and_answer", refuse)

    response = TestClient(main.app).post("/ask", json={"question": "How do I file a claim?"})

    assert response.status_code == 429
    assert "daily spending limit" in response.json()["detail"]


def test_the_api_answers_400_for_an_overlong_question(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    monkeypatch.setenv("MAX_QUESTION_CHARS", "20")
    monkeypatch.setattr(router, "_get_router_llm", lambda: pytest.fail("the classifier must not run"))

    response = TestClient(main.app).post("/ask", json={"question": "How do I file a claim for a long list?"})

    assert response.status_code == 400
    assert "too long" in response.json()["detail"]
