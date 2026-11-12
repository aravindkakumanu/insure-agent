"""
test_tracing.py — LangSmith tracing
====================================
Fully offline: a recording client keeps runs in memory (tests/fakes.py), so these
tests see exactly what would be sent to LangSmith without sending anything.

What is pinned: tracing is off by default, a coverage question produces one
connected trace (including the specialists that run in worker threads), and the
tool runs carry how many chunks came back.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
from langsmith import tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.utils import tracing_is_enabled

from src.agents import coverage_agent, exclusions_agent, router, tools
from src.tracing import (
    annotate,
    submit_in_context,
    summarize_agent_result,
    summarize_router_result,
)
from tests.fakes import (
    FakeRouterLLM,
    OneToolThenAnswerLLM,
    make_doc,
    make_result,
    recording_client,
    store_with,
)

DOCS = (
    make_doc("We cover fire damage.", chunk_id="a", page=8),
    make_doc("We do not cover flood.", chunk_id="b", page=14, kind="exclusions"),
)


@pytest.fixture
def coverage_question(monkeypatch: pytest.MonkeyPatch):
    """Everything faked: a coverage-intent question that fans out to two agents."""
    store = store_with(*DOCS)
    monkeypatch.setattr(tools, "get_vector_store", lambda: store)
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM("coverage", "COV?", "EXC?")
    )
    # One instance each: the fake counts its calls, so a new one per call would
    # request a tool forever and trip the doom-loop guard.
    coverage_llm = OneToolThenAnswerLLM("search_policy")
    exclusions_llm = OneToolThenAnswerLLM("search_exclusions")
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: coverage_llm)
    monkeypatch.setattr(exclusions_agent, "_get_llm", lambda: exclusions_llm)
    router._session_intent.clear()
    router._session_question.clear()


def traced_ask(question: str = "Is flood covered?"):
    client, log = recording_client()
    with tracing_context(enabled=True, client=client):
        result = router.route_and_answer(question, session_id="trace-test")
    client.flush()
    return result, log


# ---------------------------------------------------------------------------
# Off by default
# ---------------------------------------------------------------------------


def test_tracing_is_off_in_the_test_suite() -> None:
    assert tracing_is_enabled() is False


def test_nothing_is_recorded_when_tracing_is_off(coverage_question) -> None:
    client, log = recording_client()
    with tracing_context(enabled=False, client=client):
        router.route_and_answer("Is flood covered?", session_id="off")
    client.flush()
    assert log.runs == {}


def test_annotate_is_a_no_op_outside_a_trace() -> None:
    assert get_current_run_tree() is None
    annotate(anything="at all")  # must not raise


# ---------------------------------------------------------------------------
# One connected trace per question
# ---------------------------------------------------------------------------


def test_a_coverage_question_is_one_connected_trace(coverage_question) -> None:
    _, log = traced_ask()

    root = log.one("ask")
    assert root.get("parent_run_id") is None
    assert log.ancestors(log.one("route")) == ["ask"]

    # The specialists run in worker threads; they must still hang off the root.
    for agent in ("coverage_agent", "exclusions_agent"):
        assert log.ancestors(log.one(agent)) == ["ask"]

    # Every run belongs to the same trace: nothing is a stray root.
    roots = [r for r in log.runs.values() if not r.get("parent_run_id")]
    assert [r["name"] for r in roots] == ["ask"]


def test_tool_calls_sit_under_the_agent_that_made_them(coverage_question) -> None:
    _, log = traced_ask()

    assert "coverage_agent" in log.ancestors(log.one("search_policy"))
    assert "exclusions_agent" in log.ancestors(log.one("search_exclusions"))


def test_tool_runs_record_how_many_chunks_came_back(coverage_question) -> None:
    _, log = traced_ask()

    policy = log.one("search_policy")["metadata"]
    assert policy["chunks_returned"] == 2
    assert policy["pages"] == [8, 14]
    assert policy["closest_distance"] == pytest.approx(0.3)

    exclusions = log.one("search_exclusions")["metadata"]
    assert exclusions["chunks_returned"] >= 1
    assert exclusions["candidates"] == 2  # what the quota had to choose from


def test_tool_run_records_an_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "get_vector_store", lambda: store_with())
    client, log = recording_client()
    with tracing_context(enabled=True, client=client):
        tools.search_policy.invoke({"query": "anything"})
    client.flush()

    assert log.one("search_policy")["metadata"]["chunks_returned"] == 0


def test_the_tool_query_is_in_the_trace(coverage_question) -> None:
    _, log = traced_ask()
    # OneToolThenAnswerLLM searches for "x"; the run's inputs must show it.
    assert log.one("search_policy")["inputs"]["query"] == "x"


def test_root_run_carries_the_routing_outcome(coverage_question) -> None:
    result, log = traced_ask()

    meta = log.one("ask")["metadata"]
    assert meta["session_id"] == "trace-test"
    assert meta["intent"] == "coverage"
    assert meta["agent"] == result.agent == "coverage+exclusions"
    assert meta["confidence"] == result.confidence


def test_route_run_shows_the_decision_and_the_rewritten_questions(coverage_question) -> None:
    _, log = traced_ask()

    outputs = log.one("route")["outputs"]
    assert outputs == {
        "intent": "coverage",
        "coverage_question": "COV?",
        "exclusions_question": "EXC?",
    }


def test_agent_runs_carry_their_outcome(coverage_question) -> None:
    _, log = traced_ask()

    meta = log.one("coverage_agent")["metadata"]
    assert meta["agent"] == "coverage"
    assert meta["searches"] == 1
    assert meta["chunks_retrieved"] == 2
    assert meta["guard_tripped"] is False


def test_classifier_failure_is_visible_in_the_trace(
    coverage_question, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM(error=RuntimeError("api down"))
    )
    _, log = traced_ask()

    assert log.one("route")["metadata"]["classifier_failed"] is True


# ---------------------------------------------------------------------------
# Why submit_in_context exists
# ---------------------------------------------------------------------------


def test_a_plain_thread_pool_loses_the_trace_but_submit_in_context_keeps_it() -> None:
    client, _ = recording_client()
    with tracing_context(enabled=True, client=client), ThreadPoolExecutor(1) as pool:
        from langsmith import traceable

        @traceable(name="parent")
        def parent():
            plain = pool.submit(get_current_run_tree).result()
            kept = submit_in_context(pool, get_current_run_tree).result()
            return plain, kept

        plain, kept = parent()

    assert plain is None
    assert kept is not None and kept.name == "parent"


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def test_agent_summary_omits_chunk_text() -> None:
    from tests.fakes import make_evidence

    result = make_result("coverage", evidence=(make_evidence("SECRET CHUNK TEXT"),))
    summary = summarize_agent_result(result)

    assert "SECRET CHUNK TEXT" not in str(summary)
    assert summary["confidence"] == "high"


def test_router_summary_counts_citations_instead_of_listing_them() -> None:
    result = router.merge_results(
        [make_result("coverage")], "s", "coverage", {"coverage": "q"}
    )
    summary = summarize_router_result(result)

    assert summary["citations"] == len(result.citations)
    assert set(summary) == {"answer", "agent", "intent", "confidence", "citations"}
