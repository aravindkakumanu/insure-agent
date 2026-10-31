"""
test_router.py — Stage 3 multi-agent router tests
==================================================
Fully offline: fake LLMs and a fake vector store, no OpenAI calls.

Covers the router (classification, fallback, fan-out and structured merge,
session pass-through), each specialist (tool subset, doom-loop guard, per-turn
counter reset, per-turn sources, typed AgentResult and confidence) and the /ask
wiring in main.py.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import src.agents.langgraph_agent as stage2
from src.agents import claims_agent, coverage_agent, exclusions_agent, router
from src.agents.results import (
    AgentResult,
    RouterResult,
    assess_confidence,
    extract_sources,
    weakest_confidence,
)
from src.agents.router import IntentClassification, merge_results

SPECIALISTS = [coverage_agent, exclusions_agent, claims_agent]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class LoopingLLM:
    """Never answers: every call requests another tool call."""

    def __init__(self, tool: str = "search_policy") -> None:
        self.calls = 0
        self.tool = tool

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[{"name": self.tool, "args": {"query": "x"}, "id": f"c{self.calls}"}],
        )


class OneToolThenAnswerLLM:
    """Healthy specialist: one tool call, then a final answer."""

    def __init__(self, tool: str = "search_policy", answer: str = "an answer") -> None:
        self.calls = 0
        self.tool = tool
        self.answer = answer

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls % 2 == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": self.tool, "args": {"query": "x"}, "id": f"c{self.calls}"}],
            )
        return AIMessage(content=self.answer)


class FakeRouterLLM:
    """Stands in for the structured-output router LLM."""

    def __init__(self, intent: str | None = None, error: Exception | None = None) -> None:
        self.intent = intent
        self.error = error
        self.seen: list[list] = []

    def invoke(self, messages: list):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return IntentClassification(intent=self.intent)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tools hit a fake store; router session memory starts empty."""
    store = MagicMock()
    store.similarity_search.return_value = []
    monkeypatch.setattr(stage2, "get_vector_store", lambda: store)
    router._session_intent.clear()


# ---------------------------------------------------------------------------
# Specialists
# ---------------------------------------------------------------------------


def _tool_names(module) -> set[str]:
    return {t.name for t in module.TOOLS}


def test_tool_subsets() -> None:
    assert _tool_names(coverage_agent) == {"search_policy"}
    assert _tool_names(exclusions_agent) == {"search_exclusions"}
    assert _tool_names(claims_agent) == {"search_policy", "search_exclusions"}


def test_specialists_have_their_own_memory_and_graph() -> None:
    checkpointers = {id(m._checkpointer) for m in SPECIALISTS}
    graphs = {id(m._agent) for m in SPECIALISTS}
    assert len(checkpointers) == 3 and len(graphs) == 3
    assert id(stage2._checkpointer) not in checkpointers


@pytest.mark.parametrize("module", SPECIALISTS, ids=lambda m: m.AGENT_NAME)
def test_doom_loop_guard(module, monkeypatch: pytest.MonkeyPatch) -> None:
    # Each specialist only has its own tools, so loop on one it actually owns.
    llm = LoopingLLM(tool=sorted(_tool_names(module))[0])
    monkeypatch.setattr(module, "_get_llm", lambda: llm)

    result = module.answer_question("Does my policy cover water damage?")

    assert isinstance(result, AgentResult)
    assert result.answer == module.FALLBACK_ANSWER
    assert result.confidence == "low"  # the guard tripped
    assert llm.calls == module.MAX_ITERATIONS


@pytest.mark.parametrize("module", SPECIALISTS, ids=lambda m: m.AGENT_NAME)
def test_counter_resets_between_turns(module, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = sorted(_tool_names(module))[0]
    llm = OneToolThenAnswerLLM(tool=tool)
    monkeypatch.setattr(module, "_get_llm", lambda: llm)

    for _ in range(module.MAX_ITERATIONS):
        result = module.answer_question("Does my policy cover water damage?", "reset-test")
        assert result.answer == "an answer"


def test_sources_only_from_current_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    store.similarity_search.side_effect = [
        [Document(page_content="a", metadata={"source": "a.pdf"})],
        [Document(page_content="b", metadata={"source": "b.pdf"})],
    ]
    monkeypatch.setattr(stage2, "get_vector_store", lambda: store)
    llm = OneToolThenAnswerLLM()
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: llm)

    first = coverage_agent.answer_question("What is covered for fire?", "src-test")
    second = coverage_agent.answer_question("What about theft?", "src-test")

    assert first.sources == ["a.pdf"]
    assert second.sources == ["b.pdf"]


# ---------------------------------------------------------------------------
# Router: classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("intent", router.INTENTS)
def test_classify_intent_returns_label(intent: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM(intent))
    assert router.classify_intent("anything") == intent


def test_classify_intent_failure_falls_back_to_coverage(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM(error=RuntimeError("api down"))
    )
    assert router.classify_intent("anything") == "coverage"
    assert "classification failed" in caplog.text  # not silent


def test_classify_intent_passes_previous_intent(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRouterLLM("claims")
    monkeypatch.setattr(router, "_get_router_llm", lambda: fake)
    router.classify_intent("what about mold?", previous_intent="exclusions")
    assert "previous_intent: exclusions" in fake.seen[0][1].content


# ---------------------------------------------------------------------------
# Router: fan-out, merge, sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_agents(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mocks = {}
    for module in SPECIALISTS:
        m = MagicMock(
            side_effect=lambda question, session_id, name=module.AGENT_NAME: AgentResult(
                answer=f"{name} answer",
                sources=[f"{name}.pdf", "shared.pdf"],
                agent_name=name,
                confidence="high",
                session_id=session_id,
            )
        )
        monkeypatch.setattr(module, "answer_question", m)
        mocks[module.AGENT_NAME] = m
    return mocks


def test_coverage_fans_out_and_merges(
    fake_agents: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("coverage"))

    result = router.route_and_answer("Is water damage covered?", "s1")

    fake_agents["coverage"].assert_called_once()
    fake_agents["exclusions"].assert_called_once()
    fake_agents["claims"].assert_not_called()
    assert isinstance(result, RouterResult)
    assert result.agent == "coverage+exclusions"
    assert result.intent == "coverage"
    assert result.answer.index("coverage answer") < result.answer.index("exclusions answer")
    assert "What IS covered" in result.answer and "What is NOT covered" in result.answer
    assert result.sources == ["coverage.pdf", "exclusions.pdf", "shared.pdf"]
    assert [r.agent_name for r in result.results] == ["coverage", "exclusions"]


def test_exclusions_runs_exclusions_only(
    fake_agents: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("exclusions"))

    result = router.route_and_answer("What is excluded for floods?", "s1")

    fake_agents["exclusions"].assert_called_once()
    fake_agents["coverage"].assert_not_called()
    fake_agents["claims"].assert_not_called()
    assert result.agent == "exclusions"
    assert result.answer == "exclusions answer"


def test_claims_runs_claims_only(fake_agents: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims"))

    result = router.route_and_answer("How do I file a claim?", "s1")

    fake_agents["claims"].assert_called_once()
    fake_agents["coverage"].assert_not_called()
    fake_agents["exclusions"].assert_not_called()
    assert result.agent == "claims"


def test_session_id_passes_through_and_is_generated(
    fake_agents: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims"))

    router.route_and_answer("How do I file a claim?", "my-session")
    assert fake_agents["claims"].call_args.args[1] == "my-session"

    result = router.route_and_answer("How do I file a claim?")  # no session_id
    generated = fake_agents["claims"].call_args.args[1]
    assert generated and result.session_id == generated


def test_follow_up_gets_previous_intent(
    fake_agents: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRouterLLM("exclusions")
    monkeypatch.setattr(router, "_get_router_llm", lambda: fake)

    router.route_and_answer("What is excluded for water damage?", "s1")
    router.route_and_answer("what about mold?", "s1")

    assert "previous_intent: none" in fake.seen[0][1].content
    assert "previous_intent: exclusions" in fake.seen[1][1].content


# ---------------------------------------------------------------------------
# /ask wiring
# ---------------------------------------------------------------------------


def test_ask_endpoint_returns_agent_field(monkeypatch: pytest.MonkeyPatch) -> None:
    import main

    monkeypatch.setattr(
        router,
        "route_and_answer",
        lambda question, session_id=None: RouterResult(
            answer="merged",
            session_id="abc",
            sources=["p.pdf"],
            agent="coverage+exclusions",
            intent="coverage",
            confidence="medium",
            results=(),
        ),
    )

    response = TestClient(main.app).post("/ask", json={"question": "Is fire covered?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "merged",
        "session_id": "abc",
        "sources": ["p.pdf"],
        "agent": "coverage+exclusions",
        "confidence": "medium",
    }


# ---------------------------------------------------------------------------
# Typed results: confidence, sources, structured merge
# ---------------------------------------------------------------------------


def _tool_msg(content: str, i: int = 0) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=f"t{i}")


WITH_TEXT = "[Source: policy.pdf]\nsome policy text"
EMPTY = "No relevant policy content found for this query."


def test_assess_confidence_levels() -> None:
    q = HumanMessage(content="q")
    assert assess_confidence([q, _tool_msg(WITH_TEXT)], fell_back=False) == "high"
    assert (
        assess_confidence([q, _tool_msg(WITH_TEXT, 1), _tool_msg(EMPTY, 2)], fell_back=False)
        == "medium"
    )
    assert assess_confidence([q, _tool_msg(EMPTY)], fell_back=False) == "low"
    assert assess_confidence([q], fell_back=False) == "low"  # answered without searching
    assert assess_confidence([q, _tool_msg(WITH_TEXT)], fell_back=True) == "low"


def test_extract_sources_reads_tool_results_only() -> None:
    turn = [
        HumanMessage(content="see [Source: user.pdf]"),
        _tool_msg("[Source: a.pdf]\nx\n\n---\n\n[Source: b.pdf]\ny"),
        AIMessage(content="quoting [Source: ai.pdf]"),
    ]
    assert extract_sources(turn) == ["a.pdf", "b.pdf"]


def test_weakest_confidence() -> None:
    assert weakest_confidence(["high", "medium", "high"]) == "medium"
    assert weakest_confidence(["high", "low", "medium"]) == "low"
    assert weakest_confidence(["high"]) == "high"


@pytest.mark.parametrize(
    "module,tool", [(coverage_agent, "search_policy"), (exclusions_agent, "search_exclusions")]
)
def test_specialist_confidence_from_retrieval(
    module, tool: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = OneToolThenAnswerLLM(tool=tool)
    monkeypatch.setattr(module, "_get_llm", lambda: llm)

    # fixture store returns no docs -> tool says nothing found -> low
    empty = module.answer_question("Is fire covered?", "conf-empty")
    assert (empty.agent_name, empty.confidence, empty.sources) == (module.AGENT_NAME, "low", [])

    store = MagicMock()
    store.similarity_search.return_value = [
        Document(page_content="fire is covered. exclusions apply", metadata={"source": "p.pdf"})
    ]
    monkeypatch.setattr(stage2, "get_vector_store", lambda: store)
    grounded = module.answer_question("Is fire covered?", "conf-grounded")
    assert grounded.confidence == "high"
    assert grounded.sources == ["p.pdf"]
    assert grounded.session_id == "conf-grounded"


def _result(name: str, confidence: str, answer: str = "text", sources=None) -> AgentResult:
    return AgentResult(
        answer=answer,
        sources=sources or [],
        agent_name=name,
        confidence=confidence,
        session_id="s",
    )


def test_merge_single_result_passes_through() -> None:
    only = _result("claims", "medium", answer="file within 60 days", sources=["p.pdf"])

    merged = merge_results([only], "s", "claims")

    assert merged.answer == "file within 60 days"  # no headings, no caution
    assert (merged.agent, merged.confidence, merged.intent) == ("claims", "medium", "claims")
    assert merged.results == (only,)


def test_merge_high_confidence_has_no_caution() -> None:
    merged = merge_results(
        [_result("coverage", "high", "C"), _result("exclusions", "high", "E")], "s", "coverage"
    )

    assert merged.answer == "**What IS covered**\n\nC\n\n**What is NOT covered**\n\nE"
    assert merged.confidence == "high"
    assert "Note:" not in merged.answer


def test_merge_flags_low_confidence_exclusions() -> None:
    """A failed exclusions search must not read as 'nothing is excluded'."""
    merged = merge_results(
        [_result("coverage", "high", "C"), _result("exclusions", "low", "nothing found")],
        "s",
        "coverage",
    )

    assert merged.confidence == "low"  # weakest link
    exclusions_section = merged.answer.split("**What is NOT covered**")[1]
    assert "Do not assume nothing is excluded" in exclusions_section
    assert "Note:" not in merged.answer.split("**What is NOT covered**")[0]  # coverage is fine


def test_merge_medium_confidence_adds_no_caution_but_lowers_overall() -> None:
    merged = merge_results(
        [_result("coverage", "medium"), _result("exclusions", "high")], "s", "coverage"
    )

    assert merged.confidence == "medium"
    assert "Note:" not in merged.answer


def test_merge_unions_sources_and_keeps_breakdown() -> None:
    cov = _result("coverage", "high", sources=["b.pdf", "a.pdf"])
    exc = _result("exclusions", "high", sources=["c.pdf", "a.pdf"])

    merged = merge_results([cov, exc], "s", "coverage")

    assert merged.sources == ["a.pdf", "b.pdf", "c.pdf"]
    assert merged.results == (cov, exc)


def test_merge_requires_a_result() -> None:
    with pytest.raises(ValueError):
        merge_results([], "s", "coverage")
