"""
test_router.py — Stage 3 multi-agent router tests
==================================================
Fully offline: fake LLMs and a fake vector store, no OpenAI calls.

Covers the specialists (tool subsets, doom-loop guard, evidence, confidence,
quote checking), the router (classification, per-agent sub-questions, fan-out,
session pass-through), the structured merge (cautions, redundancy, citations)
and the /ask wiring in main.py.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.agents.langgraph_agent as stage2
from src.agents import claims_agent, coverage_agent, exclusions_agent, router, tools
from src.agents.results import AgentResult, Citation, RouterResult
from src.agents.router import RoutingDecision, merge_results
from tests.fakes import (
    FakeRouterLLM,
    LoopingLLM,
    OneToolThenAnswerLLM,
    make_doc,
    make_evidence,
    make_result,
    store_with,
)

SPECIALISTS = [coverage_agent, exclusions_agent, claims_agent]


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty fake store by default; router session memory starts empty."""
    monkeypatch.setattr(tools, "get_vector_store", lambda: store_with())
    router._session_intent.clear()
    router._session_question.clear()


@pytest.fixture
def use_store(monkeypatch: pytest.MonkeyPatch):
    def _use(*docs, **kwargs):
        store = store_with(*docs, **kwargs)
        monkeypatch.setattr(tools, "get_vector_store", lambda: store)
        return store

    return _use


def _tool_names(module) -> set[str]:
    return {t.name for t in module.TOOLS}


# ---------------------------------------------------------------------------
# Specialists
# ---------------------------------------------------------------------------


def test_tool_subsets() -> None:
    assert _tool_names(coverage_agent) == {"search_policy"}
    assert _tool_names(exclusions_agent) == {"search_exclusions"}
    assert _tool_names(claims_agent) == {"search_policy", "search_exclusions"}


def test_specialists_use_the_artifact_returning_tools() -> None:
    for module in SPECIALISTS:
        for tool in module.TOOLS:
            assert tool in (tools.search_policy, tools.search_exclusions)
            assert tool.response_format == "content_and_artifact"


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
    assert result.unverified_quotes == ()  # the fallback message is not checked
    assert llm.calls == module.MAX_ITERATIONS


@pytest.mark.parametrize("module", SPECIALISTS, ids=lambda m: m.AGENT_NAME)
def test_counter_resets_between_turns(module, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = OneToolThenAnswerLLM(tool=sorted(_tool_names(module))[0])
    monkeypatch.setattr(module, "_get_llm", lambda: llm)

    for _ in range(module.MAX_ITERATIONS):
        result = module.answer_question("Does my policy cover water damage?", "reset-test")
        assert result.answer == "an answer"


def test_specialist_returns_evidence_with_page_and_section(use_store, monkeypatch) -> None:
    use_store(
        make_doc(
            "We cover fire damage to the dwelling.",
            chunk_id="c1",
            page=8,
            section="Section I—Your Property",
            subsection="Losses We Cover Under Coverages A and B",
            kind="coverage",
        )
    )
    llm = OneToolThenAnswerLLM(answer='It says "We cover fire damage to the dwelling." Clear.')
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: llm)

    result = coverage_agent.answer_question("Is fire covered?", "ev-test")

    assert result.agent_name == "coverage" and result.session_id == "ev-test"
    assert result.sources == ["p.pdf"]
    (evidence,) = result.evidence
    assert (evidence["chunk_id"], evidence["page"], evidence["kind"]) == ("c1", 8, "coverage")
    assert evidence["subsection"] == "Losses We Cover Under Coverages A and B"
    assert result.unverified_quotes == ()
    assert result.confidence == "high"


def test_sources_and_evidence_only_from_current_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MagicMock()
    store.similarity_search_with_score.side_effect = [
        [(make_doc("a", chunk_id="a", source="a.pdf"), 0.1)],
        [(make_doc("b", chunk_id="b", source="b.pdf"), 0.1)],
    ]
    monkeypatch.setattr(tools, "get_vector_store", lambda: store)
    llm = OneToolThenAnswerLLM()
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: llm)

    first = coverage_agent.answer_question("What is covered for fire?", "src-test")
    second = coverage_agent.answer_question("What about theft?", "src-test")

    assert first.sources == ["a.pdf"] and [e["chunk_id"] for e in first.evidence] == ["a"]
    assert second.sources == ["b.pdf"] and [e["chunk_id"] for e in second.evidence] == ["b"]


def test_unverified_quote_is_surfaced_and_caps_confidence(use_store, monkeypatch) -> None:
    use_store(make_doc("We cover fire damage to the dwelling.", kind="coverage"))
    llm = OneToolThenAnswerLLM(answer='The policy states "all flood damage is covered in full."')
    monkeypatch.setattr(coverage_agent, "_get_llm", lambda: llm)

    result = coverage_agent.answer_question("Is flood covered?", "q-test")

    assert result.confidence == "medium"
    assert len(result.unverified_quotes) == 1
    assert result.unverified_quotes[0].startswith("all flood damage")


def test_no_retrieved_text_is_low(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = OneToolThenAnswerLLM(tool="search_exclusions")
    monkeypatch.setattr(exclusions_agent, "_get_llm", lambda: llm)  # store is empty

    result = exclusions_agent.answer_question("What is excluded?", "empty-test")

    assert (result.confidence, result.sources, result.evidence) == ("low", [], ())


def test_burst_pipe_coverage_grant_is_not_high_confidence_exclusions(
    use_store, monkeypatch
) -> None:
    """THE REGRESSION from the live run: the Exclusions Agent got the coverage
    grant back (it says 'not otherwise excluded') and was rated high."""
    use_store(
        make_doc(
            "we cover the direct physical damage caused by the water or steam. If loss to "
            "covered property is caused by water or steam not otherwise excluded, we will "
            "cover the cost of tearing out and replacing any part of your dwelling.",
            kind="coverage",
            subsection="Losses We Cover Under Coverages A and B",
        )
    )
    llm = OneToolThenAnswerLLM(tool="search_exclusions")
    monkeypatch.setattr(exclusions_agent, "_get_llm", lambda: llm)

    result = exclusions_agent.answer_question("What exclusions apply to a burst pipe?", "pipe")

    assert result.confidence == "medium"
    assert result.evidence[0]["exclusion_language"] is False


def test_exclusions_agent_with_real_exclusion_chunk_is_high(use_store, monkeypatch) -> None:
    use_store(make_doc("Freezing of plumbing is not covered if unoccupied.", kind="exclusions"))
    llm = OneToolThenAnswerLLM(tool="search_exclusions")
    monkeypatch.setattr(exclusions_agent, "_get_llm", lambda: llm)

    assert exclusions_agent.answer_question("What is excluded?", "x").confidence == "high"


@pytest.mark.parametrize("module", SPECIALISTS, ids=lambda m: m.AGENT_NAME)
def test_every_specialist_prompt_requires_verbatim_quotes(module) -> None:
    """Live finding: the LLM put paraphrases in quotation marks (6 of 7 flagged quotes)."""
    prompt = " ".join(module.SYSTEM_PROMPT.split())
    assert "Use quotation marks ONLY for text copied word for word" in prompt
    assert 'Use "..." where you skip words' in prompt
    assert "write it without quotation marks" in prompt


# ---------------------------------------------------------------------------
# Router: classification and sub-questions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("intent", router.INTENTS)
def test_classify_intent_returns_a_decision(intent: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM(intent, "COV?", "EXC?")
    )

    decision = router.classify_intent("anything")

    assert decision == RoutingDecision(intent, "COV?", "EXC?")


def test_blank_or_missing_rewrites_fall_back_to_the_original_question(monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("coverage", "  ", ""))

    decision = router.classify_intent("Is fire covered?")

    assert decision.coverage_question == decision.exclusions_question == "Is fire covered?"


def test_classify_intent_failure_falls_back_to_coverage_with_the_original_question(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM(error=RuntimeError("api down"))
    )

    decision = router.classify_intent("Is fire covered?")

    assert decision == RoutingDecision("coverage", "Is fire covered?", "Is fire covered?")
    assert "classification failed" in caplog.text  # not silent


def test_classify_intent_passes_previous_intent_and_question(monkeypatch) -> None:
    fake = FakeRouterLLM("claims")
    monkeypatch.setattr(router, "_get_router_llm", lambda: fake)

    router.classify_intent(
        "what about mold?", previous_intent="exclusions", previous_question="What is excluded for water?"
    )

    prompt = fake.seen[0][1].content
    assert "previous_intent: exclusions" in prompt
    assert "previous_question: What is excluded for water?" in prompt


def test_router_prompt_asks_for_standalone_rewrites() -> None:
    assert "STANDALONE" in router.ROUTER_PROMPT
    assert "coverage_question" in router.ROUTER_PROMPT and "exclusions_question" in router.ROUTER_PROMPT


# ---------------------------------------------------------------------------
# Router: fan-out, sub-questions, sessions
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_agents(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    mocks = {}
    for module in SPECIALISTS:
        name = module.AGENT_NAME
        mock = MagicMock(
            side_effect=lambda question, session_id, name=name: make_result(
                name,
                answer=f"{name} answer",
                session_id=session_id,
                evidence=(
                    make_evidence(chunk_id=f"{name}-own", source=f"{name}.pdf"),
                    make_evidence(chunk_id="shared", source="shared.pdf"),
                ),
            )
        )
        monkeypatch.setattr(module, "answer_question", mock)
        mocks[name] = mock
    return mocks


def test_coverage_fans_out_with_a_question_per_agent(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(
        router,
        "_get_router_llm",
        lambda: FakeRouterLLM("coverage", "What is covered for a burst pipe?", "What exclusions apply to a burst pipe?"),
    )

    result = router.route_and_answer("Does my policy cover a burst pipe?", "s1")

    fake_agents["coverage"].assert_called_once_with("What is covered for a burst pipe?", "s1")
    fake_agents["exclusions"].assert_called_once_with("What exclusions apply to a burst pipe?", "s1")
    fake_agents["claims"].assert_not_called()
    assert isinstance(result, RouterResult)
    assert result.agent == "coverage+exclusions" and result.intent == "coverage"
    assert result.questions == {
        "coverage": "What is covered for a burst pipe?",
        "exclusions": "What exclusions apply to a burst pipe?",
    }
    assert result.answer.index("coverage answer") < result.answer.index("exclusions answer")
    assert "What IS covered" in result.answer and "What is NOT covered" in result.answer
    assert result.sources == ["coverage.pdf", "exclusions.pdf", "shared.pdf"]
    assert [r.agent_name for r in result.results] == ["coverage", "exclusions"]


def test_exclusions_intent_runs_exclusions_with_its_rewritten_question(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM("exclusions", "COV?", "EXC?")
    )

    result = router.route_and_answer("What is excluded for floods?", "s1")

    fake_agents["exclusions"].assert_called_once_with("EXC?", "s1")
    fake_agents["coverage"].assert_not_called()
    fake_agents["claims"].assert_not_called()
    assert result.agent == "exclusions" and result.answer == "exclusions answer"


def test_claims_intent_runs_claims_with_the_original_question(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims", "COV?", "EXC?"))

    result = router.route_and_answer("How do I file a claim?", "s1")

    fake_agents["claims"].assert_called_once_with("How do I file a claim?", "s1")
    fake_agents["coverage"].assert_not_called()
    fake_agents["exclusions"].assert_not_called()
    assert result.agent == "claims"


def test_without_rewrites_both_agents_get_the_original_question(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("coverage"))

    router.route_and_answer("Is fire covered?", "s1")

    fake_agents["coverage"].assert_called_once_with("Is fire covered?", "s1")
    fake_agents["exclusions"].assert_called_once_with("Is fire covered?", "s1")


def test_session_id_passes_through_and_is_generated(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims"))

    router.route_and_answer("How do I file a claim?", "my-session")
    assert fake_agents["claims"].call_args.args[1] == "my-session"

    result = router.route_and_answer("How do I file a claim?")  # no session_id
    generated = fake_agents["claims"].call_args.args[1]
    assert generated and result.session_id == generated


def test_follow_up_gets_previous_intent_and_standalone_question(fake_agents, monkeypatch) -> None:
    fake = FakeRouterLLM("exclusions", "COV: water?", "What exclusions apply to water damage?")
    monkeypatch.setattr(router, "_get_router_llm", lambda: fake)

    router.route_and_answer("What is excluded for water damage?", "s1")
    router.route_and_answer("what about mold?", "s1")

    first, second = (call[1].content for call in fake.seen)
    assert "previous_intent: none" in first and "previous_question: none" in first
    assert "previous_intent: exclusions" in second
    # the remembered question is the STANDALONE rewrite, so chains of follow-ups keep their topic
    assert "previous_question: What exclusions apply to water damage?" in second


# ---------------------------------------------------------------------------
# Merge: structured fields
# ---------------------------------------------------------------------------


def test_merge_single_result_passes_through() -> None:
    only = make_result("claims", "medium", answer="file within 60 days")

    merged = merge_results([only], "s", "claims")

    assert merged.answer == "file within 60 days"  # no headings, no caution
    assert (merged.agent, merged.confidence, merged.intent) == ("claims", "medium", "claims")
    assert merged.results == (only,)


def test_merge_high_confidence_has_no_caution() -> None:
    merged = merge_results(
        [make_result("coverage", answer="C"), make_result("exclusions", answer="E")], "s", "coverage"
    )

    assert merged.answer == "**What IS covered**\n\nC\n\n**What is NOT covered**\n\nE"
    assert merged.confidence == "high"
    assert "Note:" not in merged.answer


def test_merge_flags_low_confidence_exclusions() -> None:
    """A failed exclusions search must not read as 'nothing is excluded'."""
    merged = merge_results(
        [make_result("coverage", answer="C"), make_result("exclusions", "low", "nothing found")],
        "s",
        "coverage",
    )

    assert merged.confidence == "low"  # weakest link
    before, after = merged.answer.split("**What is NOT covered**")
    assert "Do not assume nothing is excluded" in after
    assert "Note:" not in before


def test_merge_medium_confidence_adds_no_caution_but_lowers_overall() -> None:
    merged = merge_results(
        [make_result("coverage", "medium"), make_result("exclusions", "high")], "s", "coverage"
    )
    assert merged.confidence == "medium"
    assert "Note:" not in merged.answer


def test_merge_unions_sources_and_keeps_breakdown() -> None:
    cov = make_result("coverage", evidence=(make_evidence(chunk_id="1", source="b.pdf"), make_evidence(chunk_id="2", source="a.pdf")))
    exc = make_result("exclusions", evidence=(make_evidence(chunk_id="3", source="c.pdf"), make_evidence(chunk_id="4", source="a.pdf")))

    merged = merge_results([cov, exc], "s", "coverage", {"coverage": "q1", "exclusions": "q2"})

    assert merged.sources == ["a.pdf", "b.pdf", "c.pdf"]
    assert merged.results == (cov, exc)
    assert merged.questions == {"coverage": "q1", "exclusions": "q2"}


def test_merge_requires_a_result() -> None:
    with pytest.raises(ValueError):
        merge_results([], "s", "coverage")


# --- redundancy ------------------------------------------------------------


def test_exclusions_section_that_only_repeats_coverage_chunks_is_flagged() -> None:
    """The live Q1 problem: both agents retrieved the same coverage grant."""
    a, b = make_evidence(chunk_id="a"), make_evidence(chunk_id="b")
    merged = merge_results(
        [make_result("coverage", evidence=(a, b)), make_result("exclusions", evidence=(a,))],
        "s",
        "coverage",
    )

    after = merged.answer.split("**What is NOT covered**")[1]
    assert "drew only on policy text already used above" in after
    assert "drew only on" not in merged.answer.split("**What is NOT covered**")[0]
    assert merged.confidence == "medium"  # both were "high"; redundancy caps it
    assert merged.results[1].confidence == "high"  # the AgentResult itself is untouched
    assert "text" in after  # the section's own answer is kept, not dropped


def test_exclusions_section_with_its_own_chunk_is_not_redundant() -> None:
    a, b = make_evidence(chunk_id="a"), make_evidence(chunk_id="b")
    merged = merge_results(
        [make_result("coverage", evidence=(a,)), make_result("exclusions", evidence=(a, b))],
        "s",
        "coverage",
    )
    assert "drew only on" not in merged.answer
    assert merged.confidence == "high"


def test_a_section_with_no_evidence_is_not_called_redundant() -> None:
    merged = merge_results(
        [make_result("coverage", evidence=(make_evidence(chunk_id="a"),)), make_result("exclusions", "low")],
        "s",
        "coverage",
    )
    assert "drew only on" not in merged.answer  # low-confidence note applies instead
    assert "Do not assume nothing is excluded" in merged.answer


def test_the_first_section_is_never_redundant() -> None:
    a = make_evidence(chunk_id="a")
    merged = merge_results(
        [make_result("coverage", evidence=(a,)), make_result("exclusions", evidence=(make_evidence(chunk_id="z"),))],
        "s",
        "coverage",
    )
    assert "drew only on" not in merged.answer


# --- quote notes -----------------------------------------------------------


def test_unverified_quotes_get_a_note_in_a_merged_section() -> None:
    merged = merge_results(
        [make_result("coverage", "medium", unverified_quotes=("made up wording",)), make_result("exclusions")],
        "s",
        "coverage",
    )
    before, after = merged.answer.split("**What is NOT covered**")
    assert "1 quoted passage could not be matched" in before
    assert "could not be matched" not in after


def test_unverified_quotes_get_a_note_on_a_single_result_too() -> None:
    merged = merge_results(
        [make_result("claims", "medium", answer="Answer.", unverified_quotes=("a", "b"))], "s", "claims"
    )
    assert merged.answer.startswith("Answer.")
    assert "2 quoted passages could not be matched" in merged.answer


# --- citations -------------------------------------------------------------


def test_merge_citations_are_deduplicated_across_agents() -> None:
    shared = make_evidence(chunk_id="s1", page=13, page_end=14, subsection="Losses We Do Not Cover")
    merged = merge_results(
        [
            make_result("coverage", evidence=(make_evidence(chunk_id="c1", page=8, subsection="Losses We Cover"), shared)),
            make_result("exclusions", evidence=(make_evidence(chunk_id="s2", page=13, page_end=14, subsection="Losses We Do Not Cover"),)),
        ],
        "s",
        "coverage",
    )

    assert [(c.page, c.page_end) for c in merged.citations] == [(8, 8), (13, 14)]
    assert all(isinstance(c, Citation) for c in merged.citations)


# ---------------------------------------------------------------------------
# /ask wiring
# ---------------------------------------------------------------------------


def test_ask_endpoint_returns_agent_confidence_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
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
            citations=(Citation("p.pdf", 13, 14, "Section I—Your Property", "Losses We Do Not Cover"),),
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
        "citations": [
            {
                "source": "p.pdf",
                "page": 13,
                "page_end": 14,
                "section": "Section I—Your Property",
                "subsection": "Losses We Do Not Cover",
            }
        ],
    }
