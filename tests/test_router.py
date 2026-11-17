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
    assert result.queries == ("search_policy: x",)  # what the agent actually searched for


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


def test_exclusions_prompt_tells_the_agent_to_search_broadly_and_cover_every_exclusion() -> None:
    """Live finding: for 'water damage' the agent ran ONE short search and flood never
    came back."""
    prompt = " ".join(exclusions_agent.SYSTEM_PROMPT.split())
    assert "search SEVERAL times with different wording" in prompt
    assert "one search per distinct cause or kind of loss" in prompt
    assert "Cover EVERY distinct exclusion" in prompt


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


# ---------------------------------------------------------------------------
# Out of scope: say so, instead of padding or answering from model knowledge
# (live finding, CLAUDE.md lesson 33: 0 of 4 unanswerable questions answered cleanly)
# ---------------------------------------------------------------------------


def test_the_classifier_prompt_defines_out_of_scope_and_defaults_to_the_policy() -> None:
    assert "out_of_scope" in router.INTENTS
    prompt = " ".join(router.ROUTER_PROMPT.split())
    assert "out_of_scope - clearly NOT about this homeowners policy" in prompt
    assert "even if the policy may not answer it" in prompt  # a home question the policy may not answer
    assert "suing or disputing the insurer" in prompt  # live finding: "how long do I have to sue my insurer?"
    assert "selling the house or transferring the policy" in prompt  # live finding: buyer takes over the policy


def test_an_out_of_scope_question_runs_no_agent_and_says_so(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("out_of_scope"))

    result = router.route_and_answer("Can I deduct my premium on my taxes?", "s1")

    for mock in fake_agents.values():
        mock.assert_not_called()
    assert result.answer == router.OUT_OF_SCOPE_ANSWER
    assert result.agent == "out_of_scope" and result.results == () and result.sources == []


def test_an_out_of_scope_question_leaves_the_sessions_follow_up_context_alone(
    fake_agents, monkeypatch
) -> None:
    router._session_intent.clear()
    router._session_question.clear()
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("exclusions", "", "What excludes mold?"))
    router.route_and_answer("What about mold?", "s1")
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("out_of_scope"))
    router.route_and_answer("What's the weather?", "s1")

    assert router._session_intent.get("s1") == "exclusions"
    assert router._session_question.get("s1") == "What excludes mold?"


def test_an_answer_written_without_any_search_is_discarded(monkeypatch) -> None:
    """Live finding: the tax question reached the Claims Agent, which never searched."""
    unsearched = make_result("claims", confidence="low", answer="Premiums are generally not deductible.", queries=())
    searched = make_result("claims", answer="From the policy.")

    assert router._discard_ungrounded(unsearched).answer == router.UNGROUNDED_ANSWER
    assert router._discard_ungrounded(searched) is searched


def test_route_and_answer_discards_an_unsearched_specialist_answer(monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims"))
    monkeypatch.setattr(
        claims_agent,
        "answer_question",
        lambda q, s: make_result("claims", confidence="low", answer="model knowledge", queries=(), session_id=s),
    )

    result = router.route_and_answer("Can I deduct my premium?", "s1")

    assert "model knowledge" not in result.answer
    assert router.UNGROUNDED_ANSWER in result.answer


@pytest.mark.parametrize("module", SPECIALISTS)
def test_every_specialist_prompt_says_to_state_that_the_policy_is_silent(module) -> None:
    """Live finding: the agents padded (auto deductible) or stretched a nearby provision
    (credit monitoring) instead of saying the policy does not address the question."""
    prompt = " ".join(module.SYSTEM_PROMPT.split())
    assert "If the retrieved passages still do not address the question, say so" in prompt
    assert "Before concluding that the policy is silent, search again with different wording" in prompt
    assert "never answer from general knowledge" in prompt
    assert "A related provision is not an answer" in prompt


# ---------------------------------------------------------------------------
# A part of the question the policy cannot answer (live finding: "will asking
# about coverage raise my rates?" got a dump of coverages and exclusions)
# ---------------------------------------------------------------------------

RATES = router.OUTSIDE_TOPICS["rates_and_claim_history"]


def test_the_classifier_prompt_asks_what_the_policy_cannot_answer() -> None:
    prompt = " ".join(router.ROUTER_PROMPT.split())
    assert "outside_topic" in prompt and "names_specific_subject" in prompt
    assert "rates_and_claim_history" in prompt
    # live finding: a free-text version said the policy was silent on transferring it
    assert "selling the house, transferring or cancelling the policy" in prompt


def test_a_general_question_the_policy_cannot_answer_gets_the_note_and_an_invitation(
    fake_agents, monkeypatch
) -> None:
    router._session_intent.clear()
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM("coverage", outside_topic="rates_and_claim_history", names_specific_subject=False)
    )

    result = router.route_and_answer("Can I check coverage without it counting as a claim?", "s9")

    for mock in fake_agents.values():
        mock.assert_not_called()
    assert result.answer.startswith(f"The policy doesn't say {RATES}")
    assert "If you tell me what happened" in result.answer
    assert result.agent == "not_in_policy" and result.results == ()
    assert router._session_intent.get("s9") is None  # nothing to follow up on


def test_a_specific_question_with_an_unanswerable_part_runs_agents_after_the_note(
    fake_agents, monkeypatch
) -> None:
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM("coverage", outside_topic="rates_and_claim_history", names_specific_subject=True)
    )

    result = router.route_and_answer("Will filing a claim for my stolen laptop raise my rates?", "s1")

    fake_agents["coverage"].assert_called_once()
    assert result.answer.startswith(f"The policy doesn't say {RATES}; that is a question for")
    assert "What IS covered" in result.answer


def test_no_note_when_the_policy_can_answer(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("coverage", names_specific_subject=False))

    result = router.route_and_answer("What does my policy cover?", "s1")

    assert "The policy doesn't say" not in result.answer
    fake_agents["coverage"].assert_called_once()  # a general question WITH nothing unanswerable still runs


def test_an_out_of_scope_label_with_a_not_in_policy_phrase_says_what_the_policy_cannot_state(
    fake_agents, monkeypatch
) -> None:
    monkeypatch.setattr(
        router, "_get_router_llm", lambda: FakeRouterLLM("out_of_scope", outside_topic="rates_and_claim_history")
    )

    result = router.route_and_answer("Will my premium go up if I file a claim?", "s1")

    assert result.answer.startswith("The policy doesn't say whether or how asking about coverage")
    for mock in fake_agents.values():
        mock.assert_not_called()


def test_the_classifier_prompt_keeps_would_the_policy_pay_questions_in_scope() -> None:
    """Live finding: 'will it pay for credit monitoring?' was sent out of scope."""
    prompt = " ".join(router.ROUTER_PROMPT.split())
    assert "never out_of_scope, even if the answer is \"no\"" in prompt


def test_the_outside_topic_is_a_closed_list_the_classifier_cannot_extend() -> None:
    """Live finding: free text let the classifier call the policy silent on transferring it."""
    assert set(router.OUTSIDE_TOPICS) == {"rates_and_claim_history"}
    with pytest.raises(ValueError):
        router.IntentClassification(intent="coverage", outside_topic="policy_transfer")
    assert router.IntentClassification(intent="coverage").outside_topic == "none"


def test_the_classifier_prompt_keeps_reward_and_tip_questions_in_scope() -> None:
    """Live finding: 'does my insurance offer money for information?' (the arson reward) was refused."""
    prompt = " ".join(router.ROUTER_PROMPT.split())
    assert "including rewards, tips, payments to other people" in prompt


def test_the_rewrite_keeps_what_the_user_is_worried_about_not_only_the_event() -> None:
    """Live finding (lesson 35): 'threatening to sue me' was dropped from the rewrite, so the
    agents searched 'slip and fall' and missed the liability passage."""
    prompt = " ".join(router.ROUTER_PROMPT.split())
    assert "what the user is worried about or asking for (a threat to sue" in prompt


# ---------------------------------------------------------------------------
# A specialist that skips its search gets one explicit nudge before the text is discarded
# (live finding: "What does my policy cover?" - the coverage agent never searched, twice in a row)
# ---------------------------------------------------------------------------


def _agent_that_skips_then_searches(calls: list) -> MagicMock:
    def answer(question, session_id):
        calls.append(question)
        searched = router.SEARCH_NUDGE in question
        return make_result(
            "coverage",
            answer="searched answer" if searched else "from memory",
            queries=("search_policy: x",) if searched else (),
            session_id=session_id,
        )

    return MagicMock(side_effect=answer)


def test_a_skipped_search_is_retried_once_with_a_nudge(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(coverage_agent, "answer_question", _agent_that_skips_then_searches(calls))
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("coverage", "What is covered?", "What is excluded?"))
    monkeypatch.setattr(exclusions_agent, "answer_question", lambda q, s: make_result("exclusions", session_id=s))

    result = router.route_and_answer("What does my policy cover?", "s1")

    assert calls == ["What is covered?", "What is covered?" + router.SEARCH_NUDGE]
    assert "searched answer" in result.answer
    assert router.UNGROUNDED_ANSWER not in result.answer


def test_a_search_that_is_skipped_twice_is_still_discarded(monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(claims_agent, "answer_question", lambda q, s: (calls.append(q), make_result("claims", queries=(), answer="memory", session_id=s))[1])
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims"))

    result = router.route_and_answer("How do I file a claim?", "s1")

    assert len(calls) == 2 and calls[1].endswith(router.SEARCH_NUDGE)
    assert router.UNGROUNDED_ANSWER in result.answer and "memory" not in result.answer


def test_a_specialist_that_searched_is_called_once(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("claims"))

    router.route_and_answer("How do I file a claim?", "s1")

    fake_agents["claims"].assert_called_once()


# ---------------------------------------------------------------------------
# A stronger model must confirm "out_of_scope" before the router refuses
# (live finding: the cheap classifier refused the arson reward and selling the house)
# ---------------------------------------------------------------------------


class FakeScopeLLM:
    def __init__(self, relevant: bool = False, error: Exception = None) -> None:
        self.relevant, self.error, self.seen = relevant, error, []

    def invoke(self, messages):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return router.ScopeCheck(relevant=self.relevant)


def test_the_confirmation_prompt_defaults_to_answering(monkeypatch) -> None:
    prompt = " ".join(router.SCOPE_CHECK_PROMPT.split())
    assert "When unsure, answer true" in prompt
    assert "transferring it" in prompt and "rewards" in prompt
    assert router.CONFIRM_MODEL != router.ROUTER_MODEL


def test_a_refusal_the_stronger_model_rejects_goes_to_the_agents(fake_agents, monkeypatch) -> None:
    confirm = FakeScopeLLM(relevant=True)
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("out_of_scope", "COV?", "EXC?"))
    monkeypatch.setattr(router, "_get_confirm_llm", lambda: confirm)

    result = router.route_and_answer("Can the buyer take over my policy?", "s1")

    assert len(confirm.seen) == 1
    assert result.agent == "coverage+exclusions" and result.answer != router.OUT_OF_SCOPE_ANSWER
    fake_agents["coverage"].assert_called_once()


def test_a_refusal_the_stronger_model_confirms_stands(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("out_of_scope"))
    monkeypatch.setattr(router, "_get_confirm_llm", lambda: FakeScopeLLM(relevant=False))

    result = router.route_and_answer("Can I deduct my premium on my taxes?", "s1")

    assert result.answer == router.OUT_OF_SCOPE_ANSWER
    for mock in fake_agents.values():
        mock.assert_not_called()


def test_a_failed_confirmation_keeps_the_refusal(fake_agents, monkeypatch) -> None:
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("out_of_scope"))
    monkeypatch.setattr(router, "_get_confirm_llm", lambda: FakeScopeLLM(error=RuntimeError("api down")))

    result = router.route_and_answer("Can I deduct my premium on my taxes?", "s1")

    assert result.answer == router.OUT_OF_SCOPE_ANSWER


def test_the_confirmation_is_only_asked_when_the_cheap_model_wants_to_refuse(fake_agents, monkeypatch) -> None:
    confirm = FakeScopeLLM(relevant=False)
    monkeypatch.setattr(router, "_get_router_llm", lambda: FakeRouterLLM("coverage"))
    monkeypatch.setattr(router, "_get_confirm_llm", lambda: confirm)

    router.route_and_answer("Is fire covered?", "s1")

    assert confirm.seen == []
