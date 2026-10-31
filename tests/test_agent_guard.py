"""
test_agent_guard.py — Doom-loop scenario and the MAX_ITERATIONS fix
====================================================================
A failure-then-fix exercise. A fake LLM that ALWAYS asks for another tool call
simulates an agent stuck in a loop. Fully offline: no OpenAI calls.

  1. test_doom_loop_without_guard  — the failure: LangGraph's blunt
     GraphRecursionError is the only thing that stops the loop.
  2. test_guard_stops_doom_loop    — the fix: a graceful fallback answer, and
     the LLM is called exactly MAX_ITERATIONS times.
  3. test_normal_question_unaffected — a 2-tool question finishes untouched.
  4. test_counter_resets_between_turns — step_counter must not leak across
     /ask calls in the same session (MemorySaver persists state).
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

import src.agents.langgraph_agent as agent


class LoopingLLM:
    """Fake LLM that never answers: every call requests search_policy again."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "search_policy", "args": {"query": "water"}, "id": f"c{self.calls}"}
            ],
        )


class OneToolThenAnswerLLM:
    """Fake LLM for a healthy question: one tool call, then a final answer."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list) -> AIMessage:
        self.calls += 1
        if self.calls % 2 == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_policy", "args": {"query": "x"}, "id": f"c{self.calls}"}
                ],
            )
        return AIMessage(content="Water damage is covered.")


@pytest.fixture(autouse=True)
def offline_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tools hit a fake vector store instead of Chroma/OpenAI embeddings."""
    store = MagicMock()
    store.similarity_search.return_value = []
    monkeypatch.setattr(agent, "get_vector_store", lambda: store)


def test_doom_loop_without_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE FAILURE: with the guard disabled, only LangGraph's hard stop ends it."""
    monkeypatch.setattr(agent, "MAX_ITERATIONS", 10**6)  # guard effectively off
    llm = LoopingLLM()
    monkeypatch.setattr(agent, "_get_llm", lambda: llm)

    with pytest.raises(GraphRecursionError):
        agent.answer_question("Does my policy cover water damage?")


def test_guard_stops_doom_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE FIX: the guard returns a fallback and never makes call #MAX+1."""
    llm = LoopingLLM()
    monkeypatch.setattr(agent, "_get_llm", lambda: llm)

    result = agent.answer_question("Does my policy cover water damage?")

    assert result["answer"] == agent.FALLBACK_ANSWER
    assert llm.calls == agent.MAX_ITERATIONS


def test_normal_question_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = OneToolThenAnswerLLM()
    monkeypatch.setattr(agent, "_get_llm", lambda: llm)

    result = agent.answer_question("Does my policy cover water damage?")

    assert result["answer"] == "Water damage is covered."
    assert llm.calls == 2


def test_counter_resets_between_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many healthy questions in one session must never trip the guard."""
    llm = OneToolThenAnswerLLM()
    monkeypatch.setattr(agent, "_get_llm", lambda: llm)

    session_id = "reset-test"
    for _ in range(agent.MAX_ITERATIONS):  # 25 turns x 2 visits >> 25 total
        result = agent.answer_question("Does my policy cover water damage?", session_id)
        assert result["answer"] == "Water damage is covered."
