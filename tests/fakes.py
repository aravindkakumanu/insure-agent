"""
fakes.py — Shared fakes and builders for the offline test suite
================================================================
Fake LLMs, a fake vector store, and builders for chunks / evidence / results.
Nothing here touches OpenAI or ChromaDB.
"""

import hashlib
from typing import Optional
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.agents.results import AgentResult, Confidence, Evidence
from src.agents.router import IntentClassification

# ---------------------------------------------------------------------------
# Fake LLMs
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

    def __init__(
        self,
        intent: Optional[str] = None,
        coverage_question: str = "",
        exclusions_question: str = "",
        error: Optional[Exception] = None,
    ) -> None:
        self.intent = intent
        self.coverage_question = coverage_question
        self.exclusions_question = exclusions_question
        self.error = error
        self.seen: list[list] = []

    def invoke(self, messages: list):
        self.seen.append(messages)
        if self.error:
            raise self.error
        return IntentClassification(
            intent=self.intent,
            coverage_question=self.coverage_question,
            exclusions_question=self.exclusions_question,
        )


# ---------------------------------------------------------------------------
# Vector store, chunks, evidence
# ---------------------------------------------------------------------------


def make_doc(
    text: str,
    *,
    chunk_id: Optional[str] = None,
    source: str = "p.pdf",
    page: int = 1,
    page_end: Optional[int] = None,
    section: str = "",
    subsection: str = "",
    kind: str = "",
    has_exclusions: bool = False,
) -> Document:
    """A retrieved chunk as the vector store returns it (with an id)."""
    return Document(
        id=chunk_id or hashlib.sha1(text.encode()).hexdigest()[:12],
        page_content=text,
        metadata={
            "source": source,
            "page": page,
            "page_end": page_end or page,
            "section": section,
            "subsection": subsection,
            "kind": kind,
            "has_exclusions": has_exclusions,
        },
    )


def store_with(*docs: Document, distance: float = 0.3) -> MagicMock:
    """A fake vector store whose every search returns these docs, closest first."""
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (doc, distance + i * 0.01) for i, doc in enumerate(docs)
    ]
    return store


def make_evidence(
    text: str = "some policy text for the evidence record",
    *,
    chunk_id: str = "c1",
    source: str = "p.pdf",
    page: int = 1,
    page_end: Optional[int] = None,
    section: str = "",
    subsection: str = "",
    kind: str = "coverage",
    distance: float = 0.3,
    exclusion_language: bool = False,
    in_exclusions_section: bool = False,
) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        source=source,
        page=page,
        page_end=page_end if page_end is not None else page,
        section=section,
        subsection=subsection,
        kind=kind,
        distance=distance,
        text=text,
        exclusion_language=exclusion_language,
        in_exclusions_section=in_exclusions_section,
    )


def make_result(
    name: str,
    confidence: Confidence = "high",
    answer: str = "text",
    evidence: tuple[Evidence, ...] = (),
    unverified_quotes: tuple[str, ...] = (),
    session_id: str = "s",
) -> AgentResult:
    return AgentResult(
        answer=answer,
        sources=sorted({e["source"] for e in evidence}),
        agent_name=name,
        confidence=confidence,
        session_id=session_id,
        evidence=evidence,
        unverified_quotes=unverified_quotes,
    )
