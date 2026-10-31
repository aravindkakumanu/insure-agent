"""
results.py — Typed results shared by the specialists and the router
====================================================================
Specialists return an AgentResult; the router assembles a RouterResult from
one or more of them. Lives in its own module because the router imports the
specialists, so the specialists cannot import anything from the router.

CONFIDENCE is deterministic and measures how well GROUNDED an answer is, not
whether it is correct. It is derived from what the agent's tools returned, not
from the LLM's own opinion of itself (which is uncalibrated):

    low    - the doom-loop guard tripped, no tool was called, or no tool
             returned any policy text
    medium - some tool calls returned policy text and some returned nothing,
             or a search_exclusions result contained no exclusion language
    high   - every tool call returned policy text (and every search_exclusions
             result contained exclusion language)

Tool results that contain policy text always carry a "[Source: ...]" label
(see the tools in langgraph_agent.py), so "returned text" is detected by that
label rather than by matching the tools' empty-result wording.

SEARCH_EXCLUSIONS SPECIAL CASE: when no retrieved chunk contains exclusion
language, search_exclusions falls back to returning the top unfiltered chunks.
That is still "text", but it is not evidence that an exclusion exists, so such a
result caps confidence at medium. The keywords are the tool's own
(_EXCLUSION_KEYWORDS), imported so the list lives in one place. This is a
keyword heuristic: a match shows the text uses exclusion wording, not that the
exclusion answers the question.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from src.agents.langgraph_agent import _EXCLUSION_KEYWORDS, search_exclusions

Confidence = Literal["high", "medium", "low"]

_CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

_SOURCE_RE = re.compile(r"\[Source:\s*([^\]]+)\]")


@dataclass(frozen=True)
class AgentResult:
    """What one specialist agent returns for one question."""

    answer: str
    sources: list[str]  # PDF filenames cited by tool results in this turn
    agent_name: str  # "coverage", "exclusions", or "claims"
    confidence: Confidence
    session_id: str  # echoed back so a standalone caller learns a generated id


@dataclass(frozen=True)
class RouterResult:
    """What the router returns: the final answer plus how it was produced."""

    answer: str
    session_id: str
    sources: list[str]  # union across all specialists that ran
    agent: str  # handlers joined with "+", e.g. "coverage+exclusions"
    intent: str  # the classifier's label
    confidence: Confidence  # the weakest confidence among the specialists
    results: tuple[AgentResult, ...]  # per-specialist breakdown


def current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages from the latest HumanMessage on (the current question's turn)."""
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[last_human:]


def extract_sources(turn: Iterable[BaseMessage]) -> list[str]:
    """Sorted, de-duplicated source filenames from the turn's tool results."""
    sources: set[str] = set()
    for msg in turn:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            sources.update(s.strip() for s in _SOURCE_RE.findall(msg.content))
    return sorted(sources)


def _has_exclusion_language(content: str) -> bool:
    """True if the policy text in a tool result uses any exclusion keyword."""
    # Drop the "[Source: ...]" labels first: a filename like "exclusions.pdf"
    # is not policy text and must not count as exclusion language.
    text = _SOURCE_RE.sub("", content).lower()
    return any(keyword in text for keyword in _EXCLUSION_KEYWORDS)


def assess_confidence(turn: Iterable[BaseMessage], fell_back: bool) -> Confidence:
    """Grounding confidence for one specialist's turn (see module docstring)."""
    if fell_back:
        return "low"
    tool_results = [m for m in turn if isinstance(m, ToolMessage)]
    with_text = [
        m for m in tool_results if isinstance(m.content, str) and _SOURCE_RE.search(m.content)
    ]
    if not with_text:
        return "low"
    if len(with_text) < len(tool_results):
        return "medium"  # some tool calls came back empty
    unconfirmed_exclusions = [
        m
        for m in with_text
        if m.name == search_exclusions.name and not _has_exclusion_language(m.content)
    ]
    return "medium" if unconfirmed_exclusions else "high"


def weakest_confidence(levels: Iterable[Confidence]) -> Confidence:
    """The lowest of the given confidence levels."""
    return min(levels, key=_CONFIDENCE_RANK.__getitem__)
