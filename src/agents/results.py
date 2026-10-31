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
    medium - some tool calls returned policy text and some returned nothing
    high   - every tool call returned policy text

Tool results that contain policy text always carry a "[Source: ...]" label
(see the tools in langgraph_agent.py), so "returned text" is detected by that
label rather than by matching the tools' empty-result wording.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

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
    return "high" if len(with_text) == len(tool_results) else "medium"


def weakest_confidence(levels: Iterable[Confidence]) -> Confidence:
    """The lowest of the given confidence levels."""
    return min(levels, key=_CONFIDENCE_RANK.__getitem__)
