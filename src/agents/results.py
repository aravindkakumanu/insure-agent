"""
results.py — Typed results shared by the specialists and the router
====================================================================
Specialists return an AgentResult; the router assembles a RouterResult from
one or more of them. Lives in its own module because the router imports the
specialists, so the specialists cannot import anything from the router.

EVIDENCE: the retrieval tools (tools.py) return an ARTIFACT next to their text:
a list of Evidence records (chunk id, source, page, section, distance, text).
LangChain attaches it to the ToolMessage, so sources, citations, confidence and
quote checking all read structured data instead of regex-parsing the text the
LLM sees.

CONFIDENCE is deterministic and measures how well GROUNDED an answer is, not
whether it is correct. It is derived from what the agent's tools returned, not
from the LLM's own opinion of itself (which is uncalibrated):

    low    - the doom-loop guard tripped, no tool was called, or no tool
             returned any policy text
    medium - some tool calls returned policy text and some returned nothing;
             or a search_exclusions result held no exclusion language; or the
             answer quotes policy text that is not in the retrieved chunks
    high   - every tool call returned policy text, every search_exclusions
             result held exclusion language, and every quote was verified

Distances (Evidence["distance"]) are recorded but NOT used as a threshold: a
sensible cut-off has to be calibrated against live queries on real questions.

QUOTE CHECK: answers quote the policy. find_unverified_quotes() looks for each
quoted passage in the retrieved chunk text, ignoring case, spacing and
punctuation. A miss means the LLM altered or invented the quote, or put its own
words in quotation marks - either way the user should be told.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

Confidence = Literal["high", "medium", "low"]

_CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# Must equal the name of the search_exclusions tool in tools.py (a test pins it);
# kept as a constant here because tools.py imports this module, not vice versa.
EXCLUSIONS_TOOL_NAME = "search_exclusions"


class Evidence(TypedDict):
    """One retrieved policy chunk. Plain dict so checkpointers can serialize it."""

    chunk_id: str
    source: str  # PDF filename
    page: int  # 0 when the chunk was ingested before page metadata existed
    page_end: int
    section: str
    subsection: str
    kind: str  # "coverage" | "exclusions" | "conditions" | "definitions" | "other" | ""
    distance: float  # Chroma distance to the query; LOWER means closer
    text: str
    exclusion_language: bool  # chunk is in / contains exclusion wording


@dataclass(frozen=True)
class Citation:
    """Where in the policy a retrieved chunk came from."""

    source: str
    page: int | None
    page_end: int | None
    section: str
    subsection: str


@dataclass(frozen=True)
class AgentResult:
    """What one specialist agent returns for one question."""

    answer: str
    sources: list[str]  # PDF filenames cited by tool results in this turn
    agent_name: str  # "coverage", "exclusions", or "claims"
    confidence: Confidence
    session_id: str  # echoed back so a standalone caller learns a generated id
    evidence: tuple[Evidence, ...] = ()  # chunks retrieved this turn, de-duplicated
    unverified_quotes: tuple[str, ...] = ()  # quotes not found in the evidence
    queries: tuple[str, ...] = ()  # searches the agent ran, e.g. "search_policy: burst pipe"


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
    citations: tuple[Citation, ...] = ()  # page/section of every retrieved chunk
    questions: dict[str, str] = field(default_factory=dict)  # agent -> question it got


# ---------------------------------------------------------------------------
# Reading a turn's messages
# ---------------------------------------------------------------------------


def current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages from the latest HumanMessage on (the current question's turn)."""
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[last_human:]


def _artifact(message: ToolMessage) -> list[Evidence]:
    """The Evidence list a tool attached to its ToolMessage (empty if none)."""
    artifact = getattr(message, "artifact", None)
    return list(artifact) if isinstance(artifact, list) else []


def extract_queries(turn: Iterable[BaseMessage]) -> tuple[str, ...]:
    """The searches the agent chose to run this turn, as "tool: query" strings.

    The agent writes these itself, so they are what retrieval ACTUALLY ran on -
    not the user's question - and the first thing to read when an answer is
    missing something.
    """
    return tuple(
        f"{call['name']}: {call['args'].get('query', '')}"
        for msg in turn
        if isinstance(msg, AIMessage)
        for call in msg.tool_calls
    )


def extract_evidence(turn: Iterable[BaseMessage]) -> list[Evidence]:
    """Every chunk retrieved this turn, de-duplicated by chunk id, in order."""
    seen: set[str] = set()
    evidence: list[Evidence] = []
    for msg in turn:
        if isinstance(msg, ToolMessage):
            for item in _artifact(msg):
                if item["chunk_id"] not in seen:
                    seen.add(item["chunk_id"])
                    evidence.append(item)
    return evidence


def extract_sources(turn: Iterable[BaseMessage]) -> list[str]:
    """Sorted, de-duplicated source filenames from this turn's evidence."""
    return sorted({e["source"] for e in extract_evidence(turn)})


def citations_for(evidence: Iterable[Evidence]) -> tuple[Citation, ...]:
    """Distinct (source, page, section) citations, ordered by source then page."""
    unique: dict[tuple, Citation] = {}
    for e in evidence:
        citation = Citation(
            source=e["source"],
            page=e["page"] or None,
            page_end=e["page_end"] or None,
            section=e["section"],
            subsection=e["subsection"],
        )
        unique.setdefault(
            (citation.source, citation.page, citation.page_end, citation.section, citation.subsection),
            citation,
        )
    return tuple(sorted(unique.values(), key=lambda c: (c.source, c.page or 0, c.subsection)))


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

_MIN_QUOTE_CHARS = 25  # shorter quotes are usually scare-quoted terms, not citations
_MIN_SEGMENT_CHARS = 12  # after normalisation; skip only trivial fragments around "..."
# (kept low on purpose: an invented tail like "... and also your yacht" is short)
_MAX_QUOTE_CHARS = 1500
_CURLY_QUOTE_RE = re.compile(r"“([^“”]{%d,%d})”" % (_MIN_QUOTE_CHARS, _MAX_QUOTE_CHARS))
_ELISION_RE = re.compile(r"\.{3}|…|\[[^\]]*\]")  # "..." and editorial [brackets]


def _normalise(text: str) -> str:
    """Lowercase and drop everything but letters and digits."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extract_quotes(answer: str) -> list[str]:
    """Quoted passages in an answer: "straight" pairs and “curly” quotes."""
    quotes: list[str] = []
    positions = [m.start() for m in re.finditer('"', answer)]
    # Pair straight quotes in sequence (1st with 2nd, 3rd with 4th, ...), so the
    # prose BETWEEN two quotations is never mistaken for a quotation.
    for open_at, close_at in zip(positions[0::2], positions[1::2]):
        inner = answer[open_at + 1 : close_at]
        if _MIN_QUOTE_CHARS <= len(inner) <= _MAX_QUOTE_CHARS:
            quotes.append(inner)
    quotes.extend(m.group(1) for m in _CURLY_QUOTE_RE.finditer(answer))
    return quotes


def find_unverified_quotes(answer: str, evidence: Sequence[Evidence]) -> tuple[str, ...]:
    """
    Quoted passages in `answer` that do not appear in the retrieved evidence.

    Comparison ignores case, whitespace and punctuation. A quote containing "..."
    or [brackets] is checked piece by piece. Returns each failing quote, trimmed
    to 120 characters.
    """
    corpus = _normalise(" ".join(e["text"] for e in evidence))
    unverified: list[str] = []
    for quote in _extract_quotes(answer):
        segments = [_normalise(part) for part in _ELISION_RE.split(quote)]
        checked = [seg for seg in segments if len(seg) >= _MIN_SEGMENT_CHARS]
        if checked and not all(seg in corpus for seg in checked):
            unverified.append(quote.strip()[:120])
    return tuple(unverified)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def assess_confidence(
    turn: Iterable[BaseMessage],
    fell_back: bool,
    unverified_quotes: Sequence[str] = (),
) -> Confidence:
    """Grounding confidence for one specialist's turn (see module docstring)."""
    if fell_back:
        return "low"
    tool_results = [m for m in turn if isinstance(m, ToolMessage)]
    with_text = [m for m in tool_results if _artifact(m)]
    if not with_text:
        return "low"

    level: Confidence = "high" if len(with_text) == len(tool_results) else "medium"
    for message in with_text:
        if message.name == EXCLUSIONS_TOOL_NAME and not any(
            e["exclusion_language"] for e in _artifact(message)
        ):
            level = "medium"  # text came back, but none of it reads as an exclusion
    if unverified_quotes:
        level = weakest_confidence([level, "medium"])
    return level


def weakest_confidence(levels: Iterable[Confidence]) -> Confidence:
    """The lowest of the given confidence levels."""
    return min(levels, key=_CONFIDENCE_RANK.__getitem__)
