"""
tools.py — Retrieval tools for the Stage 3 specialists
=======================================================
Same two tools as the Stage 2 agent (langgraph_agent.py, which keeps its own
copies as the Stage 2 reference), with two differences:

  1. They return an ARTIFACT alongside the text (`content_and_artifact`): the
     text is what the LLM reads; the artifact is a list of Evidence records
     (chunk id, source, page, section, distance, text) that LangChain attaches
     to the ToolMessage. Sources, citations, confidence and quote checking read
     the artifact instead of regex-parsing the text.
  2. search_exclusions uses the section metadata stamped at ingest
     (src/rag/structure.py) as well as keywords to decide what counts as
     exclusion text, so a chunk inside an exclusions section counts even if its
     wording is unusual. Results stay ordered by distance.

Chunks ingested BEFORE the metadata existed still work: missing fields default
to empty and the keyword test alone decides. Re-ingest to get the full benefit.
"""

import hashlib
import logging
import re

from langchain_core.documents import Document
from langchain_core.tools import tool

# One keyword list for the project: the Stage 2 module owns it.
from src.agents.langgraph_agent import _EXCLUSION_KEYWORDS
from src.agents.results import Evidence
from src.rag.embedder import get_vector_store

logger = logging.getLogger(__name__)

POLICY_TOP = 4  # chunks returned by search_policy (and the exclusions fallback)
TOC_SLACK = 2  # extra candidates fetched so dropping table-of-contents chunks can't shrink the result
EXCLUSION_POOL = 30  # candidates fetched before filtering for exclusions
EXCLUSION_TOP = 6  # exclusion chunks returned; exclusions need breadth

# Affirmative uses of the word: "...loss caused by water not otherwise excluded"
# is a coverage grant, so it must not make a chunk look like an exclusion.
_NEUTRAL_EXCLUDED_RE = re.compile(r"\bnot\s+(?:otherwise\s+)?excluded\b")


def has_exclusion_language(text: str) -> bool:
    """True if the text uses exclusion wording (ignoring "not otherwise excluded")."""
    cleaned = _NEUTRAL_EXCLUDED_RE.sub("", text.lower())
    return any(keyword in cleaned for keyword in _EXCLUSION_KEYWORDS)


def _to_evidence(doc: Document, distance: float) -> Evidence:
    """Build an Evidence record from a retrieved chunk and its distance."""
    meta = doc.metadata
    page = int(meta.get("page") or 0)
    in_section = bool(meta.get("kind") == "exclusions" or meta.get("has_exclusions"))
    return Evidence(
        chunk_id=doc.id or hashlib.sha256(doc.page_content.encode()).hexdigest(),
        source=str(meta.get("source", "unknown")),
        page=page,
        page_end=int(meta.get("page_end") or page),
        section=str(meta.get("section", "")),
        subsection=str(meta.get("subsection", "")),
        kind=str(meta.get("kind", "")),
        distance=float(distance),
        text=doc.page_content,
        exclusion_language=in_section or has_exclusion_language(doc.page_content),
        in_exclusions_section=in_section,
    )


def _search(query: str, k: int) -> list[Evidence]:
    """
    Closest-first Evidence for a query, without table-of-contents chunks.

    A contents page names every section, so it matches almost any query and
    carries no policy terms. The filter is applied after the search (not as a
    Chroma `where`) so chunks ingested before the `kind` field existed still work.
    """
    hits = get_vector_store().similarity_search_with_score(query, k=k + TOC_SLACK)
    evidence = [_to_evidence(doc, distance) for doc, distance in hits]
    return [e for e in evidence if e["kind"] != "toc"][:k]


def _label(e: Evidence) -> str:
    """Human-readable citation label the LLM can quote, e.g. [Source: x.pdf, page 13, ...]."""
    parts = [e["source"]]
    if e["page"]:
        parts.append(
            f"page {e['page']}" if e["page_end"] in (0, e["page"]) else f"pages {e['page']}-{e['page_end']}"
        )
    where = e["subsection"] or e["section"]
    if where:
        parts.append(where)
    return "[Source: " + ", ".join(parts) + "]"


def _format(evidence: list[Evidence]) -> str:
    """The text the LLM reads: each chunk under its citation label."""
    return "\n\n---\n\n".join(f"{_label(e)}\n{e['text']}" for e in evidence)


@tool(response_format="content_and_artifact")
def search_policy(query: str) -> tuple[str, list[Evidence]]:
    """
    Search all insurance policy content for information relevant to a question.

    Use this tool for questions about what the policy COVERS or for general
    policy terms, definitions, and procedures.

    Args:
        query: A search query describing what policy information is needed.

    Returns:
        The most semantically similar policy excerpts, with source labels.
    """
    evidence = _search(query, POLICY_TOP)
    if not evidence:
        return "No relevant policy content found for this query.", []
    return _format(evidence), evidence


@tool(response_format="content_and_artifact")
def search_exclusions(query: str) -> tuple[str, list[Evidence]]:
    """
    Search specifically for exclusions, limitations, and things NOT covered
    by the insurance policy.

    Use this tool when the question is about what is excluded, not covered,
    or limited under the policy.

    Args:
        query: A search query describing the exclusion or limitation to find.

    Returns:
        Policy excerpts that contain exclusion or limitation language.
    """
    # Wide pool: exclusion chunks rarely rank highest for an affirmative query,
    # because negation language shifts their embeddings.
    evidence = _search(query, EXCLUSION_POOL)
    if not evidence:
        return "No exclusion-related content found in the policy.", []

    # The section / keyword test decides what COUNTS as exclusion text; distance
    # alone decides the ORDER (`evidence` is already closest-first). Ranking
    # section chunks ahead of closer matches pulled in irrelevant ones ("Satellite
    # dish antennas", distance 0.91) ahead of the page holding the $5,000 mold
    # limit (0.61).
    flagged = [e for e in evidence if e["exclusion_language"]]
    # No exclusion wording anywhere: return the closest chunks anyway so the
    # answer isn't silently empty. Their exclusion_language is False, which caps
    # the specialist's confidence at "medium" (see results.assess_confidence).
    chosen = flagged[:EXCLUSION_TOP] if flagged else evidence[:POLICY_TOP]
    return _format(chosen), chosen
