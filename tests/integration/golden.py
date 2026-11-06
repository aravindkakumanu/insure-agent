"""
golden.py — Golden questions for the live retrieval / answer evaluation
=======================================================================
Each case names a question and the policy wording that a good retrieval MUST
surface. Snippets are real policy text (validated against the ingested policy
by tests/test_golden_data.py, which runs offline), matched ignoring case,
spacing and punctuation.

WHY: retrieval tuning was being judged by eye, and two ordering rules in a row
each fixed one question and broke another. This turns "it looked right" into a
repeatable pass/fail with the rank of each required snippet.

Not every case is a regression yet - the first few are the live failures that
motivated this file (flood missing, $5,000 mold limit displaced). Add a case
whenever a live run shows a miss.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from src.agents.results import Evidence, _normalise


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    tool: str  # "search_policy" | "search_exclusions": the retrieval being scored
    must_retrieve: tuple[str, ...]  # policy wording the retrieved chunks must contain
    # End-to-end (RUN_LIVE_EVAL=full): optional, scored only when set
    expected_agent: Optional[str] = None
    answer_keywords: tuple[str, ...] = ()
    # A retrieval gap that is understood and accepted (e.g. the whole pipeline still
    # answers correctly). The live retrieval test then reports XFAIL, not FAIL, and
    # XPASS if it starts passing - so the gap stays visible without hiding real breaks.
    known_gap: str = ""
    # The text actually sent to the search. Defaults to the question; set it to
    # reproduce the SHORT query an agent wrote (e.g. "water damage"), which can
    # retrieve very differently from the full question.
    query: Optional[str] = None

    @property
    def search_text(self) -> str:
        return self.query or self.question


GOLDEN: tuple[GoldenCase, ...] = (
    # --- the live misses that motivated the quota and the eval set -----------
    GoldenCase(
        id="water-exclusions-flood",
        question="What does my policy not cover for water damage?",
        tool="search_exclusions",
        must_retrieve=(
            "Flood, including, but not limited to, surface water",
            "backs up through sewers or drains",
        ),
        expected_agent="exclusions",
        answer_keywords=("flood", "sewer"),
    ),
    GoldenCase(
        id="water-damage-short-query",
        question="What does my policy not cover for water damage?",
        query="water damage",  # what the Exclusions Agent actually searched for (live run)
        tool="search_exclusions",
        must_retrieve=(
            "Flood, including, but not limited to, surface water",
            "backs up through sewers or drains",
        ),
    ),
    GoldenCase(
        id="mold-remediation-limit",
        question="How much will the policy pay to remove mold after a water loss?",
        tool="search_policy",
        must_retrieve=("mold, fungus, wet rot or dry rot remediation",),
        answer_keywords=("5,000",),
    ),
    GoldenCase(
        id="mold-exclusion",
        question="Is mold damage excluded?",
        tool="search_exclusions",
        must_retrieve=("mold, fungus, wet rot, dry rot or bacteria",),
    ),
    # --- burst pipe ----------------------------------------------------------
    GoldenCase(
        id="burst-pipe-coverage",
        question="Does my policy cover water damage from a burst pipe?",
        tool="search_policy",
        must_retrieve=("sudden and accidental escape of water or steam",),
        expected_agent="coverage+exclusions",
        answer_keywords=("sudden and accidental",),
    ),
    GoldenCase(
        id="burst-pipe-exclusions",
        question="What exclusions or conditions apply to water damage from a burst pipe?",
        tool="search_exclusions",
        must_retrieve=(
            "continuous or repeated seepage",
            "vacant, unoccupied or under construction",
        ),
    ),
    # --- claims --------------------------------------------------------------
    GoldenCase(
        id="claim-first-steps",
        question="What do I need to do right after a loss to make a claim?",
        tool="search_policy",
        must_retrieve=(
            "immediately give us or our agent notice",
            "protect the property from further loss",
        ),
    ),
    GoldenCase(
        id="proof-of-loss",
        question="How long do I have to submit a proof of loss?",
        tool="search_policy",
        must_retrieve=("signed, sworn proof of the loss",),
    ),
    GoldenCase(
        id="legal-action-deadline",
        question="How long do I have to sue my insurer?",
        tool="search_policy",
        must_retrieve=("within one year after the inception of loss or damage",),
        known_gap=(
            "raw question vs policy wording ('sue' vs 'bring an action against us'): "
            "rank 5/30 at the default top-4. The end-to-end case passes, because the "
            "agent writes a better query. Raising POLICY_TOP would fix it at +50% context."
        ),
    ),
    GoldenCase(
        id="appraisal",
        question="What happens if I disagree with the amount of loss?",
        tool="search_policy",
        must_retrieve=("written demand for an appraisal",),
    ),
    GoldenCase(
        id="settlement-timing",
        question="How quickly will the insurer settle a covered loss?",
        tool="search_policy",
        must_retrieve=("within 60 days after the amount of loss is finally determined",),
    ),
    GoldenCase(
        id="theft-police-report",
        question="Do I need to report a theft?",
        tool="search_policy",
        must_retrieve=("Report any theft to the police",),
        known_gap=(
            "theft COVERAGE chunks outrank the claims-conditions chunk: rank 6/30 at the "
            "default top-4. The end-to-end case passes (routed to the Claims Agent)."
        ),
    ),
    # --- limits, deductible, other exclusions --------------------------------
    GoldenCase(
        id="deductible",
        question="What is my deductible?",
        tool="search_policy",
        must_retrieve=("deductible shown on the Policy Declarations",),
    ),
    GoldenCase(
        id="silverware-theft-limit",
        question="What is the theft limit for silverware?",
        tool="search_policy",
        must_retrieve=("Theft of goldware, silverware",),
    ),
    GoldenCase(
        id="collapse",
        question="Is building collapse covered?",
        tool="search_exclusions",
        must_retrieve=("Collapse, except as specifically provided",),
    ),
)

# End-to-end cases run through the router and the LLM (more expensive)
END_TO_END: tuple[GoldenCase, ...] = tuple(c for c in GOLDEN if c.expected_agent) + (
    GoldenCase(
        id="claim-storm",
        question="How do I file a claim for storm damage?",
        tool="search_policy",
        must_retrieve=(),
        expected_agent="claims",
        answer_keywords=("proof of loss", "60 days"),
    ),
    # Both failed the raw-retrieval test (lay wording vs policy wording). Does the
    # whole pipeline - router, agent-written queries - still answer them? The agent
    # is left unasserted because it is unclear how these will be classified.
    GoldenCase(
        id="legal-action-deadline-e2e",
        question="How long do I have to sue my insurer?",
        tool="search_policy",
        must_retrieve=(),
        answer_keywords=("one year",),
    ),
    GoldenCase(
        id="theft-police-report-e2e",
        question="Do I need to report a theft?",
        tool="search_policy",
        must_retrieve=(),
        answer_keywords=("police",),
    ),
)


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    ranks: dict[str, Optional[int]]  # snippet -> 1-based rank of first chunk containing it

    @property
    def missing(self) -> list[str]:
        return [snippet for snippet, rank in self.ranks.items() if rank is None]

    @property
    def passed(self) -> bool:
        return not self.missing

    def line(self) -> str:
        shown = ", ".join(f"#{r}" if r else "MISSING" for r in self.ranks.values())
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.case_id:<26} ranks: {shown}"

    def explain(self, evidence: Sequence[Evidence]) -> str:
        got = "\n".join(
            f"    #{i} p{e['page']}-{e['page_end']} d={e['distance']:.3f} {e['text'][:60]!r}"
            for i, e in enumerate(evidence, 1)
        )
        return (
            f"{self.case_id}: not retrieved: {self.missing}\n"
            f"  retrieved {len(evidence)} chunks:\n{got or '    (none)'}"
        )


def section_ranks(case: GoldenCase, evidence: Sequence[Evidence]) -> dict[str, Optional[int]]:
    """
    Rank of each required snippet among ONLY the chunks in exclusions regions.

    This is the number the exclusions quota depends on: the quota guarantees the
    closest N region chunks a place, so a snippet at region-rank 4 needs a quota
    of at least 4, and one at region-rank 12 is out of reach of any sensible quota.
    """
    in_region = [e for e in evidence if e["in_exclusions_section"]]
    normalised = [_normalise(e["text"]) for e in in_region]
    return {
        snippet: next((i for i, text in enumerate(normalised, 1) if _normalise(snippet) in text), None)
        for snippet in case.must_retrieve
    }


def evaluate(case: GoldenCase, evidence: Sequence[Evidence]) -> EvalResult:
    """Score retrieved evidence against a case: where does each required snippet appear?"""
    normalised = [_normalise(e["text"]) for e in evidence]
    ranks: dict[str, Optional[int]] = {}
    for snippet in case.must_retrieve:
        target = _normalise(snippet)
        ranks[snippet] = next((i for i, text in enumerate(normalised, 1) if target in text), None)
    return EvalResult(case.id, ranks)
