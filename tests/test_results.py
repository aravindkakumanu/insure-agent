"""
test_results.py — Evidence, citations, quote checking and confidence
=====================================================================
Pure functions over messages and evidence records: fully offline.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.results import (
    EXCLUSIONS_TOOL_NAME,
    assess_confidence,
    citations_for,
    current_turn,
    extract_evidence,
    extract_queries,
    extract_sources,
    find_unverified_quotes,
    weakest_confidence,
)
from tests.fakes import make_evidence

Q = HumanMessage(content="q")


def tool_msg(name: str, *evidence, i: int = 0) -> ToolMessage:
    """A ToolMessage as ToolNode produces it: text content plus an artifact."""
    return ToolMessage(
        content="text the LLM sees", tool_call_id=f"t{i}", name=name, artifact=list(evidence)
    )


# ---------------------------------------------------------------------------
# Reading a turn
# ---------------------------------------------------------------------------


def test_current_turn_starts_at_last_human_message() -> None:
    old, new = HumanMessage(content="old"), HumanMessage(content="new")
    answer = AIMessage(content="a")
    assert current_turn([old, AIMessage(content="x"), new, answer]) == [new, answer]


def test_extract_evidence_dedupes_by_chunk_id_in_order() -> None:
    a, b = make_evidence(chunk_id="a"), make_evidence(chunk_id="b")
    turn = [Q, tool_msg("search_policy", a, b, i=1), tool_msg("search_exclusions", b, i=2)]
    assert [e["chunk_id"] for e in extract_evidence(turn)] == ["a", "b"]


def test_extract_evidence_ignores_messages_without_artifacts() -> None:
    plain = ToolMessage(content="[Source: fake.pdf]\nlooks like text", tool_call_id="t")
    turn = [HumanMessage(content="[Source: user.pdf]"), plain, AIMessage(content="[Source: ai.pdf]")]
    assert extract_evidence(turn) == []
    assert extract_sources(turn) == []


def test_extract_sources_is_sorted_and_unique() -> None:
    turn = [
        Q,
        tool_msg(
            "search_policy",
            make_evidence(chunk_id="1", source="b.pdf"),
            make_evidence(chunk_id="2", source="a.pdf"),
            make_evidence(chunk_id="3", source="b.pdf"),
        ),
    ]
    assert extract_sources(turn) == ["a.pdf", "b.pdf"]


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_citations_dedupe_sort_and_blank_out_missing_pages() -> None:
    evidence = [
        make_evidence(chunk_id="1", page=13, page_end=14, subsection="Losses We Do Not Cover"),
        make_evidence(chunk_id="2", page=7, subsection="Property We Cover"),
        make_evidence(chunk_id="3", page=13, page_end=14, subsection="Losses We Do Not Cover"),
        make_evidence(chunk_id="4", page=0),  # ingested before page metadata existed
    ]

    citations = citations_for(evidence)

    assert [(c.page, c.page_end) for c in citations] == [(None, None), (7, 7), (13, 14)]
    assert len(citations) == 3  # chunks 1 and 3 collapse into one


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

POLICY = make_evidence(
    "We will cover sudden and accidental direct physical loss to the dwelling. "
    "This does not include damage to the defective system or appliance from which "
    "the water escaped.\nWe do not cover loss caused by wear and tear."
)


def test_verbatim_quote_is_verified() -> None:
    answer = 'It says "We do not cover loss caused by wear and tear." Clear enough.'
    assert find_unverified_quotes(answer, [POLICY]) == ()


def test_quote_matches_despite_case_whitespace_and_punctuation() -> None:
    answer = 'Quote: "WE DO NOT COVER   loss caused by wear, and tear" - verbatim enough.'
    assert find_unverified_quotes(answer, [POLICY]) == ()


def test_quote_spanning_a_line_break_is_verified() -> None:
    answer = 'Policy: "from which the water escaped. We do not cover loss caused by wear and tear."'
    assert find_unverified_quotes(answer, [POLICY]) == ()


def test_altered_quote_is_unverified() -> None:
    answer = 'It says "We do cover loss caused by wear and tear in every case."'
    unverified = find_unverified_quotes(answer, [POLICY])
    assert len(unverified) == 1 and unverified[0].startswith("We do cover loss")


def test_invented_quote_is_unverified_when_no_evidence() -> None:
    answer = 'The policy states "all water damage is covered without exception."'
    assert len(find_unverified_quotes(answer, [])) == 1


def test_ellipsis_and_brackets_are_checked_piece_by_piece() -> None:
    good = 'Quote: "We will cover sudden and accidental direct physical loss ... the water escaped."'
    assert find_unverified_quotes(good, [POLICY]) == ()
    bracketed = 'Quote: "We will cover [the] sudden and accidental direct physical loss to the dwelling"'
    assert find_unverified_quotes(bracketed, [POLICY]) == ()
    bad = 'Quote: "We will cover sudden and accidental direct physical loss ... and also your yacht."'
    assert len(find_unverified_quotes(bad, [POLICY])) == 1


def test_short_scare_quotes_are_ignored() -> None:
    answer = 'This is "sudden and accidental" damage, not "wear".'
    assert find_unverified_quotes(answer, []) == ()


def test_prose_between_two_quotations_is_not_a_quotation() -> None:
    answer = (
        'It says "We do not cover loss caused by wear and tear." '
        "and then, in my own words, there is a long sentence of paraphrase here "
        'followed by "We will cover sudden and accidental direct physical loss".'
    )
    assert find_unverified_quotes(answer, [POLICY]) == ()


def test_curly_quotes_are_checked() -> None:
    answer = "It says “We do not cover loss caused by wear and tear.” and “made up policy words here”."
    assert len(find_unverified_quotes(answer, [POLICY])) == 1


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

TEXT = make_evidence(chunk_id="t", exclusion_language=False)
EXCL = make_evidence(chunk_id="x", exclusion_language=True, kind="exclusions")


def test_exclusions_tool_name_is_pinned_to_the_real_tool() -> None:
    from src.agents import tools

    assert tools.search_exclusions.name == EXCLUSIONS_TOOL_NAME


def test_all_tool_calls_with_text_is_high() -> None:
    assert assess_confidence([Q, tool_msg("search_policy", TEXT)], fell_back=False) == "high"


def test_some_empty_tool_calls_is_medium() -> None:
    turn = [Q, tool_msg("search_policy", TEXT, i=1), tool_msg("search_policy", i=2)]
    assert assess_confidence(turn, fell_back=False) == "medium"


def test_no_text_at_all_is_low() -> None:
    assert assess_confidence([Q, tool_msg("search_policy")], fell_back=False) == "low"


def test_no_tool_call_is_low() -> None:
    assert assess_confidence([Q, AIMessage(content="from memory")], fell_back=False) == "low"


def test_guard_fallback_is_low() -> None:
    assert assess_confidence([Q, tool_msg("search_policy", TEXT)], fell_back=True) == "low"


def test_exclusions_result_with_exclusion_language_is_high() -> None:
    assert assess_confidence([Q, tool_msg(EXCLUSIONS_TOOL_NAME, EXCL)], fell_back=False) == "high"


def test_exclusions_result_without_exclusion_language_is_medium() -> None:
    """Fallback chunks: text came back (not low) but isn't evidence of an exclusion."""
    assert assess_confidence([Q, tool_msg(EXCLUSIONS_TOOL_NAME, TEXT)], fell_back=False) == "medium"


def test_special_case_only_applies_to_the_exclusions_tool() -> None:
    assert assess_confidence([Q, tool_msg("search_policy", TEXT)], fell_back=False) == "high"


def test_one_exclusions_call_without_language_lowers_a_mixed_turn() -> None:
    turn = [Q, tool_msg(EXCLUSIONS_TOOL_NAME, EXCL, i=1), tool_msg(EXCLUSIONS_TOOL_NAME, TEXT, i=2)]
    assert assess_confidence(turn, fell_back=False) == "medium"


def test_unverified_quotes_cap_confidence_at_medium() -> None:
    turn = [Q, tool_msg("search_policy", TEXT)]
    assert assess_confidence(turn, fell_back=False, unverified_quotes=("made up",)) == "medium"


def test_unverified_quotes_never_raise_a_low_confidence() -> None:
    turn = [Q, tool_msg("search_policy")]
    assert assess_confidence(turn, fell_back=False, unverified_quotes=("made up",)) == "low"


def test_weakest_confidence() -> None:
    assert weakest_confidence(["high", "medium", "high"]) == "medium"
    assert weakest_confidence(["high", "low", "medium"]) == "low"
    assert weakest_confidence(["high"]) == "high"
    with pytest.raises(ValueError):
        weakest_confidence([])


# ---------------------------------------------------------------------------
# Queries the agent chose
# ---------------------------------------------------------------------------


def test_extract_queries_lists_each_search_with_its_tool() -> None:
    turn = [
        Q,
        AIMessage(content="", tool_calls=[
            {"name": "search_policy", "args": {"query": "burst pipe coverage"}, "id": "1"},
            {"name": "search_exclusions", "args": {"query": "water damage exclusions"}, "id": "2"},
        ]),
        tool_msg("search_policy", i=1),
        AIMessage(content="final answer"),
    ]
    assert extract_queries(turn) == (
        "search_policy: burst pipe coverage",
        "search_exclusions: water damage exclusions",
    )


def test_extract_queries_with_no_tool_calls_is_empty() -> None:
    assert extract_queries([Q, AIMessage(content="from memory")]) == ()
