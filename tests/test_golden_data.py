"""
test_golden_data.py — The golden set itself, and the scoring harness
=====================================================================
Offline. Guards the evaluation's own data: a golden snippet that does not exist in
the policy would make a live run fail for the wrong reason.
"""

from pathlib import Path

import pytest

from src.agents.results import _normalise
from tests.fakes import make_evidence
from tests.integration.golden import END_TO_END, GOLDEN, GoldenCase, evaluate, section_ranks

CHROMA_DIR = Path("data/chroma_db")


def test_case_ids_are_unique_and_tools_are_valid() -> None:
    ids = [c.id for c in GOLDEN]
    assert len(ids) == len(set(ids))
    assert {c.tool for c in GOLDEN} <= {"search_policy", "search_exclusions"}
    assert all(c.must_retrieve for c in GOLDEN)  # a case with nothing to check proves nothing


def test_end_to_end_cases_check_something() -> None:
    assert END_TO_END
    assert len({c.id for c in END_TO_END}) == len(END_TO_END)
    assert all(c.expected_agent or c.answer_keywords for c in END_TO_END)  # not vacuous


def test_the_probe_module_loads_and_rejects_unknown_case_ids(capsys) -> None:
    from tests.integration import probe

    assert probe.main(["no-such-case"]) == 2  # fails before any API call
    assert "unknown case ids" in capsys.readouterr().err


@pytest.mark.skipif(not CHROMA_DIR.exists(), reason="no ingested policy on this machine")
def test_every_golden_snippet_exists_in_a_single_ingested_chunk() -> None:
    import chromadb

    docs = (
        chromadb.PersistentClient(path=str(CHROMA_DIR))
        .get_collection("insurance_policies")
        .get(include=["documents"])["documents"]
    )
    chunks = [_normalise(d) for d in docs]
    missing = [
        (case.id, snippet)
        for case in GOLDEN
        for snippet in case.must_retrieve
        if not any(_normalise(snippet) in chunk for chunk in chunks)
    ]
    assert not missing, f"snippets not found in any one ingested chunk: {missing}"


# --- the harness -------------------------------------------------------------

CASE = GoldenCase(
    id="demo", question="q", tool="search_policy",
    must_retrieve=("sudden and accidental escape", "tearing out and replacing"),
)


def test_evaluate_reports_the_rank_of_each_snippet() -> None:
    evidence = [
        make_evidence("unrelated text", chunk_id="a"),
        make_evidence("... the SUDDEN and accidental escape of water ...", chunk_id="b"),
        make_evidence("cost of tearing out, and replacing any part", chunk_id="c"),
    ]

    result = evaluate(CASE, evidence)

    assert result.ranks == {"sudden and accidental escape": 2, "tearing out and replacing": 3}
    assert result.passed and result.missing == []
    assert result.line().startswith("[PASS] demo") and "#2, #3" in result.line()


def test_evaluate_fails_when_a_snippet_is_missing_and_says_which() -> None:
    evidence = [make_evidence("sudden and accidental escape of water", chunk_id="a")]

    result = evaluate(CASE, evidence)

    assert not result.passed
    assert result.missing == ["tearing out and replacing"]
    assert "MISSING" in result.line()
    assert "tearing out and replacing" in result.explain(evidence)


def test_evaluate_with_no_evidence_fails_every_snippet() -> None:
    result = evaluate(CASE, [])
    assert result.missing == list(CASE.must_retrieve)
    assert "(none)" in result.explain([])


def test_a_snippet_split_across_two_chunks_does_not_count() -> None:
    """Each chunk is checked on its own: a split snippet means the LLM never sees it whole."""
    evidence = [make_evidence("sudden and accidental", chunk_id="a"), make_evidence("escape of water", chunk_id="b")]
    assert not evaluate(CASE, evidence).passed


def test_known_gaps_explain_themselves_and_are_real_cases() -> None:
    gaps = [c for c in GOLDEN if c.known_gap]
    assert {c.id for c in gaps} == {"legal-action-deadline", "theft-police-report"}
    assert all(len(c.known_gap) > 40 for c in gaps)  # a reason, not a placeholder


def test_search_text_defaults_to_the_question_and_can_be_overridden() -> None:
    assert CASE.search_text == "q"
    short = GoldenCase(id="s", question="a long question", query="water damage",
                       tool="search_exclusions", must_retrieve=("x",))
    assert short.search_text == "water damage"


def test_the_short_agent_query_is_a_golden_case() -> None:
    case = next(c for c in GOLDEN if c.id == "water-damage-short-query")
    assert case.search_text == "water damage" and case.tool == "search_exclusions"
    assert not case.known_gap  # this one is meant to be fixed, not accepted


def test_section_ranks_count_only_chunks_inside_exclusions_regions() -> None:
    evidence = [
        make_evidence("coverage chunk that mentions flood in passing", chunk_id="a", in_exclusions_section=False),
        make_evidence("backs up through sewers or drains", chunk_id="b", in_exclusions_section=True),
        make_evidence("Flood, including, but not limited to, surface water, waves", chunk_id="c", in_exclusions_section=True),
    ]
    case = GoldenCase(id="x", question="q", tool="search_exclusions",
                      must_retrieve=("Flood, including, but not limited to, surface water", "backs up through sewers", "not present"))

    assert section_ranks(case, evidence) == {
        "Flood, including, but not limited to, surface water": 2,  # 3rd overall, 2nd in a region
        "backs up through sewers": 1,
        "not present": None,
    }
    assert evaluate(case, evidence).ranks["Flood, including, but not limited to, surface water"] == 3
