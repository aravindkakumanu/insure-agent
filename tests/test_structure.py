"""
test_structure.py — Page & section metadata for chunks
=======================================================
Offline: synthetic pages shaped like the real policy's headings (including the
traps found on the real PDF - cross-references that look like headings, a
table of contents, and headings that wrap onto a second line).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.loader import load_pdf
from src.rag.structure import (
    _join_pages,
    annotate_chunks,
    build_regions,
    find_headings,
    is_toc_page,
    locate_chunks,
)

TOC = """Table Of Contents
Section I—Your Property.......................7
Property We Do Not Cover Under Coverage A.....7
Losses We Do Not Cover Under Coverage A, Coverage B.....13
Section II—Family Liability And Guest
Medical Protection.................................20
"""

PAGE_COVERAGE = """Section I—Your Property
Dwelling Protection–Coverage A
Property We Cover Under Coverage A:
1. The dwelling on the residence premises.
Property We Do Not Cover Under Coverage A:
1. Land, including land on which the dwelling is located.
Losses We Cover Under Coverages A and B:
We will cover sudden and accidental direct physical loss to the dwelling.
"""

PAGE_EXCLUSIONS = """Losses We Do Not Cover Under Coverage
A, Coverage B And Coverage C
1. Flood. We do not cover loss caused by flood or surface water.
The Section I—Losses We Do Not Cover Under Coverage A reference to earth movement
does not apply to the loss of land.
Additional Protection under item 11, Collapse.
2. Freezing. This exclusion applies regardless of whether or not other causes contribute.
"""

PAGE_CONDITIONS = """Section I Conditions
1.
Deductible
Section I Conditions, Mold, Fungus, Wet Rot And
Dry Rot Remediation is limited to $5,000.
This exclusion applies regardless of whether or not other causes contribute.
Section II—Family Liability And Guest
Medical Protection
Losses We Cover Under Coverage X:
We will pay damages an insured person becomes legally obligated to pay.
"""


def make_pages(*texts: str) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": "x.pdf", "page": i})
        for i, t in enumerate(texts, start=1)
    ]


PAGES = make_pages(TOC, PAGE_COVERAGE, PAGE_EXCLUSIONS, PAGE_CONDITIONS)


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


def titles(pages: list[Document] = PAGES) -> list[tuple[str, str]]:
    return [(h.level, h.title) for h in find_headings(pages)]


def test_real_headings_are_found_and_labelled_by_level() -> None:
    found = titles()
    assert ("section", "Section I—Your Property") in found
    assert ("subsection", "Property We Cover Under Coverage A") in found
    assert ("subsection", "Losses We Cover Under Coverages A and B") in found
    assert ("section", "Section I Conditions") in found
    assert ("section", "Section II—Family Liability And Guest Medical Protection") in found


def test_wrapped_heading_is_joined_with_its_continuation_line() -> None:
    assert (
        "subsection",
        "Losses We Do Not Cover Under Coverage A, Coverage B And Coverage C",
    ) in titles()


def test_a_heading_that_does_not_wrap_is_not_joined_to_the_next_line() -> None:
    # "Section I—Your Property" is followed by "Dwelling Protection–Coverage A"
    assert ("section", "Section I—Your Property") in titles()
    assert not any("Dwelling Protection" in title for _, title in titles())


def test_table_of_contents_page_yields_no_headings() -> None:
    assert find_headings(make_pages(TOC)) == []


def test_cross_references_in_running_text_are_not_headings() -> None:
    found = [title for _, title in titles()]
    assert not any(title.startswith("The Section I") for title in found)
    assert not any(title.startswith("Additional Protection under") for title in found)
    assert not any(title.startswith("Section I Conditions, Mold") for title in found)


def test_heading_offsets_point_at_the_heading_text() -> None:
    full_text, _, _ = _join_pages(PAGES)
    for heading in find_headings(PAGES):
        assert full_text[heading.offset :].startswith(heading.title.split(" ")[0])


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


def test_regions_cover_the_whole_text_without_gaps() -> None:
    full_text, _, _ = _join_pages(PAGES)
    regions = build_regions(find_headings(PAGES), len(full_text))
    assert regions[0].start == 0 and regions[-1].end == len(full_text)
    assert all(a.end == b.start for a, b in zip(regions, regions[1:]))


def test_new_section_resets_the_subsection() -> None:
    full_text, _, _ = _join_pages(PAGES)
    regions = build_regions(find_headings(PAGES), len(full_text))
    conditions = next(r for r in regions if r.section == "Section I Conditions")
    assert conditions.subsection == "" and conditions.kind == "conditions"


# ---------------------------------------------------------------------------
# Locating and annotating chunks
# ---------------------------------------------------------------------------


def chunk(text: str) -> Document:
    """A chunk as the semantic chunker emits it: newlines collapsed to spaces."""
    return Document(page_content=" ".join(text.split()), metadata={"source": "x.pdf"})


def test_locate_tolerates_collapsed_whitespace() -> None:
    full_text, _, _ = _join_pages(PAGES)
    target = "Property We Do Not Cover Under Coverage A: 1. Land, including land on which the dwelling"
    (start, end), = locate_chunks([chunk(target)], full_text)
    assert full_text[start:].startswith("Property We Do Not Cover Under Coverage A:")
    assert end > start


def test_repeated_phrases_resolve_to_the_right_occurrence() -> None:
    """Policy boilerplate repeats; two chunks starting alike must land on different pages."""
    phrase = "This exclusion applies regardless of whether or not other causes contribute."
    first = chunk("2. Freezing. " + phrase)  # page 3
    second = chunk(phrase + " Section I Conditions")  # starts with the SAME words, page 4
    pages = make_pages(TOC, PAGE_COVERAGE, PAGE_EXCLUSIONS, PAGE_CONDITIONS)

    annotate_chunks([first, second], pages)

    assert first.metadata["page"] == 3
    assert second.metadata["page"] == 4


def test_annotate_stamps_page_section_and_kind() -> None:
    cover = chunk(PAGE_COVERAGE.split("Losses We Cover")[0])
    flood = chunk("1. Flood. We do not cover loss caused by flood or surface water.")
    conditions = chunk("Section I Conditions 1. Deductible")

    annotate_chunks([cover, flood, conditions], PAGES)

    assert cover.metadata["page"] == 2
    assert cover.metadata["section"] == "Section I—Your Property"
    assert flood.metadata["page"] == 3
    assert flood.metadata["kind"] == "exclusions"
    assert flood.metadata["subsection"].startswith("Losses We Do Not Cover Under Coverage A")
    assert conditions.metadata["page"] == 4
    assert conditions.metadata["kind"] == "conditions"
    assert [c.metadata["chunk_index"] for c in (cover, flood, conditions)] == [0, 1, 2]


def test_property_we_do_not_cover_is_an_exclusions_kind() -> None:
    land = chunk("1. Land, including land on which the dwelling is located.")
    annotate_chunks([land], PAGES)
    assert land.metadata["kind"] == "exclusions"


def test_chunk_spanning_a_heading_reports_the_page_range() -> None:
    spanning = chunk(
        "Additional Protection under item 11, Collapse. 2. Freezing. This exclusion applies "
        "regardless of whether or not other causes contribute. Section I Conditions 1. Deductible"
    )
    annotate_chunks([spanning], PAGES)
    assert (spanning.metadata["page"], spanning.metadata["page_end"]) == (3, 4)


def test_chunk_that_starts_under_coverage_but_contains_exclusions_is_flagged() -> None:
    """The real 4-page chunk: starts in a coverage region, swallows an exclusions one."""
    straddler = chunk(
        "We will cover sudden and accidental direct physical loss to the dwelling. "
        "Losses We Do Not Cover Under Coverage A, Coverage B And Coverage C "
        "1. Flood. We do not cover loss caused by flood or surface water."
    )
    annotate_chunks([straddler], PAGES)
    assert straddler.metadata["kind"] == "coverage"  # where it STARTS
    assert straddler.metadata["has_exclusions"] is True  # but it overlaps an exclusions region


def test_chunk_wholly_inside_coverage_has_no_exclusions_flag() -> None:
    inside = chunk("We will cover sudden and accidental direct physical loss to the dwelling.")
    annotate_chunks([inside], PAGES)
    assert inside.metadata["has_exclusions"] is False


def test_unlocatable_chunk_is_logged_not_fatal(caplog: pytest.LogCaptureFixture) -> None:
    ghost = chunk("this text appears nowhere in the document at all, truly nowhere")
    annotate_chunks([ghost], PAGES)
    assert "not found in document text" in caplog.text
    assert ghost.metadata["page"] >= 1  # still annotated


def test_metadata_values_are_chroma_compatible() -> None:
    """Chroma accepts only str / int / float / bool metadata."""
    c = chunk("Property We Cover Under Coverage A: 1. The dwelling")
    annotate_chunks([c], PAGES)
    assert all(isinstance(v, (str, int, float, bool)) for v in c.metadata.values())


# ---------------------------------------------------------------------------
# Through load_pdf (fixed_token keeps it offline)
# ---------------------------------------------------------------------------


def test_load_pdf_stamps_metadata_on_every_chunk() -> None:
    mock_pages = []
    for text in (PAGE_COVERAGE, PAGE_EXCLUSIONS):
        page = MagicMock()
        page.get_text.return_value = text
        mock_pages.append(page)
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)

    with patch("src.rag.loader.fitz.open", return_value=mock_doc), \
         patch.object(Path, "exists", return_value=True), \
         patch("src.rag.loader.CHUNK_STRATEGY", "fixed_token"):
        chunks = load_pdf("data/policies/fake.pdf")

    assert [c.metadata["page"] for c in chunks] == [1, 2]
    assert chunks[0].metadata["section"] == "Section I—Your Property"
    assert chunks[1].metadata["kind"] == "exclusions"
    assert all(c.metadata["source"] == "fake.pdf" for c in chunks)


# ---------------------------------------------------------------------------
# Table of contents
# ---------------------------------------------------------------------------


def test_toc_page_is_recognised() -> None:
    assert is_toc_page(make_pages(TOC)[0])
    assert not is_toc_page(make_pages(PAGE_EXCLUSIONS)[0])


def test_chunk_starting_on_the_toc_page_is_kind_toc_and_not_exclusions() -> None:
    toc_chunk = chunk("Table Of Contents Section I—Your Property.......7 Property We Do Not Cover Under Coverage A.....7")
    real = chunk("1. Flood. We do not cover loss caused by flood or surface water.")

    annotate_chunks([toc_chunk, real], PAGES)

    assert toc_chunk.metadata["kind"] == "toc"
    assert toc_chunk.metadata["has_exclusions"] is False
    assert real.metadata["kind"] == "exclusions"  # real content is unaffected


BIG_TOC = TOC + "".join(f"Topic number {i} of the policy.........{i + 20}\n" for i in range(8))


def test_chunk_that_swallows_the_toc_without_starting_on_it_is_still_toc() -> None:
    """The real first chunk can start on the cover page and run through the contents."""
    cover_and_contents = chunk(
        "APC220 Homeowners Policy cover page text. " + " ".join(BIG_TOC.split())
    )
    pages = make_pages("APC220 Homeowners Policy cover page text.", BIG_TOC, PAGE_COVERAGE)

    annotate_chunks([cover_and_contents], pages)

    assert cover_and_contents.metadata["page"] == 1  # it does NOT start on the contents page
    assert cover_and_contents.metadata["kind"] == "toc"


def test_ordinary_chunk_with_a_few_dots_is_not_toc() -> None:
    body = "Wait... see section 3.... for the deductible.... and limits...."
    pages = make_pages(BIG_TOC, PAGE_COVERAGE, body)  # contents page first, body later
    normal = chunk(body)

    annotate_chunks([normal], pages)

    assert normal.metadata["page"] == 3
    assert normal.metadata["kind"] != "toc"
