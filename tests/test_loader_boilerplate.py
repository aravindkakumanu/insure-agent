"""
test_loader_boilerplate.py — Running header / footer stripping
===============================================================
Offline. Pages are built to look like the sample policy: every page ends with the
same footer block, and one sentence runs across a page break.
"""

from langchain_core.documents import Document

from src.rag.loader import _is_substantive, _line_key, _strip_boilerplate

FOOTER = (
    "Homeowners Policy\nPolicy number:\nPolicy effective date:\n"
    "Your Allstate agency is\nPage {n} of 26\nSample Document\n"
)


def page(n: int, body: str) -> Document:
    return Document(page_content=body + FOOTER.format(n=n), metadata={"source": "x.pdf", "page": n})


def pages(*bodies: str) -> list[Document]:
    return [page(i, b) for i, b in enumerate(bodies, start=1)]


BODIES = [
    "1. Intro text.\nWe will settle within 60 days and\n",
    "b) the action is commenced within one year.\nMore body text here.\n",
    "Third page body.\nAnother line.\n",
    "Fourth page body.\nLast line.\n",
]


def test_footer_block_is_removed_from_every_page() -> None:
    out = _strip_boilerplate(pages(*BODIES))
    for doc in out:
        for gone in ("Homeowners Policy", "Policy number:", "Policy effective date:",
                     "Your Allstate agency is", "Sample Document", "of 26"):
            assert gone not in doc.page_content


def test_a_sentence_across_a_page_break_is_no_longer_interrupted() -> None:
    """The reason this exists: the footer used to sit between 'and' and 'b)'."""
    joined = "\n\n".join(d.page_content for d in _strip_boilerplate(pages(*BODIES)))
    assert "within 60 days and b) the action is commenced" in " ".join(joined.split())
    assert "Page" not in joined and "Sample Document" not in joined


def test_body_text_and_metadata_survive() -> None:
    out = _strip_boilerplate(pages(*BODIES))
    assert out[0].page_content == BODIES[0]
    assert [d.metadata["page"] for d in out] == [1, 2, 3, 4]


def test_digits_are_normalised_so_changing_page_numbers_still_match() -> None:
    assert _line_key("Page 7 of 26") == _line_key("Page 18 of 26") == "Page # of #"


def test_same_words_in_the_body_are_not_removed() -> None:
    bodies = list(BODIES)
    bodies[0] += "Our Homeowners Policy covers this.\n"  # similar words, different line
    bodies[1] = "\n".join(["filler"] * 20) + "\nHomeowners Policy\n" + "\n".join(["filler"] * 20) + "\n"
    out = _strip_boilerplate(pages(*bodies))
    assert "Homeowners Policy\n" in out[1].page_content  # mid-page occurrence stays
    assert "Our Homeowners Policy covers this." in out[0].page_content


def test_numbering_lines_are_never_boilerplate() -> None:
    assert not _is_substantive("a)")
    assert not _is_substantive("1.")
    assert _is_substantive("Sample Document")
    bodies = ["a)\n1.\nBody one.\n", "a)\n1.\nBody two.\n", "a)\n1.\nBody three.\n", "a)\n1.\nBody four.\n"]
    out = _strip_boilerplate(pages(*bodies))
    assert all("a)\n1." in d.page_content for d in out)


def test_a_line_on_too_few_pages_is_kept() -> None:
    bodies = list(BODIES)
    bodies[0] = "Special Header Line\n" + bodies[0]  # only page 1
    out = _strip_boilerplate(pages(*bodies))
    assert "Special Header Line" in out[0].page_content


def test_short_documents_are_left_alone() -> None:
    three = pages(*BODIES[:3])
    assert _strip_boilerplate(three) is three


def test_document_without_repeated_lines_is_unchanged() -> None:
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    plain = [Document(page_content=f"Unique {w} body text.\nSecond {w} line here.\n",
                      metadata={"page": i}) for i, w in enumerate(words, start=1)]
    assert _strip_boilerplate(plain) is plain


def test_a_page_is_never_stripped_down_to_nothing() -> None:
    """Short pages whose lines differ only in digits all look like page furniture."""
    furniture = [Document(page_content=f"Unique body number {i}.\nSecond line {i}.\n",
                          metadata={"page": i}) for i in range(1, 6)]
    out = _strip_boilerplate(furniture)
    assert [d.page_content for d in out] == [d.page_content for d in furniture]
