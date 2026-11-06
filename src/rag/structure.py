"""
structure.py — Page & Section Metadata for Policy Chunks
=========================================================
Responsibility: after a PDF is split into chunks, work out WHERE each chunk
came from and stamp that onto its metadata:

    page         - page number the chunk starts on (1-based)
    page_end     - page number it ends on
    section      - top-level section, e.g. "Section I—Your Property"
    subsection   - e.g. "Losses We Do Not Cover Under Coverage A, ..."
    kind         - "coverage" | "exclusions" | "conditions" | "definitions" | "other"
                   (the heading in effect where the chunk STARTS)
    has_exclusions - True if ANY part of the chunk lies in an exclusions region
    chunk_index  - position of the chunk in the document

WHY: the exclusions tool used to find exclusion text by keyword alone, and
citations could only name the PDF. With `kind` the tool can see which chunks live
in an exclusions section, and answers can cite a page.

WHY NOT SemanticChunker(add_start_index=True): it computes start_index by adding
up chunk lengths, but the chunker drops the whitespace between sentences, so its
offsets drift away from the real text. We locate each chunk in the page text
ourselves instead.

LIMITS - the heading rules below are written for the Allstate-style
homeowners jacket this project ingests ("Section I—Your Property",
"Losses We Do Not Cover Under Coverage ..."). Another policy layout needs its own
rules; unrecognised headings just mean `section`/`kind` stay at the last known
value. Heading patterns are anchored to the START of a line so a cross-reference
inside a sentence ("...under the Section I—Losses We Do Not Cover...") is never
mistaken for a heading.

Semantic chunks can be large (one real chunk spans four pages), so a chunk may
straddle several headings. `section`/`subsection`/`kind` describe where it
starts; `has_exclusions` says whether it also overlaps an exclusions region, so
a chunk that starts under a coverage heading but contains the exclusions list is
still recognised.
"""

import bisect
import logging
import re
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heading rules
# ---------------------------------------------------------------------------

# Section titles must be the WHOLE line: a line that merely starts with
# "Section I Conditions, Mold, Fungus..." is body text, not a heading.
_SECTION_RE = re.compile(
    r"^(?:Definitions Used In This Policy"
    r"|Section I—Your Property"
    r"|Section I Conditions"
    r"|Section II—Family Liability And Guest"
    r"|Section II Conditions"
    r"|Section III—Optional Protection)$"
)

# Subsection titles start the line; "Additional Protection" must be the whole
# line (it also begins running text like "Additional Protection under item 11").
_SUBSECTION_RE = re.compile(
    r"^(?:[A-Z]\.\s+)?(?:"
    r"Losses We (?:Do Not )?Cover Under Coverages?"
    r"|Property We (?:Do Not )?Cover Under Coverage"
    r"|Additional Protection$"
    r"|Limitations On Certain Personal Property"
    r")"
)

# A heading that ends on one of these words wraps onto the next line
# ("Losses We Do Not Cover Under Coverage" / "A, Coverage B And Coverage C").
_WRAP_ENDINGS = ("Coverage", "Guest", "Under", "And")
_MAX_CONTINUATION_CHARS = 45

# Table-of-contents lines end in dot leaders and a page number.
_TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$")
_TOC_PAGE_MIN_LINES = 3


@dataclass(frozen=True)
class Region:
    """A stretch of the document under one heading context."""

    start: int
    end: int
    section: str
    subsection: str
    kind: str


@dataclass(frozen=True)
class Heading:
    """A heading found in the document, at a character offset into full_text."""

    offset: int
    level: str  # "section" | "subsection"
    title: str


def _kind_for(title: str) -> str:
    """Classify a heading title into a chunk `kind`."""
    if "Do Not Cover" in title:
        return "exclusions"
    if "Cover Under" in title or title.startswith("Additional Protection"):
        return "coverage"
    if title.startswith("Limitations On Certain Personal Property"):
        return "coverage"
    if "Conditions" in title:
        return "conditions"
    if title.startswith("Definitions"):
        return "definitions"
    return "other"


def _join_pages(pages: list[Document]) -> tuple[str, list[int], list[int]]:
    """Join page texts as the semantic chunker sees them; return page offsets."""
    starts: list[int] = []
    numbers: list[int] = []
    parts: list[str] = []
    offset = 0
    for i, page in enumerate(pages):
        starts.append(offset)
        numbers.append(int(page.metadata.get("page", i + 1)))
        parts.append(page.page_content)
        offset += len(page.page_content) + 2  # the "\n\n" separator
    return "\n\n".join(parts), starts, numbers


def find_headings(pages: list[Document]) -> list[Heading]:
    """Find section and subsection headings, with offsets into the joined text."""
    _, starts, _ = _join_pages(pages)
    headings: list[Heading] = []
    for page, page_start in zip(pages, starts):
        raw_lines = page.page_content.splitlines(keepends=True)
        if sum(bool(_TOC_LINE_RE.search(line.strip())) for line in raw_lines) >= _TOC_PAGE_MIN_LINES:
            continue  # a table-of-contents page lists headings without being them
        offset = page_start
        for i, raw in enumerate(raw_lines):
            line = raw.strip()
            level = (
                "section" if _SECTION_RE.match(line)
                else "subsection" if _SUBSECTION_RE.match(line)
                else None
            )
            if level:
                title = line
                nxt = raw_lines[i + 1].strip() if i + 1 < len(raw_lines) else ""
                if (
                    title.endswith(_WRAP_ENDINGS)
                    and nxt
                    and len(nxt) <= _MAX_CONTINUATION_CHARS
                    and not nxt.endswith(".")
                ):
                    title = f"{title} {nxt}"
                headings.append(Heading(offset, level, title.rstrip(":").strip()))
            offset += len(raw)
    return headings


def build_regions(headings: list[Heading], text_length: int) -> list[Region]:
    """Split the document into regions, each with its section/subsection/kind."""
    regions: list[Region] = []
    section = subsection = ""
    region_start = 0

    def close(end: int) -> None:
        if end > region_start or not regions:
            title = subsection or section
            regions.append(
                Region(region_start, end, section, subsection, _kind_for(title) if title else "other")
            )

    for heading in headings:
        close(heading.offset)
        region_start = heading.offset
        if heading.level == "section":
            section, subsection = heading.title, ""
        else:
            subsection = heading.title
    close(text_length)
    return regions


# ---------------------------------------------------------------------------
# Locating chunks in the document text
# ---------------------------------------------------------------------------

_ANCHOR_WORDS = 8


def _word_pattern(words: list[str]) -> re.Pattern[str]:
    """Regex matching these words separated by any whitespace."""
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def locate_chunks(chunks: list[Document], full_text: str) -> list[tuple[int, int]]:
    """
    Find each chunk's (start, end) offsets in full_text.

    Chunks arrive in document order, so we search forward only, matching the
    chunk's first few words with whitespace tolerance (the chunker may have
    collapsed newlines into spaces). The chunk's end is found by matching its
    last few words inside a window around where it should end, NOT anywhere
    later in the document: policy boilerplate repeats, and an unbounded search
    would jump the cursor past the following chunks. The cursor then moves to
    ~80% through the chunk, so a phrase repeated inside it can't be matched as
    the next chunk's start. A chunk that can't be found falls back to the cursor
    and is logged - its page/section may be slightly off, but ingestion still
    succeeds.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, chunk in enumerate(chunks):
        words = chunk.page_content.split()
        if not words:
            spans.append((cursor, cursor))
            continue
        match = _word_pattern(words[:_ANCHOR_WORDS]).search(full_text, cursor)
        if match is None:
            logger.warning("Chunk %d not found in document text; using cursor position", index)
            start = cursor
            end = min(len(full_text), start + len(chunk.page_content))
        else:
            start = match.start()
            length = len(chunk.page_content)
            end = min(len(full_text), start + length)
            tail = _word_pattern(words[-_ANCHOR_WORDS:]).search(
                full_text, start + int(0.7 * length), start + int(1.3 * length) + 50
            )
            if tail is not None:
                end = tail.end()
        spans.append((start, end))
        cursor = max(cursor, start + int(0.8 * (end - start)))
    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_chunks(chunks: list[Document], pages: list[Document]) -> list[Document]:
    """
    Stamp page / section / kind / chunk_index metadata onto each chunk, in place.

    Args:
        chunks: Chunks in document order (output of either chunking strategy).
        pages:  The per-page Documents the chunks were made from.

    Returns:
        The same chunk list, for chaining.
    """
    full_text, page_starts, page_numbers = _join_pages(pages)
    headings = find_headings(pages)
    regions = build_regions(headings, len(full_text))
    region_starts = [r.start for r in regions]

    def page_at(offset: int) -> int:
        i = max(0, bisect.bisect_right(page_starts, offset) - 1)
        return page_numbers[i]

    for index, (chunk, (start, end)) in enumerate(zip(chunks, locate_chunks(chunks, full_text))):
        here = regions[max(0, bisect.bisect_right(region_starts, start) - 1)]
        chunk.metadata.update(
            {
                "page": page_at(start),
                "page_end": page_at(max(start, end - 1)),
                "section": here.section,
                "subsection": here.subsection,
                "kind": here.kind,
                "has_exclusions": any(
                    r.kind == "exclusions" and r.start < end and r.end > start for r in regions
                ),
                "chunk_index": index,
            }
        )
    logger.info(
        "Annotated %d chunks with page/section metadata (%d headings found)",
        len(chunks), len(headings),
    )
    return chunks
