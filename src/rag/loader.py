"""
loader.py — Document Loading & Chunking
========================================
Responsibility: Take a PDF file path, extract its text with PyMuPDF,
and split it into chunks suitable for embedding.

CHUNKING STRATEGY (set CHUNK_STRATEGY below):
  "fixed_token"  — Split by exact token count using tiktoken.
                   Fast and predictable, but may cut sentences mid-thought.
                   Good baseline to understand where chunking breaks down.

  "semantic"     — Split by meaning using OpenAI embeddings to detect topic
                   boundaries. Slower and costs a few embedding calls, but
                   chunks stay semantically coherent. Better for retrieval.

WHY TOKENS INSTEAD OF CHARACTERS?
LLMs think in tokens, not characters. A token is roughly 0.75 words.
Measuring in tokens gives you precise control over how much fits in each
embedding call and in the LLM's context window.
"""

import logging
import math
import re
from collections import Counter
from pathlib import Path

import pymupdf as fitz  # PyMuPDF — newer versions prefer `import pymupdf`
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.structure import annotate_chunks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — change these to switch strategies
# ---------------------------------------------------------------------------
CHUNK_STRATEGY = "semantic"       # "fixed_token" | "semantic"

# Fixed-token settings
CHUNK_SIZE = 512        # tokens per chunk  (was 1000 chars; now true token count)
CHUNK_OVERLAP = 0       # no overlap — intentional for the first experiment
ENCODING_MODEL = "cl100k_base"  # tiktoken encoding used by GPT-4o & embeddings


# ---------------------------------------------------------------------------
# Header / footer stripping
# ---------------------------------------------------------------------------
# Every page of the sample policy ends with a block ("Homeowners Policy / Policy
# number: / Policy effective date: / Your Allstate agency is / Page 7 of 26 /
# Sample Document"). When a sentence runs across a page break, that block lands
# in the MIDDLE of the sentence in the chunk text: it pollutes embeddings and
# makes accurate quotes fail verification. 25 such blocks were found in the
# stored chunks, 11 of them mid-sentence.
HEADER_LINES = 6            # lines at the top of a page that may be a header
FOOTER_LINES = 10           # lines at the bottom of a page that may be a footer
BOILERPLATE_MIN_PAGES = 4   # too few pages to tell a footer from content
BOILERPLATE_PAGE_FRACTION = 0.5  # a line must repeat on at least this share of pages


def _line_key(line: str) -> str:
    """Normalise a line so "Page 7 of 26" and "Page 8 of 26" compare equal."""
    return re.sub(r"\d+", "#", line.strip())


def _is_substantive(key: str) -> bool:
    """At least two words, so numbering like "a)" or "1." is never boilerplate."""
    return len(re.findall(r"[A-Za-z]{2,}", key)) >= 2


def _strip_boilerplate(pages: list[Document]) -> list[Document]:
    """
    Remove running headers and footers from the pages.

    A line is boilerplate if, after digits are normalised, it sits within the
    first HEADER_LINES or last FOOTER_LINES lines of a page on at least half of
    all pages, and has two or more words. Only those edge lines are removed:
    the same words in the body of a page are left alone. A page is never stripped
    down to nothing, and documents with fewer than BOILERPLATE_MIN_PAGES pages are
    returned untouched.

    LIMIT: this is a repetition heuristic. A body line that genuinely repeats at
    the edge of most pages (differing at most in digits) would be removed.
    """
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return pages

    page_lines = [p.page_content.splitlines(keepends=True) for p in pages]

    def edges(lines: list[str]) -> list[int]:
        n = len(lines)
        return sorted(set(range(min(HEADER_LINES, n))) | set(range(max(0, n - FOOTER_LINES), n)))

    counts: Counter[str] = Counter()
    for lines in page_lines:
        counts.update({_line_key(lines[i]) for i in edges(lines)})  # once per page

    threshold = max(2, math.ceil(BOILERPLATE_PAGE_FRACTION * len(pages)))
    boilerplate = {k for k, n in counts.items() if n >= threshold and _is_substantive(k)}
    if not boilerplate:
        return pages

    stripped: list[Document] = []
    removed = 0
    for page, lines in zip(pages, page_lines):
        drop = {i for i in edges(lines) if _line_key(lines[i]) in boilerplate}
        if len(drop) >= sum(bool(line.strip()) for line in lines):
            stripped.append(page)  # never strip a page down to nothing
            continue
        removed += len(drop)
        text = "".join(line for i, line in enumerate(lines) if i not in drop)
        stripped.append(Document(page_content=text, metadata=page.metadata))
    logger.info(
        "Stripped %d header/footer lines (%d distinct) from %d pages",
        removed, len(boilerplate), len(pages),
    )
    return stripped


def _extract_pages(pdf_path: Path) -> list[Document]:
    """Extract raw text from every page of a PDF using PyMuPDF."""
    raw_pages: list[Document] = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():  # skip blank pages
                raw_pages.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_num},
                    )
                )
    if not raw_pages:
        raise ValueError(f"No extractable text in {pdf_path.name}. Is it a scanned PDF?")
    raw_pages = [p for p in _strip_boilerplate(raw_pages) if p.page_content.strip()]
    if not raw_pages:
        raise ValueError(f"No text left in {pdf_path.name} after removing headers and footers.")
    return raw_pages


def _split_fixed_token(pages: list[Document]) -> list[Document]:
    """
    Split documents into fixed-size token chunks using tiktoken.

    Uses RecursiveCharacterTextSplitter.from_tiktoken_encoder so chunk_size
    is measured in real tokens (not characters). This is the correct way to
    guarantee you never exceed embedding or context window limits.

    KNOWN WEAKNESS: With overlap=0, a clause like
      "...flood damage is NOT covered" may be split so that "NOT covered"
      lands in a different chunk than the thing it modifies. Retrieval then
      returns incomplete context and the LLM may give a wrong answer.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    logger.info(
        "Fixed-token split: %d chunks (size=%d tokens, overlap=%d, encoding=%s)",
        len(chunks), CHUNK_SIZE, CHUNK_OVERLAP, ENCODING_MODEL,
    )
    return chunks


def _split_semantic(pages: list[Document]) -> list[Document]:
    """
    Split documents at natural topic boundaries using SemanticChunker.

    Instead of counting tokens, SemanticChunker embeds every sentence and
    looks for places where the embedding distance between consecutive sentences
    spikes — that spike signals a topic change, and a new chunk starts there.

    COST: Makes one embedding API call per sentence. For a 26-page PDF that's
    ~300–600 calls to text-embedding-3-small (~$0.001 total — negligible).

    BENEFIT: Chunks stay semantically coherent. "Flood damage is NOT covered"
    will never be split from its context because it's part of the same topic.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "SemanticChunker requires langchain-experimental.\n"
            "Install it with: pip install langchain-experimental"
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Combine all pages into one document so SemanticChunker can detect
    # topic boundaries that cross page breaks
    full_text = "\n\n".join(p.page_content for p in pages)
    source = pages[0].metadata.get("source", "unknown")

    chunker = SemanticChunker(
        embeddings=embeddings,
        # breakpoint_threshold_type options:
        #   "percentile"   — split where distance > Nth percentile (default 95)
        #   "standard_deviation" — split where distance > mean + N*stddev
        #   "interquartile" — split where distance > Q3 + 1.5*IQR
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,  # split top 10% of distance spikes
    )

    chunks = chunker.create_documents([full_text], metadatas=[{"source": source}])
    logger.info("Semantic split: %d chunks (breakpoint=percentile/90)", len(chunks))
    return chunks


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Load a PDF and return a list of LangChain Document objects, one per chunk.

    The chunking strategy is controlled by the CHUNK_STRATEGY constant at the
    top of this file: "fixed_token" or "semantic".

    Args:
        pdf_path: Path to the PDF file to load.

    Returns:
        List of Document objects with `page_content` (the text chunk) and
        `metadata`: "source", plus "page", "page_end", "section", "subsection",
        "kind" and "chunk_index" (see src/rag/structure.py).

    Raises:
        FileNotFoundError: If the PDF doesn't exist.
        ValueError: If the PDF has no extractable text (e.g. scanned image).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Loading PDF: %s (strategy=%s)", pdf_path.name, CHUNK_STRATEGY)

    pages = _extract_pages(pdf_path)
    logger.info("Extracted %d pages from %s", len(pages), pdf_path.name)

    if CHUNK_STRATEGY == "fixed_token":
        chunks = _split_fixed_token(pages)
    elif CHUNK_STRATEGY == "semantic":
        chunks = _split_semantic(pages)
    else:
        raise ValueError(f"Unknown CHUNK_STRATEGY: {CHUNK_STRATEGY!r}. Use 'fixed_token' or 'semantic'.")

    # Stamp page / section / kind onto every chunk (see src/rag/structure.py)
    return annotate_chunks(chunks, pages)
