# InsureAgent — AI Guidance File

This file tells AI assistants (Claude, Cursor, etc.) how to work on this project.
Keep it updated as the project evolves.

---

## Stage Status

| Stage | Description                              | Status         |
|-------|------------------------------------------|----------------|
| 1     | Project skeleton, RAG pipeline, REST API | ✅ Complete     |
| 2     | LangGraph agent, tool routing, memory    | ✅ Complete     |
| 3     | Auth, multi-tenant policies, prod deploy | 🔜 Planned     |

### Stage 2 — Completed Oct 13, 2026
- Replaced LCEL chain with a LangGraph ReAct agent (`src/agents/langgraph_agent.py`)
- Two retrieval tools: `search_policy` (general) and `search_exclusions` (exclusion post-filter)
- LLM decides which tool(s) to call based on the question; can call both for broad queries
- Conversation memory via `MemorySaver` keyed by `session_id` (in-process, not durable)
- `/ask` endpoint now accepts optional `session_id` and always returns one
- `qa_agent.py` preserved as Stage 1 reference but no longer wired to the endpoint
- Unit test patched to use `fixed_token` strategy (semantic strategy requires live OpenAI calls)

### Stage 1 — Completed Oct 8, 2026
- Python venv + all dependencies installed and pinned in `requirements.txt`
- PDF ingestion pipeline: PyMuPDF → chunking → OpenAI embeddings → ChromaDB
- FastAPI app with `/health`, `/ingest`, `/ask` endpoints
- LangChain LCEL-based RAG chain (see "Decisions" below)
- 3 unit tests passing (`tests/test_loader.py`)
- Verified end-to-end with `apc220-ho-policy-jacket.pdf` (137 chunks ingested)

---

## Project Overview

**InsureAgent** is a multi-agent insurance policy Q&A system built with:
- **LangChain** — orchestration of LLM chains and agents
- **ChromaDB** — local vector store for RAG (Retrieval-Augmented Generation)
- **FastAPI** — REST API layer
- **PyMuPDF** — PDF parsing for insurance policy documents
- **OpenAI** — LLM backend (GPT-4o for answers, text-embedding-3-small for embeddings)

The system ingests insurance PDFs → chunks & embeds them → stores in Chroma → serves
answers through a FastAPI endpoint backed by LangChain LCEL chains.

---

## Architecture

```
data/policies/          ← Raw PDF files (never commit sensitive docs)
data/chroma_db/         ← ChromaDB persisted vector store (auto-created on ingest)
src/
  rag/                  ← Document loading, chunking, embedding, retrieval
  agents/               ← LangChain agent definitions and tool wiring
tests/                  ← Unit and integration tests (pytest)
main.py                 ← FastAPI app entry point
.env                    ← Secrets (OPENAI_API_KEY, etc.) — never commit
```

---

## Tech Stack & Key Decisions

| Layer         | Choice                    | Why                                                  |
|--------------|--------------------------|------------------------------------------------------|
| LLM          | OpenAI GPT-4o            | Strong reasoning; easy via langchain-openai          |
| Embeddings   | text-embedding-3-small   | Cheap, fast, high quality; enough for insurance docs |
| Vector DB    | ChromaDB (local)         | Zero-infra, runs in-process during development       |
| Orchestration| LangChain LCEL           | Modern pipe-based chains; RetrievalQA removed in 1.x |
| API          | FastAPI                  | Async, typed, auto-docs at /docs                     |
| PDF Parsing  | PyMuPDF (`import pymupdf`)| Fastest and most accurate Python PDF library        |
| Env Secrets  | python-dotenv            | Load .env files safely into os.environ               |
| Testing      | pytest                   | Industry standard; fast and readable                 |

---

## Decisions & Lessons Learned (Stage 2)

### 7. LangGraph vs LCEL — use LangGraph when you need loops or decisions
An LCEL chain is a straight pipeline: `A | B | C`. It cannot loop or branch.
A LangGraph graph has **nodes** (functions that read/write state) and **edges**
(connections between nodes, optionally conditional). A conditional edge is what
makes the agent "decide" — route to the tools node or end.

```
START → agent_node → [tool_calls?] → tools_node → agent_node (loop)
                   → [no tools]    → END
```

Use LCEL for simple, deterministic retrieval. Use LangGraph when the agent
needs to pick between tools, loop, or maintain memory.

### 8. Two-tool design — general + exclusion-specific
`search_policy`: standard similarity search, top-k=4.
`search_exclusions`: similarity search with k=20 then post-filter for exclusion
keywords ("exclud", "not covered", "we do not cover", etc.). Retrieves k=20
first because exclusion chunks don't always rank highest in similarity — the
negation language shifts their embeddings away from affirmative queries.

### 9. Conversation memory — MemorySaver with thread_id
`MemorySaver` is a dict in the server process keyed by `thread_id` (our
`session_id`). Each `/ask` call with the same `session_id` rehydrates the
full message history, so follow-up questions have context.
**Limitation**: sessions are lost on server restart. For production, swap
`MemorySaver` for `SqliteSaver` or `RedisSaver` (same API, durable storage).

### 10. Memory window — trim to MEMORY_WINDOW before LLM call
The full message history grows without bound. In `call_model` we slice to
`MEMORY_WINDOW = 10` messages (≈ 3–4 full Q&A turns including tool calls)
before sending to the LLM. The full history stays in MemorySaver; only the
LLM context is trimmed. We walk forward to the first HumanMessage after
slicing so the context never starts mid-tool-call.

### 11. LLM lazy initialization
`ChatOpenAI` must not be instantiated at module level — it reads `OPENAI_API_KEY`
from the environment at construction time. In FastAPI, `load_dotenv()` runs
before the deferred import, so it works. But it breaks bare `python -c` imports
and test runs. Solution: `_get_llm()` singleton that initializes on first call.

### 12. Unit tests must not call OpenAI
The `test_load_pdf_returns_chunks` test patches `CHUNK_STRATEGY = "fixed_token"`
so it never hits the semantic chunker (which calls the OpenAI embeddings API).
Rule: unit tests must be fully offline. Integration tests that need a real API
key belong in a separate `tests/integration/` folder (not yet created).

---

## Decisions & Lessons Learned (Stage 1)

### 1. LangChain 1.x removed `RetrievalQA` — use LCEL instead
`langchain.chains.RetrievalQA` no longer exists in LangChain 1.x. The modern
replacement is **LCEL** (LangChain Expression Language), which uses the pipe `|`
operator to compose steps explicitly:

```python
# Old (broken in 1.x):
from langchain.chains import RetrievalQA

# New (LCEL):
chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
chain.invoke(question)
```

LCEL is clearer — data flow is visible left-to-right, like Unix pipes.

### 2. `langchain.text_splitter` moved to `langchain-text-splitters`
In LangChain 1.x, text splitters are in a separate package:

```python
# Old (broken):
from langchain.text_splitter import RecursiveCharacterTextSplitter

# New:
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

`langchain-text-splitters` must be installed separately — it is now in `requirements.txt`.

### 3. PyMuPDF import changed
PyMuPDF now prefers the `pymupdf` namespace over `fitz`:

```python
# Old (triggers deprecation warning):
import fitz

# New:
import pymupdf as fitz   # alias keeps rest of code unchanged
```

### 4. Pydantic 2.14 requires Python 3.12+
The latest pydantic uses `ForwardRef(is_class=True)` which only exists in Python 3.12.
Our venv runs Python 3.10, so we pin `pydantic==2.10.6`.
If you ever upgrade to Python 3.12, you can remove this pin.

### 5. ChromaDB persists to `data/chroma_db/`
After a successful `/ingest` call, ChromaDB saves its index to disk at `data/chroma_db/`.
This means:
- You only need to ingest a PDF once per machine — data survives server restarts
- That folder is in `.gitignore` — each developer ingests locally
- Deleting `data/chroma_db/` wipes all vectors; you must re-ingest

### 6. Chunking strategy — use semantic, not fixed-token
Controlled by `CHUNK_STRATEGY` at the top of `src/rag/loader.py`.

**Experiment run Oct 9, 2026** — same query, two strategies, same PDF:

| Strategy | Chunks | Answer quality |
|---|---|---|
| `fixed_token` (512 tok, overlap=0) | 61 | Found coverage grant + one exclusion. Missed tear-out clause. |
| `semantic` (percentile/90) | 69 | Found coverage grant + tear-out clause in same chunk. More complete. |

**Why semantic wins for insurance docs:** Policy clauses like "we cover X. We also cover the cost of Y." are semantically related but may straddle a token boundary. Semantic chunking keeps them together because their sentence embeddings are similar. Fixed-token splits them arbitrarily.

**Default is `semantic`.** Only switch to `fixed_token` for debugging or speed testing.

After changing `CHUNK_STRATEGY`, always wipe and re-ingest:
```bash
rm -rf data/chroma_db
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "data/policies/apc220-ho-policy-jacket.pdf"}'
```

Other tunable parameters in `loader.py`:
- `CHUNK_SIZE` — only applies to `fixed_token`; measured in tokens via tiktoken
- `CHUNK_OVERLAP` — currently 0; add overlap if fixed-token misses cross-boundary context
- `breakpoint_threshold_amount` — only applies to `semantic`; lower = more chunks

---

## Coding Conventions

- Python 3.10 (venv uses 3.10; see pydantic pin above)
- Use **type hints** everywhere — this is a learning project, hints act as documentation
- Follow PEP 8; 88-char line limit (Black-compatible)
- Every module gets a docstring explaining its purpose in plain English
- Use `logging` (not `print`) for runtime output
- Raise descriptive exceptions; never silently swallow errors
- Tests live in `tests/` and use **pytest**

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
uvicorn main:app --reload
```

---

## Key Files to Know

| File                        | Purpose                                               |
|-----------------------------|-------------------------------------------------------|
| `src/rag/loader.py`         | PDF → Document chunks (PyMuPDF + RecursiveCharacterTextSplitter) |
| `src/rag/embedder.py`       | Chunks → Vectors → ChromaDB (text-embedding-3-small)  |
| `src/rag/retriever.py`      | Query ChromaDB, return top-k semantically similar docs |
| `src/agents/qa_agent.py`    | LCEL RAG chain: retriever + GPT-4o + prompt           |
| `main.py`                   | FastAPI routes: /health, /ingest, /ask                |
| `tests/test_loader.py`      | Unit tests for PDF loading and chunking               |
| `.env.example`              | Template for required environment variables           |

---

## Do Not

- Commit `.env` or any real insurance PDF files
- Commit `data/chroma_db/` (large binary files, machine-specific)
- Use synchronous blocking I/O inside FastAPI async routes
- Hard-code API keys anywhere in source code
- Skip type hints to save time — they help you learn faster
- Import from `langchain.chains` or `langchain.text_splitter` — those modules no longer exist

---

## Learning Notes

This project is intentionally built to teach:
1. **RAG pipelines** — how to turn PDFs into searchable knowledge bases
2. **LCEL chains** — how LangChain composes steps with the `|` pipe operator
3. **FastAPI** — how to build production-grade REST APIs in Python
4. **Async Python** — why `async def` matters for high-throughput services
5. **Vector embeddings** — how text becomes numbers that capture meaning
6. **ChromaDB** — how similarity search works without a cloud database
