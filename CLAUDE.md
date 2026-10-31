# InsureAgent — AI Guidance File

This file tells AI assistants (Claude, Cursor, etc.) how to work on this project.
Keep it updated as the project evolves.

---

## Stage Status

| Stage | Description                              | Status         |
|-------|------------------------------------------|----------------|
| 1     | Project skeleton, RAG pipeline, REST API | ✅ Complete     |
| 2     | LangGraph agent, tool routing, memory    | ✅ Complete     |
| 3     | Multi-agent router — lightweight LLM classifier routes to Coverage Agent, Exclusions Agent, or Claims Agent, each a separate LangGraph graph with its own system prompt and tool subset. | 🚧 In progress |

### Stage 3 — Multi-agent router (in progress)
- `src/agents/router.py`: `classify_intent()` (one structured-output LLM call, not a
  graph) + `route_and_answer()` (fan-out, merge, session pass-through)
- `coverage_agent.py` (`search_policy`), `exclusions_agent.py` (`search_exclusions`),
  `claims_agent.py` (both tools): each its own compiled graph, MemorySaver,
  `answer_question()`, and `MAX_ITERATIONS` guard
- `/ask` now goes through the router; `QuestionResponse` has `agent` and `confidence`
- Specialists return typed `AgentResult`s; the router merges them into a `RouterResult`
- Guardrails (`src/agents/guardrails.py`): injection check before routing, PII
  scrubbing + low-confidence warning after; `GuardrailViolation` → HTTP 400
- `langgraph_agent.py` is untouched (Stage 2 reference; the specialists import its tools)
- Tests: `tests/test_router.py` (offline)
- Not yet verified against the real OpenAI API: run the curl checks in lesson 20

### Stage 2 hardening — Oct 30, 2026
- `/ask` and `/ingest` are plain `def` (threadpool) instead of `async def`
- `get_vector_store()` cached with `@lru_cache`; ingestion is idempotent
  (SHA-256 chunk IDs + delete-by-source before add)
- Memory trimming uses `trim_messages(start_on="human")`
- Doom-loop guard: `step_counter` in `AgentState` + `MAX_ITERATIONS = 25`
  (see lesson 16); tests in `tests/test_agent_guard.py`

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

### 13. Plain `def` routes for blocking work
FastAPI runs `def` routes in a threadpool; `async def` routes run on the event
loop. Our agent and PDF code are synchronous and block on OpenAI, so `async def`
froze every other request. Use `async def` only with an async client end to end
(`ainvoke` + async tools), which would have been a much bigger change.

### 14. Cache clients with `@lru_cache`, build lazily
`get_vector_store()` was rebuilding the embeddings client and Chroma handle on
every tool call. `@lru_cache(maxsize=1)` builds it once on first call (lazy, so
imports still work without `OPENAI_API_KEY`).

### 15. Idempotent ingestion
Chunk ID = `sha256(f"{source}:{index}")`, plus `delete(where={"source": ...})`
before `add_documents`. IDs alone are not enough: a re-ingested PDF that yields
fewer chunks would leave stale high-index chunks behind.
`trim_messages(start_on="human")` can also return an EMPTY list when the current
turn alone exceeds the window; `call_model` falls back to the whole current turn.

### 16. Doom-loop guard — failure-then-fix
**Failure:** an agent whose LLM keeps requesting tools never reaches END. Each
loop costs an LLM call plus a tool call, so a stuck agent burns tokens and
latency until something external stops it.
**Fix:** `step_counter` in `AgentState`, incremented on every `call_model` visit.
Past `MAX_ITERATIONS` (25) it returns `FALLBACK_ANSWER` with no tool_calls, so
`should_continue` routes to END without another LLM call.
Lessons:
- **Reset per question.** State persists in `MemorySaver` per session. A plain
  `int` field (no reducer) is overwritten by whatever is passed at invoke time,
  so `answer_question` sends `"step_counter": 0` every time. Without that, a
  healthy session would trip the guard after ~10 questions
  (`test_counter_resets_between_turns` pins this).
- **Pick the limit above legitimate use.** Tool-using questions take a few agent
  visits; 25 only fires on genuine loops, not complex questions.
- **Set `recursion_limit` explicitly.** LangGraph also has its own step limit
  (every node run counts, so one agent+tool cycle = 2 steps), and its default
  differs by version: 10007 in langgraph 1.2.x, 25 in older releases. We pass
  `RECURSION_LIMIT = 2 * MAX_ITERATIONS + 5` so it is a backstop above our
  guard in any version. I first assumed the default was 25 and was wrong; an
  empirical check caught it.
- **Test with a fake LLM.** `LoopingLLM` always requests a tool and the vector
  store is patched, so the scenario is fully offline (rule 12).

### 17. Router = a classifier call, not a graph
Routing is one decision with no loop, so `classify_intent()` is a single
structured-output call (`with_structured_output(IntentClassification)`, `Literal`
labels) on a small model (`gpt-4o-mini`). Use LangGraph only where there is a loop.
If classification fails it logs a warning and returns `coverage`, never raising,
because coverage fans out to exclusions too and is the least one-sided default.

### 18. Fan out coverage questions
Routing a coverage question to the Coverage Agent alone reintroduces the problem
the Stage 2 prompt solved: an answer about what IS covered that hides what is NOT.
So `coverage` → Coverage Agent AND Exclusions Agent, run in parallel threads
(each blocks on OpenAI), merged as "What IS covered" / "What is NOT covered".
Limits and conditions come from the Coverage prompt, which must include them.
`exclusions` → Exclusions Agent only. `claims` → Claims Agent only (it already
uses both tools). The merge is deterministic, not an LLM call: cheaper, and no
risk of the merge step inventing text. It now works on typed `AgentResult`s
(lesson 22). Trade-off: slightly more repetitive than a synthesized answer.

### 19. No new tool for claims
The vector store holds one policy document with no separate claims section, so a
`search_claims_procedures` tool would imply data that doesn't exist. The Claims
Agent reuses `search_policy` + `search_exclusions` and gets its claims focus from
its system prompt (queries like "duties after loss", "proof of loss").

### 20. Sessions across several agents
Each specialist has its own `MemorySaver`; the same `session_id` is the
`thread_id` in each, so histories are independent per agent. The router keeps
`_session_intent[session_id]` and passes the previous label to the classifier so
a bare follow-up ("what about mold?") returns the same intent and lands on the
same agent(s). Known limits (both in-process, lost on restart): that map, and
a session whose intent changes mid-conversation will hit an agent that has not
seen the earlier turns.
Also: sources are collected from the current turn only. Stage 2 scanned the
whole session history, so sources from earlier questions leaked into later ones.
To check against the real API:
```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Is water damage covered?"}'   # expect agent=coverage+exclusions
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "How do I file a claim?"}'     # expect agent=claims
```

### 21. Duplication is deliberate (for now)
The three specialist files are near-identical (prompt and tool subset differ).
That was the requested structure, and it keeps each agent readable on its own.
Extract a shared `build_agent(name, prompt, tools)` factory if a fourth agent
appears or the loop/guard logic changes.

### 22. Typed results: AgentResult → RouterResult
Specialists return `AgentResult(answer, sources, agent_name, confidence,
session_id)`; `merge_results()` in `router.py` builds a `RouterResult` from them
(`answer`, `sources`, `agent`, `intent`, `confidence`, per-agent `results`).
Types live in `src/agents/results.py` because the router imports the
specialists, so the specialists can't import from the router (cycle). That module
also holds the shared `current_turn`, `extract_sources`, `assess_confidence`.
- `session_id` rides on `AgentResult` (a fifth field) so a specialist called
  standalone still returns the id it generated.
- **Confidence measures grounding, not correctness.** It is derived from tool
  results, never from the LLM's self-rating (uncalibrated): `low` = guard tripped,
  no tool called, or no tool returned text; `medium` = some returned text, some
  didn't; `high` = every tool call returned text. "Returned text" is detected by
  the `[Source: ...]` label every non-empty tool result carries, so it does not
  depend on the tools' empty-result wording.
- **`search_exclusions` special case.** The tool falls back to unfiltered chunks
  when no retrieved chunk has exclusion wording. That is text, but not evidence
  of an exclusion, so a `search_exclusions` result with none of
  `_EXCLUSION_KEYWORDS` caps confidence at `medium` (not `low`: text did come
  back). The check strips `[Source: ...]` labels first so a filename like
  `exclusions.pdf` can't count, keys on `ToolMessage.name` (set by `ToolNode`),
  and imports the keyword list from `langgraph_agent.py` so it lives in one place.
  Still a heuristic: a keyword hit means exclusion wording is present, not that
  it answers the question.
- **Keyword-list fix (Oct 30, 2026):** `_EXCLUSION_KEYWORDS` had a comment claiming
  the stem `"exclud"` catches "exclusion", but `"exclud" in "exclusion"` is False
  (exclu-s-ion), so "Exclusions" headings were missed by the tool's filter and
  flagged `medium` by the check above. Added `"exclusion"` to the list (the one
  deliberate edit to the Stage 2 file) and corrected the comment. On the 69
  ingested chunks, matches went 31 → 32. Lesson: verify a "stem" claim with `in`
  before trusting a comment; `test_exclusion_keywords_match_exclusion_and_exclusions`
  pins it.

### 23. Guardrails
`route_and_answer()` runs `check_prompt_injection()` FIRST (before the classifier,
the agents, or any session state), then `apply_output_guardrails()` on the merged
result. `GuardrailViolation` is mapped to a 400 in `main.py` (a client error, not
a 500); its message is written to be shown to the user.
- **Injection detection is a tripwire, not a defence.** It is regex phrase
  matching ("ignore your instructions", "pretend you are", "disregard/forget the
  policy", "say the policy covers everything", plus close variants), so rewording
  evades it, and it has false positives ("should I forget the policy
  deductible?"). The real protection is the specialists' prompts, which answer
  only from retrieved text. Extend `_INJECTION_PATTERNS` in one place.
- **SSN:** `XXX-XX-XXXX` → `[REDACTED-SSN]`. Misses dashless SSNs, names, addresses.
- **Policy-number-like IDs:** a bare "8+ alphanumerics" rule would redact ordinary
  words, so a token must be 8+ chars of letters/digits/hyphens containing at least
  one letter AND one digit, as a whole token. Deliberately left alone: plain words,
  plain numbers, dates, PDF filenames (`apc220-ho-policy-jacket.pdf`; no match if
  followed by `.<letter>`), and UUID session ids (otherwise every session id in the
  logs would be redacted). Measured on the 69 ingested policy chunks: 0 matches,
  so no false positives on the real document. True positives are only
  covered by synthetic tests, since the real policy has no IDs in it.
- **Logs are scrubbed in a `Formatter`, not at call sites.** The question is logged
  in several places; `PIIScrubbingFormatter` scrubs the final formatted line, so
  messages, args, tracebacks and library logs are all covered.
  `install_log_scrubbing()` is called in `main.py` right after `basicConfig`.
  It only covers handlers on the root logger.
- The answer AND each specialist's answer in `RouterResult.results` are scrubbed.
- **Confidence gate:** `low` prepends "⚠️ Warning: limited policy text was found
  for this question. This answer may be incomplete." For a coverage+exclusions
  merge this can sit alongside the per-section cautions from `merge_results`
  (lesson 22): the section note says WHICH part is weak, the warning says the
  whole answer is.

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
| `src/agents/langgraph_agent.py` | Stage 2 agent + the two retrieval tools (reference) |
| `src/agents/router.py`      | Intent classifier, fan-out, `merge_results`, session pass-through |
| `src/agents/guardrails.py`  | Injection check, PII scrubbing (answers + logs), confidence gate |
| `tests/test_guardrails.py`  | Offline guardrail tests                               |
| `src/agents/results.py`     | `AgentResult` / `RouterResult` + confidence and source helpers |
| `src/agents/{coverage,exclusions,claims}_agent.py` | Stage 3 specialists |
| `tests/test_router.py`      | Offline router and specialist tests                   |
| `tests/test_agent_guard.py` | Offline doom-loop scenario and guard tests            |
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
