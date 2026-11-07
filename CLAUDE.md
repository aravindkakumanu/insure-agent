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
- Golden-question evaluation in `tests/integration/` (lesson 30): run it after any
  change to retrieval, chunking or prompts.
- Router rewrites the question per specialist (lesson 24); retrieval tools return
  evidence artifacts (lesson 25); chunks carry page/section metadata (lesson 26);
  page headers/footers and the table of contents are kept out of the chunks
  (lesson 27). **Re-ingest after pulling these changes**: `/ingest` replaces a
  PDF's chunks, no need to wipe `data/chroma_db`.
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
`tests/conftest.py` enforces this: an autouse fixture makes the real
`get_vector_store` raise, so a test that forgets to patch fails loudly. (It was
added after a test run silently reached the real ChromaDB and OpenAI embeddings
API, because `main.py`'s `load_dotenv()` puts the key in the environment once any
test imports `main`.) Shared fakes live in `tests/fakes.py`.

### 13. Plain `def` routes for blocking work
FastAPI runs `def` routes in a threadpool; `async def` routes run on the event
loop. Our agent and PDF code are synchronous and block on OpenAI, so `async def`
froze every other request. Use `async def` only with an async client end to end
(`ainvoke` + async tools), which would have been a much bigger change.

### 14. Cache clients with `@lru_cache`, build lazily
`get_vector_store()` was rebuilding the embeddings client and Chroma handle on
every tool call. `@lru_cache(maxsize=1)` builds it once on first call (lazy, so
imports still work without `OPENAI_API_KEY`).
**Thread-safety (found live):** `lru_cache` does not stop several threads from each
building the value on a cold cache, and ChromaDB's client setup is not safe to run
concurrently. An agent issuing parallel searches in a fresh process crashed with
`RustBindingsAPI object has no attribute 'bindings'` / `Could not connect to tenant
default_tenant` (the same race was latent in the coverage+exclusions fan-out on the
first request after a restart). `get_vector_store()` now serialises construction
with a lock around the cached builder; `test_embedder_threadsafety.py` hammers it
from 8 threads and includes a check that the same test DOES see a double build when
the lock is removed. Lesson: a cache is not a lock.

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
  no tool called, or no tool returned text; `medium` = some returned text and some
  didn't, or a `search_exclusions` result held no exclusion language, or the answer
  quotes text that wasn't retrieved; `high` = none of those. "Returned text" means
  the tool's evidence artifact is non-empty (lesson 25).
- **`search_exclusions` special case.** The tool falls back to the closest chunks
  when none has exclusion wording. That is text, but not evidence of an exclusion,
  so such a result caps confidence at `medium` (not `low`: text did come back).
  Whether a chunk reads as an exclusion is decided in `tools.py`
  (`has_exclusion_language` + section metadata) and recorded on each Evidence as
  `exclusion_language`; `assess_confidence` just reads it and keys on
  `ToolMessage.name` (set by `ToolNode`; `EXCLUSIONS_TOOL_NAME` is pinned to the
  real tool by a test). Still a heuristic: a hit means exclusion wording is
  present, not that it answers the question.
- **Keyword-list fix (Oct 30, 2026):** `_EXCLUSION_KEYWORDS` had a comment claiming
  the stem `"exclud"` catches "exclusion", but `"exclud" in "exclusion"` is False
  (exclu-s-ion), so "Exclusions" headings were missed by the tool's filter and
  flagged `medium` by the check above. Added `"exclusion"` to the list (the one
  deliberate edit to the Stage 2 file) and corrected the comment. On the 69
  ingested chunks, matches went 31 → 32. Lesson: verify a "stem" claim with `in`
  before trusting a comment; `test_exclusion_keywords_match_exclusion_and_exclusions`
  pins it.
- **"Not otherwise excluded" is not an exclusion (found in a live run).** The
  burst-pipe coverage grant says "...water or steam not otherwise excluded", which
  matched the `"exclud"` keyword, so the Exclusions Agent was handed the coverage
  grant and rated `high`. `has_exclusion_language` now strips "not (otherwise)
  excluded" before matching (pinned by
  `test_burst_pipe_coverage_grant_is_not_high_confidence_exclusions`).

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

### 24. Per-agent sub-questions (inter-agent communication)
Live finding: sending a coverage-framed question ("Does my policy cover a burst
pipe?") unchanged to the Exclusions Agent made it search for coverage text, so
its "NOT covered" section quoted the coverage grant again and missed seepage.
The classifier call now also returns `coverage_question` and `exclusions_question`
(standalone, same subject, no added facts). The coverage intent sends each
specialist its own; `exclusions` uses the rewrite; `claims` gets the original.
Missing/blank rewrites fall back to the user's question, so a weak classifier
output degrades to the old behaviour instead of failing.
- The rewrite also resolves bare follow-ups: the router remembers the previous
  STANDALONE question per session (`_session_question`) and passes it with
  `previous_intent`, so "what about mold?" then "and fungus?" keeps its topic.
  (In-process, lost on restart, like `_session_intent`.)
- Side effect to know: a specialist's own history stores the rewritten question,
  not the user's literal words. `RouterResult.questions` records what each got.
- Cost: no extra LLM call, only a larger structured output from the same call.

### 25. Evidence artifacts instead of parsing strings
(`AgentResult.queries` also records the searches the agent chose to run - what
retrieval ACTUALLY ran on, as opposed to the user's question.)
`src/agents/tools.py` has the specialists' tools (the Stage 2 tools in
`langgraph_agent.py` are untouched). They use `response_format=
"content_and_artifact"`: the text the LLM reads, plus a list of `Evidence`
records (chunk id, source, page, section, kind, distance, text,
exclusion_language) that LangChain attaches to the `ToolMessage` (verified to
survive `ToolNode`). Everything downstream reads the artifact:
- `AgentResult.evidence`, `.sources`, and `RouterResult.citations`
- **Quote check:** `find_unverified_quotes` looks for each quoted passage in the
  retrieved text, ignoring case/spacing/punctuation, and checks `...`/`[brackets]`
  piece by piece (an invented tail after `...` is only ~16 chars, so the
  per-piece minimum is deliberately low). A miss caps confidence at `medium`,
  lists the quote on `AgentResult.unverified_quotes`, and adds a note to the
  answer. Straight quotes are paired in sequence so prose BETWEEN two quotations
  isn't mistaken for one.
- **Redundancy:** if a later specialist's chunks are all a subset of an earlier
  one's, `merge_results` keeps its text but flags it ("drew only on policy text
  already used above") and caps its effective confidence at `medium`. Kept, not
  dropped: the text may still be a valid reading.
- Evidence carries Chroma's `distance` (squared L2 on unit vectors; LOWER is
  closer) but it is NOT used as a confidence threshold: a defensible cut-off needs
  calibrating against live queries. Treat that as the next step.
- Both tools skip `kind == "toc"` chunks (lesson 27) by over-fetching `TOC_SLACK`
  extra candidates and filtering afterwards, so a result is never short because a
  contents chunk was dropped.
- `search_exclusions` returns up to 6 chunks (was 4) from a pool of 30. What
  COUNTS as exclusion text is decided by section metadata or keywords; which chunks
  are RETURNED is decided by a quota (lesson 29).
- `/ask` also returns `citations` (source, page, page_end, section, subsection) for
  every retrieved chunk. These are what was RETRIEVED, not necessarily what the
  answer quoted.

### 26. Page and section metadata (`src/rag/structure.py`)
Semantic chunking lost page numbers and nothing recorded sections. After either
chunking strategy, `annotate_chunks` stamps: `page`, `page_end`, `section`,
`subsection`, `kind` (coverage / exclusions / conditions / definitions / other),
`has_exclusions`, `chunk_index`.
- **Do not use `SemanticChunker(add_start_index=True)`:** it sums chunk lengths,
  but the chunker drops inter-sentence whitespace, so offsets drift. Chunks are
  located in the page text by matching their first words (whitespace-tolerant).
- **Locator bug found on the real PDF:** an unbounded search for a chunk's last
  words matched a repeated boilerplate phrase pages later, the cursor jumped, and
  half the chunks became unlocatable. The tail search is now confined to a window
  around where the chunk should end.
- **Heading rules are policy-specific** (this Allstate-style jacket: "Section
  I—Your Property", "Losses We Do Not Cover Under Coverage ..."). They are
  anchored to the whole line / line start so cross-references inside sentences
  ("...the Section I—Losses We Do Not Cover Under...", "Additional Protection under
  item 11") aren't taken for headings, wrapped headings are joined, and the table
  of contents is skipped. Another policy layout needs its own rules; unrecognised
  headings leave `kind` at the last known value.
- **`kind` is where a chunk STARTS, `has_exclusions` is whether it overlaps any
  exclusions region.** One real semantic chunk spans pages 20-23 and starts under
  a coverage heading while containing the whole Section II exclusions list.
- Chunks ingested before this exist without the new fields; the tools default them
  and fall back to keywords alone. **Re-ingest to get the benefit:**
  `curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json"
  -d '{"pdf_path": "data/policies/apc220-ho-policy-jacket.pdf"}'` (idempotent: it
  replaces that PDF's chunks, so no need to wipe `data/chroma_db`; this makes live
  embedding calls). Offline checks on the real PDF found all 69 chunks located
  with correct page and section; retrieval quality on real embeddings is unmeasured.

### 27. Strip page furniture at ingest; keep the contents page out of search
Two live findings, one fix pass, one re-ingest.
- **Running headers/footers corrupt chunk text.** Every page of the sample policy
  ends with "Homeowners Policy / Policy number: / Policy effective date: / Your
  Allstate agency is / Page N of 26 / Sample Document". A sentence that crosses a
  page break therefore had that block spliced into its middle: 25 blocks in the
  stored chunks, 11 mid-sentence. Effects seen: embeddings polluted, and the quote
  check flagged two ACCURATE quotes as "unverified" (the LLM quoted the sentence
  correctly; the stored text was the broken one). `loader._strip_boilerplate`
  removes a line only if, digit-normalised ("Page # of #"), it sits in the first 6
  or last 10 lines of a page on at least half the pages and has 2+ words (so `a)`
  and `1.` never qualify, and body text elsewhere on a page is untouched). A page
  is never stripped to nothing, and documents under 4 pages are left alone. On the
  real PDF it removed 153 lines and no footer text remains. LIMIT: a repetition
  heuristic; a body line that genuinely repeats at the edge of most pages would go.
- **The contents page matches every query.** It names every section, so it ranked
  for almost anything (a "what about mold?" answer cited page 2-3). Chunks are now
  `kind="toc"` if they start on a contents page OR their own text has 5+
  dot-leader runs ("......7"). The second rule matters: the first real chunk can
  start on the cover page and swallow the contents, so a start-page rule alone
  misses it (found by running the pipeline on the real PDF). Both tools drop them.
- Verified offline on the real PDF (fake embeddings): both previously false-flagged
  quotes now verify, and the cover+contents chunk is marked `toc`. Chunk
  boundaries will differ after re-ingesting, since the text changed.

### 28. The quote check works: the LLM was putting paraphrases in quotation marks
After the footer fix, the "N quoted passages could not be matched" notes stopped
being false alarms and became true positives. Of the 6 quotes the check flagged
(4 on one answer, 2 on another), every one differed from the policy: dropped words ("perils **12, 13, and 14**" omitted), a
list marker removed, or a sentence that doesn't exist at all (a "quote" about
sump pumps assembled from several clauses). The meaning was mostly right; the
quotation marks were not. All three specialist prompts now say: use quotation
marks ONLY for text copied word for word, use "..." to skip words, and write
anything reworded without quotation marks (pinned by
`test_every_specialist_prompt_requires_verbatim_quotes`). **Result (live,
Nov 2026): 0 unverified quotes across all 5 end-to-end answers** (was 4 and 2 on
the runs before the rule) - the rule worked. Keep watching `unverified_quotes` in
the live answer test; it is printed, not asserted.
Note the order of events: the check first looked broken (footer noise, lesson 27),
then looked right once the data was clean. Fix the data before tuning the check.

### 29. Exclusion retrieval: a quota, because neither pure ordering works
`search_exclusions` has been through three rules, each driven by a live run:
1. Section chunks first: pulled in distant, irrelevant ones ("Satellite dish
   antennas", distance 0.91) ahead of the page with the $5,000 mold limit (0.61).
2. Distance only: seven coverage-region chunks that merely contain "we do not
   cover" (pages 9-10 list perils) crowded out the only chunk that mentions flood.
3. **Quota (current):** the closest `EXCLUSION_SECTION_QUOTA` (**4**) chunks from
   genuine exclusions regions are guaranteed a place; the remaining slots (of
   `EXCLUSION_TOP` = 6) go to the closest of everything else flagged; the result is
   returned closest-first. The quota was 3 and is **4, tuned from the probe**: for
   the agent's real query `water damage` the flood chunk was raw rank 8/30 but only
   exclusions-region rank 4, so 3 missed it by one. `Evidence.in_exclusions_section` records region
   membership (kind == "exclusions" OR `has_exclusions`), separately from
   `exclusion_language` (which also counts keyword hits).
Replaying stored vectors as proxy queries showed the quota keeps both the $5,000
chunk and the flood chunk (still true at quota 4), but proxies are not real
queries. The tests derive their expectations from the constants, so retuning means
changing one number and re-running the golden set (lesson 30). Re-run it after ANY
change here. `EXCLUSION_TOP` = 6 is still judgment, not tuned.

### 30. Golden-question evaluation (`tests/integration/`)
Retrieval was being judged by eye, and rules 1 and 2 above each fixed one question
and broke another. `golden.py` lists questions with the policy wording a good
retrieval must surface (real snippets, matched ignoring case/spacing/punctuation,
each required to be inside ONE chunk, since a snippet split across chunks is never
seen whole by the LLM). `evaluate()` reports the rank of each snippet.
- **Offline (every run):** `tests/test_golden_data.py` checks the harness and that
  every snippet exists in a single ingested chunk (skips if nothing is ingested).
  A bad snippet would fail a live run for the wrong reason.
- **Live retrieval (cheap: embeddings only, no LLM):**
  `RUN_LIVE_EVAL=1 pytest tests/integration/test_live_retrieval.py -v -s`
- **Live end to end (router + GPT-4o, several calls per question):**
  `RUN_LIVE_EVAL=full pytest tests/integration/test_live_answers.py -v -s`
  asserts the right agent, confidence != low, and required facts in the answer;
  it prints (not asserts) the unverified-quote count so the trend is visible.
- Live tests carry the `live` marker, are skipped unless `RUN_LIVE_EVAL` is set, and
  are exempt from the conftest guard that forbids the real vector store.
- **Probe:** when a case FAILS, `python -m tests.integration.probe <case-id>` prints
  how deep the required wording sits in the raw similarity ranking (default top 30):
  rank 5-8 suggests returning more chunks; rank 30+/absent means the question and
  the policy use different words (query rewriting or hybrid keyword search).
- **First live result (Nov 2026): 12 of 14 retrieval cases pass**, including both
  motivating misses (flood retrieved, rank #6 of 6 - the last slot, so thin margin;
  $5,000 mold limit #1). Two fail, both plain top-4 `search_policy` with lay wording
  against legalese: "How long do I have to sue my insurer?" (policy says "bring an
  action against us") and "Do I need to report a theft?" (theft COVERAGE chunks win;
  the wording is in the claims-conditions chunk). All four chunks returned for each
  had distances of 1.17-1.25, i.e. weak matches. CAVEAT: the eval sends the raw
  question to a plain search, while the live system has the agent write its own
  query and routes claims questions to the Claims Agent, so these may not fail end
  to end; both are also in the end-to-end set (`*-e2e`, agent left unasserted).
- **Second live result:** the probe put the two failing snippets at raw rank 5/30 and
  6/30 - close, so raising `POLICY_TOP` (4) would fix them. Decision: **don't.** Both
  end-to-end cases PASS (routed to the Claims Agent, whose own queries find them), so
  the retrieval failures are not user-visible, and +50% context on every
  `search_policy` call would buy a green eval row, not a better answer. Instead
  `GoldenCase.known_gap` records why; those rows report XFAIL (XPASS if they start
  passing) so the gap stays visible without hiding real breaks. Revisit if a real
  question of this kind starts failing end to end.
- **End-to-end result: 4 of 5 pass.** The failure is `water-exclusions-flood`: the
  answer lists backups, ground water, mold, freezing and defective systems but NOT
  flood, even though the retrieval-level case passes (rank #6 of 6, the last slot)
  and flood is the classic water exclusion. Chunk 51 (the only flood chunk) is 833
  chars, but it OPENS with soil-conditions text and the flood item sits in its last
  fifth (character ~653 of 817), so dilution by other content may well be part of why
  it ranks poorly for "water damage" (raw rank 8/30). An earlier version of this note
  said the opposite after checking only the chunk's size and the count of numbered
  items, never where flood sits in the text - that was wrong.
  **Diagnosed (live): NEVER retrieved.** The Exclusions Agent ran ONE search with the
  query `water damage`; the flood chunk didn't make the six results (the sewer
  chunk did, at #4). An agent's query can be far terser than the user's question and
  retrieve very differently, which the whole-question retrieval case could not show.
  Two responses:
  1. Prompt (cause I can see): the Exclusions Agent must search SEVERAL times with
     different wording for a broad topic (one search per distinct cause or kind of
     loss) and cover EVERY distinct exclusion it retrieved. Unverified until a live
     rerun.
  2. Measurement (so the quota is tuned, not guessed): `GoldenCase.query` reproduces
     the agent's short query (`water-damage-short-query`, currently expected to FAIL
     - it is the regression to fix, not an accepted gap), and the probe now also
     prints each snippet's **exclusions-region rank**: the rank among only
     chunks inside exclusions regions, which is the number `EXCLUSION_SECTION_QUOTA`
     must reach. Region rank 4-5 -> raise the quota; 10+ -> no sensible quota helps
     and the fix is multi-search / smaller chunks.
  Run: `python -m tests.integration.probe water-damage-short-query`
  **Probe result:** flood raw rank 8/30, exclusions-region rank 4 (sewer: 4/30, region
  3) -> quota raised 3 -> 4 (lesson 29). The end-to-end rerun then crashed in
  ChromaDB, not in retrieval: the new multi-search prompt made the agent issue
  parallel searches in a fresh process and exposed the cache race (lesson 14), now
  fixed.
  **Confirmed live after both fixes:** the flood end-to-end case PASSES. The Exclusions
  Agent now runs five searches (`water damage exclusions`, `flood damage exclusions`,
  `sewer backup exclusions`, `rainwater damage exclusions`, `groundwater damage
  exclusions`), flood is retrieved, and the answer states it (0 unverified quotes,
  confidence high). Retrieval suite: **13 pass, 2 xfail (the accepted gaps), 0 fail**;
  `water-damage-short-query` passes with flood in the LAST slot of six (#6) and sewer
  at #5, i.e. the quota is a safety net with a thin margin. The multi-search prompt is
  the more robust fix, because each query reaches different chunks. Cost: a broad
  exclusions question now makes ~5 searches (~5x the retrieval tokens, run in parallel;
  the e2e case took ~10 s).
- Add a case whenever a live run shows a miss (retrieval-level snippets first).
- LIMIT: the queries are the questions as written, not what the router or an agent
  would actually send, and snippets prove retrieval, not answer quality.

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
| `src/agents/tools.py`       | Specialists' retrieval tools returning evidence artifacts |
| `src/rag/structure.py`      | Page / section / kind metadata for chunks             |
| `tests/integration/golden.py` | Golden questions + `evaluate()` scoring; live tests beside it |
| `tests/test_golden_data.py` | Offline checks of the golden set and harness          |
| `tests/test_loader_boilerplate.py` | Header/footer stripping tests              |
| `tests/conftest.py`, `tests/fakes.py` | Offline safety net; shared fakes             |
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
