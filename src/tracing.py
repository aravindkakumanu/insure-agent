"""
tracing.py — LangSmith tracing helpers
=======================================
Tracing is OFF unless LANGSMITH_TRACING=true is in the environment (add
LANGSMITH_API_KEY too; see .env.example). With it off, `@traceable` calls the
function directly, `annotate` finds no run to write to, and the summarisers below
are never called, so normal runs and the offline tests pay nothing and send
nothing.

What a trace of one /ask shows:

    ask                          root: question, session, intent, agents, confidence
    ├── route                    the router decision (and the classifier LLM call
    │   └── ChatOpenAI           under it: tokens + latency)
    ├── coverage_agent           one run per specialist that ran, in parallel
    │   ├── LangGraph → ChatOpenAI   each agent step's LLM call: tokens + latency
    │   └── search_policy        each tool call: its query, and in metadata how
    │                            many chunks came back, their pages and distance
    └── exclusions_agent         (same shape)

Token counts and per-step latency come from LangSmith itself (the OpenAI client
reports usage on every LLM run; every run has a start and end time).

Two things in here exist because of how this app runs:
- `submit_in_context`: the coverage question runs two specialists in a thread
  pool. A new thread does not inherit the trace context, so without copying it the
  specialists would be detached from the `ask` run (and not traced at all unless
  tracing was enabled process-wide). Found by experiment, pinned by a test.
- The summarisers keep the trace readable: an AgentResult carries every retrieved
  chunk's full text, which the tool runs already show.

PRIVACY: a trace holds the user's question, the rewritten questions, the answer and
the retrieved policy text, and is sent to LangSmith. The answer is PII-scrubbed
(guardrails.py) before it is returned, and so before it is traced; the question and
the sub-questions are not scrubbed. Enable tracing only where that is acceptable.
"""

import contextvars
from concurrent.futures import Executor, Future
from typing import Any, Callable

from langsmith.run_helpers import get_current_run_tree


def annotate(**metadata: Any) -> None:
    """Add filterable metadata to the current run; does nothing when not tracing."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(metadata)


def submit_in_context(pool: Executor, fn: Callable[..., Any], *args: Any) -> Future:
    """pool.submit(fn, *args), but the worker thread joins the caller's trace."""
    return pool.submit(contextvars.copy_context().run, fn, *args)


def annotate_retrieval(chunks: list, **extra: Any) -> None:
    """Record what a search tool returned: how many chunks, from where, how close."""
    annotate(
        chunks_returned=len(chunks),
        pages=sorted({c["page"] for c in chunks if c["page"]}),
        closest_distance=min((c["distance"] for c in chunks), default=None),
        **extra,
    )


def annotate_agent_result(result: Any, fell_back: bool) -> None:
    """Record a specialist's outcome on its run."""
    annotate(
        agent=result.agent_name,
        confidence=result.confidence,
        searches=len(result.queries),
        chunks_retrieved=len(result.evidence),
        unverified_quotes=len(result.unverified_quotes),
        guard_tripped=fell_back,
    )


def summarize_agent_result(result: Any) -> dict:
    """Trace output for a specialist: the answer and how it was grounded."""
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "queries": list(result.queries),
        "sources": list(result.sources),
        "unverified_quotes": list(result.unverified_quotes),
    }


def summarize_decision(decision: Any) -> dict:
    """Trace output for the router decision."""
    return {
        "intent": decision.intent,
        "coverage_question": decision.coverage_question,
        "exclusions_question": decision.exclusions_question,
    }


def summarize_router_result(result: Any) -> dict:
    """Trace output for the whole request."""
    return {
        "answer": result.answer,
        "agent": result.agent,
        "intent": result.intent,
        "confidence": result.confidence,
        "citations": len(result.citations),
    }
