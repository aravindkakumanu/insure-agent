"""
router.py — Stage 3: Intent Router
===================================
Decides which specialist agent(s) should answer a question. This is NOT a
LangGraph graph: classify_intent() is one small LLM call with structured output
that returns a label.

    coverage   → Coverage Agent AND Exclusions Agent (run in parallel, merged)
    exclusions → Exclusions Agent only
    claims     → Claims Agent (uses both retrieval tools internally)

Each specialist returns a typed AgentResult (answer, sources, agent_name,
confidence); merge_results() assembles those into a RouterResult from the
structured fields rather than by gluing strings together. See results.py.

GUARDRAILS (guardrails.py): route_and_answer() rejects prompt injections before
anything else runs, and passes every result through PII scrubbing and
confidence gating on the way out.

Coverage questions always fan out to the Exclusions Agent too, for the same
reason as the Stage 2 system prompt: an answer about what IS covered that
leaves out what is NOT covered misleads the user.

SESSIONS: the session_id is passed straight through to the specialist(s), each
of which keeps its own history under that id. The router remembers the last
intent per session and hands it to the classifier, so a bare follow-up like
"what about mold?" stays with the same specialist(s). Like MemorySaver, this
map is in-process and lost on restart.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agents import claims_agent, coverage_agent, exclusions_agent
from src.agents.guardrails import apply_output_guardrails, check_prompt_injection
from src.agents.results import AgentResult, RouterResult, weakest_confidence

logger = logging.getLogger(__name__)

Intent = Literal["coverage", "exclusions", "claims"]
INTENTS: tuple[str, ...] = ("coverage", "exclusions", "claims")

# Classification is a trivial task; a small model keeps the extra hop cheap.
ROUTER_MODEL = "gpt-4o-mini"

# Safe default when classification fails: coverage fans out to the Exclusions
# Agent as well, so it is the least likely to give a one-sided answer.
DEFAULT_INTENT: Intent = "coverage"

ROUTER_PROMPT = """\
You classify the intent of a question about an insurance policy. Reply with
exactly one label:

  coverage   - what the policy covers, protects, or pays for; whether something
               is covered; limits and deductibles
  exclusions - what is NOT covered, excluded, limited, or what could void or
               reduce coverage
  claims     - how to file or handle a claim: reporting a loss, deadlines,
               documents or proof required, how claims are paid or settled

You are also given previous_intent, the label of the previous question in this
conversation (or "none"). If the question is a short follow-up with no intent
of its own (for example "what about mold?"), return previous_intent.
"""


class IntentClassification(BaseModel):
    """Structured output the router LLM must return."""

    intent: Intent = Field(description="The single best intent label")


_router_llm = None  # lazy: needs OPENAI_API_KEY at construction time


def _get_router_llm():
    """Return the structured-output router LLM, creating it on first call."""
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0).with_structured_output(
            IntentClassification
        )
    return _router_llm


# session_id -> intent of that session's previous question
_session_intent: dict[str, str] = {}


def classify_intent(question: str, previous_intent: Optional[str] = None) -> Intent:
    """
    Classify a question as "coverage", "exclusions", or "claims".

    On any classifier failure (API error, unparseable output) this logs a
    warning with the traceback and returns DEFAULT_INTENT rather than failing
    the user's request.
    """
    try:
        result = _get_router_llm().invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(
                    content=f"previous_intent: {previous_intent or 'none'}\n"
                    f"question: {question}"
                ),
            ]
        )
        intent = getattr(result, "intent", None)
        if intent not in INTENTS:
            raise ValueError(f"router returned unusable result: {result!r}")
    except Exception:
        logger.warning(
            "Intent classification failed; defaulting to %r", DEFAULT_INTENT, exc_info=True
        )
        return DEFAULT_INTENT

    logger.info("Router: %r -> %s (previous=%s)", question[:60], intent, previous_intent)
    return intent


# How each specialist's answer is headed when several are merged, in the
# Stage 2 structure: what IS covered (with limits and conditions, per the
# Coverage prompt), then what is NOT covered.
_SECTION_HEADINGS: dict[str, str] = {
    "coverage": "What IS covered",
    "exclusions": "What is NOT covered",
}

# Appended under a section whose confidence is "low". Exclusions gets the
# stronger wording: a failed exclusions search merged into a confident-looking
# answer reads as "nothing is excluded", which is the misleading case.
_LOW_CONFIDENCE_NOTES: dict[str, str] = {
    "coverage": "Note: coverage could not be reliably confirmed from the policy text.",
    "exclusions": (
        "Note: exclusions could not be reliably confirmed from the policy text. "
        "Do not assume nothing is excluded."
    ),
}


def _section(result: AgentResult) -> str:
    """One headed block of a merged answer, with a caution if confidence is low."""
    heading = _SECTION_HEADINGS.get(result.agent_name, result.agent_name.title())
    block = f"**{heading}**\n\n{result.answer}"
    if result.confidence == "low":
        note = _LOW_CONFIDENCE_NOTES.get(
            result.agent_name,
            f"Note: the {result.agent_name} answer could not be reliably "
            "confirmed from the policy text.",
        )
        block += f"\n\n_{note}_"
    return block


def merge_results(results: list[AgentResult], session_id: str, intent: str) -> RouterResult:
    """
    Assemble a RouterResult from one or more specialist AgentResults.

    - answer: a single result passes through unchanged; several become headed
      sections, each with a caution when that specialist's confidence is low
    - sources: the sorted union of every specialist's sources
    - agent: the specialist names joined with "+", in the order given
    - confidence: the weakest confidence among the specialists
    - results: the per-specialist breakdown, kept for callers that want it
    """
    if not results:
        raise ValueError("merge_results needs at least one AgentResult")

    answer = (
        results[0].answer
        if len(results) == 1
        else "\n\n".join(_section(r) for r in results)
    )
    return RouterResult(
        answer=answer,
        session_id=session_id,
        sources=sorted({src for r in results for src in r.sources}),
        agent="+".join(r.agent_name for r in results),
        intent=intent,
        confidence=weakest_confidence(r.confidence for r in results),
        results=tuple(results),
    )


def route_and_answer(question: str, session_id: Optional[str] = None) -> RouterResult:
    """
    Classify the question, run the right specialist(s), and return one result.

    The RouterResult's `agent` names what handled it: "coverage+exclusions",
    "exclusions", or "claims".

    Raises:
        GuardrailViolation: the question looks like a prompt injection. Checked
            first, so nothing (classifier, agents, session state) sees it.
    """
    check_prompt_injection(question)

    if session_id is None:
        session_id = str(uuid.uuid4())

    intent = classify_intent(question, _session_intent.get(session_id))
    _session_intent[session_id] = intent

    if intent == "exclusions":
        results = [exclusions_agent.answer_question(question, session_id)]
    elif intent == "claims":
        results = [claims_agent.answer_question(question, session_id)]
    else:
        # coverage: run both specialists concurrently (each blocks on OpenAI)
        with ThreadPoolExecutor(max_workers=2) as pool:
            cov_future = pool.submit(coverage_agent.answer_question, question, session_id)
            exc_future = pool.submit(exclusions_agent.answer_question, question, session_id)
            results = [cov_future.result(), exc_future.result()]

    return apply_output_guardrails(merge_results(results, session_id, intent))
