"""
router.py — Stage 3: Intent Router
===================================
Decides which specialist agent(s) should answer a question. This is NOT a
LangGraph graph: classify_intent() is one small LLM call with structured output
that returns a label.

    coverage   → Coverage Agent AND Exclusions Agent (run in parallel, merged)
    exclusions → Exclusions Agent only
    claims     → Claims Agent (uses both retrieval tools internally)

SUB-QUESTIONS: the classifier also rewrites the question once per specialist.
A coverage-framed question ("Does my policy cover a burst pipe?") sent as-is to
the Exclusions Agent makes it search for coverage text and return the coverage
grant. So each specialist is asked its own standalone question ("What exclusions
or conditions apply to water damage from a burst pipe?"). The rewrite also
resolves bare follow-ups using the previous question.

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
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agents import claims_agent, coverage_agent, exclusions_agent
from src.agents.guardrails import apply_output_guardrails, check_prompt_injection
from src.agents.results import (
    AgentResult,
    Confidence,
    RouterResult,
    citations_for,
    weakest_confidence,
)

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

You are also given previous_intent and previous_question, from the previous turn
of this conversation (or "none"). If the question is a short follow-up with no
intent of its own (for example "what about mold?"), return previous_intent.

Also rewrite the question as two STANDALONE questions, each understandable
without the conversation, keeping every specific the user gave (the peril, the
item, amounts) and adding no facts of your own. Resolve follow-ups using
previous_question.
  coverage_question   - asks what the policy covers for the subject, including
                        limits, sublimits and conditions
  exclusions_question - asks what exclusions, limitations or conditions remove
                        or reduce coverage for the same subject
Example: "Does my policy cover water damage from a burst pipe?"
  coverage_question:   "What does the policy cover for water damage from a burst
                        pipe, including limits and conditions?"
  exclusions_question: "What exclusions, limitations or conditions apply to water
                        damage from a burst pipe?"
"""


class IntentClassification(BaseModel):
    """Structured output the router LLM must return."""

    intent: Intent = Field(description="The single best intent label")
    # Defaults keep the schema lenient: a missing rewrite falls back to the
    # user's own question rather than failing the request.
    coverage_question: str = Field(
        default="", description="Standalone question about what is covered"
    )
    exclusions_question: str = Field(
        default="", description="Standalone question about exclusions and limitations"
    )


@dataclass(frozen=True)
class RoutingDecision:
    """The classifier's output, with rewrites already defaulted to the original."""

    intent: Intent
    coverage_question: str
    exclusions_question: str

    @property
    def main_question(self) -> str:
        """The standalone phrasing to remember as 'the previous question'."""
        return self.exclusions_question if self.intent == "exclusions" else self.coverage_question


_router_llm = None  # lazy: needs OPENAI_API_KEY at construction time


def _get_router_llm():
    """Return the structured-output router LLM, creating it on first call."""
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0).with_structured_output(
            IntentClassification
        )
    return _router_llm


# session_id -> intent / standalone question of that session's previous turn
_session_intent: dict[str, str] = {}
_session_question: dict[str, str] = {}


def classify_intent(
    question: str,
    previous_intent: Optional[str] = None,
    previous_question: Optional[str] = None,
) -> RoutingDecision:
    """
    Classify a question and rewrite it once per specialist.

    A missing or blank rewrite falls back to the original question. On any
    classifier failure (API error, unparseable output) this logs a warning with
    the traceback and returns DEFAULT_INTENT with the original question for
    every specialist, rather than failing the user's request.
    """
    try:
        result = _get_router_llm().invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(
                    content=f"previous_intent: {previous_intent or 'none'}\n"
                    f"previous_question: {previous_question or 'none'}\n"
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
        return RoutingDecision(DEFAULT_INTENT, question, question)

    decision = RoutingDecision(
        intent=intent,
        coverage_question=(getattr(result, "coverage_question", "") or "").strip() or question,
        exclusions_question=(getattr(result, "exclusions_question", "") or "").strip() or question,
    )
    logger.info("Router: %r -> %s (previous=%s)", question[:60], intent, previous_intent)
    return decision


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


def _quote_note(result: AgentResult) -> str:
    """Caution when the answer quotes policy wording that wasn't retrieved."""
    n = len(result.unverified_quotes)
    if not n:
        return ""
    noun = "passage" if n == 1 else "passages"
    return (
        f"Note: {n} quoted {noun} could not be matched to the retrieved policy "
        "text, so treat that wording as unverified."
    )


def _section(result: AgentResult, redundant: bool) -> str:
    """One headed block of a merged answer, with any cautions that apply."""
    heading = _SECTION_HEADINGS.get(result.agent_name, result.agent_name.title())
    notes: list[str] = []
    if result.confidence == "low":
        notes.append(
            _LOW_CONFIDENCE_NOTES.get(
                result.agent_name,
                f"Note: the {result.agent_name} answer could not be reliably "
                "confirmed from the policy text.",
            )
        )
    if redundant:
        notes.append(
            "Note: this section drew only on policy text already used above, so "
            "no separate language was found for it."
        )
    if quote_note := _quote_note(result):
        notes.append(quote_note)
    block = f"**{heading}**\n\n{result.answer}"
    return block + "".join(f"\n\n_{note}_" for note in notes)


def merge_results(
    results: list[AgentResult],
    session_id: str,
    intent: str,
    questions: Optional[dict[str, str]] = None,
) -> RouterResult:
    """
    Assemble a RouterResult from one or more specialist AgentResults.

    - answer: a single result passes through (plus a note if its quotes were
      unverified); several become headed sections, each with cautions for low
      confidence, unverified quotes, or REDUNDANCY
    - redundancy: a later section whose retrieved chunks were all already used
      by an earlier one found nothing of its own. Its text is kept (it may still
      be a valid reading) but it is flagged and its confidence is capped at
      medium
    - sources / citations: the union across specialists, de-duplicated
    - agent: the specialist names joined with "+", in the order given
    - confidence: the weakest effective confidence among the specialists
    - results: the per-specialist breakdown, kept for callers that want it
    """
    if not results:
        raise ValueError("merge_results needs at least one AgentResult")

    seen_chunks: set[str] = set()
    redundant_flags: list[bool] = []
    for result in results:
        ids = {e["chunk_id"] for e in result.evidence}
        redundant_flags.append(bool(ids) and ids <= seen_chunks)
        seen_chunks |= ids

    if len(results) == 1:
        note = _quote_note(results[0])
        answer = results[0].answer + (f"\n\n_{note}_" if note else "")
    else:
        answer = "\n\n".join(_section(r, red) for r, red in zip(results, redundant_flags))

    effective: list[Confidence] = [
        weakest_confidence([r.confidence, "medium"]) if red else r.confidence
        for r, red in zip(results, redundant_flags)
    ]
    return RouterResult(
        answer=answer,
        session_id=session_id,
        sources=sorted({src for r in results for src in r.sources}),
        agent="+".join(r.agent_name for r in results),
        intent=intent,
        confidence=weakest_confidence(effective),
        results=tuple(results),
        citations=citations_for(e for r in results for e in r.evidence),
        questions=dict(questions or {}),
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

    decision = classify_intent(
        question, _session_intent.get(session_id), _session_question.get(session_id)
    )
    intent = decision.intent
    _session_intent[session_id] = intent
    _session_question[session_id] = decision.main_question

    # Each specialist gets its own standalone question (see module docstring).
    if intent == "exclusions":
        questions = {"exclusions": decision.exclusions_question}
    elif intent == "claims":
        questions = {"claims": question}
    else:
        questions = {
            "coverage": decision.coverage_question,
            "exclusions": decision.exclusions_question,
        }
    logger.info("Router sub-questions: %s", questions)

    specialists = {
        "coverage": coverage_agent,
        "exclusions": exclusions_agent,
        "claims": claims_agent,
    }
    if len(questions) == 1:
        (name, text), = questions.items()
        results = [specialists[name].answer_question(text, session_id)]
    else:
        # coverage: run both specialists concurrently (each blocks on OpenAI)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(specialists[name].answer_question, text, session_id)
                for name, text in questions.items()
            ]
            results = [f.result() for f in futures]

    return apply_output_guardrails(merge_results(results, session_id, intent, questions))
