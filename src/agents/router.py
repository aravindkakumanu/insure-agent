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
"what about mold?" stays with the same specialist(s). Both that map and each
specialist's history are stored in SQLite (sessions.py), so they survive a restart.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from src.agents import claims_agent, coverage_agent, exclusions_agent
from src.agents.guardrails import apply_output_guardrails, check_prompt_injection, scrub_pii
from src.agents.sessions import SessionMap
from src.agents.results import (
    AgentResult,
    Confidence,
    RouterResult,
    citations_for,
    weakest_confidence,
)
from src.tracing import (
    annotate,
    submit_in_context,
    summarize_decision,
    summarize_router_result,
)

logger = logging.getLogger(__name__)

Intent = Literal["coverage", "exclusions", "claims", "out_of_scope"]
INTENTS: tuple[str, ...] = ("coverage", "exclusions", "claims", "out_of_scope")

# Classification is a trivial task; a small model keeps the extra hop cheap.
ROUTER_MODEL = "gpt-4o-mini"

# Said instead of running any agent when the question is not about this policy.
OUT_OF_SCOPE_ANSWER = (
    "That question isn't about your homeowners policy, so I can't answer it from the "
    "policy. I can help with what the policy covers, what it excludes, and how to make "
    "a claim."
)

# Topics a policy document is reliably silent on. A CLOSED list on purpose: a free-text "the
# policy doesn't say X" from the classifier was wrong live (it said the policy was silent on
# transferring the policy to a buyer, which the Policy Transfer clause covers).
OUTSIDE_TOPICS: dict[str, str] = {
    "rates_and_claim_history": (
        "whether or how asking about coverage, filing a claim or your claim history "
        "affects your premium or rates"
    ),
}


def not_in_policy_note(phrase: str) -> str:
    """The sentence that says the policy cannot answer part of a question."""
    return f"The policy doesn't say {phrase}; that is a question for your insurer or agent."


# Added after the note when the question names no loss, item or topic at all: there is
# nothing for the agents to look up, so none is run.
ASK_FOR_DETAIL = (
    " If you tell me what happened, or what you are thinking of claiming for, I can check "
    "what the policy says about it."
)

# Said in place of an agent's text when it answered without searching the policy at all:
# that text came from the model's own knowledge, not from the policy.
UNGROUNDED_ANSWER = (
    "I couldn't find policy text that addresses this, and I won't answer it from general "
    "knowledge. Try rephrasing, or ask about a specific coverage, exclusion or claim step."
)

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
  out_of_scope - clearly NOT about this homeowners policy: another kind of
               insurance (auto, life, health), taxes, legal or financial advice,
               or general knowledge. Use it ONLY when the question has nothing to
               do with what this policy covers, excludes or how to claim under
               it. If the question is about the home, a loss, money the insurance
               might pay in connection with a loss (including rewards, tips,
               payments to other people), someone being hurt or suing, your rights or deadlines under the policy (suing or
               disputing the insurer, cancelling, renewing, premiums, selling the
               house or transferring the policy, what happens to the policy
               itself), or the policy itself, even if the policy may not answer it, use
               coverage, exclusions or claims instead. A question asking whether
               THIS policy would pay for, cover or allow something is never
               out_of_scope, even if the answer is "no".

You are also given previous_intent and previous_question, from the previous turn
of this conversation (or "none"). If the question is a short follow-up with no
intent of its own (for example "what about mold?"), return previous_intent.

Also fill in two fields about what the POLICY DOCUMENT can answer:
  outside_topic - "rates_and_claim_history" when the question (or part of it) asks
                  how asking about coverage, filing a claim or your claim
                  history affects your premium, rates or record with the
                  insurer. A policy document never says that. Otherwise "none":
                  do NOT use it for anything else, even if you think the policy
                  might not mention it (selling the house, transferring or
                  cancelling the policy, and so on are policy terms).
  names_specific_subject - true when the question names a specific loss, item,
                  event or policy topic; false for a general request such as
                  "check whether something would be covered" with nothing named.

Also rewrite the question as two STANDALONE questions, each understandable
without the conversation, keeping every specific the user gave (the peril, the
item, amounts) and ALSO what the user is worried about or asking for (a threat to
sue, a request for payment, a demand), not only the event, and adding no facts of
your own. Resolve follow-ups using
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
    outside_topic: Literal["none", "rates_and_claim_history"] = Field(
        default="none", description="A topic a policy document never states, or 'none'"
    )
    names_specific_subject: bool = Field(
        default=True, description="Whether the question names a specific loss, item or topic"
    )


@dataclass(frozen=True)
class RoutingDecision:
    """The classifier's output, with rewrites already defaulted to the original."""

    intent: Intent
    coverage_question: str
    exclusions_question: str
    not_in_policy: str = ""  # what the question asks that a policy document cannot state
    names_specific_subject: bool = True

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


# session_id -> intent / standalone question of that session's previous turn.
# Persistent (sessions.py), so a bare follow-up still works after a restart.
_session_intent = SessionMap("intent")
_session_question = SessionMap("question")


@traceable(name="route", run_type="chain", process_outputs=summarize_decision)
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
        annotate(classifier_failed=True)
        return RoutingDecision(DEFAULT_INTENT, question, question)

    decision = RoutingDecision(
        intent=intent,
        coverage_question=(getattr(result, "coverage_question", "") or "").strip() or question,
        exclusions_question=(getattr(result, "exclusions_question", "") or "").strip() or question,
        not_in_policy=OUTSIDE_TOPICS.get(getattr(result, "outside_topic", "none"), ""),
        names_specific_subject=bool(getattr(result, "names_specific_subject", True)),
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


def _discard_ungrounded(result: AgentResult) -> AgentResult:
    """Replace an answer that was written without a single policy search."""
    if result.queries:
        return result
    logger.warning("[%s] answered without searching the policy; discarding the text", result.agent_name)
    return replace(result, answer=UNGROUNDED_ANSWER)


@traceable(name="ask", run_type="chain", process_outputs=summarize_router_result)
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

    annotate(session_id=session_id)
    decision = classify_intent(
        question, _session_intent.get(session_id), _session_question.get(session_id)
    )
    intent = decision.intent
    if intent == "out_of_scope":
        # No agent runs and the session's remembered intent/question are left alone, so a
        # stray off-topic question doesn't break the follow-up that comes after it.
        annotate(intent=intent, agent="out_of_scope", confidence="high")
        if decision.not_in_policy:  # say WHAT the policy cannot state, not just "not about your policy"
            answer = not_in_policy_note(scrub_pii(decision.not_in_policy)) + ASK_FOR_DETAIL
        else:
            answer = OUT_OF_SCOPE_ANSWER
        return RouterResult(
            answer=answer,
            session_id=session_id,
            sources=[],
            agent="out_of_scope",
            intent=intent,
            confidence="high",  # grounding does not apply: the reply makes no claim about the policy
            results=(),
        )
    note = scrub_pii(not_in_policy_note(decision.not_in_policy)) if decision.not_in_policy else ""
    if note and not decision.names_specific_subject:
        # Only a question the policy cannot answer, with nothing specific to look up.
        annotate(intent=intent, agent="not_in_policy", confidence="high")
        return RouterResult(
            answer=note + ASK_FOR_DETAIL,
            session_id=session_id,
            sources=[],
            agent="not_in_policy",
            intent=intent,
            confidence="high",
            results=(),
        )
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
                submit_in_context(pool, specialists[name].answer_question, text, session_id)
                for name, text in questions.items()
            ]
            results = [f.result() for f in futures]

    results = [_discard_ungrounded(r) for r in results]
    merged = apply_output_guardrails(merge_results(results, session_id, intent, questions))
    if note:
        merged = replace(merged, answer=f"{note}\n\n{merged.answer}")
    annotate(
        intent=merged.intent,
        agent=merged.agent,
        confidence=merged.confidence,
        unverified_quotes=sum(len(r.unverified_quotes) for r in merged.results),
    )
    return merged
