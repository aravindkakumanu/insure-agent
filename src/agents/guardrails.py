"""
guardrails.py — Input and output safety checks for the router
==============================================================
Three checks, wired into route_and_answer() in router.py:

  1. check_prompt_injection(question)   BEFORE routing - raises
     GuardrailViolation so an injection never reaches the router LLM or any
     agent.
  2. scrub_pii(text)                    AFTER - redacts SSNs and policy-number-
     like IDs from the answer, and (via PIIScrubbingFormatter) from the logs.
  3. apply_confidence_gate(...)         AFTER - prepends a visible warning when
     confidence is "low".

LIMITS - read before relying on these:
  - Injection detection is phrase matching. It catches blatant attempts and is
    trivially evaded by rewording; it is a tripwire, not a defence. The
    specialists' "answer ONLY from retrieved policy text" prompts remain the
    real protection.
  - Phrase matching also has false positives (e.g. "should I forget the policy
    deductible?" matches "forget the policy").
  - PII scrubbing is regex-based: it will miss formats it doesn't know (SSNs
    without dashes, names, addresses) and may redact harmless look-alikes.
"""

import logging
import re
from dataclasses import replace
from typing import Optional

from src.agents.results import AgentResult, RouterResult

logger = logging.getLogger(__name__)


class GuardrailViolation(Exception):
    """Raised when an input is blocked. str(exc) is safe to show the client."""


# ---------------------------------------------------------------------------
# 1. Prompt injection detection
# ---------------------------------------------------------------------------

# (label, pattern). Matched against the lowercased, whitespace-collapsed text.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore your instructions",
        re.compile(
            r"\bignore\s+(?:all\s+|any\s+)?(?:your|previous|prior|earlier|the\s+above|above)"
            r"\s+(?:\w+\s+)?(?:instructions|rules|prompt)\b"
        ),
    ),
    (
        "pretend you are",
        re.compile(r"\bpretend\s+(?:that\s+)?(?:you\s+are|you're|to\s+be)\b"),
    ),
    (
        "disregard / forget the policy",
        re.compile(
            r"\b(?:disregard|forget)\s+(?:all\s+)?(?:the\s+|your\s+)?(?:policy|instructions)\b"
        ),
    ),
    (
        "say the policy covers everything",
        re.compile(r"\bsay\s+(?:that\s+)?the\s+policy\s+covers\s+(?:everything|all)\b"),
    ),
]


def check_prompt_injection(question: str) -> None:
    """
    Raise GuardrailViolation if the question looks like a prompt injection.

    Returns None when the question is clean.
    """
    normalized = re.sub(r"\s+", " ", question.lower())
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            logger.warning("Prompt injection blocked (pattern=%r)", label)
            raise GuardrailViolation(
                f"Your question was blocked because it appears to ask the assistant "
                f"to ignore its instructions or the policy text (matched: '{label}'). "
                "Please rephrase it as a question about the policy."
            )


# ---------------------------------------------------------------------------
# 2. PII scrubbing
# ---------------------------------------------------------------------------

SSN_REDACTION = "[REDACTED-SSN]"
POLICY_NUMBER_REDACTION = "[REDACTED-POLICY-NUMBER]"

_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

# An ID-like token: 8+ chars of letters/digits/hyphens containing at least one
# letter AND one digit (so ordinary words, plain numbers and dates are left
# alone). It must be a whole token, and must not be followed by ".<letter>" so
# filenames like "apc220-ho-policy-jacket.pdf" survive.
_POLICY_NUMBER = re.compile(
    r"(?<![\w-])"
    r"(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])"
    r"[A-Za-z0-9][A-Za-z0-9-]{7,}"
    r"(?![\w-]|\.[A-Za-z])"
)

# Session ids are UUIDs and match the pattern above; keep them readable in logs.
_UUID = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def _redact_policy_number(match: re.Match[str]) -> str:
    token = match.group(0)
    return token if _UUID.fullmatch(token) else POLICY_NUMBER_REDACTION


def scrub_pii(text: str) -> str:
    """Redact SSNs (XXX-XX-XXXX) and policy-number-like IDs from text."""
    text = _SSN.sub(SSN_REDACTION, text)
    return _POLICY_NUMBER.sub(_redact_policy_number, text)


class PIIScrubbingFormatter(logging.Formatter):
    """
    Log formatter that scrubs the FINAL formatted line, so PII is removed
    wherever it came from: the message, its args, a traceback, or a library.
    """

    def format(self, record: logging.LogRecord) -> str:
        return scrub_pii(super().format(record))


def install_log_scrubbing(root: Optional[logging.Logger] = None) -> None:
    """Swap every handler on the root logger to a PIIScrubbingFormatter."""
    root = root or logging.getLogger()
    for handler in root.handlers:
        existing = handler.formatter
        if isinstance(existing, PIIScrubbingFormatter):
            continue
        fmt = existing._fmt if existing is not None else None
        datefmt = existing.datefmt if existing is not None else None
        handler.setFormatter(PIIScrubbingFormatter(fmt, datefmt))


# ---------------------------------------------------------------------------
# 3. Confidence gating
# ---------------------------------------------------------------------------

LOW_CONFIDENCE_WARNING = (
    "⚠️ Warning: limited policy text was found for this question. "
    "This answer may be incomplete."
)


def apply_confidence_gate(answer: str, confidence: str) -> str:
    """Prepend LOW_CONFIDENCE_WARNING to the answer when confidence is "low"."""
    if confidence == "low":
        return f"{LOW_CONFIDENCE_WARNING}\n\n{answer}"
    return answer


# ---------------------------------------------------------------------------
# Output pipeline
# ---------------------------------------------------------------------------


def apply_output_guardrails(result: RouterResult) -> RouterResult:
    """
    Scrub PII from the answer (and from each specialist's answer in the
    breakdown), then gate on confidence. Returns a new RouterResult.
    """
    scrubbed_results: tuple[AgentResult, ...] = tuple(
        replace(r, answer=scrub_pii(r.answer)) for r in result.results
    )
    answer = apply_confidence_gate(scrub_pii(result.answer), result.confidence)
    return replace(result, answer=answer, results=scrubbed_results)
