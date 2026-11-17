# End-to-end, after the "policy doesn't say" note (ask-before-filing), 2026-11-17

Change: the classifier also reports `not_in_policy` (what the question asks that a policy
document cannot state) and `names_specific_subject`. With no subject named, the router
replies with the note plus an invitation to name a loss (no agent runs). With a subject, the
agents run and the note is put in front of their answer. An `out_of_scope` label that carries
a `not_in_policy` phrase says WHAT the policy cannot state instead of the generic reply.
The `out_of_scope` prompt also says a "would this policy pay for / cover X" question is
never out of scope, even when the answer is no.

## Runs
- `2026-11-17-e2e-after-not-in-policy-raw.txt`: first run. VALID, 24 cases, 23 pass
  (the one failure is the known burglary routing case). It showed two side effects:
  `near-miss-credit-monitoring` was sent out of scope, and `rates-after-claim` got the generic
  out-of-scope reply.
- `2026-11-17-e2e-after-not-in-policy-fix-raw.txt`: second run, after fixing those two.
  **PARTLY INVALID: the OpenAI account ran out of credits mid-run (HTTP 429
  `credit_balance_exhausted`)**. Everything from `h-icy-driveway-lawsuit` on failed with
  classifier errors and is meaningless. The behaviour cases ran before that and are valid.

## Valid results (live)
| case | result |
|---|---|
| h-ask-before-filing-a-claim | FIXED: "The policy doesn't say whether filing a claim affects your rates; that is a question for your insurer or agent. If you tell me what happened ... I can check what the policy says about it." No agents, no padding. |
| rates-after-claim (dev) | same tailored reply (after the fix; the first run gave the generic out-of-scope reply) |
| rates-with-a-named-loss (dev) | note first, then a normal coverage answer for the stolen laptop |
| general-coverage-question-is-still-answered (dev) | normal overview, no note: no false positive |
| near-miss-credit-monitoring | correct again after the fix: the policy does not mention credit monitoring |
| auto deductible, tax, injection, SSN | unchanged and good |

## Full re-run after the credits were topped up (valid, 28 cases, no API errors)

Files: `2026-11-17-e2e-full-after-not-in-policy-raw.txt` (24 of 28), then
`2026-11-17-e2e-full-after-closed-topic-raw.txt` (25 of 28) and
`2026-11-17-e2e-selling-house-check-raw.txt` (8 selected cases, 8 pass).

**A defect in this change, found by the fresh held-out set.** The first full run sent
`h2-selling-the-house` ("Can the buyer take over my policy?") to the note: "The policy
doesn't say whether a buyer can take over your policy". It does: the Policy Transfer clause.
A free-text "the policy doesn't say X" from the classifier can simply be wrong. Fix: the
classifier now picks `outside_topic` from a CLOSED list (only `rates_and_claim_history`,
mapped to a fixed phrase); anything else cannot become a claim of silence. A second fix: the
in-scope list now includes selling the house, transferring the policy and what happens to the
policy itself (the label `out_of_scope` had also swallowed that question). Both are
principled changes, but `h2-selling-the-house` was seen before them.

Final live state of the changed behaviour: ask-before-filing, `rates-after-claim`, a named
loss with a rates question, "what does my policy cover?" (no note), credit monitoring,
auto, tax, selling the house: all good.

## Fresh held-out set (h2), first exposure, human read
| case | read |
|---|---|
| h2-stolen-keys-locks | GOOD (re-key locks, $500); the NOT-covered section just repeats the coverage |
| h2-selling-the-house | wrong at first (above); good after the two fixes |
| h2-code-upgrade-after-fire | WEAK / flaky. The agents never searched the policy's own term ("building codes"). Run 1: said the policy has nothing on upgrades, retrieval failed. Run 2: retrieved the 10% building-codes coverage by luck (rank 15 of 16) but still said "the policy does not provide specific coverage" before mentioning it. The mechanical check passed run 2; the answer is contradictory |
| h2-ground-washed-out | GOOD ($10,000 land stability found, with the earth-movement caveats) |

## Still failing / flaky
- `h-burglary-laptop-jewelry`: fails every time (claims routing; skipped on request).
- `h-icy-driveway-lawsuit`: passed in some runs and failed in others (2 of the last 3 full
  runs): the coverage agent's lay-wording queries do not reach the liability chunk, and when
  they miss, the answer says the policy has no liability coverage. The new "search again"
  rule does not reliably cure that; the underlying problem is query wording.
- Every run is a single sample of a non-deterministic system.
