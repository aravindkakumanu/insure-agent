# Search-retry and scope confirmation, 2026-11-17

Changes: (1) a specialist that answers without a single search is retried ONCE with an explicit
"search first" nudge; text still unsearched after that is discarded (`UNGROUNDED_ANSWER`).
(2) When the cheap classifier (gpt-4o-mini) labels a question `out_of_scope`, a stronger model
(gpt-4o, `ScopeCheck`) must confirm before the router refuses; "when unsure, answer true". If the
confirmation call fails, the refusal stands.

Targeted live check (7 cases, ~$0.30; raw: `2026-11-17-e2e-targeted-after-retry-and-confirm-raw.txt`):
7 of 7 pass, 0 API errors.
- `h2-selling-the-house`: was refused every time (0/8); now reaches the agents and cites the
  transfer clause ("may not transfer the policy ... without the insurer's written consent").
- `general-coverage-question-is-still-answered`: a normal overview with the coverage section
  present (it had been replaced by "couldn't find policy text" in the two previous runs).
- `h3-tip-money-for-arsonist`, auto deductible, market value, tax, `rates-after-claim`: correct.
  The refusals that should stand were confirmed by the stronger model.

NOT verified: the retry path itself did not fire in this run (the coverage agent searched), so it
is proven only by offline tests that fail when it is removed. One run per case; the full
31-case suite was not re-run (cost). A tax question that the classifier tags with the rates topic
would still show the rates note: untouched.
