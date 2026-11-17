# End-to-end, after the out-of-scope rule, 2026-11-17

Raw outputs (every answer in full, unfiltered):
`2026-11-17-e2e-after-out-of-scope-raw.txt` (first attempt) and
`2026-11-17-e2e-after-out-of-scope-fix-raw.txt` (after the two fixes below). Compare with
`2026-11-17-e2e.md` (before). Single runs; the LLM calls are not deterministic.

## What changed
1. Router: a fourth label `out_of_scope` (fixed reply, no agent runs, session context untouched).
2. All three specialist prompts: say plainly that the policy is silent, do not pad, do not
   answer from general knowledge, and treat a nearby provision as a different thing.
3. Router guard: an agent answer written without a single search is replaced by a fixed message.

## First attempt: 17 of 21 pass, and it broke things
- `legal-action-deadline-e2e` ("How long do I have to sue my insurer?", a dev case) was
  routed out of scope: a false positive on a real policy question.
- Two previously good lay-wording answers (icy driveway, baseball) turned into "the policy
  does not address liability": the new "say it is silent" rule made the agents declare
  silence after one weak search.

## Fixes (principled, not aimed at a case)
- Router prompt now lists what is IN scope (someone hurt or suing, suing or disputing the
  insurer, cancelling, renewing, premiums, deadlines).
- Specialist prompts: "Before concluding that the policy is silent, search again with
  different wording" (the policy uses formal terms, not everyday words).

## After the fixes: 20 of 21 pass (the 21 = 16 behaviour/H cases + 5 older end-to-end cases)

| case | before | after |
|---|---|---|
| out-of-scope-auto-deductible | WEAK (padded, "your auto policy") | clean: fixed out-of-scope reply, no agents |
| out-of-scope-market-value | WEAK (padded) | out-of-scope reply (see caveat) |
| out-of-scope-tax-deduction | WRONG (model knowledge, no search) | clean: fixed out-of-scope reply |
| near-miss-credit-monitoring | FAIL (claimed credit monitoring) | correct: says it does not cover it, names the $1,000 card section as different; still padded by the two-section layout |
| h-ask-before-filing-a-claim | POOR | unchanged: still a dump of coverages, never says the policy is silent on rates |
| h-icy-driveway-lawsuit, h-baseball-neighbors-window | good | good (after the regression and its fix) |
| h-burglary-laptop-jewelry | FAIL | FAIL, unchanged (claims routing, pattern 3) |
| legal-action-deadline-e2e | pass | pass (after the fix) |
| the other older e2e cases | pass | pass |

## Caveats
- The market-value question is about the home, not another product, and the reply says it
  "isn't about your homeowners policy": acceptable, slightly off.
- The "no search, discard the text" guard did not fire live (the router caught the tax
  question first). It is proven only offline, by tests that fail when it is removed.
- The `out_of_scope` reply reports confidence "high" because it makes no claim about the policy.
- The held-out lay-wording cases were seen in three full runs now. The two regressions were
  found through them. The fix is generic, but they are no longer unseen: treat later H
  numbers as development evidence, and the next fresh held-out set as the real test.
- `h-ask-before-filing-a-claim` shows the rule cannot detect "the policy is silent about
  rates" when retrieval returns lots of loosely related text. That needs a different mechanism.
