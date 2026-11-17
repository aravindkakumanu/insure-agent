# Query wording and retrieval stability, 2026-11-17

Question: why does "someone slipped on my icy driveway and is threatening to sue me" retrieve the
liability section in some runs and not in others, and can the agents' queries be fixed?
Tool: `python -m tests.integration.stability <case ids> -n N [--show-queries]` (LIVE; asks the
same question N times through the router and counts how often the required wording was retrieved).

## What was tried, in order
| change | icy driveway | code upgrade | notes |
|---|---|---|---|
| before (committed state) | 3/6 | 3/6 | queries all in everyday words ("slip and fall", "wiring upgrades") |
| specialist prompts: "write a query in formal insurance wording" | 0/6 | 0/6 | WORSE: agents wrote "bodily injury slip and fall icy driveway" (still the event) |
| specialist prompts: "run a query naming only the KIND of protection, details stripped" | 6/6 then 11/12 | 6/6 then 12/12 | helped those two, but a 9-case comparison showed car and ground-washed-out fall 6/6 -> 0/6 with it, and the agents did not actually write stripped queries in those cases. REVERTED |
| router rewrite keeps what the user is worried about or asking for | 0/8 -> 6/8 | 6/8 | kept. Cause found: the rewrite turned "threatening to sue me" into "someone slipping on my icy driveway", so the agents searched slip-and-fall and never "being sued" |

## What this shows
1. **Probe (raw similarity rank of the required passage for the agents' real queries):** liability
   chunk 12-16 for "slip and fall" queries, building-codes chunk 6-11, land stability 4-8, transfer
   clause 4-6, with `search_policy` returning 4. Queries that name the kind of protection ("someone is
   suing me", "building codes coverage", "legally obligated to pay damages") put it in the top 4. The
   policy never mentions ice, wiring or decks, so incidental detail drags the query toward other text.
2. **At temperature 0 the agents' behaviour is deterministic per prompt, not random:** 3 repeats of
   the same question wrote identical queries. A prompt change therefore flips cases (car 6/6 -> 0/6)
   without being "noise" and without being a reliable improvement. Repeat-sampling (the stability
   tool) measures residual API nondeterminism and the effect of ONE prompt version; it cannot show
   that a prompt edit is a net win. Compare prompt versions over many cases, as done above.
3. **A prompt rule asking for a certain query style is not reliably followed.** The reliable fix was
   upstream (what the rewrite keeps), where the cause was visible.

## State at the end, and what is NOT fixed
- `h2-selling-the-house`: 0/8. The classifier labels it `out_of_scope` every time, although the
  prompt lists selling the house as in scope. It passed once earlier in the day, so the classifier
  (gpt-4o-mini) is not stable on it either. This is a regression in committed code.
- `out-of-scope-tax-deduction` and similar: the classifier sometimes fills `outside_topic` with the
  rates topic for a tax question (seen when calling the classifier directly), which would show the
  rates note. In the full runs the tax answer was the generic out-of-scope reply, but this is fragile.
- The full end-to-end regression after the rewrite change could NOT be completed: the OpenAI account
  ran out of credits again mid-run (`credit_balance_exhausted`), and that partial run was discarded.
- Fresh `h3-*` held-out cases were seen in several runs now; arson-reward only passes since the
  in-scope list gained "rewards, tips, payments to other people".

## Full end-to-end regression after the rewrite change (credits topped up; clean, 0 API errors)
File: `2026-11-17-e2e-full-after-rewrite-keeps-the-worry-raw.txt`. **29 of 31 pass.** The two failures:
`h-burglary-laptop-jewelry` (claims routing, known and skipped) and `h2-selling-the-house` (the
classifier sends it out of scope). Human read of what the mechanical check does not show:
- `h-icy-driveway-lawsuit`: GOOD this run (Family Liability Protection, defence, limit).
- `h2-code-upgrade-after-fire`: the check PASSES (the building-codes text was retrieved) but the
  answer still says the policy does not address wiring upgrades. A lenient check; the answer is wrong.
- `general-coverage-question-is-still-answered` ("What does my policy cover?"): REGRESSION in the
  answer. The coverage agent made no `search_policy` call, so the no-search guard replaced its
  section with "I couldn't find policy text that addresses this" and the low-confidence warning was
  shown, while the exclusions section answered normally. The mechanical check (no "The policy
  doesn't say") passes. So the guard works as designed, but an agent skipping its search on a
  broad question now produces a visibly wrong section; before the guard it would have answered
  from memory. Not seen in the earlier full runs: the agent's behaviour on this question varies.
