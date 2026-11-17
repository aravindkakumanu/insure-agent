# Retrieval baseline, category H (lay wording), 2026-11-17

Command: `RUN_LIVE_EVAL=1 pytest tests/integration/test_live_retrieval.py -v -s -k "h-"`
Code state: commit 59d06c1 plus the uncommitted eval expansion (see the 2026-11-13 baseline).
Run once, nothing tuned, no case edited after the run. All nine retrieval cases are
`heldout`: the questions are the project owner's own words; the tool and snippet were
chosen from the stored chunks before any result was seen (`search_exclusions` where the
right answer is a "no"). The tenth H case (`h-ask-before-filing-a-claim`) is a behaviour
case and waits for the end-to-end tier.

**5 of 9 pass, 4 fail.** (Dev categories A-F scored 16 of 17 on the same code: the gap
is the wording, not the tool.)

| case | tool | ranks | result | raw rank in top 30 (probe) |
|---|---|---|---|---|
| h-burst-pipes-on-vacation | search_policy | #1 | pass | |
| h-icy-driveway-lawsuit | search_policy | #3 | pass | |
| h-basement-flood-rainstorm | search_exclusions | #4 of 6 | pass | |
| h-leaking-roof-mold | search_exclusions | #3 of 6 | pass | |
| h-baseball-neighbors-window | search_policy | #2 | pass | |
| h-tree-on-fence | search_policy | missing | FAIL | 16 (d=1.435) |
| h-burglary-laptop-jewelry | search_policy | jewelry #1, actual cash value missing | FAIL | jewelry 1, ACV 6 (d=1.372) |
| h-candle-fire | search_policy | missing | FAIL | 24 (d=1.304) |
| h-vandalized-car-in-driveway | search_exclusions | missing | FAIL | not in top 30 |

## Reading the failures

- **vandalized car: a real vocabulary gap.** The policy says "motorized land vehicles";
  the question says "car". The wording is not in the top 30 at all, so raising the number
  of chunks would not help. Query rewriting is the candidate fix (the agents write their
  own queries, so the end-to-end tier may differ).
- **burglary: half a pass.** The jewelry limit is #1; the "actual cash value" payment
  rule sits at raw rank 6 and the tool returns 4. Two required snippets in one case is a
  deliberately demanding ground truth.
- **tree on fence, candle fire: the ground truth may be too narrow.** Each required one
  generic sentence ("fences are Coverage B"; the all-risk grant for Coverages A and B).
  Other chunks retrieved (Windstorm or Hail, Collapse, Fire Department Charges) are
  relevant to a human reader. Not edited, per the rule: the record of what failed is the
  point. Whether a looser snippet would be fairer is a question for the owner.
- **Caveat as before:** the raw question goes to one plain search; the agents write
  their own queries. These are retrieval-tier numbers, not answer quality.

## Addendum: two cases loosened at the owner's request (same day)

The owner decided that returning any genuinely relevant chunk should pass for
`h-tree-on-fence` and `h-candle-fire`; the other two failures stay as they are. Both
cases now use `any_of=True` with three alternative snippets (the original, plus the
fallen-tree debris clause / the A-and-B grant for the tree; the A-and-B grant / the
wall-to-wall carpet clause / additional living expense for the fire).

Re-run: **both now pass** (tree via the debris-removal chunk at #2; fire via the
carpet clause at #3 and living expense at #4). The original run above is what the
case first scored: 5 of 9. With the loosening the same code scores **7 of 9**.

Read the two passes with care: the alternatives were chosen AFTER seeing what was
retrieved, so they are weaker evidence than the other seven cases. They show that
relevant text is returned, not that the question was predicted blind.
