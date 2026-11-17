# Retrieval baseline, 2026-11-13

Retrieval tier only (`RUN_LIVE_EVAL=1 pytest tests/integration/test_live_retrieval.py -v -s`): the raw question goes to one plain search, embeddings only, no LLM. Run once, before any new case was used for tuning. Code state: commit `59d06c1` plus the uncommitted eval expansion (CLAUDE.md lesson 33).

**All 32 cases:** 29 pass, 2 xfail (the two accepted gaps), 1 fail.  
**The 17 new cases (categories A-F):** 16 pass, 1 fail.  
**Held-out (5 cases):** 5 of 5 pass.  
**Not scored here:** categories G and K (behaviour cases, end-to-end tier), H (to be written by the owner), I and J (not written yet).

Ranks are the position of each required snippet among the chunks returned (`search_policy` returns 4, `search_exclusions` 6); a case with several snippets lists one rank each.

| case | cat | split | tool | ranks | result |
|---|---|---|---|---|---|
| water-exclusions-flood | B | dev | search_exclusions | #6, #1 | PASS |
| water-damage-short-query | B | dev | search_exclusions | #6, #5 | PASS |
| mold-remediation-limit | B | dev | search_policy | #1 | PASS |
| mold-exclusion | B | dev | search_exclusions | #1 | PASS |
| burst-pipe-coverage | A | dev | search_policy | #1 | PASS |
| burst-pipe-exclusions | B | dev | search_exclusions | #3, #2 | PASS |
| claim-first-steps | C | dev | search_policy | #1, #1 | PASS |
| proof-of-loss | C | dev | search_policy | #2 | PASS |
| legal-action-deadline | C | dev | search_policy | MISSING | XFAIL (accepted gap) |
| appraisal | C | dev | search_policy | #3 | PASS |
| settlement-timing | C | dev | search_policy | #1 | PASS |
| theft-police-report | C | dev | search_policy | MISSING | XFAIL (accepted gap) |
| deductible | F | dev | search_policy | #1 | PASS |
| silverware-theft-limit | B | dev | search_policy | #1 | PASS |
| collapse | B | dev | search_exclusions | #1 | PASS |
| additional-living-expense | A | dev | search_policy | #1 | PASS |
| fire-department-charges | A | dev | search_policy | #1 | PASS |
| temporary-repairs | A | heldout | search_policy | #1 | PASS |
| belongings-away-from-home | A | dev | search_policy | #3, #3 | PASS |
| hoa-loss-assessment | A | dev | search_policy | #3 | PASS |
| earth-movement | B | heldout | search_exclusions | #1 | PASS |
| jewelry-theft-limit | B | dev | search_policy | #2 | PASS |
| other-insurance | C | dev | search_policy | #2 | PASS |
| insurer-cancellation | C | heldout | search_policy | #1 | PASS |
| define-residence-premises | D | dev | search_policy | #1 | PASS |
| define-insured-person | D | dev | search_policy | #1 | PASS |
| define-business-boarders | D | heldout | search_policy | #1 | PASS |
| liability-what-it-pays | E | dev | search_policy | #4 | PASS |
| guest-medical-payments | E | dev | search_policy | #1 | PASS |
| liability-business-exclusion | E | heldout | search_exclusions | #2 | PASS |
| dwelling-limit-amount | F | dev | search_policy | MISSING | FAIL |
| premium-amount | F | dev | search_policy | #1 | PASS |

## The one failure: `dwelling-limit-amount` (category F, a new dev case)
"What is the dollar limit on my dwelling coverage?" retrieved Collapse, Temporary Repairs, Increased Coverage On Money and
Breakage Of Glass (distances 0.97-1.08, all weak). The required sentence ("the Policy Declarations shows the location of the
residence premises, applicable coverages, limits of liability and premiums") sits in chunk 6, a 5,935-character chunk that also
holds most of the definitions and the insuring agreement; the probe puts it at raw rank 13/30 (d=1.145).
Two things to keep in mind before reading this as a retrieval bug: the case demands that ONE specific sentence be retrieved, while
several other chunks also say limits are "shown on the Policy Declarations", so the ground truth may be too narrow; and the
retrieval tier cannot say whether the agent still answers correctly, which only the end-to-end tier can. The case was NOT
edited after the run.

## Margins worth knowing
- `liability-what-it-pays` passes at #4 of 4 (zero slack). `belongings-away-from-home` and `hoa-loss-assessment` pass at #3.
- Ten of the 17 new cases are #1; thirteen are in the top two.
- Passing says nothing about chunk size: `liability-business-exclusion` passes at #2, but its chunk is about 11,000 characters.
