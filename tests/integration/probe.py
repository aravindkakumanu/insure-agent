"""
probe.py — How deep is each required snippet in the raw similarity ranking?
===========================================================================
LIVE (embeddings only). The golden test says PASS or FAIL against the 4 chunks
a tool returns; when it FAILS you need to know HOW FAR the required wording is
from being returned: rank 5 means "return more chunks", rank 40 means "the
question and the policy use different words".

    python -m tests.integration.probe                      # every golden case
    python -m tests.integration.probe legal-action-deadline theft-police-report
    python -m tests.integration.probe -k 50 appraisal

Ranks come from RAW similarity search over the whole collection (closest first),
not from a tool's own selection, so for search_exclusions cases they show how
close the wording is, not what that tool would return. "exclusions-region rank" is
the rank among ONLY chunks inside exclusions regions - the number the exclusions
quota (EXCLUSION_SECTION_QUOTA) has to reach. Needs OPENAI_API_KEY (.env)
and an ingested policy.
"""

import argparse
import os
import sys
from typing import Optional, Sequence

from dotenv import load_dotenv

from src.agents.results import Evidence
from tests.integration.golden import GOLDEN, evaluate, section_ranks


def ranked_evidence(question: str, k: int) -> list[Evidence]:
    """The k closest chunks to the question, closest first."""
    from src.agents import tools

    hits = tools.get_vector_store().similarity_search_with_score(question, k=k)
    return [tools._to_evidence(doc, distance) for doc, distance in hits]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("case_ids", nargs="*", help="golden case ids (default: all)")
    parser.add_argument("-k", type=int, default=30, help="how deep to search (default 30)")
    args = parser.parse_args(argv)

    known = {c.id for c in GOLDEN}
    unknown = [i for i in args.case_ids if i not in known]
    if unknown:  # checked first: needs no key and touches no environment
        print(f"unknown case ids: {unknown}\nknown: {sorted(known)}", file=sys.stderr)
        return 2

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set (expected in .env)", file=sys.stderr)
        return 2

    for case in GOLDEN:
        if args.case_ids and case.id not in args.case_ids:
            continue
        evidence = ranked_evidence(case.search_text, args.k)
        print(f"\n{case.id}  (query {case.search_text!r})  best distance {evidence[0]['distance']:.3f}")
        in_region = section_ranks(case, evidence)
        for snippet, rank in evaluate(case, evidence).ranks.items():
            where = (
                f"rank {rank}/{args.k}  d={evidence[rank - 1]['distance']:.3f}"
                if rank
                else f"NOT in the top {args.k}"
            )
            region = in_region[snippet]
            print(
                f"   {snippet[:52]!r:<56} {where:<26}"
                f" exclusions-region rank: {region if region else '-'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
