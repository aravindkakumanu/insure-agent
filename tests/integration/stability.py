"""
stability.py — How often does a question get retrieved, run after run?
=====================================================================
The end-to-end tier is one sample of a non-deterministic system (the agents write their
own queries), so a single pass or fail is weak evidence: the icy-driveway question passed
in some full runs and failed in others. This asks the SAME question N times through the
router and counts how often the required wording was retrieved by whatever queries the
agents wrote. LIVE; costs N times the usual end-to-end cost per case.

    python -m tests.integration.stability h-icy-driveway-lawsuit h2-code-upgrade-after-fire -n 6
"""

import argparse
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from tests.integration.golden import GOLDEN, evaluate


def _once(case) -> tuple[bool, list[str]]:
    from src.agents.router import route_and_answer

    result = route_and_answer(case.question)
    evidence = [e for r in result.results for e in r.evidence]
    queries = [q for r in result.results for q in r.queries]
    return evaluate(case, evidence).passed, queries


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("case_ids", nargs="+")
    parser.add_argument("-n", type=int, default=6)
    parser.add_argument("--show-queries", action="store_true")
    args = parser.parse_args()
    by_id = {c.id: c for c in GOLDEN}
    for case_id in args.case_ids:
        case = by_id[case_id]
        with ThreadPoolExecutor(max_workers=3) as pool:
            runs = list(pool.map(lambda _: _once(case), range(args.n)))
        print(f"{case_id}: retrieved {sum(ok for ok, _ in runs)}/{args.n}")
        if args.show_queries:
            for ok, queries in runs:
                print(f"  {'ok  ' if ok else 'MISS'} {queries}")


if __name__ == "__main__":
    main()
