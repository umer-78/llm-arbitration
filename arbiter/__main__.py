"""python -m arbiter bench                          the panel on every model's answers, written to results/
python -m arbiter explain KEY [--answerer NAME]   one arbitration, critique by critique"""
import argparse
import json
import sys
from dataclasses import asdict

from . import bench, data
from .critics import RecordedCritic
from .pipeline import Adjudicator, Arbiter


def main(argv=None):
    ap = argparse.ArgumentParser(prog="arbiter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bench")
    e = sub.add_parser("explain")
    e.add_argument("key", help="a question key, e.g. gsm8k//id1234 (see results/summary.json for models)")
    e.add_argument("--answerer", default="gpt-4o-mini")
    args = ap.parse_args(argv)
    if args.cmd == "bench":
        print(bench.bench())
        return 0
    table = data.table()
    others = [k for k in table if k != args.key]          # the adjudicator never sees the question it judges
    panel = bench.panel_for(table, args.answerer)
    critics = [RecordedCritic(c, {k: q["models"][c]["answer"] for k, q in table.items()}) for c in panel]
    history = [(it["task"], args.answerer, table[it["key"]]["models"][args.answerer]["correct"], {c.name: c(it).passed for c in critics})
               for it in bench.items(table, others, args.answerer)]
    item = bench.items(table, [args.key], args.answerer)[0]
    verdict = Arbiter(critics, Adjudicator().fit(history)).run(item)
    print(json.dumps({"question": item["question"][:300], "answer": item["answer"],
                      "actually_right": bool(table[args.key]["models"][args.answerer]["correct"]), "verdict": asdict(verdict)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
