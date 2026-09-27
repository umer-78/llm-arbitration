"""python -m arbiter.demo   write the live demo's data (docs/data.json): the bench's rows, plus for each
answering model how its out-of-fold P(right) splits right and wrong answers (40 bins, for the
threshold slider) and how often the answer was right under each pattern of critic verdicts."""
import json
from pathlib import Path

import numpy as np

from .bench import run

ROOT = Path(__file__).resolve().parent.parent
BINS = 40


def build(out=ROOT / "docs"):
    rows = run(detail=True)
    for r in rows:
        d = r.pop("detail")
        p, right, dis = np.array(d["p"]), np.array(d["right"]), np.array(d["disagree"], bool)
        b = np.minimum((p * BINS).astype(int), BINS - 1)
        r["bins"] = {"right": np.bincount(b[right == 1], minlength=BINS).tolist(), "wrong": np.bincount(b[right == 0], minlength=BINS).tolist()}
        patterns = {}
        for row, y, q in zip(dis, right, p):
            key = "".join("x" if x else "v" for x in row)          # per critic, in panel order: v passed, x objected
            c = patterns.setdefault(key, {"n": 0, "right": 0, "p": 0.0})
            c["n"] += 1
            c["right"] += int(y)
            c["p"] += float(q)
        r["patterns"] = [{"verdicts": k, "n": c["n"], "right": c["right"], "p": round(c["p"] / c["n"], 4)} for k, c in sorted(patterns.items())]
    out.mkdir(exist_ok=True)
    (out / "data.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out / 'data.json'}: {len(rows)} answering models")


if __name__ == "__main__":
    build()
