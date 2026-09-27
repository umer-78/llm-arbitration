"""How well a panel of critics catches wrong answers, on recorded answers.

Every model's answers to HELM Lite's 4,551 questions are put under review in turn. The panel
is the three other models with the best overall accuracy, each a verification critic. The
adjudicator is cross-fitted: it is trained on four fifths of the questions and scores the
fifth it never saw.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from . import data
from .critics import RecordedCritic
from .pipeline import Adjudicator, Arbiter

RESULTS = Path(__file__).resolve().parent.parent / "results"


def auroc(score, positive):
    """Chance a random positive scores above a random negative (Mann-Whitney)."""
    positive = np.asarray(positive, bool)
    ranks = rankdata(score)
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def flags(flagged, wrong):
    return {"flagged": float(flagged.mean()), "precision": float(wrong[flagged].mean()) if flagged.any() else None,
            "recall": float(flagged[wrong].mean())}


def panel_for(table, answerer, size=3):
    names = sorted(next(iter(table.values()))["models"])
    acc = {n: np.mean([q["models"][n]["correct"] for q in table.values()]) for n in names}
    return sorted((n for n in names if n != answerer), key=lambda n: -acc[n])[:size]


def items(table, keys, answerer):
    return [{"key": k, "task": table[k]["task"], "answerer": answerer, "question": table[k]["question"],
             "answer": table[k]["models"][answerer]["answer"]} for k in keys]


def run(folds=5, seed=0, detail=False):
    """One row per answering model; with `detail`, also each question's out-of-fold P(right), label and critic verdicts."""
    table = data.table()
    keys = sorted(table)
    fold = np.random.default_rng(seed).permutation(len(keys)) % folds
    names = sorted(next(iter(table.values()))["models"])
    rows, out = [], []
    for answerer in names:
        panel = panel_for(table, answerer)
        critics = [RecordedCritic(c, {k: table[k]["models"][c]["answer"] for k in keys}) for c in panel]
        its = items(table, keys, answerer)
        right = np.array([table[k]["models"][answerer]["correct"] for k in keys])
        history = [(it["task"], answerer, r, {c.name: c(it).passed for c in critics}) for it, r in zip(its, right)]
        p = np.zeros(len(keys))
        for f in range(folds):
            arbiter = Arbiter(critics, Adjudicator().fit([h for h, g in zip(history, fold) if g != f]))
            for i in np.flatnonzero(fold == f):
                p[i] = arbiter.run(its[i]).p_correct
        wrong = right == 0
        disagree = np.array([[not h[3][c] for c in panel] for h in history])
        split = disagree.any(1) & ~disagree.all(1)
        row = {"answerer": answerer, "panel": panel, "accuracy": float(right.mean()), "wrong": int(wrong.sum()),
               "auroc": auroc(-p, wrong), "ece": ece(p, right), "arbiter": flags(p < 0.5, wrong),
               "majority": flags(disagree.sum(1) >= 2, wrong), "best_critic": flags(disagree[:, 0], wrong),
               "split_panel": float(split.mean()), "accuracy_when_split": float(right[split].mean()),
               "accuracy_when_all_pass": float(right[~disagree.any(1)].mean())}
        if detail:
            row["detail"] = {"p": p.tolist(), "right": right.tolist(), "disagree": disagree.tolist()}
        out.append(row)
    return out


def report(rows):
    pct = lambda x: "—" if x is None else f"{100 * x:.0f}%"
    pr = lambda f: f"{pct(f['precision'])} / {pct(f['recall'])}"
    lines = ["| Answers from | Accuracy | Panel | AUROC | Arbiter flags | Arbiter precision / recall | Majority vote | Best single critic |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['answerer']} | {100 * r['accuracy']:.1f}% | {', '.join(r['panel'])} | {r['auroc']:.3f} | "
                     f"{pct(r['arbiter']['flagged'])} | {pr(r['arbiter'])} | {pr(r['majority'])} | {pr(r['best_critic'])} |")
    lines += ["", "| Answers from | Panel split | Accuracy when split | Accuracy when all critics pass | Calibration error (ECE) |",
              "|---|---|---|---|---|"]
    lines += [f"| {r['answerer']} | {pct(r['split_panel'])} | {pct(r['accuracy_when_split'])} | {pct(r['accuracy_when_all_pass'])} | "
              f"{r['ece']:.3f} |" for r in rows]
    return "\n".join(lines)


def bench():
    rows = run()
    RESULTS.mkdir(exist_ok=True)
    text = report(rows)
    (RESULTS / "bench.md").write_text(text + "\n")
    (RESULTS / "summary.json").write_text(json.dumps(rows, indent=1))
    return text
