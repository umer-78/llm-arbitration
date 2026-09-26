"""Arbitration: fan out to the critics in parallel, find where they disagree, and turn their
critiques into one verdict.

The adjudicator is statistical. From labelled history it learns, for each critic and task,
how often the critic passes answers that are right and answers that are wrong, and the prior
accuracy of the model that wrote the answer. Treating critics as independent given the truth
(naive Bayes), that gives P(answer is right | critiques). Its verdict upholds the issues
raised by critics on the losing side of the evidence and dismisses the others, saying why.
Critics that fail after their retries are left out, and the verdict lists them.

An `escalate` hook, such as an LLM adjudicator that re-reads the question, is called only
when critics disagree; it can revise the verdict.
"""
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Verdict:
    score: int                    # 1 to 10
    p_correct: float
    confidence: float
    confirmed: list = field(default_factory=list)
    dismissed: list = field(default_factory=list)
    disagreements: list = field(default_factory=list)
    degraded: list = field(default_factory=list)
    summary: str = ""


class Adjudicator:
    def __init__(self, smoothing=1.0):
        self.smoothing, self.prior, self.rates = smoothing, {}, {}

    def fit(self, rows):
        """rows: (task, answerer, answer was right 0/1, {critic: passed})."""
        a = self.smoothing
        by_prior, by_rate = {}, {}
        for task, answerer, right, passes in rows:
            by_prior.setdefault((answerer, task), []).append(right)
            for critic, ok in passes.items():
                by_rate.setdefault((critic, task), {0: [], 1: []})[right].append(ok)
        self.prior = {k: (sum(v) + a) / (len(v) + 2 * a) for k, v in by_prior.items()}
        self.rates = {k: tuple((sum(v[r]) + a) / (len(v[r]) + 2 * a) for r in (1, 0)) for k, v in by_rate.items()}
        return self

    def p_correct(self, task, answerer, passes):
        p = self.prior.get((answerer, task), 0.5)
        logit = np.log(p / (1 - p))
        for critic, ok in passes.items():
            if (critic, task) in self.rates:
                right, wrong = self.rates[(critic, task)]      # P(pass | right), P(pass | wrong)
                logit += np.log(right / wrong) if ok else np.log((1 - right) / (1 - wrong))
        return float(1 / (1 + np.exp(-logit)))


class Arbiter:
    def __init__(self, critics, adjudicator, retries=2, escalate=None, store=None):
        self.critics, self.adjudicator, self.retries, self.escalate = critics, adjudicator, retries, escalate
        self.store = Store(store) if store else None

    def _run(self, critic, item):
        for attempt in range(self.retries + 1):
            try:
                return critic(item)
            except Exception as e:           # a critic's failure must not sink the verdict
                error = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    time.sleep(0.1 * 2 ** attempt)
        return error

    @staticmethod
    def disagreements(critiques):
        out = []
        scores = [c.score for c in critiques]
        if scores and max(scores) - min(scores) > 2:
            out.append(f"scores range {min(scores)}-{max(scores)}")
        failed = [c.critic for c in critiques if not c.passed]
        if failed and len(failed) < len(critiques):
            out.append(f"only {', '.join(failed)} found problems")
        return out

    def run(self, item):
        """item: {"key", "task", "answerer", "question", "answer"}."""
        with ThreadPoolExecutor(max(1, len(self.critics))) as pool:
            results = list(pool.map(lambda c: (c, self._run(c, item)), self.critics))
        critiques = [r for _, r in results if not isinstance(r, str)]
        degraded = [f"{c.name}: {r}" for c, r in results if isinstance(r, str)]
        p = self.adjudicator.p_correct(item["task"], item["answerer"], {c.critic: c.passed for c in critiques})
        passing = [c.critic for c in critiques if c.passed]
        flagged = [(c.critic, i) for c in critiques if not c.passed for i in c.issues]
        if p >= 0.5:
            confirmed, dismissed = [], [{"critic": n, **asdict(i), "why": f"outweighed by {', '.join(passing) or 'the prior'}"}
                                        for n, i in flagged]
        else:
            confirmed, dismissed = [{"critic": n, **asdict(i)} for n, i in flagged], []
        verdict = Verdict(1 + round(9 * p), p, max(p, 1 - p), confirmed, dismissed, self.disagreements(critiques), degraded,
                          f"{'Likely right' if p >= 0.5 else 'Likely wrong'} (P(right) = {p:.2f}); "
                          f"{len(passing)} of {len(critiques)} critics passed it"
                          + (f"; {len(degraded)} critic(s) unavailable" if degraded else "") + ".")
        if self.escalate and verdict.disagreements:
            verdict = self.escalate(item, critiques, verdict)
        if self.store:
            self.store.add(item, critiques, verdict)
        return verdict


class Store:
    """Every arbitration, for audit: the item, each critique, the verdict."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS arbitrations (id INTEGER PRIMARY KEY, at REAL, item TEXT, critiques TEXT, verdict TEXT)")

    def add(self, item, critiques, verdict):
        self.db.execute("INSERT INTO arbitrations (at, item, critiques, verdict) VALUES (?, ?, ?, ?)",
                        (time.time(), json.dumps(item), json.dumps([asdict(c) for c in critiques]), json.dumps(asdict(verdict))))
        self.db.commit()
