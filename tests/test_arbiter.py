import sqlite3

import numpy as np

from arbiter.bench import auroc
from arbiter.critics import Critique, RecordedCritic
from arbiter.pipeline import Adjudicator, Arbiter


def item(answer="42", key="q1"):
    return {"key": key, "task": "gsm8k", "answerer": "small", "question": "6 x 7?", "answer": answer}


def history(n=400, seed=0):
    """A strong critic (passes 95% of right answers, 10% of wrong) and a weak one (60% / 40%)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        right = int(rng.random() < 0.6)
        rows.append(("gsm8k", "small", right, {"strong": rng.random() < (0.95 if right else 0.10),
                                               "weak": rng.random() < (0.60 if right else 0.40)}))
    return rows


def test_adjudicator_trusts_the_reliable_critic():
    adj = Adjudicator().fit(history())
    assert adj.p_correct("gsm8k", "small", {"strong": True, "weak": False}) > 0.8
    assert adj.p_correct("gsm8k", "small", {"strong": False, "weak": True}) < 0.2
    assert abs(adj.p_correct("unseen-task", "small", {"strong": True}) - 0.5) < 1e-9


def test_recorded_critic_flags_a_different_answer():
    critic = RecordedCritic("m", {"q1": "42"})
    assert critic(item("42")).passed
    c = critic(item("41"))
    assert not c.passed and "42" in c.issues[0].problem


def test_verdict_confirms_or_dismisses_issues_and_survives_a_failing_critic(tmp_path):
    def broken(_):
        raise TimeoutError("no reply")
    broken.name = "broken"
    arbiter = Arbiter([RecordedCritic("strong", {"q1": "42"}), RecordedCritic("weak", {"q1": "41"}), broken],
                      Adjudicator().fit(history()), retries=1, store=tmp_path / "audit.db")
    v = arbiter.run(item("42"))
    assert v.p_correct > 0.5 and v.dismissed and not v.confirmed
    assert v.degraded and "broken" in v.degraded[0] and v.disagreements
    v = arbiter.run(item("41"))
    assert v.p_correct < 0.5 and v.confirmed[0]["critic"] == "strong"
    assert sqlite3.connect(tmp_path / "audit.db").execute("SELECT COUNT(*) FROM arbitrations").fetchone()[0] == 2


def test_disagreement_rules():
    agree = [Critique("a", "logic", 5), Critique("b", "accuracy", 4)]
    assert Arbiter.disagreements(agree) == []
    split = [Critique("a", "logic", 5), Critique("b", "accuracy", 1)]
    assert len(Arbiter.disagreements(split)) == 2


def test_auroc():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.5, 0.5], [0, 1]) == 0.5
