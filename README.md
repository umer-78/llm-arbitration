# llm-arbitration

**Live demo:** https://umer-78.github.io/llm-arbitration/ (set the review threshold and see what the critics' verdicts are worth)

A second opinion on LLM answers. A panel of critics checks each answer in parallel, disagreements between them are flagged, and an adjudicator turns their critiques into one verdict. The verdict carries a calibrated probability that the answer is right, the issues it upholds, and the ones it dismissed and why. Every arbitration is written to an SQLite audit log.

It was measured on real recorded answers: six models on HELM Lite's 4,551 questions (GSM8K, MATH, MMLU, MedQA, OpenBookQA, LegalBench).

## Results

`python -m arbiter bench` puts each model's answers under review. The panel is the three other models with the best overall accuracy. Each critic solves the question itself and flags the answer when its own final answer differs. The adjudicator is cross-fitted: trained on four fifths of the questions, it scores the fifth it never saw.

| Answers from | Accuracy | AUROC | Flagged | Arbiter precision / recall | Majority vote | Best single critic |
|---|---|---|---|---|---|---|
| deepseek-v3 | 78.7% | 0.792 | 12% | 56% / 32% | 43% / 31% | 52% / 34% |
| gemini-1.5-flash | 62.6% | 0.890 | 31% | 80% / 67% | 80% / 66% | 80% / 66% |
| gpt-4o | 77.5% | 0.810 | 12% | 60% / 31% | 51% / 34% | 60% / 38% |
| gpt-4o-mini | 70.9% | 0.866 | 22% | 76% / 56% | 80% / 44% | 76% / 48% |
| llama-3.1-70b | 74.0% | 0.849 | 22% | 67% / 57% | 68% / 43% | 70% / 46% |
| llama-3.1-8b | 49.2% | 0.958 | 46% | 93% / 85% | 93% / 84% | 93% / 85% |

"Precision" is the share of flagged answers that were really wrong. "Recall" is the share of wrong answers that got flagged.

- **The verdict is a usable probability.** Calibration error (ECE) is 0.04–0.08. When the adjudicator says 80%, about 80% of those answers are right, so a threshold can be chosen for the cost of a miss.
- **Weighing critics beats counting them.** On Llama 3.1 70B's answers, the arbiter catches 57% of wrong answers against majority vote's 43%, at about the same precision. On GPT-4o-mini's, 56% against 44%.
- **Disagreement is the signal.** When the panel splits, the answer under review is right only 43–64% of the time. When all three critics pass it, 86–89%.
- **Shared blind spots are the limit.** About 14% of answers all three critics pass are still wrong: the critics make the same mistake. Critics from different model families help. No panel removes this.

Full tables are in `results/bench.md` and `results/summary.json`.

## How it works

- **Critics.** A critic returns a structured critique: dimension, a 1–5 score, issues (each a quote from the answer, the problem and a severity), and its confidence.
  - `RecordedCritic` is a verification critic backed by a model's recorded answers. It is what the bench measures.
  - `LLMCritic` asks any OpenAI-compatible endpoint for a JSON critique on accuracy, logic or completeness. It is pluggable and not measured here, since the repository runs without API keys.
- **Dispatch.** All critics run in parallel, each with retries. A critic that still fails is left out, and the verdict names it under `degraded`.
- **Disagreement detection.** A disagreement is scores more than 2 apart, or problems found by only some of the critics.
- **Adjudication.** From labelled history, the adjudicator learns how often each critic passes right answers and wrong answers on each task, plus the answering model's prior accuracy. Treating critics as independent given the truth (naive Bayes), it combines them into P(right).
  - The independence is an approximation: the critics are models that fail on the same questions. Calibrated overall (ECE 0.044–0.083), it is overconfident when all three object: for DeepSeek-V3's answers it gives those 20% on average and 48% were right. The live demo shows the gap for every pattern of verdicts.
  - Issues from critics on the losing side are dismissed, with the reason.
  - An `escalate` hook, such as an LLM adjudicator, runs only when critics disagree.

## Use

```python
from arbiter.critics import LLMCritic
from arbiter.pipeline import Adjudicator, Arbiter

critics = [LLMCritic("accuracy", "accuracy", "gpt-4o", "https://api.openai.com/v1", key), ...]
arbiter = Arbiter(critics, Adjudicator().fit(labelled_history), retries=2, store="audit.db")
verdict = arbiter.run({"key": "t-1", "task": "support", "answerer": "gpt-4o-mini", "question": q, "answer": a})
verdict.p_correct, verdict.confirmed, verdict.dismissed, verdict.degraded
```

`labelled_history` holds rows of `(task, answerer, right 0/1, {critic: passed})` from answers you have graded.

```bash
pip install -e '.[dev]'
python -m arbiter bench                            # about 30 s once the data is cached
python -m arbiter explain gsm8k//id8660            # one arbitration, with every critique
python -m arbiter.demo                              # rebuild the live demo's data in docs/
```

HELM Lite's public results are downloaded on first use into `~/.cache/arbiter`; nothing is committed.
