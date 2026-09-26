| Answers from | Accuracy | Panel | AUROC | Arbiter flags | Arbiter precision / recall | Majority vote | Best single critic |
|---|---|---|---|---|---|---|---|
| deepseek-v3 | 78.7% | gpt-4o, llama-3.1-70b, gpt-4o-mini | 0.792 | 12% | 56% / 32% | 43% / 31% | 52% / 34% |
| gemini-1.5-flash | 62.6% | deepseek-v3, gpt-4o, llama-3.1-70b | 0.890 | 31% | 80% / 67% | 80% / 66% | 80% / 66% |
| gpt-4o | 77.5% | deepseek-v3, llama-3.1-70b, gpt-4o-mini | 0.810 | 12% | 60% / 31% | 51% / 34% | 60% / 38% |
| gpt-4o-mini | 70.9% | deepseek-v3, gpt-4o, llama-3.1-70b | 0.866 | 22% | 76% / 56% | 80% / 44% | 76% / 48% |
| llama-3.1-70b | 74.0% | deepseek-v3, gpt-4o, gpt-4o-mini | 0.849 | 22% | 67% / 57% | 68% / 43% | 70% / 46% |
| llama-3.1-8b | 49.2% | deepseek-v3, gpt-4o, llama-3.1-70b | 0.958 | 46% | 93% / 85% | 93% / 84% | 93% / 85% |

| Answers from | Panel split | Accuracy when split | Accuracy when all critics pass | Calibration error (ECE) |
|---|---|---|---|---|
| deepseek-v3 | 23% | 64% | 86% | 0.054 |
| gemini-1.5-flash | 19% | 56% | 86% | 0.055 |
| gpt-4o | 23% | 62% | 86% | 0.044 |
| gpt-4o-mini | 21% | 43% | 86% | 0.050 |
| llama-3.1-70b | 20% | 54% | 86% | 0.060 |
| llama-3.1-8b | 12% | 47% | 89% | 0.083 |
