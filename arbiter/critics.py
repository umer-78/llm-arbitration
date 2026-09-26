"""Critics check an answer independently and return a structured critique.

RecordedCritic is a verification critic: a model that solved the same question itself and
flags the answer under review when its own final answer differs. It is backed by that
model's recorded HELM answers, which is what the bench measures.

LLMCritic asks any OpenAI-compatible chat endpoint for a JSON critique along one dimension
(accuracy, logic or completeness). It is pluggable and not measured here: this repository
runs without API keys.
"""
import json
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Issue:
    quote: str        # the part of the answer the issue is about
    problem: str
    severity: int     # 1 (minor) to 5 (makes the answer wrong)


@dataclass
class Critique:
    critic: str
    dimension: str    # verification, accuracy, logic or completeness
    score: int        # 1 (fails) to 5 (passes)
    issues: list = field(default_factory=list)
    confidence: float = 0.5   # the critic's confidence in its own assessment

    @property
    def passed(self):
        return self.score >= 4


class RecordedCritic:
    dimension = "verification"

    def __init__(self, name, answers):
        self.name, self.answers = name, answers        # {question key: this model's final answer}

    def __call__(self, item):
        mine = self.answers.get(item["key"])
        if mine is None:
            raise LookupError(f"{self.name} has no recorded answer for {item['key']}")
        if mine == item["answer"]:
            return Critique(self.name, self.dimension, 5)
        return Critique(self.name, self.dimension, 1,
                        [Issue(item["answer"], f"solving it independently gives {mine!r}", 5)])


class LLMCritic:
    PROMPT = ("You are the {dimension} critic. Judge only {dimension}. Question:\n{question}\n\nAnswer under review:\n"
              "{answer}\n\nReply with JSON: {{\"score\": 1-5, \"confidence\": 0-1, \"issues\": [{{\"quote\": exact words "
              "from the answer, \"problem\": what is wrong, \"severity\": 1-5}}]}}")

    def __init__(self, name, dimension, model, base_url, api_key, timeout=60):
        self.name, self.dimension, self.model = name, dimension, model
        self.url, self.key, self.timeout = base_url.rstrip("/") + "/chat/completions", api_key, timeout

    def __call__(self, item):
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": self.PROMPT.format(dimension=self.dimension, **item)}]}
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
        issues = [Issue(str(i.get("quote", "")), str(i.get("problem", "")), min(5, max(1, int(i.get("severity", 3)))))
                  for i in out.get("issues", [])]
        return Critique(self.name, self.dimension, min(5, max(1, int(out["score"]))), issues, float(out.get("confidence", 0.5)))
