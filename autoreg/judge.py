from typing import List
from dataclasses import dataclass
from transformers import pipeline

@dataclass
class JudgeInput:
    query: str
    context: str   # concatenated retrieved chunks
    predicted: str
    gold: str

class LLMJudge:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        # text2text pipeline works for scoring prompts
        self.scorer = pipeline("text2text-generation", model=self.model_name, device=-1)

    def build_prompt(self, inp: JudgeInput) -> str:
        # instruct the model to score from 0 to 1 with short justification
        prompt = (
            "You are an evaluator. Given the context, a predicted answer, and a gold answer,\n"
            "rate the predicted answer's correctness and faithfulness to the context on a scale\n"
            "from 0.0 (completely incorrect or hallucinated) to 1.0 (fully correct and supported).\n"
            "Return only a JSON with fields: score (0.0-1.0) and short_reason.\n\n"
            f"Context:\n{inp.context}\n\n"
            f"Question: {inp.query}\n\n"
            f"Predicted Answer: {inp.predicted}\n\n"
            f"Gold Answer: {inp.gold}\n\n"
            "JSON:"
        )
        return prompt

    def score(self, inp: JudgeInput) -> float:
        prompt = self.build_prompt(inp)
        out = self.scorer(prompt, max_length=128, do_sample=False)
        text = out[0]["generated_text"].strip()

        # try to extract a float from model output robustly
        import re
        m = re.search(r"([0-1](?:\.\d+)?)", text)
        if m:
            try:
                val = float(m.group(1))
                if val < 0: val = 0.0
                if val > 1: val = 1.0
                return val
            except:
                pass

        # fallback: if model didn't give a number, use a heuristic (0.0-1.0)
        # very simple: 0.0 if 'not' in text, else 0.5
        if "not" in text.lower():
            return 0.0
        return 0.5
