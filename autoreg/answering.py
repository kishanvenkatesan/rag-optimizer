from typing import List
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@dataclass
class RAGSampleInput:
    query: str
    retrieved_docs: List[str]

class AnswerGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        max_new_tokens: int = 256,
        device: int = -1,  # -1 = CPU, >=0 = CUDA device index
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        if device >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # model max length 
        self.max_input_len = getattr(self.tokenizer, "model_max_length", 16384)

    def build_prompt(self, sample: RAGSampleInput) -> str:
        ctx = []
        for i, doc in enumerate(sample.retrieved_docs):
            ctx.append(f"[DOC {i+1}]\n{doc.strip()}")
        context_text = "\n\n".join(ctx)

        return (
            "You are a precise QA system.\n"
            "Answer using ONLY the provided context.\n"
            "Answer in ONE short sentence (max 20 words).\n"
            "Do NOT add extra background. Do NOT quote the context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {sample.query}\n"
            "Answer:"
        )


    def generate_answer(self, sample: RAGSampleInput) -> str:
        prompt = self.build_prompt(sample)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_len - self.max_new_tokens,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if len(text.split()) < 3:
            return " ".join(sample.retrieved_docs[:2])[:2000]

        return text
