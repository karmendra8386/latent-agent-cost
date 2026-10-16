"""Model runners for the harness.

``HFRunner`` runs a local Hugging Face causal LM (transformers) --
the Week-1 explicit-CoT baseline runner. ``MockRunner`` returns
canned chain-of-thought text so the full pipeline (logging, costing,
tables) can be validated on machines where even a tiny live model
is infeasible; mock runs are pipeline validation, never results.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from .netenv import sanitize_proxy_env

sanitize_proxy_env()


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class MockRunner:
    """Deterministic canned-CoT runner (pipeline validation only).

    The canned trace solves tasks whose id hashes to a multiple of
    three *only if* the canned final number happens to match the
    gold -- it does not peek at gold answers. It exists so every
    harness path (verifier included) executes end to end.
    """

    name = "mock-cot-0"

    def n_params(self) -> int:
        return 0  # mock: FLOPs estimator reports 0 by construction

    def generate(self, messages: list[dict], *, seed: int,
                 max_new_tokens: int = 256) -> Generation:
        t0 = time.perf_counter()
        h = int(hashlib.sha256(
            (messages[-1]["content"] + str(seed)).encode()).hexdigest(), 16)
        n_steps = 2 + h % 3
        lines = [f"Step {i + 1}: work through part {i + 1} of the problem."
                 for i in range(n_steps)]
        canned_answer = [72, 18, 5][h % 3]
        lines.append(f"#### {canned_answer}")
        text = "\n".join(lines)
        prompt_tokens = sum(len(m["content"].split()) for m in messages) + 8
        completion_tokens = len(text.split())
        return Generation(text=text,
                          prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens,
                          latency_s=time.perf_counter() - t0)


class HFRunner:
    """A local transformers causal LM, sampled per-seed."""

    def __init__(self, model_id: str, *, dtype: str = "bfloat16",
                 device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = model_id
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self._n_params = sum(p.numel() for p in self.model.parameters())

    def n_params(self) -> int:
        return self._n_params

    def generate(self, messages: list[dict], *, seed: int,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.95) -> Generation:
        torch = self._torch
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        gen = torch.Generator(device=self.device).manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                generator=gen,
                pad_token_id=self.tokenizer.eos_token_id)
        latency = time.perf_counter() - t0
        new_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return Generation(text=text,
                          prompt_tokens=int(prompt_tokens),
                          completion_tokens=int(new_ids.shape[0]),
                          latency_s=latency)
