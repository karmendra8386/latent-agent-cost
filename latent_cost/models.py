"""Model runners for the harness.

``HFRunner`` runs a local Hugging Face causal LM (transformers) --
the Week-1 explicit-CoT baseline runner, extended in Week 2 with
the latent / ablation / control generation modes below.
``MockRunner`` returns canned output so the full pipeline (logging,
costing, tables) can be validated on machines where even a tiny live
model is infeasible; mock runs are pipeline validation, never
results.

Week-2 generation modes (selected per condition by the harness)
---------------------------------------------------------------
``explicit``
    Standard sampled chain-of-thought via ``model.generate`` --
    the Week-1 protocol, unchanged.

``latent`` / ``latent_zero`` / ``latent_perturb``
    Training-free latent reasoning in the Coconut style (Hao et
    al., 2024): after the prompt prefill, the model's last hidden
    state is fed back as the next input embedding for ``K``
    latent steps -- no token is sampled, so the reasoning leaves
    no text trace -- and the runner then switches to explicit
    decoding to produce the answer segment. Variants:

    - ``latent``: the hidden state is fed back unchanged
      (identity transform).
    - ``latent_zero``: the fed-back state is replaced with zeros.
      The steps still run (same schedule, same forward passes),
      so this is the compute-matched zero-latent ablation: any
      difference from ``latent`` is information the states
      carried, not compute. (The outline's phrase "zero latent
      steps" is the other reading; ``latent`` with
      ``latent_steps=0`` covers it and degenerates to decoding
      straight from the prompt.)
    - ``latent_perturb``: Gaussian noise of scale
      ``perturb_scale`` (in hidden-state units) is added to the
      fed-back state at every step -- the perturbation ablation
      the 2026 critiques demand.

    Note on naming: the mechanism implemented here is Coconut's
    last-hidden-state feedback. *Soft Thinking* (NeurIPS 2025),
    the training-free reference, feeds a probability-weighted
    mixture of token embeddings instead; the two are documented
    as distinct in the paper outline, and this runner implements
    the feedback rule, used training-free.

``pause``
    Pause-token control: ``K`` copies of a fixed filler token
    are appended to the prompt and processed before ordinary
    explicit decoding -- the same number of extra forward passes
    as the latent condition's budget, but spent on visible
    filler tokens instead of latent states. If this control
    matches the latent condition, the latent effect is not in
    the vectors. The filler is an existing vocabulary token
    (see :meth:`HFRunner._pause_token_id`), not an added
    special token: no embedding is trained or resized for it.

GPT-2 control conditions run through this same runner with
``mode="explicit"`` / ``mode="latent"`` on ``gpt2``: GPT-2
predates chat templates, so its prompt is the same content
(system + user) rendered as plain text (see
:meth:`HFRunner._prompt_text`). That rendering difference, and
the absence of any Coconut curriculum training, are the control's
documented deviations: base GPT-2 was never trained to reason in
latent space, so its latent condition is expected to be weak --
that weakness is the control's role.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from .data import final_answer_line
from .metrics import StepLog
from .netenv import sanitize_proxy_env

sanitize_proxy_env()

#: Latent-feedback variants keyed by generation mode.
LATENT_VARIANTS = {
    "latent": "identity",
    "latent_zero": "zero",
    "latent_perturb": "perturb",
}

#: Default latent-step budget and perturbation scale (both are
#: recorded per attempt in the attempt log's meta fields).
DEFAULT_LATENT_STEPS = 16
DEFAULT_PERTURB_SCALE = 0.1


def transform_latent_state(state, variant: str, *, sigma: float = 0.0,
                           torch_module=None):
    """Apply one latent-feedback transform to a hidden state.

    ``state`` is a torch tensor of shape ``[batch, 1, hidden]``
    (any tensor-like object with ``*`` works for identity/zero).
    ``variant`` is ``"identity"`` (Coconut-style feedback),
    ``"zero"`` (zero-latent ablation), or ``"perturb"`` (add
    ``sigma``-scaled standard Gaussian noise, drawn from the
    caller's seeded RNG).
    """
    if variant == "identity":
        return state
    if variant == "zero":
        return state * 0.0
    if variant == "perturb":
        t = torch_module
        if t is None:
            import torch as t
        return state + sigma * t.randn_like(state)
    raise ValueError(f"unknown latent variant {variant!r}")


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    # Week-2 accounting: forward passes spent in latent space
    # (charged like token positions by the FLOPs estimator) and
    # forced filler-token positions (pause control).
    latent_steps: int = 0
    pause_tokens: int = 0
    # Step decomposition for the auditability rubric, when the
    # runner knows it; None lets the harness build the default
    # explicit-CoT decomposition.
    steps: Optional[list] = None
    # Extra per-attempt provenance (variant, sigma, filler token)
    # merged into the JSONL attempt record by the harness.
    meta: dict = field(default_factory=dict)


class MockRunner:
    """Deterministic canned-output runner (pipeline validation only).

    The canned trace solves tasks whose id hashes to a multiple of
    three *only if* the canned final number happens to match the
    gold -- it does not peek at gold answers. It exists so every
    harness path (verifier included) executes end to end, for every
    Week-2 condition as well as the explicit baseline.
    """

    name = "mock-cot-0"

    def n_params(self) -> int:
        return 0  # mock: FLOPs estimator reports 0 by construction

    def generate(self, messages: list[dict], *, seed: int,
                 max_new_tokens: int = 768,
                 mode: str = "explicit",
                 latent_steps: int = DEFAULT_LATENT_STEPS,
                 perturb_scale: float = DEFAULT_PERTURB_SCALE
                 ) -> Generation:
        t0 = time.perf_counter()
        h = int(hashlib.sha256(
            (messages[-1]["content"] + str(seed)).encode()).hexdigest(), 16)
        n_steps = 2 + h % 3
        canned_answer = [72, 18, 5][h % 3]
        reasoning_lines = [
            f"Step {i + 1}: work through part {i + 1} of the problem."
            for i in range(n_steps)]
        prompt_tokens = sum(len(m["content"].split()) for m in messages) + 8

        if mode in LATENT_VARIANTS:
            # Latent reasoning leaves no text: the completion is
            # only the explicitly decoded answer segment.
            text = f"#### {canned_answer}"
            steps = [
                StepLog(kind="latent_reasoning", trace_text=None),
                StepLog(kind="final_answer", trace_text=text),
            ]
            return Generation(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(text.split()),
                latency_s=time.perf_counter() - t0,
                latent_steps=latent_steps,
                steps=steps,
                meta={"latent_variant": LATENT_VARIANTS[mode],
                      "perturb_scale":
                          perturb_scale if mode == "latent_perturb"
                          else None})
        if mode == "pause":
            text = "\n".join(reasoning_lines + [f"#### {canned_answer}"])
            steps = [
                StepLog(kind="pause", trace_text=None),
                StepLog(kind="reasoning", trace_text=text),
                StepLog(kind="final_answer",
                        trace_text=f"#### {canned_answer}"),
            ]
            return Generation(
                text=text,
                prompt_tokens=prompt_tokens,
                # Pause positions are visible output positions and
                # are counted as completion tokens (see module
                # docstring); the budget matches the latent one.
                completion_tokens=len(text.split()) + latent_steps,
                latency_s=time.perf_counter() - t0,
                pause_tokens=latent_steps,
                steps=steps,
                meta={"pause_tokens": latent_steps})
        if mode != "explicit":
            raise ValueError(f"unknown generation mode {mode!r}")
        text = "\n".join(reasoning_lines + [f"#### {canned_answer}"])
        return Generation(text=text,
                          prompt_tokens=prompt_tokens,
                          completion_tokens=len(text.split()),
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

    # -- prompt rendering -----------------------------------------
    def _prompt_text(self, messages: list[dict]) -> str:
        """Render the fixed harness prompt for this tokenizer.

        Chat models get the chat template (the Week-1 protocol).
        Models that predate chat templates (GPT-2, the control
        family) get the *same content* as plain text -- system and
        user turns joined, with a trailing blank line as the
        completion cue. That rendering difference is a documented
        deviation of the GPT-2 control, forced by the model, not
        a protocol change: content, suffix instruction, splits,
        and verifier are identical across conditions.
        """
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(m["content"] for m in messages) + "\n\n"

    # -- architecture adapter --------------------------------------
    def _backbone(self):
        """(backbone, input-embedding layer, lm_head) for this model.

        Works across the families used here: Qwen2/Qwen3/Llama
        expose the transformer as ``model.model`` and GPT-2 as
        ``model.transformer``; ``base_model`` resolves both, and
        the embedding / head accessors are architecture-generic.
        """
        return (self.model.base_model,
                self.model.get_input_embeddings(),
                self.model.get_output_embeddings())

    def _pause_token_id(self) -> int:
        """The pause control's filler token: the first of a few
        neutral candidates that this tokenizer encodes as a single
        token. An existing vocabulary token, deliberately -- the
        control spends forward passes on filler the model already
        has an embedding for, and no embedding is added or trained.
        """
        for cand in ("…", ".", "..."):
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return self.tokenizer.encode(".", add_special_tokens=False)[0]

    # -- sampling ---------------------------------------------------
    def _sample_token(self, logits, temperature: float, top_p: float):
        """One sampled token id from a ``[1, vocab]`` logits row,
        mirroring the explicit path's temperature / top-p sampling
        (draws use the global RNG, seeded per attempt). A manual
        sampler keeps the latent decode loop independent of the
        ``generate()`` API, which changed across transformers
        versions (v5 dropped per-call generators)."""
        torch = self._torch
        logits = logits.float()
        if temperature and temperature > 0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(
                probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            keep = cumsum <= top_p
            keep[..., 1:] = keep[..., :-1].clone()
            keep[..., 0] = True
            sorted_probs = torch.where(
                keep, sorted_probs,
                torch.zeros_like(sorted_probs))
            probs = torch.zeros_like(probs).scatter_(
                -1, sorted_idx, sorted_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return int(torch.multinomial(probs, 1).item())

    # -- generation modes -------------------------------------------
    def _generate_explicit(self, messages: list[dict], *, seed: int,
                           max_new_tokens: int, temperature: float,
                           top_p: float) -> Generation:
        torch = self._torch
        prompt_text = self._prompt_text(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        # transformers v5 removed the per-call `generator` kwarg from
        # generate(); seed the global RNG per attempt instead.
        # Attempts run sequentially, so draws stay reproducible per
        # (task, seed).
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id)
        latency = time.perf_counter() - t0
        new_ids = out[0][prompt_tokens:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return Generation(text=text,
                          prompt_tokens=int(prompt_tokens),
                          completion_tokens=int(new_ids.shape[0]),
                          latency_s=latency)

    def _generate_pause(self, messages: list[dict], *, seed: int,
                        max_new_tokens: int, temperature: float,
                        top_p: float, n_pause: int) -> Generation:
        """Pause-token control: ``n_pause`` filler tokens are
        appended to the prompt and processed first; explicit
        decoding then proceeds exactly as in the baseline. The
        decoded text (filler excluded) is what the verifier sees,
        so prompts, splits, and verifier are untouched -- only the
        first ``n_pause`` forward passes are spent on filler."""
        torch = self._torch
        prompt_text = self._prompt_text(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.device)
        prompt_ids = inputs["input_ids"]
        prompt_tokens = prompt_ids.shape[1]
        pause_id = self._pause_token_id()
        filler = torch.full((1, n_pause), pause_id,
                            dtype=prompt_ids.dtype, device=self.device)
        input_ids = torch.cat([prompt_ids, filler], dim=1)
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id)
        latency = time.perf_counter() - t0
        new_ids = out[0][prompt_tokens + n_pause:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        steps = [
            StepLog(kind="pause", trace_text=None),
            StepLog(kind="reasoning", trace_text=text),
            StepLog(kind="final_answer",
                    trace_text=final_answer_line(text)),
        ]
        return Generation(
            text=text,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(n_pause + new_ids.shape[0]),
            latency_s=latency,
            pause_tokens=n_pause,
            steps=steps,
            meta={"pause_tokens": n_pause, "pause_token_id": pause_id})

    def _decode_from_state(self, h_last, past, *, max_new_tokens: int,
                           temperature: float, top_p: float):
        """Explicit decoding from a latent state: sample tokens one
        at a time from the lm_head on the current last hidden state,
        feeding each sampled token's embedding back through the
        backbone with the accumulated KV cache. Returns
        (text, n_tokens)."""
        torch = self._torch
        backbone, embed_layer, lm_head = self._backbone()
        eos = self.tokenizer.eos_token_id
        ids: list[int] = []
        for _ in range(max_new_tokens):
            logits = lm_head(h_last[:, -1, :])
            tok = self._sample_token(logits, temperature, top_p)
            if tok == eos:
                break
            ids.append(tok)
            emb = embed_layer(
                torch.tensor([[tok]], device=self.device))
            out = backbone(inputs_embeds=emb, past_key_values=past,
                           use_cache=True)
            past = out.past_key_values
            h_last = out.last_hidden_state[:, -1:, :]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text, len(ids)

    def _generate_latent(self, messages: list[dict], *, seed: int,
                         max_new_tokens: int, temperature: float,
                         top_p: float, latent_steps: int,
                         variant: str,
                         perturb_scale: float) -> Generation:
        """Latent reasoning (see module docstring): prefill the
        prompt, run ``latent_steps`` hidden-state feedback steps
        under ``variant``, then decode the answer explicitly.
        Only the decoded segment is text; the latent phase leaves
        no trace, which the step log records."""
        torch = self._torch
        prompt_text = self._prompt_text(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.device)
        input_ids = inputs["input_ids"]
        prompt_tokens = input_ids.shape[1]
        backbone, _embed, _head = self._backbone()
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = backbone(input_ids=input_ids, use_cache=True)
            past = out.past_key_values
            h = out.last_hidden_state[:, -1:, :]
            for _ in range(latent_steps):
                h_in = transform_latent_state(
                    h, variant, sigma=perturb_scale,
                    torch_module=torch)
                out = backbone(inputs_embeds=h_in,
                               past_key_values=past, use_cache=True)
                past = out.past_key_values
                h = out.last_hidden_state[:, -1:, :]
            text, n_decoded = self._decode_from_state(
                h, past, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p)
        latency = time.perf_counter() - t0
        steps = [
            # The reasoning happened in latent space: no explicit
            # trace exists for it, and the log says so.
            StepLog(kind="latent_reasoning", trace_text=None),
            # The answer segment is explicit decoded text -- the
            # attempt's action-bearing step still carries a trace
            # (or doesn't, if the decode produced no answer line,
            # which the auditability score then reflects).
            StepLog(kind="final_answer",
                    trace_text=final_answer_line(text)),
        ]
        return Generation(
            text=text,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=n_decoded,
            latency_s=latency,
            latent_steps=latent_steps,
            steps=steps,
            meta={"latent_variant": variant,
                  "perturb_scale":
                      perturb_scale if variant == "perturb" else None})

    def generate(self, messages: list[dict], *, seed: int,
                 max_new_tokens: int = 768,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 mode: str = "explicit",
                 latent_steps: int = DEFAULT_LATENT_STEPS,
                 perturb_scale: float = DEFAULT_PERTURB_SCALE
                 ) -> Generation:
        if mode == "explicit":
            return self._generate_explicit(
                messages, seed=seed, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p)
        if mode == "pause":
            return self._generate_pause(
                messages, seed=seed, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                n_pause=latent_steps)
        if mode in LATENT_VARIANTS:
            return self._generate_latent(
                messages, seed=seed, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                latent_steps=latent_steps,
                variant=LATENT_VARIANTS[mode],
                perturb_scale=perturb_scale)
        raise ValueError(f"unknown generation mode {mode!r}")
