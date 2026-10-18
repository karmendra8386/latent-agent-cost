"""Coconut-style curriculum training for a small causal LM.

The Week-2 pilot showed that *training-free* hidden-state
feedback collapses a model that was never trained to consume its
own latent states (ProsQA latent 0.014 vs explicit 0.202 on
Qwen2.5-1.5B). Coconut (Hao et al., 2024) exists precisely to
make latent inference in-distribution: the model is *trained*
with a curriculum in which the first ``k`` reasoning steps are
progressively replaced by latent feedback steps, and the loss
falls on the text that remains. This module implements that
training for a small GPT-2, producing the one trained comparison
the paper's design calls for (the pre-registered GPT-2 control
pair, ``gpt2_cot`` / ``gpt2_latent``, pointed at a checkpoint
that has actually been through the curriculum).

Recipe implemented (following the paper)
-----------------------------------------
* An example is a question plus the dataset's own reasoning
  chain (ProsQA ``steps``; see ``scripts/prepare_prosqa_train.py``)
  and its final answer, rendered as the harness renders it: the
  fixed prompt, then the chain as text, then the
  ``#### <answer>`` line, then EOS.
* At curriculum stage ``k``, the first ``k`` chain steps are
  removed from the text and replaced by ``k * thoughts_per_step``
  latent positions (Coconut's ``c`` thoughts per replaced step;
  default 1). A latent position's input embedding is the model's
  last hidden state at the previous position -- the same feedback
  rule the harness's latent condition uses at inference.
* Loss is cross-entropy on the *remaining* text tokens (the
  unreplaced steps and the answer line) plus the final EOS --
  never on the prompt, never on latent positions.
* Stages run in order, weights carrying forward, each for a
  configurable number of epochs.

Deliberate deviations from the paper, forced by the harness
-----------------------------------------------------------
* No ``<bot>`` / ``<eot>`` marker tokens. The harness's latent
  inference inserts none: it prefills the prompt, feeds hidden
  states back for a fixed step budget, then decodes. Training
  therefore uses exactly that interface, so the trained
  checkpoint and the evaluation path agree by construction.
  Adding markers would require changing the harness's condition
  behavior, which is out of scope.
* Training forwards recompute the full prefix at every latent
  step instead of carrying a KV cache. For these sequence
  lengths (~<=350 positions) the cost is negligible, autograd
  through a mutable cache is a needless hazard, and the math is
  identical to the cached inference path.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .data import Task
from .harness import build_messages
from .netenv import sanitize_proxy_env

sanitize_proxy_env()


# ---------------------------------------------------------------------
# Training examples
# ---------------------------------------------------------------------
@dataclass
class TrainExample:
    question: str
    steps: list[str]
    answer: str

    def to_dict(self) -> dict:
        return {"question": self.question, "steps": list(self.steps),
                "answer": self.answer}


def load_examples_jsonl(path) -> list[TrainExample]:
    """Load ``{question, steps[], answer}`` JSONL (the schema
    ``scripts/prepare_prosqa_train.py`` writes)."""
    examples = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(TrainExample(question=row["question"],
                                     steps=list(row["steps"]),
                                     answer=row["answer"]))
    return examples


def completion_text(example: TrainExample) -> str:
    """The trained completion: chain steps as text, then the
    harness's final-answer line."""
    return "\n".join(example.steps) + "\n#### " + example.answer


def completion_segments(example: TrainExample) -> list[tuple[str, int, int]]:
    """(kind, char_start, char_end) segments tiling
    :func:`completion_text`: one ``"step"`` segment per chain
    step (each after the first carries its leading newline) and
    one ``"answer"`` segment (leading newline + ``#### `` line).
    """
    segments: list[tuple[str, int, int]] = []
    pos = 0
    for i, step in enumerate(example.steps):
        text = step if i == 0 else "\n" + step
        segments.append(("step", pos, pos + len(text)))
        pos += len(text)
    answer_text = "\n#### " + example.answer
    segments.append(("answer", pos, pos + len(answer_text)))
    return segments


def tokenize_completion(tokenizer, example: TrainExample):
    """Tokenize the completion and split the ids by segment.

    Returns ``(step_token_ids, answer_token_ids)``: one id list
    per chain step, plus the answer line's ids. Token assignment
    uses the tokenizer's offset mapping over the jointly
    tokenized completion text: a token belongs to the segment
    its first character falls in (deterministic at the newline
    boundaries, where a merged token stays with the segment it
    starts in).
    """
    text = completion_text(example)
    enc = tokenizer(text, return_offsets_mapping=True,
                    add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    segments = completion_segments(example)
    step_ids: list[list[int]] = [
        [] for kind, _, _ in segments if kind == "step"]
    answer_ids: list[int] = []
    seg_idx = 0
    for tok_id, (start, _end) in zip(ids, offsets):
        while (seg_idx < len(segments) - 1
               and start >= segments[seg_idx][2]):
            seg_idx += 1
        kind = segments[seg_idx][0]
        if kind == "step":
            # Steps come first, in order; the running segment
            # index maps to the step index directly.
            step_ids[seg_idx].append(tok_id)
        else:
            answer_ids.append(tok_id)
    return step_ids, answer_ids


def render_prompt_text(tokenizer, question: str) -> str:
    """The harness's prompt for a ProsQA question, rendered
    exactly as :meth:`HFRunner._prompt_text` renders it (chat
    template when the tokenizer has one, plain-text join for
    GPT-2). Reuses the method itself -- on a namespace shim, so
    no model is loaded -- to keep training and evaluation
    prompts identical by construction."""
    from .models import HFRunner

    task = Task(task_id="train", dataset="prosqa", question=question,
                gold="")
    shim = SimpleNamespace(tokenizer=tokenizer)
    return HFRunner._prompt_text(shim, build_messages(task))


# ---------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------
def replaced_steps(stage: int, n_steps: int) -> int:
    """Chain steps replaced by latent positions at ``stage``:
    the first ``stage`` steps, capped at the chain's length (a
    short chain is fully replaced; the answer always remains)."""
    if stage < 0:
        raise ValueError("stage must be >= 0")
    return min(stage, n_steps)


def latent_positions(stage: int, n_steps: int,
                     thoughts_per_step: int = 1) -> int:
    """Latent feedback positions at ``stage`` (Coconut's ``c``
    thoughts per replaced step)."""
    if thoughts_per_step < 1:
        raise ValueError("thoughts_per_step must be >= 1")
    return replaced_steps(stage, n_steps) * thoughts_per_step


# ---------------------------------------------------------------------
# One-example loss
# ---------------------------------------------------------------------
def example_loss(model, tokenizer, example: TrainExample, *,
                 stage: int, thoughts_per_step: int = 1):
    """Cross-entropy loss for one example at one curriculum
    stage (see the module docstring for the exact construction).

    Returns a scalar tensor with gradients attached. Positions:
    prompt tokens (no loss) -> ``latent_positions(...)`` feedback
    steps (no loss) -> remaining chain steps + answer line as
    text (loss on every token, each predicted from the previous
    position) -> EOS (loss).
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    backbone = model.base_model
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    prompt_text = render_prompt_text(tokenizer, example.question)
    prompt_ids = tokenizer(prompt_text)["input_ids"]
    step_ids, answer_ids = tokenize_completion(tokenizer, example)
    n_replaced = replaced_steps(stage, len(example.steps))
    n_latent = latent_positions(stage, len(example.steps),
                                thoughts_per_step)
    text_ids = [t for s in step_ids[n_replaced:] for t in s]
    text_ids += answer_ids

    seq = embed(torch.tensor([prompt_ids], device=device))
    out = backbone(inputs_embeds=seq, use_cache=False)
    h = out.last_hidden_state[:, -1:, :]
    for _ in range(n_latent):
        seq = torch.cat([seq, h], dim=1)
        out = backbone(inputs_embeds=seq, use_cache=False)
        h = out.last_hidden_state[:, -1:, :]

    # The position after the latent phase predicts the first
    # remaining text token; each text position predicts the next;
    # the last predicts EOS.
    boundary_logits = lm_head(h)  # [1, 1, V]
    if text_ids:
        text_emb = embed(torch.tensor([text_ids], device=device))
        full = torch.cat([seq, text_emb], dim=1)
        out = backbone(inputs_embeds=full, use_cache=False)
        text_logits = lm_head(out.last_hidden_state[:, -len(text_ids):, :])
        logits = torch.cat([boundary_logits, text_logits], dim=1)
    else:
        logits = boundary_logits
    targets = torch.tensor(
        [text_ids + [tokenizer.eos_token_id]], device=device)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def count_positions(tokenizer, example: TrainExample, *, stage: int,
                    thoughts_per_step: int = 1) -> int:
    """Total sequence positions an example occupies at ``stage``
    (prompt + latent + remaining text + EOS); used for the
    length guard in training."""
    prompt_ids = tokenizer(
        render_prompt_text(tokenizer, example.question))["input_ids"]
    step_ids, answer_ids = tokenize_completion(tokenizer, example)
    n_replaced = replaced_steps(stage, len(step_ids))
    remaining_text = (sum(len(s) for s in step_ids[n_replaced:])
                      + len(answer_ids))
    return (len(prompt_ids)
            + latent_positions(stage, len(example.steps),
                               thoughts_per_step)
            + remaining_text + 1)


# ---------------------------------------------------------------------
# Curriculum training loop
# ---------------------------------------------------------------------
def train_curriculum(model, tokenizer, examples: list[TrainExample],
                     *, stages: list[int], epochs_per_stage: int = 1,
                     lr: float = 1e-4, thoughts_per_step: int = 1,
                     grad_accum: int = 8, max_len: int = 512,
                     seed: int = 0, log=print,
                     on_stage_end=None) -> list[dict]:
    """Train ``model`` through the curriculum stages in order.

    Per-example micro-batches with gradient accumulation (the
    latent phase makes per-example sequence shapes differ; at
    this scale the simple loop is the correct trade). Returns a
    history of per-(stage, epoch) records with mean loss, the
    example count used, and the length-guard skip count.
    ``on_stage_end(stage, model)``, if given, fires after each
    stage's epochs (the CLI uses it for per-stage checkpoints).
    """
    import torch

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict] = []
    model.train()
    for stage in stages:
        for epoch in range(epochs_per_stage):
            order = list(range(len(examples)))
            random.Random(seed + stage * 10007 + epoch).shuffle(order)
            optimizer.zero_grad()
            loss_sum, used, skipped, micro = 0.0, 0, 0, 0
            for idx in order:
                ex = examples[idx]
                if count_positions(tokenizer, ex, stage=stage,
                                   thoughts_per_step=thoughts_per_step
                                   ) > max_len:
                    skipped += 1
                    continue
                loss = example_loss(
                    model, tokenizer, ex, stage=stage,
                    thoughts_per_step=thoughts_per_step)
                (loss / grad_accum).backward()
                loss_sum += float(loss.detach())
                used += 1
                micro += 1
                if micro == grad_accum:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    micro = 0
            if micro:  # final partial accumulation window
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            record = {"stage": stage, "epoch": epoch,
                      "loss": loss_sum / used if used else None,
                      "examples": used, "skipped": skipped}
            history.append(record)
            log(f"stage {stage} epoch {epoch}: "
                f"loss {record['loss']} "
                f"({used} examples, {skipped} skipped)")
        if on_stage_end is not None:
            on_stage_end(stage, model)
    return history


def save_checkpoint(model, tokenizer, out_dir, *, meta: dict | None = None
                    ) -> Path:
    """Save an HF-format checkpoint the harness loads unchanged
    (``AutoModelForCausalLM`` / ``AutoTokenizer``
    ``from_pretrained`` on the directory), plus a
    ``coconut_training.json`` provenance sidecar (ignored by the
    loaders)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    if meta is not None:
        (out / "coconut_training.json").write_text(
            json.dumps(meta, indent=2))
    return out
