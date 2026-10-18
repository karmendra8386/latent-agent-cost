#!/usr/bin/env python3
"""Train a small GPT-2 with the Coconut curriculum (ProsQA).

The training-free Week-2 pilot collapsed because the model had
never been trained to consume its own hidden states. This script
runs the curriculum from :mod:`latent_cost.coconut` -- chain
steps progressively replaced by latent feedback, loss on the
remaining text -- and saves an HF-format checkpoint that the
existing harness evaluates unchanged::

    .venv/bin/python scripts/prepare_prosqa_train.py \
        --split train --out data/prosqa_train.jsonl
    .venv/bin/python scripts/train_coconut_gpt2.py \
        --data data/prosqa_train.jsonl \
        --out checkpoints/coconut-gpt2-prosqa \
        --stages 0,1,2,3 --epochs-per-stage 1 --device cuda

Evaluate the checkpoint through the harness's GPT-2 control pair
(same weights, text vs latent -- the trained comparison):

    .venv/bin/python scripts/run_baseline.py --dataset prosqa \
        --model checkpoints/coconut-gpt2-prosqa \
        --condition gpt2_cot ...
    .venv/bin/python scripts/run_baseline.py --dataset prosqa \
        --model checkpoints/coconut-gpt2-prosqa \
        --condition gpt2_latent --latent-steps 3 ...

``--init config`` builds the base model from its config with
random weights instead of downloading pretrained ones (smoke
runs / pipeline validation only -- a config-init "checkpoint" is
not a trained model and must never be reported as one).
``--config-overrides`` takes a JSON object merged into the
config (e.g. a tiny ``n_layer``/``n_embd`` for CPU smokes).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_cost.coconut import (load_examples_jsonl, save_checkpoint,
                                 train_curriculum)
from latent_cost.netenv import sanitize_proxy_env

sanitize_proxy_env()


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="training JSONL ({question, steps[], answer}; "
                         "see scripts/prepare_prosqa_train.py)")
    ap.add_argument("--base-model", default="openai-community/gpt2",
                    help="HF model id or local path for weights + "
                         "tokenizer")
    ap.add_argument("--init", choices=["pretrained", "config"],
                    default="pretrained",
                    help="pretrained: load base weights; config: "
                         "random init from the base config (smoke "
                         "runs only)")
    ap.add_argument("--config-overrides", default=None,
                    help="JSON object merged into the model config "
                         "(with --init config; e.g. tiny sizes for "
                         "a CPU smoke)")
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--stages", type=_int_list, default=[0, 1, 2, 3],
                    help="curriculum stages, comma-separated "
                         "(stage k replaces the first k chain steps "
                         "with latent feedback)")
    ap.add_argument("--epochs-per-stage", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--thoughts-per-step", type=int, default=1,
                    help="latent positions per replaced step "
                         "(Coconut's c)")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="examples per optimizer step")
    ap.add_argument("--max-len", type=int, default=512,
                    help="skip examples longer than this many "
                         "positions at the current stage")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="cap on training examples (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-each-stage", action="store_true",
                    help="also save a checkpoint after each stage "
                         "to <out>-stage<k> (resumable staging "
                         "for long GPU runs)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    examples = load_examples_jsonl(args.data)
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
    print(f"loaded {len(examples)} training examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if args.init == "pretrained":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.float32)
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.base_model)
        if args.config_overrides:
            for key, value in json.loads(args.config_overrides).items():
                setattr(config, key, value)
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.float32)
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"base: {args.base_model} (init={args.init}); "
          f"params: {n_params:,}; device: {args.device}")

    meta = {
        "base_model": args.base_model, "init": args.init,
        "data": str(args.data), "n_examples": len(examples),
        "stages": args.stages,
        "epochs_per_stage": args.epochs_per_stage, "lr": args.lr,
        "thoughts_per_step": args.thoughts_per_step,
        "grad_accum": args.grad_accum, "max_len": args.max_len,
        "seed": args.seed,
    }

    def on_stage_end(stage, trained_model):
        if args.save_each_stage:
            path = save_checkpoint(
                trained_model, tokenizer, f"{args.out}-stage{stage}",
                meta={**meta, "through_stage": stage})
            print(f"saved stage-{stage} checkpoint to {path}")

    t0 = time.time()
    history = train_curriculum(
        model, tokenizer, examples,
        stages=args.stages, epochs_per_stage=args.epochs_per_stage,
        lr=args.lr, thoughts_per_step=args.thoughts_per_step,
        grad_accum=args.grad_accum, max_len=args.max_len,
        seed=args.seed, on_stage_end=on_stage_end)
    meta["history"] = history
    meta["wall_time_s"] = round(time.time() - t0, 1)
    out = save_checkpoint(model, tokenizer, args.out, meta=meta)
    print(f"saved final checkpoint to {out} "
          f"({meta['wall_time_s']}s of training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
