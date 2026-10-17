#!/usr/bin/env python3
"""Run one condition over a dataset slice and write a baseline table.

Usage (smoke):
    .venv/bin/python scripts/run_baseline.py --dataset gsm8k \
        --n 20 --seeds 0 1 --model Qwen/Qwen2.5-0.5B-Instruct \
        --out results/smoke_gsm8k --label SMOKE

Every output is an experiment artifact, not a paper result; the
--label is written into the table title so slices cannot be
mistaken for headline numbers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_cost.data import load_gsm8k, load_prosqa
from latent_cost.harness import run_condition, run_kind_note, write_tables
from latent_cost.metrics import ModelSpec, PriceCard, compute_metrics
from latent_cost.models import HFRunner, MockRunner

# Stated rate for local runs (the outline requires prices to be
# stated so traces can be re-priced): a flat illustrative token rate.
# The card's label carries the run label so a table always shows
# which run its prices belong to. Full runs re-state their card in
# the results file header.
def stated_card(label: str) -> PriceCard:
    return PriceCard(
        input_usd_per_token=0.10 / 1e6,
        output_usd_per_token=0.10 / 1e6,
        label=f"flat stated rate $0.10/1M tokens ({label.strip().lower()})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "prosqa"],
                    default="gsm8k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the HF runner (cpu, cuda)")
    ap.add_argument("--dtype", default="bfloat16",
                    help="model dtype for the HF runner "
                         "(bfloat16, float16, float32)")
    ap.add_argument("--mock", action="store_true",
                    help="use the canned-CoT mock runner")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="RUN")
    args = ap.parse_args()

    loader = load_gsm8k if args.dataset == "gsm8k" else load_prosqa
    tasks = loader(split=args.split, limit=args.n)
    print(f"loaded {len(tasks)} {args.dataset} tasks ({args.split})")

    if args.mock:
        import latent_cost.models as m
        runner = MockRunner()
        n_params = 494_032_768  # Qwen2.5-0.5B spec, for FLOPs scale
        spec = ModelSpec(name="mock (Qwen2.5-0.5B-scale FLOPs)",
                         n_params=n_params)
        _ = m
    else:
        runner = HFRunner(args.model, dtype=args.dtype,
                          device=args.device)
        spec = ModelSpec(name=args.model, n_params=runner.n_params())
    print(f"runner: {runner.name}; params: {spec.n_params:,}")

    out_prefix = Path(args.out)
    log_path = out_prefix.with_suffix(".attempts.jsonl")
    if log_path.exists():
        log_path.unlink()
    attempts = run_condition(tasks, runner=runner,
                             seeds=tuple(args.seeds),
                             max_new_tokens=args.max_new_tokens,
                             log_path=log_path)
    result = compute_metrics(attempts,
                             price_card=stated_card(args.label),
                             model=spec)
    title = (f"{args.label} baseline — {args.dataset} / explicit_cot / "
             f"{spec.name} {run_kind_note(args.label)}")
    csv_path, md_path = write_tables([result], out_prefix, title=title)
    print(f"attempts: {result.attempts}  solves: {result.solves}  "
          f"accuracy: {result.accuracy:.3f}")
    print(f"wrote {csv_path} and {md_path}; log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
