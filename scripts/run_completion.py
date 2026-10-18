#!/usr/bin/env python3
"""Run ONLY a task slice of a dataset, under the identical protocol
as ``scripts/run_baseline.py`` -- the completion path for a run
whose attempts log survived a cut-off session.

The Week-1R combined Kaggle run was killed at the ~12 h session
limit during the ProsQA cell, but the runner's per-attempt
checkpointing meant the attempts log persisted with 1,449 of 1,500
attempts: every (task, seed) for tasks 0..482. This script runs
exactly the missing slice -- tasks[483:500] x seeds {0, 1, 2} --
with the same loader, prompts, verifier, condition, and stated
price card as the original run, so the two attempts logs merge
into one Week-1R ProsQA baseline (merge with
``scripts/aggregate_attempts.py``).

Usage (Week-1R ProsQA completion):
    python scripts/run_completion.py --dataset prosqa \
        --task-start 483 --task-stop 500 --seeds 0 1 2 \
        --model Qwen/Qwen2.5-1.5B-Instruct --device cuda \
        --dtype float16 --out results/week1r_prosqa_completion \
        --label WEEK1R
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_cost.data import Task, load_gsm8k, load_prosqa
from latent_cost.harness import (CONDITION_MODES, run_condition,
                                 run_kind_note, write_tables)
from latent_cost.metrics import ModelSpec, compute_metrics
from latent_cost.models import HFRunner, MockRunner
from run_baseline import stated_card


def slice_tasks(tasks: list[Task], start: int, stop: int) -> list[Task]:
    """The ``tasks[start:stop]`` slice, with bounds checked.

    The runner is task-major, so a run cut off after task ``k``
    leaves exactly the tail slice ``[k+1:]`` missing; selecting by
    list position (not by task id) keeps the completion slice in
    the same task order the original run would have visited.
    """
    if not (0 <= start < stop <= len(tasks)):
        raise ValueError(
            f"invalid slice [{start}:{stop}] for {len(tasks)} tasks")
    return tasks[start:stop]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "prosqa"],
                    default="prosqa")
    ap.add_argument("--split", default="test")
    ap.add_argument("--task-start", type=int, required=True,
                    help="first task index of the slice (inclusive)")
    ap.add_argument("--task-stop", type=int, required=True,
                    help="last task index of the slice (exclusive); "
                         "the task list is loaded up to this index")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the HF runner (cpu, cuda)")
    ap.add_argument("--dtype", default="bfloat16",
                    help="model dtype for the HF runner "
                         "(bfloat16, float16, float32)")
    ap.add_argument("--mock", action="store_true",
                    help="use the canned-CoT mock runner")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--condition", default="explicit_cot",
                    choices=sorted(CONDITION_MODES))
    ap.add_argument("--latent-steps", type=int, default=16)
    ap.add_argument("--perturb-scale", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="WEEK1R")
    args = ap.parse_args()

    loader = load_gsm8k if args.dataset == "gsm8k" else load_prosqa
    tasks = loader(split=args.split, limit=args.task_stop)
    tasks = slice_tasks(tasks, args.task_start, args.task_stop)
    print(f"loaded slice tasks[{args.task_start}:{args.task_stop}] "
          f"= {len(tasks)} {args.dataset} tasks ({args.split}): "
          f"{tasks[0].task_id} .. {tasks[-1].task_id}")

    if args.mock:
        runner = MockRunner()
        n_params = 494_032_768  # Qwen2.5-0.5B spec, for FLOPs scale
        spec = ModelSpec(name="mock (Qwen2.5-0.5B-scale FLOPs)",
                         n_params=n_params)
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
                             condition=args.condition,
                             seeds=tuple(args.seeds),
                             max_new_tokens=args.max_new_tokens,
                             latent_steps=args.latent_steps,
                             perturb_scale=args.perturb_scale,
                             log_path=log_path)
    result = compute_metrics(attempts,
                             price_card=stated_card(args.label),
                             model=spec)
    title = (f"{args.label} baseline — {args.dataset} / "
             f"{args.condition} / "
             f"{spec.name} {run_kind_note(args.label)}")
    csv_path, md_path = write_tables([result], out_prefix, title=title)
    print(f"attempts: {result.attempts}  solves: {result.solves}  "
          f"accuracy: {result.accuracy:.3f}")
    print(f"wrote {csv_path} and {md_path}; log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
