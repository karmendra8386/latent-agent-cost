#!/usr/bin/env python3
"""Recompute a baseline table (.md + .csv) from attempts JSONL logs.

The runner writes its summary tables only at the end of a run, so
a run cut off mid-way (a Kaggle session limit, a crash) loses the
table even when the per-attempt checkpoint log survives. This
script rebuilds the table from one or more attempts logs -- e.g.
the Week-1R ProsQA baseline merges the 1,449 attempts recovered
from the killed combined run with the 51 attempts of the
completion slice -- using the same ``compute_metrics`` /
``write_tables`` code path, the same stated price card, and the
same table title as ``scripts/run_baseline.py``, so an aggregated
table is indistinguishable from a directly-run one.

Validation: aggregating ``results/week1r_gsm8k.attempts.jsonl``
reproduces ``results/week1r_gsm8k.md`` / ``.csv`` byte for byte;
``tests/test_completion.py`` pins that reproduction.

Usage:
    python scripts/aggregate_attempts.py \
        results/week1r_prosqa.attempts.partial.jsonl \
        results/week1r_prosqa_completion.attempts.jsonl \
        --out results/week1r_prosqa --label WEEK1R
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_cost.harness import run_kind_note, write_tables
from latent_cost.metrics import AttemptLog, ModelSpec, compute_metrics
from run_baseline import stated_card

# The Week-1 / Week-1R runner spec: HFRunner.n_params() for
# Qwen/Qwen2.5-1.5B-Instruct, i.e. what the original runs priced
# FLOPs with. Override for runs on other models.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MODEL_PARAMS = 1_543_714_304

_ATTEMPT_FIELDS = {f.name for f in fields(AttemptLog)}


def load_attempts(paths: list[Path]) -> list[AttemptLog]:
    """Read attempts from JSONL logs, in file order.

    Records carry the ``AttemptLog`` fields plus runner extras
    (``completion_text`` and per-attempt meta such as
    ``latent_variant``); only the ``AttemptLog`` fields are kept,
    which is exactly what the run's own in-memory aggregation saw.
    """
    attempts: list[AttemptLog] = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                d = {k: v for k, v in record.items()
                     if k in _ATTEMPT_FIELDS}
                attempts.append(AttemptLog.from_dict(d))
    return attempts


def aggregate_files(paths: list[Path], *,
                    label: str,
                    model_name: str = DEFAULT_MODEL_NAME,
                    model_params: int = DEFAULT_MODEL_PARAMS):
    """Aggregate one or more attempts logs into a MetricsResult.

    All attempts must belong to a single (dataset, condition)
    cell -- the same precondition ``compute_metrics`` assumes of
    a single run.
    """
    attempts = load_attempts(paths)
    if not attempts:
        raise ValueError("no attempts found in the given logs")
    cells = {(a.dataset, a.condition) for a in attempts}
    if len(cells) != 1:
        raise ValueError(
            f"attempts span multiple (dataset, condition) cells: "
            f"{sorted(cells)}")
    spec = ModelSpec(name=model_name, n_params=model_params)
    return compute_metrics(attempts,
                           price_card=stated_card(label),
                           model=spec)


def table_title(label: str, dataset: str, condition: str,
                model_name: str) -> str:
    """The run_baseline.py table title, rebuilt from its parts."""
    return (f"{label} baseline — {dataset} / {condition} / "
            f"{model_name} {run_kind_note(label)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="attempts JSONL logs, in run order")
    ap.add_argument("--out", required=True,
                    help="output prefix for the .md / .csv tables")
    ap.add_argument("--label", default="WEEK1R")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--model-params", type=int,
                    default=DEFAULT_MODEL_PARAMS,
                    help="n_params the run priced FLOPs with")
    args = ap.parse_args()

    result = aggregate_files(args.inputs, label=args.label,
                             model_name=args.model_name,
                             model_params=args.model_params)
    title = table_title(args.label, result.dataset, result.condition,
                        args.model_name)
    csv_path, md_path = write_tables([result], Path(args.out),
                                     title=title)
    print(f"attempts: {result.attempts}  solves: {result.solves}  "
          f"accuracy: {result.accuracy:.3f}")
    print(f"wrote {csv_path} and {md_path} "
          f"from {len(args.inputs)} log(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
