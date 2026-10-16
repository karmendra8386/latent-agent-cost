"""Experiment harness: tasks x seeds x condition -> attempt logs
-> five-metric baseline table.

Week 1 implements the ``explicit_cot`` condition only (the baseline
every later condition is priced against). Prompts, splits, and the
verifier are fixed here so later conditions differ only in the
reasoning mode, per the outline's reporting rules.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .data import Task, verify
from .metrics import (AttemptLog, MetricsResult, ModelSpec, PriceCard,
                      StepLog, compute_metrics)
from .models import Generation

# ---------------------------------------------------------------------
# Fixed prompt templates (identical across conditions and seeds)
# ---------------------------------------------------------------------
SYSTEM_PROMPT = ("You are a careful problem solver. Reason step by "
                 "step, then state the final answer.")

USER_SUFFIX = {
    "gsm8k": "\n\nGive the final answer on its own line in the form: #### <number>",
    "prosqa": "\n\nGive the final answer on its own line in the form: #### <answer>",
}


def build_messages(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": task.question + USER_SUFFIX[task.dataset]},
    ]


def final_answer_line(completion: str) -> str:
    if "####" in completion:
        return "####" + completion.rsplit("####", 1)[-1]
    lines = [ln for ln in completion.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def run_condition(tasks: list[Task], *,
                  runner,
                  condition: str = "explicit_cot",
                  seeds: tuple[int, ...] = (0, 1),
                  max_new_tokens: int = 256,
                  log_path: Path | None = None) -> list[AttemptLog]:
    """Run every (task, seed) once; log each attempt as JSONL."""
    attempts: list[AttemptLog] = []
    log_fh = open(log_path, "a") if log_path else None
    try:
        for task in tasks:
            messages = build_messages(task)
            for seed in seeds:
                g: Generation = runner.generate(
                    messages, seed=seed, max_new_tokens=max_new_tokens)
                solved = verify(task, g.text)
                # Step decomposition for the auditability rubric:
                # in explicit CoT the reasoning is written text, and
                # the final-answer step -- the attempt's one
                # action-bearing step -- carries that explicit
                # trace, so explicit CoT scores 1.0 by construction
                # of the rubric (later conditions are where the
                # score can fall).
                steps = [
                    StepLog(kind="reasoning", trace_text=g.text),
                    StepLog(kind="final_answer",
                            trace_text=final_answer_line(g.text)),
                ]
                attempt = AttemptLog(
                    task_id=task.task_id,
                    dataset=task.dataset,
                    condition=condition,
                    seed=seed,
                    prompt_tokens=g.prompt_tokens,
                    completion_tokens=g.completion_tokens,
                    latency_s=g.latency_s,
                    solved=solved,
                    latent_steps=0,
                    steps=steps,
                )
                attempts.append(attempt)
                if log_fh:
                    record = attempt.to_dict()
                    record["completion_text"] = g.text
                    log_fh.write(json.dumps(record) + "\n")
                    log_fh.flush()
    finally:
        if log_fh:
            log_fh.close()
    return attempts


# ---------------------------------------------------------------------
# Baseline tables
# ---------------------------------------------------------------------
COLUMNS = ["dataset", "condition", "model", "attempts", "solves",
           "accuracy", "usd_per_solve", "latency_per_solve_median_s",
           "latency_per_solve_iqr_s", "flops_per_solve",
           "tokens_per_solve", "total_tokens_per_solve",
           "auditability", "price_card_label"]


def write_tables(results: list[MetricsResult], out_prefix: Path,
                 *, title: str) -> tuple[Path, Path]:
    """Write the baseline table as CSV and Markdown."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    rows = [r.to_dict() for r in results]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in COLUMNS})

    def fmt(k: str, v) -> str:
        if v is None:
            return "n/a"
        if k == "accuracy":
            return f"{v:.3f}"
        if k == "auditability":
            return f"{v:.2f}"
        if k == "flops_per_solve":
            return f"{v:.3e}"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    buf = io.StringIO()
    buf.write(f"# {title}\n\n")
    buf.write("| " + " | ".join(COLUMNS) + " |\n")
    buf.write("|" + "---|" * len(COLUMNS) + "\n")
    for row in rows:
        buf.write("| " + " | ".join(fmt(k, row[k]) for k in COLUMNS)
                  + " |\n")
    md_path.write_text(buf.getvalue())
    return csv_path, md_path
