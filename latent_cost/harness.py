"""Experiment harness: tasks x seeds x condition -> attempt logs
-> five-metric baseline table.

Week 1 implemented the ``explicit_cot`` condition (the baseline
every later condition is priced against). Week 2 adds the latent
condition, its zero-latent and perturbation ablations, the
pause-token control, and the GPT-2 Coconut-style control pair --
all behind the same :func:`run_condition` interface, so every
condition lands in the same five-metric tables. Prompts, splits,
and the verifier are fixed here so conditions differ only in the
reasoning mode, per the outline's reporting rules.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .data import Task, final_answer_line, verify
from .metrics import (AttemptLog, MetricsResult, ModelSpec, PriceCard,
                      StepLog, compute_metrics)
from .models import (DEFAULT_LATENT_STEPS, DEFAULT_PERTURB_SCALE,
                      Generation)

__all__ = ["CONDITION_MODES", "build_messages", "final_answer_line",
           "run_condition", "run_kind_note", "write_tables"]

#: Condition name (as it appears in tables and attempt logs) ->
#: runner generation mode. The GPT-2 control conditions reuse the
#: explicit / latent modes on a GPT-2 runner; the model is part of
#: the run configuration, the condition names keep the control's
#: identity explicit in every output.
CONDITION_MODES = {
    "explicit_cot": "explicit",
    "latent": "latent",
    "latent_zero": "latent_zero",
    "latent_perturb": "latent_perturb",
    "pause_token": "pause",
    "gpt2_cot": "explicit",
    "gpt2_latent": "latent",
}

# ---------------------------------------------------------------------
# Fixed prompt templates (identical across conditions and seeds)
# ---------------------------------------------------------------------
SYSTEM_PROMPT = ("You are a careful problem solver. Reason step by "
                 "step, then state the final answer.")

USER_SUFFIX = {
    "gsm8k": "\n\nFinish with a final line in the form: #### <number>",
    "prosqa": ("\n\nFinish with a final line in the form: "
               "#### <full statement> "
               "(for example: #### Sally is a sterpus)"),
}


def build_messages(task: Task) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": task.question + USER_SUFFIX[task.dataset]},
    ]


def run_kind_note(label: str) -> str:
    """Heading note for a run's table: smoke runs are pipeline
    validation; any other label is a baseline run."""
    if label.strip().upper() == "SMOKE":
        return "(pipeline validation, not a result)"
    return "(baseline run)"


def _explicit_steps(g: Generation) -> list[StepLog]:
    """Default step decomposition (explicit CoT): the reasoning is
    written text, and the final-answer step -- the attempt's one
    action-bearing step -- carries that explicit trace, so
    explicit CoT scores 1.0 by construction of the rubric (the
    Week-2 conditions are where the score can fall). Runners that
    know a richer decomposition (latent, pause) supply their own
    via ``Generation.steps``."""
    return [
        StepLog(kind="reasoning", trace_text=g.text),
        StepLog(kind="final_answer",
                trace_text=final_answer_line(g.text)),
    ]


def run_condition(tasks: list[Task], *,
                  runner,
                  condition: str = "explicit_cot",
                  seeds: tuple[int, ...] = (0, 1),
                  max_new_tokens: int = 768,
                  latent_steps: int = DEFAULT_LATENT_STEPS,
                  perturb_scale: float = DEFAULT_PERTURB_SCALE,
                  log_path: Path | None = None) -> list[AttemptLog]:
    """Run every (task, seed) once under ``condition``; log each
    attempt as JSONL.

    ``latent_steps`` is the shared reasoning budget: latent
    forward passes for the latent conditions, filler-token
    positions for the pause-token control. ``perturb_scale`` is
    the Gaussian noise scale for the perturbation ablation.
    """
    if condition not in CONDITION_MODES:
        raise ValueError(
            f"unknown condition {condition!r}; "
            f"known: {sorted(CONDITION_MODES)}")
    mode = CONDITION_MODES[condition]
    attempts: list[AttemptLog] = []
    log_fh = open(log_path, "a") if log_path else None
    try:
        for task in tasks:
            messages = build_messages(task)
            for seed in seeds:
                g: Generation = runner.generate(
                    messages, seed=seed, max_new_tokens=max_new_tokens,
                    mode=mode, latent_steps=latent_steps,
                    perturb_scale=perturb_scale)
                solved = verify(task, g.text)
                steps = (g.steps if g.steps is not None
                         else _explicit_steps(g))
                attempt = AttemptLog(
                    task_id=task.task_id,
                    dataset=task.dataset,
                    condition=condition,
                    seed=seed,
                    prompt_tokens=g.prompt_tokens,
                    completion_tokens=g.completion_tokens,
                    latency_s=g.latency_s,
                    solved=solved,
                    latent_steps=g.latent_steps,
                    steps=steps,
                )
                attempts.append(attempt)
                if log_fh:
                    record = attempt.to_dict()
                    record["completion_text"] = g.text
                    record.update(g.meta)
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
