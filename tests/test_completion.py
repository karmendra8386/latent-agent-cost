"""Tests for the Week-1R completion machinery.

Two pieces are pinned here:

1. The completion slice (``scripts/run_completion.py``): the
   Week-1R ProsQA attempts log recovered from the killed Kaggle
   run holds tasks 0..482 x seeds {0,1,2}; the completion run
   must select exactly tasks 483..499 x seeds {0,1,2} and write
   records in the identical JSONL schema.

2. The aggregator (``scripts/aggregate_attempts.py``): the
   validation gate -- aggregating the retrieved Week-1R GSM8K
   attempts log must reproduce the run's own published tables
   (``results/week1r_gsm8k.md`` / ``.csv``) byte for byte. If
   this gate fails, merged tables cannot be trusted either.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
RESULTS = REPO / "results"
sys.path.insert(0, str(SCRIPTS))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_completion = _load_script("run_completion")
aggregate_attempts = _load_script("aggregate_attempts")

from latent_cost.data import Task  # noqa: E402
from latent_cost.harness import run_condition  # noqa: E402
from latent_cost.models import MockRunner  # noqa: E402

PARTIAL = RESULTS / "week1r_prosqa.attempts.partial.jsonl"
GSM8K_LOG = RESULTS / "week1r_gsm8k.attempts.jsonl"


def _synthetic_prosqa(n=500):
    return [Task(task_id=f"prosqa-test-{i}", dataset="prosqa",
                 question=f"Question {i}?",
                 gold="X is a sterpus.",
                 target_concept="sterpus",
                 neg_concept="druffle")
            for i in range(n)]


def _read_records(path):
    return [json.loads(line) for line in
            Path(path).read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------
# Slice logic
# ---------------------------------------------------------------------
def test_slice_selects_exactly_tasks_483_to_499():
    tasks = _synthetic_prosqa(500)
    sliced = run_completion.slice_tasks(tasks, 483, 500)
    assert [t.task_id for t in sliced] == [
        f"prosqa-test-{i}" for i in range(483, 500)]
    assert len(sliced) == 17


def test_slice_rejects_out_of_bounds():
    tasks = _synthetic_prosqa(10)
    for start, stop in [(-1, 5), (5, 5), (9, 11), (8, 3)]:
        with pytest.raises(ValueError):
            run_completion.slice_tasks(tasks, start, stop)


# ---------------------------------------------------------------------
# Completion run: schema + coverage against the recovered log
# ---------------------------------------------------------------------
@pytest.mark.skipif(not PARTIAL.exists(),
                    reason="recovered partial attempts log not present")
def test_completion_covers_exactly_the_missing_attempts(tmp_path):
    partial_records = _read_records(PARTIAL)
    done = {(r["task_id"], r["seed"]) for r in partial_records}
    full_grid = {(f"prosqa-test-{i}", s)
                 for i in range(500) for s in (0, 1, 2)}
    missing = full_grid - done
    # The recovered log's gap, as diagnosed from the killed run.
    assert missing == {(f"prosqa-test-{i}", s)
                       for i in range(483, 500) for s in (0, 1, 2)}

    sliced = run_completion.slice_tasks(_synthetic_prosqa(500),
                                        483, 500)
    log_path = tmp_path / "completion.attempts.jsonl"
    run_condition(sliced, runner=MockRunner(),
                  condition="explicit_cot", seeds=(0, 1, 2),
                  log_path=log_path)
    records = _read_records(log_path)
    assert {(r["task_id"], r["seed"]) for r in records} == missing
    assert len(records) == 51
    # Identical schema: same keys, in the same proportions of types.
    assert {tuple(sorted(r)) for r in records} == \
        {tuple(sorted(r)) for r in partial_records}


@pytest.mark.skipif(not PARTIAL.exists(),
                    reason="recovered partial attempts log not present")
def test_completion_main_writes_identical_schema(tmp_path,
                                                  monkeypatch):
    monkeypatch.setattr(
        run_completion, "load_prosqa",
        lambda split, limit: _synthetic_prosqa(500)[:limit])
    out = tmp_path / "week1r_prosqa_completion"
    monkeypatch.setattr(sys, "argv", [
        "run_completion.py", "--dataset", "prosqa",
        "--task-start", "483", "--task-stop", "500",
        "--seeds", "0", "1", "2", "--mock",
        "--out", str(out), "--label", "WEEK1R"])
    assert run_completion.main() == 0
    records = _read_records(
        out.with_suffix(".attempts.jsonl"))
    assert len(records) == 51
    partial_keys = {tuple(sorted(r))
                    for r in _read_records(PARTIAL)}
    assert {tuple(sorted(r)) for r in records} == partial_keys


# ---------------------------------------------------------------------
# Aggregator: the GSM8K reproduction gate
# ---------------------------------------------------------------------
@pytest.mark.skipif(not GSM8K_LOG.exists(),
                    reason="Week-1R GSM8K attempts log not present")
def test_aggregator_reproduces_week1r_gsm8k_tables(tmp_path):
    from latent_cost.harness import write_tables

    result = aggregate_attempts.aggregate_files(
        [GSM8K_LOG], label="WEEK1R")
    title = aggregate_attempts.table_title(
        "WEEK1R", result.dataset, result.condition,
        "Qwen/Qwen2.5-1.5B-Instruct")
    out = tmp_path / "week1r_gsm8k"
    write_tables([result], out, title=title)

    assert out.with_suffix(".md").read_bytes() == \
        (RESULTS / "week1r_gsm8k.md").read_bytes()
    assert out.with_suffix(".csv").read_bytes() == \
        (RESULTS / "week1r_gsm8k.csv").read_bytes()

    # The headline numbers, checked explicitly as well.
    assert result.attempts == 1500
    assert result.solves == 992
    assert result.accuracy == pytest.approx(0.661, abs=5e-4)
    with open(RESULTS / "week1r_gsm8k.csv") as fh:
        row = next(csv.DictReader(fh))
    assert result.usd_per_solve == float(row["usd_per_solve"])
    assert result.latency_per_solve_median_s == \
        float(row["latency_per_solve_median_s"])
    assert result.latency_per_solve_iqr_s == \
        float(row["latency_per_solve_iqr_s"])
    assert result.flops_per_solve == float(row["flops_per_solve"])
    assert result.tokens_per_solve == float(row["tokens_per_solve"])
    assert result.total_tokens_per_solve == \
        float(row["total_tokens_per_solve"])
    assert result.auditability == float(row["auditability"])
    assert result.price_card_label == \
        "flat stated rate $0.10/1M tokens (week1r)"


def test_aggregator_merges_logs_in_order(tmp_path):
    # Two halves of one synthetic log aggregate exactly like the
    # whole: the merge the ProsQA salvage relies on.
    sliced = _synthetic_prosqa(6)
    log_all = tmp_path / "all.attempts.jsonl"
    run_condition(sliced, runner=MockRunner(),
                  condition="explicit_cot", seeds=(0, 1),
                  log_path=log_all)
    lines = log_all.read_text().splitlines()
    part_a = tmp_path / "a.jsonl"
    part_b = tmp_path / "b.jsonl"
    part_a.write_text("\n".join(lines[:7]) + "\n")
    part_b.write_text("\n".join(lines[7:]) + "\n")
    whole = aggregate_attempts.aggregate_files([log_all],
                                               label="WEEK1R")
    merged = aggregate_attempts.aggregate_files([part_a, part_b],
                                                label="WEEK1R")
    assert merged == whole


def test_aggregator_rejects_mixed_cells(tmp_path):
    log = tmp_path / "mixed.jsonl"
    rec = {"task_id": "t", "dataset": "gsm8k",
           "condition": "explicit_cot", "seed": 0,
           "prompt_tokens": 1, "completion_tokens": 1,
           "latency_s": 1.0, "solved": True, "latent_steps": 0,
           "cost_usd": None, "steps": []}
    other = dict(rec, dataset="prosqa")
    log.write_text(json.dumps(rec) + "\n" + json.dumps(other) + "\n")
    with pytest.raises(ValueError):
        aggregate_attempts.aggregate_files([log], label="WEEK1R")
