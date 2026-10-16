# latent-agent-cost

Experiment code for **"Tokens Are Not Dollars: Latent vs Explicit
Reasoning Inside LLM Agents, Measured in Cost per Solved Task"**
(Karmendra Pandey). The governing document is the approved outline
(October 2026); it extends the tokenecon toolkit (PyPI
`tokenecon==0.2.0`, Zenodo DOI 10.5281/zenodo.23427025) and the cost
model of *Token-Economics for Agentic AI* (Zenodo DOI
10.5281/zenodo.23195511).

## Status

Week 1: harness and costing.

- `latent_cost/metrics.py` — the five per-solved-task metrics from
  outline Section 3: $/solve (failed attempts count in the
  numerator), latency/solve (median + IQR), FLOPs/solve (documented
  estimator: token positions x 2 x n_params; latent steps charged
  as forward passes), tokens/solve, auditability score (share of
  action-bearing steps with an explicit trace, per the stated
  rubric). Includes an adapter for tokenecon-style run records.
- `latent_cost/harness.py` — tasks x seeds x condition runner with
  fixed prompts and verifier; writes per-attempt JSONL logs and
  baseline tables (CSV + Markdown). Week 1 ships the
  `explicit_cot` condition.
- `latent_cost/data.py` — GSM8K (HF `openai/gsm8k`, main, test)
  and ProsQA (canonical distribution: the Coconut authors'
  repository, `facebookresearch/coconut`, `data/prosqa_*.json`).

## Layout

```
latent_cost/   metrics, harness, data loaders/verifiers, runners
tests/         pytest unit tests for the metric math
scripts/       run_baseline.py CLI
results/       run artifacts (labeled; smoke runs are pipeline
               validation, not results)
```

## Running

```
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install transformers datasets pytest tokenecon==0.2.0
.venv/bin/python -m pytest tests/ -q
.venv/bin/python scripts/run_baseline.py --dataset gsm8k --n 20 \
    --seeds 0 1 --model Qwen/Qwen2.5-0.5B-Instruct \
    --out results/smoke_gsm8k --label SMOKE
```

Reporting rules (from the outline): all five metrics for every
condition x task cell, plus accuracy; multi-seed with spread for
every headline comparison; same prompts, splits, and verifier
across conditions.
