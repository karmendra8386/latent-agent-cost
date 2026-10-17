# Results artifacts

Everything in this directory is an **experiment artifact, not a
paper result**. Smoke tables are pipeline validation only and are
labeled SMOKE in their titles.

## smoke_gsm8k (2026-10-16)

- Command: `scripts/run_baseline.py --dataset gsm8k --n 20
  --seeds 0 1 --model Qwen/Qwen2.5-0.5B-Instruct --label SMOKE`
- Slice: first 20 GSM8K test problems x seeds {0, 1} = 40 attempts,
  explicit CoT, fixed harness prompt, temperature 0.7 / top-p 0.95,
  max 256 new tokens.
- Machine: this VM -- no GPU, 2 CPU cores (AMD EPYC 9D25), 7.7 GB
  RAM; torch 2.14.1+cpu, bf16 weights; model RSS ~1.4 GB.
- Price card (stated, per the outline's re-pricing rule): flat
  $0.10 / 1M tokens, input and output. The $/solve figure is an
  artifact of that stated rate on a 40-attempt slice; it is not a
  finding.
- Files: `smoke_gsm8k.md` / `.csv` (the five-metric table),
  `smoke_gsm8k.attempts.jsonl` (per-attempt logs incl. completion
  text), `smoke_gsm8k.log` (runner log).

## smoke_prosqa (2026-10-16)

- Same protocol, `--dataset prosqa --n 10`: first 10 ProsQA test
  items x seeds {0, 1} = 20 attempts. ProsQA loaded from the
  canonical Coconut authors' distribution
  (facebookresearch/coconut, data/prosqa_test.json; 500 items).
