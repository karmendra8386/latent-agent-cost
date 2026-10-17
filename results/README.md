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

## week1_gsm8k + week1_prosqa (2026-10-16/17, Kaggle T4 x2)

- Command (per dataset): `scripts/run_baseline.py --dataset {gsm8k,prosqa}
  --n 500 --seeds 0 1 2 --model Qwen/Qwen2.5-1.5B-Instruct
  --device cuda --dtype float16 --label WEEK1`
- Runtime: 27,062.6s total (GSM8K stage 13,808s; ProsQA 13,245s).
- GSM8K: 1500 attempts, 687 solves, accuracy 0.458, usd/solve 0.0001,
  latency median 9.93s, FLOPs/solve 2.274e12, tokens/solve 501.5,
  auditability 1.00.
- ProsQA: 1500 attempts, 216 solves, accuracy 0.144, usd/solve 0.0004,
  latency median 9.64s, FLOPs/solve 1.289e13, tokens/solve 1614.0,
  auditability 1.00.
- $ figures use the flat stated rate $0.10/1M tokens (carried over from
  the smoke price card; label bug — see below). Re-priced at analysis
  time per the outline.
- DIAGNOSIS (2026-10-17, from the attempts logs): ProsQA 0.144 is a
  harness artifact, NOT a model result. Of 1,284 failures: 972 (75.7%)
  truncated at the 256-token cap; 284 (22.1%) contained the target
  concept in the full text but not in the extracted final line; 27
  (2.1%) killed by the verifier's both-concepts rule; 1 (0.1%) never
  reached the target concept. GSM8K capped rate was also 51.1% (its
  verifier's last-number fallback rescues many truncations).
- Known label bugs in these artifacts: the .md headings say
  "(pipeline validation, not a result)" and the price card says
  "(smoke)" — hardcoded strings, fixed in the harness after this run.
- ACTION: protocol fix (max_new_tokens 256 -> 768; shared prompt gains
  an explicit "#### <answer>" final-line instruction; label bugs fixed),
  then BOTH baselines rerun under the corrected protocol before Week 2.

## smoke2_prosqa (2026-10-17, protocol-fix validation)

- Command: `scripts/run_baseline.py --dataset prosqa --n 10 --seeds 0
  --model Qwen/Qwen2.5-0.5B-Instruct --label SMOKE` under the FIXED
  protocol (cap 768, "Finish with a final line in the form: #### ..."
  closing instruction, label fixes). CPU, this VM; pipeline
  validation only -- accuracy on 10 items is not a finding.
- Truncation: 0 of 10 attempts hit the 768 cap (completion lengths
  140-518 tokens, all terminated naturally), vs 3 of 10 seed-0
  attempts capped at 256 in smoke_prosqa above. The cap fix works.
- Format: 3 of 10 completions contain a `####` line; 1 ends with a
  textbook `#### Alex is a zhorpus.` Most others end with a
  "Final answer: ..." line instead -- at 0.5B the marker instruction
  is only partly followed. Extraction no longer loses these (the
  last-line fallback reads them cleanly); the remaining failures are
  wrong-concept answers, not truncation/extraction losses. Marker
  compliance at the baseline's 1.5B scale is for the Kaggle rerun
  to show.
- Files: `smoke2_prosqa.md` / `.csv` / `.attempts.jsonl` / `.log`.

## smoke2-week2_* (2026-10-17, Week-2 conditions smoke)

- Command (per condition): `scripts/run_baseline.py --dataset prosqa
  --n 10 --seeds 0 --model Qwen/Qwen2.5-0.5B-Instruct
  --condition {explicit_cot, latent, latent_zero, latent_perturb,
  pause_token} --latent-steps 16 --perturb-scale 0.1 --label SMOKE`.
  CPU, this VM. Pipeline validation only -- solves on 10 items
  (1/2/3/4 across the latent family and pause) are noise, NOT
  findings; the latent conditions' decoded text at 0.5B is largely
  degenerate, as expected for training-free hidden-state feedback
  on a small model.
- Accounting verified in the attempt logs: latent conditions record
  `latent_steps=16` and the variant (`identity` / `zero` /
  `perturb`) on every attempt; the pause control records
  `pause_tokens=16` in the record meta, with filler positions
  counted inside `completion_tokens`; FLOPs charge latent steps
  like token positions.
- Auditability: 1.00 for every condition on ProsQA. The latent
  attempts' step logs record the reasoning step as
  `latent_reasoning` with NO trace text -- the tracelessness is in
  the record. The rubric scores only action-bearing steps, and on
  these reasoning-only datasets the one action-bearing step is the
  final answer, which every condition still produces as explicit
  decoded text (a non-empty answer line). The score differentiates
  once Week 3's agent tasks add tool/action steps.
- GPT-2 control paths (`gpt2_cot`, `gpt2_latent`) also validated
  end-to-end on the real `openai-community/gpt2` weights (GSM8K
  n=2, throwaway outputs, not filed here): prompt renders as plain
  text (no chat template), latent steps recorded, base-model
  output degenerate -- the control behaving as a control.
- Files: `smoke2-week2_{explicit,latent,zero,perturb,pause}.md` /
  `.csv` / `.attempts.jsonl` / `.log`.
