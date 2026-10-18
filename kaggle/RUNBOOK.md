# Week 1 baselines on Kaggle — runbook

Goal: run the Week-1 explicit-CoT baselines (Qwen2.5-1.5B-Instruct;
GSM8K 500 × seeds {0,1,2} and ProsQA 500 × seeds {0,1,2}) on a free
Kaggle GPU, headless, and bring the result tables back.

Code source of truth: https://github.com/karmendra8386/latent-agent-cost
(branch `main`). The notebook `week1-baselines.ipynb` in this folder
clones it fresh on every run.

## 1. Create the notebook

1. Kaggle → Notebooks → **New Notebook**.
2. Import `kaggle/week1-baselines.ipynb` from the GitHub repo above
   (File → Import Notebook → GitHub, or upload the file).
3. Notebook **Settings** (right-hand panel) — both are required:
   - **Accelerator: GPU** (T4 ×2 or P100; a single GPU is used).
   - **Internet: ON** — the run clones from GitHub and downloads the
     GSM8K dataset, the ProsQA file, and the model weights from
     Hugging Face. With Internet off, cell 2 fails immediately.

## 2. Run it headless (commit mode)

1. **Save Version → Save & Run All (Commit)**.
   This executes the whole notebook on Kaggle's servers and persists
   the outputs — you can close the tab. (Interactive "Run All" also
   works but dies with the session.)
2. Wait for the version to reach **Complete** on the notebook's
   Versions list. The first cell hard-fails if no GPU is visible, so a
   fast failure there means the Accelerator setting didn't stick.

The two run commands the notebook executes (after cloning and
installing `requirements.txt` minus the local CPU `torch` pin, which
must not replace Kaggle's CUDA torch):

```
python scripts/run_baseline.py --dataset gsm8k --n 500 --seeds 0 1 2 --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype float16 --out results/week1_gsm8k --label WEEK1
python scripts/run_baseline.py --dataset prosqa --n 500 --seeds 0 1 2 --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype float16 --out results/week1_prosqa --label WEEK1
```

## 3. Retrieve the results

1. Open the finished version → **Output** tab.
2. Download from the cloned repo's `results/` folder:
   - `week1_gsm8k.md` / `week1_gsm8k.csv` / `week1_gsm8k.attempts.jsonl`
   - `week1_prosqa.md` / `week1_prosqa.csv` / `week1_prosqa.attempts.jsonl`
3. Drop them into `results/` in the local checkout, alongside the
   SMOKE artifacts, and commit.

Notes:

- **Checkpointing:** the runner appends and flushes each attempt to
  the `.attempts.jsonl` as it completes, so a session cut off
  mid-run keeps every completed attempt (the `.md`/`.csv` tables are
  only written at the end of a run; a lost table can be recomputed
  from the attempts log).
- **Price card:** the tables carry the runner's stated flat rate
  ($0.10/1M tokens) in the `price_card_label` column so traces can be
  re-priced later, per the paper outline's re-pricing rule.
- **If git clone is blocked** in the notebook: upload
  `dist/latent-agent-cost.zip` (repo snapshot, no `.venv`/`.git`) as
  a Kaggle dataset, attach it to the notebook, and replace the clone
  cell with an unzip of the dataset copy into the working directory.

## Week 1R — ProsQA salvage: completion run (supersedes the full rerun)

The corrected-protocol Week-1R rerun was first committed as one
combined notebook (Version 3 of `week1-baselines.ipynb`: GSM8K then
ProsQA, labels WEEK1R). It was **killed at Kaggle's ~12 h session
limit during the ProsQA cell** — GSM8K finished and its WEEK1R
results were retrieved (`results/week1r_gsm8k.*`, accuracy 0.661),
but ProsQA never printed a table.

Recovery turned out better than a rerun: the runner checkpoints
every attempt, so the ProsQA attempts log persisted in the killed
run's Output with **1,449 of 1,500 attempts** — all of tasks
0..482 × seeds {0,1,2} (saved locally as
`results/week1r_prosqa.attempts.partial.jsonl`). Only tasks
483..499 × seeds {0,1,2} (51 attempts) are missing.

**Current plan — completion, not rerun:**

1. `week1r-prosqa-completion.ipynb` runs only the missing slice
   via `scripts/run_completion.py` (same loader, task list sliced
   by position, same protocol and label):

   ```
   python scripts/run_completion.py --dataset prosqa --task-start 483 --task-stop 500 --seeds 0 1 2 --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype float16 --out results/week1r_prosqa_completion --label WEEK1R
   ```

   Minutes-to-tens-of-minutes at the v3 pace. Settings and
   commit-mode running are as in §1–2; retrieve
   `week1r_prosqa_completion.attempts.jsonl` from the Output tab.
2. Merge locally with `scripts/aggregate_attempts.py`, which
   recomputes the table from attempts logs through the same
   metrics/table code as the runner:

   ```
   python scripts/aggregate_attempts.py results/week1r_prosqa.attempts.partial.jsonl results/week1r_prosqa_completion.attempts.jsonl --out results/week1r_prosqa --label WEEK1R
   ```

   The aggregator is validated by a hard gate
   (`tests/test_completion.py`): aggregating
   `results/week1r_gsm8k.attempts.jsonl` reproduces the retrieved
   `results/week1r_gsm8k.md` / `.csv` **byte for byte**.

**Superseded:** `week1r-prosqa.ipynb` (the full ProsQA-only rerun,
~7–8 h) is kept in this folder for the record but should **not**
be run — the completion plan above recovers the identical 1,500
attempts in a fraction of the GPU time, and a full rerun would
also redraw the 1,449 already-logged attempts under new sampling
noise instead of preserving them. Lesson kept from the v3 kill:
keep one dataset per Kaggle commit at the corrected protocol's
pace — and prefer completing a checkpointed log over rerunning it.

## Week 2 — staged plan

The full Week-2 matrix (`week2-conditions.ipynb`: latent,
latent_zero, latent_perturb, pause_token × GSM8K + ProsQA × 500
tasks × 3 seeds = 12,000 attempts) is estimated at roughly 25–30
GPU-hours — beyond Kaggle's ~12 h single-session limit and close to
the 30 h/week free quota — so it runs in stages, one **Save & Run
All (Commit)** per stage. Staging is lossless: the runner appends
and flushes every attempt to each run's `.attempts.jsonl`, so a
stage that is cut off keeps all completed attempts.

1. **Stage 1 (pilot)** — `week2-stage1.ipynb`: latent, latent_zero,
   and pause_token on ProsQA only (500 tasks × seed 0, label
   `WEEK2S1`). Purpose: measure per-condition wall-clock per attempt
   and get a first latent-vs-controls signal. Three short runs, well
   inside one session.
2. **Timing measured** — extrapolate Stage 1's per-attempt times to
   the full matrix and split it into stage-sized commits that fit
   the remaining weekly quota.
3. **Full matrix in stages** — the remaining `week2-conditions.ipynb`
   cells, committed stage by stage (datasets/conditions split across
   commits as the timing dictates), until all 8 condition × dataset
   tables exist.

## Week 3 — trained comparison: Coconut-curriculum GPT-2

The stage-1 pilot's verdict (latent 0.014 vs explicit 0.202 on
ProsQA) is a statement about *training-free* feedback: the model
had never been trained to consume its own hidden states. The
paper's design calls for one trained comparison — a small GPT-2
(124M) put through Coconut's curriculum, then evaluated through
the harness's existing `gpt2_cot` / `gpt2_latent` pair with the
checkpoint as `--model`. No harness changes were needed: the
runner loads a local checkpoint directory exactly like a Hub id.

### Data and training (run in the notebook, after the clone/install cell)

Training data is the dataset's own reasoning chains — the
canonical ProsQA train split ships a gold `steps` chain per item
(17,886 rows; the prep script validates each chain mechanically
and dropped 0):

```
python scripts/prepare_prosqa_train.py --split train --out data/prosqa_train.jsonl
python scripts/train_coconut_gpt2.py --data data/prosqa_train.jsonl --base-model openai-community/gpt2 --out checkpoints/coconut-gpt2-prosqa --stages 0,1,2,3 --epochs-per-stage 1 --lr 1e-4 --grad-accum 8 --device cuda --save-each-stage
```

`--save-each-stage` writes `checkpoints/coconut-gpt2-prosqa-stage<k>`
after each stage, so a session that dies mid-curriculum resumes
from the last stage checkpoint (`--base-model` accepts the local
path) instead of restarting.

### Evaluation (same notebook pattern as Weeks 1–2)

```
python scripts/run_baseline.py --dataset prosqa --n 500 --seeds 0 1 2 --model checkpoints/coconut-gpt2-prosqa --condition gpt2_cot --device cuda --dtype float32 --out results/week3_gpt2cot_trained_prosqa --label WEEK3
python scripts/run_baseline.py --dataset prosqa --n 500 --seeds 0 1 2 --model checkpoints/coconut-gpt2-prosqa --condition gpt2_latent --latent-steps 3 --device cuda --dtype float32 --out results/week3_gpt2latent_trained_prosqa --label WEEK3
```

Notes:

- **`--latent-steps` must match the trained interface.** The
  checkpoint was trained (final stage 3) to consume up to 3 latent
  feedback positions before text resumes; evaluating with a very
  different budget measures a different interface than the one
  trained. A small sweep (2/3/4) is a legitimate secondary table.
- **Same protocol as every other cell:** same test split, seeds,
  prompts, verifier, and price card — only the weights differ
  from the untrained GPT-2 control pair, which is the point.
- The untrained GPT-2 pair (`--model openai-community/gpt2`,
  conditions `gpt2_cot` / `gpt2_latent`) should be run in the same
  stage if its tables don't exist yet, so trained-vs-untrained is
  a same-week comparison.

### GPU-hour expectations (estimates, from measured pilot pacing)

- **Training:** ~1–2 GPU-hours per epoch at stages 0–1, up to
  ~2–3 at stage 3 (later stages recompute longer prefixes).
  The full 4-stage × 1-epoch curriculum: **~5–8 GPU-hours** —
  one Kaggle session, inside the ~12 h cap. Two epochs per stage
  (~10–16 GPU-hours) should be split across two sessions;
  per-stage checkpoints make the split lossless.
- **Evaluation:** GPT-2 is ~12× smaller than the Week-2 pilot's
  1.5B model; expect roughly 0.5–1 GPU-hour per condition for
  500 × 3 seeds, so **~1–2 GPU-hours** for the trained pair.
- Total Week-3 budget: **~7–10 GPU-hours** for the 1-epoch
  curriculum plus both evaluations — about a third of one week's
  free quota. Do not start it in a week whose quota is already
  committed to the Week-2 matrix stages.

