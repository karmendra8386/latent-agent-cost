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
