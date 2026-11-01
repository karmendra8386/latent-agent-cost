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

### Evaluation — trained checkpoints (WEEK3EVAL notebook)

The training run completed in `latent-agent-cost-week3-gpt2-train`
Version 1 (recorded runtime **5h51m49s**; recorded stage losses
0.089 / 0.073 / 0.094 / 0.128 for stages 0 / 1 / 2 / 3). Its four
stage checkpoints and final checkpoint live only in that
version's Output — they are not in this repository. Use the
dedicated notebook `kaggle/week3-gpt2-eval.ipynb`; do not
substitute the older generic three-seed commands.

#### Attach the training output as an input

1. Open `latent-agent-cost-week3-gpt2-train`, select **Version
   1**, and confirm its Output contains the checkpoint
   directories.
2. In the evaluation notebook, use **Add Input** and select the
   training notebook's Version 1 Output (Kaggle's menu wording
   may show this under Notebook Output / Your Work). Attach it
   before Save & Run All.
3. The eval notebook's locator cell searches `/kaggle/input`
   recursively, so either a direct `checkpoints/` tree or the
   full training working directory (including a nested
   `latent-agent-cost/checkpoints/` tree) is fine. It accepts a
   directory only when the directory name matches and it
   contains `config.json`, `coconut_training.json`, and model
   weights:
   - stage 0: `coconut-gpt2-prosqa-stage0`
   - final: `coconut-gpt2-prosqa` (preferred), or the
     stage-3 snapshot `coconut-gpt2-prosqa-stage3`
4. If the locator fails, fix the attachment — do not replace it
   with a guessed mounted path. Alternatively set
   `WEEK3_CKPT_ROOT` at the top of the locator cell to a parent
   directory containing the checkpoint directories.

#### Evaluation matrix

ProsQA, **n=500, seed 0**, label `WEEK3EVAL`, dtype `float32`.
The notebook resolves `STAGE0_CKPT` and `FINAL_CKPT` from the
attached input and runs:

```
python scripts/run_baseline.py --dataset prosqa --n 500 --seeds 0 --model "$STAGE0_CKPT" --condition gpt2_cot --device cuda --dtype float32 --out results/week3eval_stage0_gpt2_cot_prosqa --label WEEK3EVAL
python scripts/run_baseline.py --dataset prosqa --n 500 --seeds 0 --model "$FINAL_CKPT" --condition gpt2_cot --device cuda --dtype float32 --out results/week3eval_final_gpt2_cot_prosqa --label WEEK3EVAL
python scripts/run_baseline.py --dataset prosqa --n 500 --seeds 0 --model "$FINAL_CKPT" --condition gpt2_latent --latent-steps 3 --device cuda --dtype float32 --out results/week3eval_final_gpt2_latent_prosqa --label WEEK3EVAL
```

The final cell writes `results/week3eval_summary.csv` and
`results/week3eval_summary.md`, placing accuracy and $/solve for
all three runs side by side with the recorded training runtime
and stage losses labelled as training-run constants. Before
tabulating, it asserts from the attempts logs (not the command
strings) that every run has 500 seed-0 attempts, the expected
condition, and the expected latent-step count — 0 for both CoT
runs and exactly **3** for the latent run.

#### Context-limit guard — why WEEK3EVAL Version 1 failed

Version 1 died during the stage-0 `gpt2_cot` run with
`torch.AcceleratorError: CUDA error: device-side assert
triggered`. Root cause (reproduced locally on CPU with the
first real ProsQA prompt): GPT-2 has only **1,024 position
embeddings**, the ProsQA prompts are long (208–593 GPT-2
tokens; the first test prompt is 591), and the harness's
standard 768-token generation request therefore lets a
non-terminating generation run past position 1,023 — an
embedding index out of range, which surfaces on CUDA as a
device-side assert. It was not a checkpoint/tokenizer mismatch:
the trainer adds no tokens, and the saved config's
`vocab_size` and the tokenizer length are both 50,257; every
prompt token id is below that.

The harness now clamps every generation budget to the
positions actually remaining under the model's config limit
(`min(requested, limit - prompt_tokens - reserved_positions)`,
where latent steps and pause filler count as reserved
positions), and raises a clear `ValueError` if a prompt alone
fills the context. The requested cap is still 768; the
*effective* cap for a 591-token GPT-2 prompt is 433 (430 for
the latent run, which reserves its 3 feedback positions).
Gold ProsQA completions are at most 77 tokens, so the clamp
binds only on generations that would otherwise loop into the
position wall.

The notebook's **pre-flight cell** (immediately after the
checkpoint locator) is mandatory: it checks all 500 prompts
against the stage-0 checkpoint's limit and runs one short
generation from each evaluated path (stage-0 CoT and final
latent, 3 steps). If it does not print `PREFLIGHT PASSED`, do
not start the 500-attempt cells.

#### Environment guard — why WEEK3EVAL Version 2 failed

Version 2 got past the install and the checkpoint locator, then
died in the **pre-flight cell** after ~85 s — before any
500-attempt cell ran — with:

```text
AttributeError: module 'numpy._core._multiarray_umath' has no
attribute '_blas_supports_fpe'  (numpy/testing/_private/utils.py)
```

cascading into `ModuleNotFoundError: Could not import module
'GenerationMixin'` and then the same for `AutoModelForCausalLM`
at the first `HFRunner(...)` construction. That attribute exists
only when numpy's Python files and its compiled core are the
**same** version: the Python side on disk was new, the compiled
core already loaded in the kernel was old — a mixed numpy
install.

Mechanism: the old install cell filtered only the `torch` line
out of `requirements.txt`, leaving `numpy==2.5.4` — the *local*
CPU stack's pin — to be force-installed over Kaggle's
preinstalled numpy. The kernel had already imported the
preinstalled numpy (the notebook's first cell imports torch,
which imports numpy), so pip swapped the files on disk under a
live, old compiled core. The break surfaced only when something
lazily imported a not-yet-loaded numpy submodule
(`numpy.testing`, pulled in during transformers' dynamic import
of `GenerationMixin` in the pre-flight cell). Whether that lazy
import lands before or after the swap is import-order luck —
which is why Version 1 (same install cell, ~35 min earlier) and
the Week-3 training notebook (Oct 23–24) ran fine while Version
2 did not. The install is therefore hardened rather than merely
retried. Note the old cell's only post-install check,
`!python -c "import torch"`, ran in a fresh subprocess against a
consistent on-disk install and was blind to the in-kernel mix.

Fix (in the notebook's install cell): filter **both** `torch`
and `numpy` out of the requirements; capture Kaggle's
preinstalled `numpy` / `scipy` / `pandas` versions *before* pip
runs and pass them as a **constraints file**, so no transitive
dependency (e.g. `datasets` → pandas) can swap the compiled
stack either — a genuine conflict now fails the install cell
loudly instead of corrupting the session. `transformers` stays
pinned at `==5.20.1`: it is proven on this exact Kaggle stack
(torch 2.11.0+cu128) by the training run and by Version 1 — the
failure was never a transformers/torch incompatibility. A new
**smoke-check cell** immediately after the install imports — in
this kernel, in this process — `numpy`, `numpy.testing`,
`torch`, `transformers`, `AutoModelForCausalLM`, and
`GenerationMixin`, prints all versions, and hard-fails with a
clear message if the stack is inconsistent. It must print
`ENV SMOKE OK`; treat it as a launch gate exactly like
`PREFLIGHT PASSED` below.

#### Launch procedure

1. Check the account's remaining GPU quota first; do not start
   unless it comfortably exceeds the runtime estimate below.
2. Import `kaggle/week3-gpt2-eval.ipynb`, attach the training
   Version 1 Output as above, and set **Accelerator = GPU
   (T4 ×2)** and **Internet = ON**.
3. **Save Version → Save & Run All (Commit)**.
4. Immediately open the Versions list and verify the new version
   is **Running** (or Queued). If importing/auto-start produced
   a duplicate version, cancel the duplicate so only one version
   is consuming quota; a duplicate launched without a GPU should
   hard-fail in its first cell, but do not leave two live
   versions running.
5. Confirm the early output shows a CUDA device and the
   locator's printed stage-0 and final checkpoint paths. A
   locator `FileNotFoundError` means the input was not attached
   correctly.
6. Confirm the smoke-check cell prints `ENV SMOKE OK` with the
   numpy / torch / transformers versions. If it fails, stop: the
   environment, not the checkpoints, is broken — see failure #2
   above. Do not let the pre-flight cell be the first thing to
   discover a broken install.
7. Confirm the pre-flight cell prints `PREFLIGHT PASSED`
   (both short generations, the 1,024-position limit, and the
   worst-case clamped budget). If it fails, stop and fix that
   failure — do not let a 500-attempt cell be the first thing
   to discover a checkpoint or context problem.

**Expected runtime (estimate, not a result): ~0.5–1.5
GPU-hours total.** Basis: the earlier estimate below of
0.5–1 GPU-hour for a 1,500-attempt GPT-2 condition scales to
roughly 10–20 minutes for each of these 500-attempt conditions
(~30–60 minutes for all three), plus checkpoint loading and
extra allowance for the latent condition's token-by-token
decode loop. This remains far inside Kaggle's ~12 h session cap,
and the runner flushes every attempt to its JSONL log as it
completes.

#### Retrieve the results

From the finished eval version's Output, download from the
cloned repository's `results/` directory:

- `week3eval_stage0_gpt2_cot_prosqa.md` / `.csv` /
  `.attempts.jsonl`
- `week3eval_final_gpt2_cot_prosqa.md` / `.csv` /
  `.attempts.jsonl`
- `week3eval_final_gpt2_latent_prosqa.md` / `.csv` /
  `.attempts.jsonl`
- `week3eval_summary.md` / `.csv`

Drop them into `results/` in the local checkout and commit.

#### Hard-stop rule — read stage 0 first

The stage-0 `gpt2_cot` row is the explicit-fine-tune sanity
baseline and is read **before** either final-checkpoint row is
interpreted. No numeric sanity threshold was pre-registered in
the project record, so do not invent one after seeing the
number. If stage-0 CoT is degenerate or otherwise not a
credible ProsQA fine-tune, stop the trained-arm interpretation:
report that the trained arm failed the stage-0 sanity gate, do
not present the final latent-vs-CoT difference as evidence
about trained latent reasoning, and proceed with the paper
without the trained arm, saying so. Only if stage 0 is sane
does the same-weights final CoT vs final latent comparison
answer the trained-interface question.

Notes:

- **`--latent-steps` must be 3.** The checkpoint was trained
  (final stage 3) to consume 3 latent feedback positions before
  text resumes; evaluating with a different budget measures a
  different interface than the one trained. A small sweep
  (2/3/4) would be a separate secondary table, not WEEK3EVAL.
- **Same protocol inside the matrix:** same ProsQA test split,
  n, seed, prompts, verifier, max-token cap, and stated price
  card. The intended differences are checkpoint (stage 0 vs
  final) and, on the final checkpoint, generation mode.

### GPU-hour expectations (estimates, from measured pilot pacing)

- **Training:** ~1–2 GPU-hours per epoch at stages 0–1, up to
  ~2–3 at stage 3 (later stages recompute longer prefixes).
  The full 4-stage × 1-epoch curriculum: **~5–8 GPU-hours** —
  one Kaggle session, inside the ~12 h cap. Two epochs per stage
  (~10–16 GPU-hours) should be split across two sessions;
  per-stage checkpoints make the split lossless.
- **Evaluation:** GPT-2 is ~12× smaller than the Week-2 pilot's
  1.5B model. The older planning estimate was roughly
  0.5–1 GPU-hour per condition for 500 × 3 seeds. For the
  actual WEEK3EVAL matrix (three conditions × 500 × seed 0),
  use the dedicated estimate in the evaluation section above:
  **~0.5–1.5 GPU-hours total**, including loading and latent
  decode allowance.
- Total Week-3 budget: the completed 1-epoch training run took
  **5h51m49s** (recorded), and WEEK3EVAL is estimated at
  **~0.5–1.5 GPU-hours**, for **~6.4–7.4 GPU-hours** overall.
  Do not start a comparable run in a week whose quota is
  already committed to the Week-2 matrix stages.

