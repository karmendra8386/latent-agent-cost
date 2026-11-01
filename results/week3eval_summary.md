# WEEK3EVAL summary — trained GPT-2 checkpoints (ProsQA, n=500, seed 0)

**Hard-stop gate:** read the stage-0 `gpt2_cot` row first (accuracy 0.570). If that explicit fine-tune is not a sane ProsQA baseline, report the failed sanity gate and do not interpret the final latent-vs-CoT difference as a trained-latent effect.

**Training context (recorded constants from the completed Kaggle training run, not recomputed in this notebook):** runtime 5h51m49s; stage losses stage 0 0.089, stage 1 0.073, stage 2 0.094, stage 3 0.128.

| checkpoint | checkpoint_kind | checkpoint_path | condition | attempts | solves | accuracy | usd_per_solve | latency_per_solve_median_s | latency_per_solve_iqr_s | flops_per_solve | tokens_per_solve | total_tokens_per_solve | auditability |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Stage 0 | stage-0 checkpoint | /kaggle/input/notebooks/karmendra8386/latent-agent-cost-week3-gpt2-train/latent-agent-cost/checkpoints/coconut-gpt2-prosqa-stage0 | gpt2_cot | 500 | 285 | 0.570 | 0.000084 | 0.4735 | 0.1783 | 2.089e+11 | 121.89473684210526 | 839.5228070175439 | 1.0000 |
| Final (stage-3-trained) | final checkpoint | /kaggle/input/notebooks/karmendra8386/latent-agent-cost-week3-gpt2-train/latent-agent-cost/checkpoints/coconut-gpt2-prosqa | gpt2_cot | 500 | 384 | 0.768 | 0.000054 | 0.0977 | 0.0070 | 1.356e+11 | 12.049479166666666 | 544.6640625 | 1.0000 |
| Final (stage-3-trained) | final checkpoint | /kaggle/input/notebooks/karmendra8386/latent-agent-cost-week3-gpt2-train/latent-agent-cost/checkpoints/coconut-gpt2-prosqa | gpt2_latent | 500 | 338 | 0.676 | 0.000064 | 0.2075 | 0.0221 | 1.595e+11 | 31.33431952662722 | 636.4349112426036 | 1.0000 |
