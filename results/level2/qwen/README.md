# Level-2 results: Qwen2.5-0.5B

Produced by `scripts/run_all.py --models Qwen/Qwen2.5-0.5B --configs imdb_bal` on a Colab T4 (see `run_info.json`). Same grid, procedures and external metrics as the GPT-2 runs in `../gpt2/`; the adapter targets the four attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with rank 8, alpha 16, dropout 0.1, learning rate 2e-4, 3 epochs and masked padding.

- `summary.csv`: aggregate metrics per (method, lambda/gamma/temperature); the `human` row is the held-out reference.
- `entropy_matched.csv`: DynaLoRA at each lambda against the plain adapter at the closest mean entropy.
- `metrics__qwen__imdb_bal.csv.gz`: every generation with its per-generation metrics (gzipped CSV).

The adapter weights and the third-party corpora are not stored here (see `../gpt2/README.md`).
