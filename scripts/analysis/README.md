# Analysis scripts of the revision

Run from the repository root; they read the result files under `results/level2/` and write tables and figures to `results/level2/analysis/`.

| Script | Output | Content |
|---|---|---|
| `curves_ci.py` (uses `ci_lib.py`) | `curves_ci.csv` | prompt-clustered bootstrap intervals of the response curves (unified domain classifier, vocabulary score, tag rate, perplexity under GPT-2 large and SmolLM2-1.7B) |
| `eqdiv.py` | `eqdiv.csv` | logit- vs weight-space perplexity at equal KL divergence from the base model, paired bootstrap over texts |
| `bench_eq.py` | `bench_eq.csv` | zero-shot accuracy at equal KL divergence on general text, bootstrap over documents within tasks |
| `entmatch.py`, `entci.py` | `entmatch.csv`, `entci.csv` | continuous entropy-matched ablation (plain adapter interpolated in mean entropy over the 0.05-step temperature grid) and its prompt-clustered intervals |
| `figcurves.py`, `figsweep.py`, `figeq.py`, `figbench.py` | `fig_*.pdf` | figures of the paper |
