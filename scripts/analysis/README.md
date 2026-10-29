# Analysis scripts of the revision

Run from the repository root; they read the result files under `results/level2/` and write tables and figures to `results/level2/analysis/`.

| Script | Output | Content |
|---|---|---|
| `curves_ci.py` (uses `ci_lib.py`) | `curves_ci.csv` | prompt-clustered bootstrap intervals of the response curves (unified domain classifier, vocabulary score, tag rate, perplexity under GPT-2 large and SmolLM2-1.7B) |
| `eqdiv.py` | `eqdiv.csv` | logit- vs weight-space perplexity at equal KL divergence from the base model, paired bootstrap over texts |
| `bench_eq.py` | `bench_eq.csv` | zero-shot accuracy at equal KL divergence on general text, bootstrap over documents within tasks |
| `entmatch.py`, `entci.py` | `entmatch.csv`, `entci.csv` | continuous entropy-matched ablation (plain adapter interpolated in mean entropy over the 0.05-step temperature grid) and its prompt-clustered intervals |
| `format_metrics.py` | `format_rates.csv`, `paired_ppl2_fmt.csv`, `eqdiv_gen.csv` | corpus formatting rate (tags, literal line-break escapes, runs of spaces) and repetition per grid point, paired tests on SmolLM2 perplexity and formatting, generation metrics at equal divergence (run after `eqdiv.py`) |
| `replication.py` | `replication_eqdiv.csv`, `replication_bench_eq.csv`, `replication_curves.csv`, `replication_eqdiv_gen.csv`, `replication_paired.csv` | the same comparisons for every adapter, including the seed replicates of GPT-2 and the Qwen2.5-1.5B and Qwen2.5-7B adapters, and paired tests on the replication runs |
| `pubmedqa.py` | `pubmedqa_curves.csv`, `pubmedqa_eq.csv`, `pubmedqa_by_answer.csv` (with `--by_answer`) | downstream task: PubMedQA accuracy (1,000 labeled questions, yes/no/maybe by option log-likelihood) per adapter, method and lambda with its change from the base model, bootstrap over questions; logit minus weight at equal KL divergence, matched as in `bench_eq.py` on general text and also on domain text, paired bootstrap over questions. Reads `results/level2/pubmedqa/bench__*.csv.gz` from `notebooks/colab_pubmedqa.ipynb` (`scripts/zeroshot_bench.py --tasks pubmedqa_labeled`; `--inputs`, `--ll`, `--out` to change the paths) |
| `figcurves.py`, `figsweep.py`, `figeq.py`, `figbench.py` | `fig_*.pdf` | figures of the paper |
