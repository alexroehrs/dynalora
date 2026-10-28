# Replication over model size: Qwen2.5-7B (4-bit)

Produced by `notebooks/colab_qwen7b.ipynb` on an NVIDIA L4 (see `run_info.json`): `scripts/run_all.py --models Qwen/Qwen2.5-7B --configs imdb_bal pubmed --load_in_4bit --methods dynalora weight --lambdas 0 0.5 1 1.5 2 3`. The recipes are those of `imdb_bal` and `pubmed`, with the attention projections of the Qwen2.5-0.5B adapter as targets (5,046,272 trainable parameters); the base weights are quantized to 4-bit NormalFloat with double quantization and bfloat16 computation (QLoRA) for training and for every evaluation.

- `summary.csv`: aggregate metrics per configuration, method, and lambda, as in `results/level2/qwen15/`.
- `metrics__Qwen2.5-7B__qwen7b.csv.gz`: every generation with its per-generation metrics; rows with `model == human` keep the scores of the held-out reference texts but not their text.
- Related files elsewhere: `results/level2/likelihood/ll__Qwen2.5-7B__*` and `llgen__Qwen2.5-7B__*` (rows of `ll_summary.csv` with `model == Qwen2.5-7B`), `results/level2/bench/bench__Qwen2.5-7B__*` (first 500 documents per task, lambda step 0.5), `results/level2/rescore/rescore__qwen7b.csv.gz`, and the rows of `results/level2/analysis/mauve__gpt2-large.csv` with `run == qwen7b` (the reference split of this run is `reference_qwen7b`).
- Comparisons at equal divergence and paired tests: rows with `model == Qwen2.5-7B` in `results/level2/analysis/replication_*.csv` (`scripts/analysis/replication.py`).
