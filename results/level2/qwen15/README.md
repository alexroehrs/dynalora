# Replication over model size: Qwen2.5-1.5B

Produced by `scripts/run_all.py --models Qwen/Qwen2.5-1.5B --configs imdb_bal pubmed --methods dynalora weight` (see `run_info.json`). The recipes are those of `imdb_bal` and `pubmed`, with the attention projections of the Qwen2.5-0.5B adapter as targets.

- `summary.csv`: aggregate metrics per configuration, method, and lambda, as in `results/level2/qwen/`.
- `metrics__Qwen2.5-1.5B__qwen15.csv.gz`: every generation with its per-generation metrics; rows with `model == human` keep the scores of the held-out reference texts but not their text.
