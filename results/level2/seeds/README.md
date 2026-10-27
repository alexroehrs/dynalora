# Replication over training seeds: GPT-2 small

Produced by `scripts/run_all.py --models gpt2 --configs imdb_bal__seed43 imdb_bal__seed44 pubmed__seed43 pubmed__seed44 --methods dynalora weight` (see `run_info.json`). The suffix `__seed<n>` changes only the training seed (LoRA initialization, dropout masks, order of the training texts); the corpus sample and all other settings are those of `imdb_bal` and `pubmed`, whose main adapters used seed 42.

- `summary.csv`: aggregate metrics per configuration, method, and lambda, as in `results/level2/gpt2/`.
- `metrics__gpt2__seeds.csv.gz`: every generation with its per-generation metrics; rows with `model == human` keep the scores of the held-out reference texts but not their text.
