# Level-2 results: GPT-2 small

Produced by `scripts/run_all.py --models gpt2 --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed` on a Colab T4 (see `run_info.json` for the exact arguments and library versions).

- `summary.csv`: one row per (config, model, method, lambda/gamma/temperature) with the aggregate metrics reported in the article (artifact rate, collapse rate, vocabulary score, length, mean sampling entropy, external perplexity under GPT-2 large, domain probability, positive share, dist-1/2/3). Rows with `model == human` are the held-out human references.
- `entropy_matched.csv`: DynaLoRA at each lambda against the plain adapter at the temperature whose mean entropy is closest.
- `metrics__gpt2__<config>.csv.gz`: every generation (prompt, continuation, method, lambda/gamma/temperature, seed) with its per-generation metrics; 200 generations per grid point. Gzipped CSV, split by configuration.

The adapters themselves (`adapters/`) are not stored here; they are retrained by the script in a few minutes each. The third-party corpora (IMDB, Yelp, PubMedQA, WikiText) are downloaded from the Hugging Face Hub and are not redistributed.
