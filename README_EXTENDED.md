# DynaLoRA: extended experiments

This fork extends the original DynaLoRA study (`RodrigoMFischer/dynalora-inference-control-llms`)
with a reusable pipeline and a broader evaluation. The original notebooks and released
generations are kept unchanged at the repository root.

## Layout

```
dynalora/            package: data configs, LoRA training, generation methods, metrics, aggregation
scripts/run_all.py   end-to-end run (train adapters, generate, evaluate, summarize)
scripts/level1_original_generations.py   metrics over the original released generations
notebooks/colab_run_all.ipynb            Colab driver (GPU)
results/level1/      metrics computed over the original 840 generations
results/level2/gpt2/ GPT-2 small: summary tables, entropy-matched ablation, per-generation metrics (gzipped)
results/level2/qwen/ Qwen2.5-0.5B (balanced IMDB): same files
```

## Methods compared

* `dynalora`: logit-space scaling, final = y_base + lambda (y_lora - y_base), two forward passes.
* `weight`: the LoRA scaling alpha/r multiplied by lambda in weight space, one forward pass.
* `cfg`: classifier-free guidance on the base model with a domain-description prefix (gamma grid).
* `lora` / `base`: adapted and base models at several temperatures (entropy-matched ablation).
* `human`: held-out human texts from the same corpus, scored with the same metrics.

## Corpus configurations

`imdb_neg` (original recipe: first 10,000 IMDB train rows, all negative, lr 5e-6, unmasked padding),
`imdb_bal` (5,000 positive + 5,000 negative), `imdb_bal_nohtml` (same, HTML tags removed),
`yelp` (Yelp Review Full), `pubmed` (PubMedQA abstracts). Datasets are loaded from the
Hugging Face Hub and are not redistributed here.

## Metrics

HTML artifact rate, generated length and EOS collapse, domain vocabulary score (fixed lexicons,
identical rule for all domains), perplexity of the continuation under an external language model
(GPT-2 large by default), probability of the target domain under a TF-IDF + logistic regression
classifier trained on held-out human texts of all domains plus generic text, positive probability
under an SST-2 sentiment classifier (review domains), dist-1/2/3, and the mean entropy of the
sampling distribution, used to match temperatures between DynaLoRA and the plain adapter.

## Runs reported in the article

Both runs were executed on a Google Colab T4 (Python 3.13, PyTorch 2.11, transformers 5.20,
peft 0.21, datasets 4.8, scikit-learn 1.6) with 20 samples x 10 prompts per grid point:
42,000 generations for GPT-2 small (five configurations) and 8,400 for Qwen2.5-0.5B.
`run_info.json` in each results folder records the arguments and the domain-classifier accuracy.

## Running

```
pip install -r requirements.txt
python scripts/run_all.py --out results/level2 --models gpt2 --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed
python scripts/run_all.py --out results/level2 --models Qwen/Qwen2.5-0.5B --configs imdb_bal
python scripts/level1_original_generations.py --out results/level1
python scripts/run_all.py --smoke --out /tmp/smoke     # CPU smoke test with a random tiny model
```
