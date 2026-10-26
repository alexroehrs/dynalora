# Controlling LoRA Adapter Strength at Inference Time through Logit-Space Scaling, Weight-Space Scaling, and Temperature

Code, generated texts, and per-generation scores for the paper. DynaLoRA scales the contribution of a LoRA adapter at decoding time as

    y = y_base + lambda * (y_lora - y_base)

where `y_base` and `y_lora` are the next-token logits of the same model with the adapter disabled and enabled. The study sweeps lambda from 0 to 3 on six adapters and compares DynaLoRA with weight-space scaling of the LoRA update, classifier-free guidance with a domain prefix, and the plain adapter sampled at matched entropy.

## Layout

```
dynalora/                 package: corpus configurations, LoRA training, generation procedures, metrics, aggregation
scripts/run_all.py        end-to-end run: train adapters, generate, score, summarize
scripts/llm_judge.py      LLM-judge ratings (fluency, genre fit) on a stratified sample
scripts/bootstrap_ci.py   prompt-clustered bootstrap intervals and paired tests
scripts/likelihood_tradeoff.py  teacher-forced likelihood and KL divergence on domain and general text, per lambda
scripts/likelihood_summary.py   token-weighted perplexity with bootstrap intervals over texts
notebooks/colab_run_all.ipynb   driver used on a Colab T4 GPU
results/level2/gpt2/      GPT-2 small: summary tables, entropy-matched ablation, per-generation scores
results/level2/qwen/      Qwen2.5-0.5B (balanced IMDB): same files
results/level2/stats/     bootstrap intervals, paired tests, LLM-judge ratings
results/level2/likelihood/ per-text likelihoods (ll__: domain text, llgen__: general text) and ll_summary.csv
```

## Corpus configurations

`imdb_neg` (10,000 negative IMDB reviews, learning rate 5e-6, unmasked padding: an under-trained contrast), `imdb_bal` (5,000 positive and 5,000 negative), `imdb_bal_nohtml` (same, HTML tags removed), `yelp` (Yelp Review Full), `pubmed` (PubMedQA abstracts). The corpora are loaded from the Hugging Face Hub and are not redistributed; the per-generation files keep the scores of the held-out reference texts but not their text, which the scripts reload from the public sources.

## Procedures compared

* `dynalora`: logit-space scaling, two forward passes per token, lambda in {0, 0.25, ..., 3}.
* `weight`: the LoRA scaling alpha/r multiplied by lambda, one forward pass, same grid.
* `cfg`: classifier-free guidance on the base model with a one-sentence domain prefix, gamma in {1, 1.5, 2, 3}.
* `lora` / `base`: adapted and base models at temperatures 0.7 to 0.2 (entropy-matched ablation).

## Metrics

HTML artifact rate, generated length and EOS collapse, domain vocabulary score, perplexity of the continuation under GPT-2 large, probability of the target domain under a TF-IDF and logistic-regression classifier trained on held-out texts, positive probability under an SST-2 sentiment classifier, dist-1/2/3, mean sampling entropy, and, on a stratified sample, fluency and genre-fit ratings from two open LLM judges (Qwen2.5-7B-Instruct and OLMo-2-7B-Instruct). Without sampling, the likelihood scripts score 500 held-out domain texts and 500 WikiText-103 paragraphs under logit-space and weight-space scaling at every lambda, with the per-token KL divergence from the base model.

## Running

```
pip install -r requirements.txt
python scripts/run_all.py --out results/level2/gpt2 --models gpt2 --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed
python scripts/run_all.py --out results/level2/qwen --models Qwen/Qwen2.5-0.5B --configs imdb_bal
python scripts/bootstrap_ci.py --inputs 'results/level2/*/metrics__*.csv.gz' --out results/level2/stats
python scripts/llm_judge.py --inputs <gpt2>/generations_all_metrics.csv <qwen>/generations_all_metrics.csv --out results/level2/judge
python scripts/likelihood_tradeoff.py --model gpt2 --adapters_dir <gpt2>/adapters --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed --out results/level2/likelihood
python scripts/likelihood_tradeoff.py --model Qwen/Qwen2.5-0.5B --adapters_dir <qwen>/adapters --configs imdb_bal --batch_size 8 --out results/level2/likelihood
python scripts/likelihood_summary.py results/level2/likelihood
python scripts/run_all.py --smoke --out /tmp/smoke     # CPU smoke test with a random tiny model
```

All runs reported in the paper used a single NVIDIA T4 GPU (Python 3.13, PyTorch 2.11, transformers 5.20, peft 0.21, datasets 4.8, scikit-learn 1.6); `run_info.json` in each results folder records the arguments and the domain-classifier accuracy.
