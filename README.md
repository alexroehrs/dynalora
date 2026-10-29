# Inference-Time Strength Control of LoRA Adapters: Comparison of Logit-Space Scaling, Weight-Space Scaling, Temperature, and Prompt Guidance

Code, generated texts, and per-generation scores for the paper. DynaLoRA scales the contribution of a LoRA adapter at decoding time as

```
y = y_base + lambda * (y_lora - y_base)
```

where `y_base` and `y_lora` are the next-token logits of the same model with the adapter disabled and enabled. The study sweeps lambda from 0 to 3 on six adapters (five on GPT-2 small, one on Qwen2.5-0.5B), on a learning-rate sweep, and on a replication with two further training seeds and on Qwen2.5-1.5B and Qwen2.5-7B (4-bit), and compares DynaLoRA with weight-space scaling of the LoRA update, classifier-free guidance with a domain prefix, and the plain adapter sampled at matched entropy, on generation metrics, LLM-judge ratings, teacher-forced likelihood, and zero-shot benchmarks.

## Layout

| Path | Content |
|---|---|
| `dynalora/` | package: corpus configurations, LoRA training, generation procedures, metrics, aggregation |
| `scripts/run_all.py` | end-to-end run: train adapters, generate, score, summarize (`--methods`, `--adapters_dir`, `--clf_configs`; configuration suffixes `__lr<value>` and `__seed<n>` override the learning rate and the training seed) |
| `scripts/likelihood_tradeoff.py`, `scripts/likelihood_summary.py` | teacher-forced likelihood and KL divergence per lambda on domain text, general text, and negative IMDB reviews |
| `scripts/rescore.py` | re-scoring with SmolLM2-1.7B and a unified domain classifier |
| `scripts/zeroshot_bench.py` | zero-shot LAMBADA, HellaSwag, PIQA, ARC-Easy for both forms of scaling (requests from `lm-evaluation-harness`) |
| `scripts/bootstrap_ci.py` | prompt-clustered bootstrap intervals and Holm-adjusted paired tests |
| `scripts/llm_judge.py` | LLM-judge ratings (fluency, genre fit) on a stratified sample |
| `scripts/analysis/` | analyses and figures of the paper (see its README) |
| `notebooks/colab_run_all.ipynb`, `notebooks/colab_revision.ipynb`, `notebooks/colab_replication.ipynb` | drivers used on a Colab T4 GPU |
| `notebooks/colab_pubmedqa.ipynb` | downstream task: PubMedQA accuracy along lambda (`scripts/zeroshot_bench.py --tasks pubmedqa_labeled`) |
| `notebooks/colab_qwen7b.ipynb` | Qwen2.5-7B with a 4-bit NF4 base model (QLoRA, `--load_in_4bit`): training, likelihood, benchmarks, reduced generation grid, re-scoring and MAUVE |
| `results/level2/gpt2/`, `results/level2/qwen/` | main runs: summary tables, per-generation scores |
| `results/level2/sweep/` | learning-rate sweep on balanced IMDB (DynaLoRA only) |
| `results/level2/seeds/` | balanced IMDB and PubMed adapters of GPT-2 retrained with seeds 43 and 44 (DynaLoRA and weight-space scaling) |
| `results/level2/qwen15/` | balanced IMDB and PubMed adapters on Qwen2.5-1.5B (DynaLoRA and weight-space scaling) |
| `results/level2/qwen7b/` | balanced IMDB and PubMed adapters on Qwen2.5-7B with 4-bit weights (DynaLoRA and weight-space scaling, lambda in {0, 0.5, 1, 1.5, 2, 3}) |
| `results/level2/temps_gpt2/`, `results/level2/temps_qwen/` | plain adapter on a 0.05-step temperature grid |
| `results/level2/rescore/` | per-generation scores with SmolLM2-1.7B perplexity (`ext2_*`) and unified domain probability (`domain_prob_u`) |
| `results/level2/likelihood/` | per-text likelihoods (`ll__` domain, `llgen__` general, `llneg__` negative IMDB reviews) and `ll_summary.csv` |
| `results/level2/bench/` | per-document benchmark scores for every method and lambda |
| `results/level2/pubmedqa/` | per-question PubMedQA scores (PQA-L, 1,000 questions) of the PubMed and balanced IMDB adapters of GPT-2, Qwen2.5-1.5B and Qwen2.5-7B, for every method and lambda (`notebooks/colab_pubmedqa.ipynb`) |
| `results/level2/stats/` | bootstrap intervals, paired tests, LLM-judge ratings |
| `results/level2/analysis/` | derived tables (equal divergence, entropy matching, formatting rates, paired tests on SmolLM2 perplexity, replication, PubMedQA) |

## Data

The corpora (IMDB, Yelp Review Full, PubMedQA, WikiText-103) and benchmark datasets are loaded from the Hugging Face Hub and are not redistributed. The per-generation files keep the scores of the held-out reference texts but not their text, which the scripts reload from the public sources; reference formatting rates in `results/level2/analysis/format_rates.csv` were computed before the text was removed.

## Running

```
pip install -r requirements.txt lm-eval
python scripts/run_all.py --out results/level2/gpt2 --models gpt2 --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed
python scripts/run_all.py --out results/level2/qwen --models Qwen/Qwen2.5-0.5B --configs imdb_bal
python scripts/run_all.py --out results/level2/sweep --models gpt2 --configs imdb_bal__lr5e-06 imdb_bal__lr1.5e-05 imdb_bal__lr5e-05 imdb_bal__lr1e-04 --methods dynalora --clf_configs imdb_bal yelp pubmed
python scripts/run_all.py --out results/level2/temps_gpt2 --adapters_dir results/level2/gpt2/adapters --models gpt2 --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed --methods lora_temp --temps 0.65 0.55 0.45 0.35 0.25
python scripts/likelihood_tradeoff.py --model gpt2 --adapters_dir results/level2/gpt2/adapters --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed --out results/level2/likelihood
python scripts/likelihood_summary.py results/level2/likelihood
python scripts/rescore.py --inputs "results/level2/*/generations_all_metrics.csv" --out results/level2/rescore
python scripts/zeroshot_bench.py --model gpt2 --adapters_dir results/level2/gpt2/adapters --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed --out results/level2/bench
python scripts/bootstrap_ci.py --inputs "results/level2/gpt2/metrics__*.csv.gz" "results/level2/qwen/metrics__*.csv.gz" --out results/level2/stats
python scripts/run_all.py --out results/level2/seeds --models gpt2 --configs imdb_bal__seed43 imdb_bal__seed44 pubmed__seed43 pubmed__seed44 --methods dynalora weight --clf_configs imdb_bal yelp pubmed
python scripts/run_all.py --out results/level2/qwen15 --models Qwen/Qwen2.5-1.5B --configs imdb_bal pubmed --methods dynalora weight --clf_configs imdb_bal yelp pubmed
python scripts/zeroshot_bench.py --model Qwen/Qwen2.5-1.5B --adapters_dir results/level2/qwen15/adapters --configs imdb_bal pubmed --out results/level2/bench --batch_size 8 --limit 500 --lambdas 0 0.5 1 1.5 2 2.5 3
# Qwen2.5-7B with 4-bit weights (NVIDIA L4; see notebooks/colab_qwen7b.ipynb)
python scripts/run_all.py --out results/level2/qwen7b --models Qwen/Qwen2.5-7B --configs imdb_bal pubmed --load_in_4bit --train_only
python scripts/likelihood_tradeoff.py --model Qwen/Qwen2.5-7B --adapters_dir results/level2/qwen7b/adapters --configs imdb_bal pubmed --load_in_4bit --batch_size 4 --out results/level2/likelihood
python scripts/zeroshot_bench.py --model Qwen/Qwen2.5-7B --adapters_dir results/level2/qwen7b/adapters --configs imdb_bal pubmed --load_in_4bit --batch_size 4 --limit 500 --lambdas 0 0.5 1 1.5 2 2.5 3 --out results/level2/bench
python scripts/run_all.py --out results/level2/qwen7b --models Qwen/Qwen2.5-7B --configs imdb_bal pubmed --load_in_4bit --methods dynalora weight --lambdas 0 0.5 1 1.5 2 3 --clf_configs imdb_bal yelp pubmed --n_samples 20
python scripts/analysis/replication.py
# PubMedQA along lambda (see notebooks/colab_pubmedqa.ipynb for the three models)
python scripts/zeroshot_bench.py --model gpt2 --adapters_dir results/level2/gpt2/adapters --configs pubmed imdb_bal --tasks pubmedqa_labeled --max_len 1024 --batch_size 8 --out results/level2/pubmedqa
python scripts/analysis/pubmedqa.py --inputs results/level2/pubmedqa --by_answer
python scripts/run_all.py --smoke --out /tmp/smoke   # CPU smoke test with a random tiny model
```

All runs reported in the paper used a single NVIDIA T4 GPU, except Qwen2.5-7B (NVIDIA L4, 4-bit NF4 base weights with bitsandbytes) (Python 3.13, PyTorch 2.11, transformers 5.20, peft 0.21, datasets 4.8, scikit-learn 1.6, lm-eval 0.4); `run_info.json` in each results folder records the arguments and the domain-classifier accuracy.
