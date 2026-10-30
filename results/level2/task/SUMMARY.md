# Task adapter `pubmedqa_task` and decoding latency: summary

Task adapter: one LoRA adapter per model (GPT-2, Qwen2.5-1.5B) trained on the PubMedQA question-answering task itself (`scripts/train_task_adapter.py`, `notebooks/colab_task_adapter.ipynb`), evaluated along lambda for weight-space and logit-space scaling on the 500 test questions of the labeled set (harness task `pubmedqa_test500`, yes/no/maybe scored by option log-likelihood). Lambda grid: 0, 0.25, ..., 3 for GPT-2 and 0, 0.5, ..., 3 for Qwen2.5-1.5B on the task; general benchmarks on the first 1,000 (GPT-2) or 500 (Qwen2.5-1.5B) documents of each of LAMBADA, HellaSwag (acc_norm), PIQA and ARC-Easy; general-text KL on the 0.25 grid for both models. All numbers come from `results/level2/analysis/task_adapter_curves.csv`, `task_adapter_by_answer.csv`, `task_adapter_eq.csv` (`scripts/analysis/task_adapter.py`, 2,000 bootstrap resamples, seed 0) and `results/level2/latency/`. Accuracies in %, intervals are 95% bootstrap over questions (paired for differences). Lambda 0 is the base model and lambda 1 the plain adapter, identical for both forms.

## Accuracy on the PubMedQA test half along lambda

Qwen2.5-1.5B (n = 500):

| lambda | logit | weight |
|---|---|---|
| 0 | 65.0 [61.0, 69.0] | 65.0 [61.0, 69.0] |
| 0.5 | 74.6 [70.6, 78.4] | 74.8 [71.0, 78.6] |
| 1 | 75.2 [71.4, 78.8] | 75.2 [71.4, 78.8] |
| 1.5 | 74.6 [70.8, 78.4] | 74.2 [70.2, 78.0] |
| 2 | 70.0 [65.8, 74.2] | 74.0 [70.0, 77.8] |
| 2.5 | 59.0 [54.6, 63.2] | 73.6 [69.6, 77.4] |
| 3 | 46.0 [41.6, 50.6] | 73.0 [69.0, 77.0] |

The adapter raises accuracy by 10.2 points over the base model (paired [6.8, 13.8]). Beyond lambda 1 neither form improves on the plain adapter; weight-space accuracy stays within the interval of lambda 1 up to 3 (change from lambda 1 at 3: -2.2 [-5.0, 0.4]), while logit-space accuracy falls from lambda 2 (-5.2 [-8.2, -2.2]) to 46.0 at lambda 3 (-29.2 [-34.0, -24.0]).

GPT-2 (n = 500):

| lambda | logit | weight |
|---|---|---|
| 0 | 44.8 [40.6, 49.4] | 44.8 [40.6, 49.4] |
| 0.25 | 52.0 [47.6, 56.4] | 52.8 [48.6, 57.0] |
| 0.5 | 57.4 [53.0, 61.6] | 56.4 [51.8, 60.6] |
| 0.75 | 55.0 [50.6, 59.2] | 54.8 [50.4, 59.0] |
| 1 | 54.2 [49.6, 58.4] | 54.2 [49.6, 58.4] |
| 1.25 | 54.2 [49.6, 58.4] | 54.2 [49.6, 58.4] |
| 1.5 | 54.2 [49.6, 58.4] | 54.2 [49.6, 58.4] |
| 1.75 | 54.2 [49.6, 58.4] | 54.4 [50.0, 58.6] |
| 2 | 54.2 [49.6, 58.4] | 54.4 [50.0, 58.6] |
| 2.25 | 54.2 [49.6, 58.4] | 54.2 [49.6, 58.4] |
| 2.5 | 53.8 [49.2, 58.0] | 53.4 [48.8, 57.6] |
| 2.75 | 53.6 [49.2, 57.8] | 53.4 [49.0, 57.6] |
| 3 | 53.4 [49.0, 57.6] | 51.2 [46.8, 55.4] |

The plain adapter is 9.4 points above the base model ([2.4, 16.0]), but at lambda 1 it answers yes to nearly every question (next section): 54.2% is close to the 54.0% share of yes answers in the test half. The best point of both forms is lambda 0.5 (logit 57.4, +3.2 [0.8, 5.8] over lambda 1; weight 56.4, +2.2 [0.2, 4.4]); from lambda 1 to 3 both forms stay at the always-yes plateau.

## Accuracy by gold answer

Test half: 270 yes (54.0%), 174 no (34.8%), 56 maybe (11.2%). Accuracy within each answer:

| model | answer | base (lambda 0) | plain adapter (lambda 1) |
|---|---|---|---|
| Qwen2.5-1.5B | yes | 95.6 | 94.1 |
| Qwen2.5-1.5B | no | 38.5 | 70.1 |
| Qwen2.5-1.5B | maybe | 0.0 | 0.0 |
| GPT-2 | yes | 30.0 | 100.0 |
| GPT-2 | no | 82.2 | 0.6 |
| GPT-2 | maybe | 0.0 | 0.0 |

For Qwen2.5-1.5B the gain of the adapter comes from the no questions (38.5 to 70.1) with yes unchanged, so it reflects a better reading of the abstracts rather than a shift of the answer prior. Neither model answers maybe at lambda 1 (`pqa_artificial` has no maybe answers). For GPT-2 the adapter moves the base model from mostly no to always yes, a change of answer prior. Past lambda 1 in logit space, Qwen2.5-1.5B starts to answer maybe (maybe 26.8 / no 60.3 / yes 85.2 at lambda 2; 53.6 / 32.2 / 53.3 at lambda 3), while weight space keeps the lambda 1 pattern (maybe 3.6 / no 60.9 / yes 95.2 at lambda 3).

## General benchmarks (equal-weight mean of LAMBADA, HellaSwag, PIQA, ARC-Easy)

| lambda | Qwen2.5-1.5B logit | Qwen2.5-1.5B weight | GPT-2 logit | GPT-2 weight |
|---|---|---|---|---|
| 0 | 68.6 [66.4, 70.6] | 68.6 [66.4, 70.6] | 43.8 [42.4, 45.4] | 43.8 [42.4, 45.4] |
| 0.5 | 68.8 [66.8, 70.8] | 68.9 [66.9, 70.9] | 42.4 [41.0, 44.0] | 42.5 [41.0, 44.0] |
| 1 | 68.7 [66.7, 70.7] | 68.7 [66.7, 70.7] | 40.0 [38.5, 41.4] | 40.0 [38.5, 41.4] |
| 1.5 | 68.0 [66.0, 70.0] | 68.4 [66.4, 70.3] | 38.2 [36.8, 39.6] | 37.2 [35.8, 38.6] |
| 2 | 67.0 [65.0, 68.9] | 67.1 [65.1, 69.0] | 35.1 [33.8, 36.5] | 33.8 [32.5, 35.2] |
| 2.5 | 65.3 [63.3, 67.3] | 64.4 [62.4, 66.5] | 33.1 [31.8, 34.4] | 31.2 [30.0, 32.6] |
| 3 | 64.0 [62.0, 66.0] | 61.6 [59.5, 63.6] | 30.9 [29.6, 32.2] | 29.4 [28.1, 30.7] |

n = 2,000 documents for Qwen2.5-1.5B and 4,000 for GPT-2 (bootstrap stratified within benchmark); the GPT-2 quarter points are in `task_adapter_curves.csv`. The Qwen2.5-1.5B adapter costs no general accuracy at lambda 1 (+0.2 [-0.8, 1.2] from the base model); the GPT-2 adapter costs 3.9 points ([-4.9, -3.0]), mostly on LAMBADA (31.5 to 21.7). Both forms degrade beyond lambda 1, weight space faster at the high end (lambda 3: -7.1 vs -4.7 points from lambda 1 for Qwen2.5-1.5B, -10.5 vs -9.0 for GPT-2).

## Logit vs weight at equal general-text KL

For each weight-space lambda (`lam_w`), the logit-space lambda (`lam_star`) with the same mean per-token KL from the base model on general text; difference logit minus weight with 95% paired bootstrap interval. At `lam_w` = 3 the weight-space KL exceeds the largest logit-space KL of the grid, so `lam_star` is clamped to 3 and the comparison is not at equal KL (the logit point has less divergence).

| model | lam_w | KL | lam_star | PubMedQA logit / weight | diff | general mean logit / weight | diff |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1.5 | 0.028 | 1.53 | 74.4 / 74.2 | +0.2 [-1.6, 2.0] | 68.0 / 68.4 | -0.4 [-1.2, 0.3] |
| Qwen2.5-1.5B | 2 | 0.061 | 2.07 | 68.5 / 74.0 | -5.5 [-8.9, -2.3] | 66.7 / 67.1 | -0.4 [-1.5, 0.7] |
| Qwen2.5-1.5B | 3 | 0.192 | 3 (clamped) | 46.0 / 73.0 | -27.0 [-32.2, -21.8] | 64.0 / 61.6 | +2.4 [0.8, 4.0] |
| GPT-2 | 1.5 | 0.521 | 1.53 | 54.2 / 54.2 | 0.0 [0.0, 0.0] | 38.0 / 37.2 | +0.8 [0.3, 1.3] |
| GPT-2 | 2 | 1.462 | 2.15 | 54.2 / 54.4 | -0.2 [-0.6, 0.0] | 34.6 / 33.8 | +0.8 [0.0, 1.5] |
| GPT-2 | 3 | 7.122 | 3 (clamped) | 53.4 / 51.2 | +2.2 [-0.2, 4.6] | 30.9 / 29.4 | +1.5 [0.5, 2.5] |

At equal general-text KL the two forms do not differ on the general benchmarks for Qwen2.5-1.5B within the grid, and logit space is slightly ahead for GPT-2 (+0.8 points at `lam_w` 1.5 and 2). On the task itself, logit space is behind for Qwen2.5-1.5B from `lam_w` = 2 (-5.5 points), while for GPT-2 both forms sit on the same always-yes plateau.

## Decoding latency

`scripts/latency.py`, PubMed adapters of the main runs, batch of 20 prompts, 50 new tokens, fp16 on a Colab GPU, with KV caches (two caches for logit space); median ms per generated token (the whole batch per step divided by the batch size), peak memory in MB.

| model | base | plain adapter (lambda 1) | weight (lambda 2) | logit (lambda 2) | logit / weight | peak memory base / weight / logit |
|---|---|---|---|---|---|---|
| GPT-2 | 1.000 | 1.004 | 0.997 | 1.967 | 1.97 | 311 / 311 / 360 |
| Qwen2.5-1.5B | 3.210 | 3.239 | 3.220 | 6.381 | 1.98 | 3,039 / 3,039 / 3,092 |

Weight-space scaling costs the same as the base model; logit-space scaling, two forward passes per step, about twice as much (x1.97 and x1.98), with 16% (GPT-2) and 2% (Qwen2.5-1.5B) more peak memory for the second KV cache.

## Files

`results/level2/task/pubmedqa/`, `bench/`, `likelihood/`: per-question and per-text scores (doc_id or text_id, method, lambda, accuracy or NLL/KL), no question text, abstracts or gold labels. The training logs of the adapters (`adapters/*/train_log.txt`, `train_info.json`) were not included in the exported archive (the notebook's zip step excludes `task/adapters/`) and are not in the repository.
