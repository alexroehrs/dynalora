# Results format

Results are not kept in this repository. The scripts and notebooks read and write a results root, by default the Google Drive folder `MyDrive/dynalora_results` (see `dynalora/paths.py`: `--results DIR`, or the environment variable `DYNALORA_RESULTS`). The data generated in the study are available from the authors on reasonable request. Third-party datasets and models are never stored: they are loaded from the Hugging Face Hub at run time.

All outputs are under `level4/` of the results root. They are written by `notebooks/colab_pilot.ipynb`, `notebooks/colab_run.ipynb` and `notebooks/colab_gsm8k_budget.ipynb`, and analysed by `notebooks/colab_analysis.ipynb`. Adapter and score directories are named `<model>__<task>__s<seed>`, with the suffix `__ckpt<pct>` for an adapter saved at that percentage of the optimizer steps. Tasks: `pubmedqa`, `yelp5`, `gsm8k`, `samsum` (`dynalora/tasks.py`).

### `adapters/`

One peft adapter per directory (`scripts/train_task_chat.py`), with the tokenizer and `train_log.json`: the arguments, `n_train`, `n_val`, `truncated_inputs`, `steps`, `lora_targets`, `trainable_parameters`, `saved_at_step`, `val_loss` (list of `[step, loss]`, step 0 is the base model) and `seconds`.

### `scores/`

Written by `scripts/eval_task.py`, one directory per adapter.

- `own__<form>.csv` (`form` is `logit` or `weight`): one row per test item and factor on the adapter's own task. Columns: `task`, `form`, `factor`, `item`, `gold`, `pred`, `score` (main metric of the task: 1 or 0 for accuracy and exact match, ROUGE-L F1 for `samsum`). Multiple choice adds `p_gold`, the probability of the gold option among the options. Generation adds the metric columns (`em`, or `rouge1`, `rouge2`, `rougeL`), `n_tokens` (generated tokens before the stop token), `ended` (a stop token was generated within the budget) and `rep2` (share of repeated bigrams of the generated tokens).
- `own__<form>__outputs.jsonl`: `{factor, item, text}`, the generated answers (generation tasks).
- `cross_<task>__<form>.csv` and `cross_<task>__<form>__outputs.jsonl`: the same for the adapter scored on another task.
- `kl__<form>.csv`: per item and factor, `nll` (negative log-likelihood of the reference answer under teacher forcing), `n_tok` (answer tokens, end-of-turn token included) and `kl_sum` (KL(p_factor || p_base) summed over the answer positions, temperature 1).
- `<prefix>__<form>__ext.csv` and `<prefix>__<form>__ext_outputs.jsonl` (`scripts/extend_cut_outputs.py`): generated answers that did not end within the token budget, at factors up to 1, decoded again with a larger budget. Same columns as the score file plus `budget`; the analysis uses these rows in place of the rows of the same factor and item. `_base/<model>__<task>__n<items>__ext.csv` holds the base model.
- `timing.csv`: `what`, `task`, `form`, `factor`, `items`, `seconds`, `gpu` of every unit of work.
- `_base/<model>__<task>__n<items>.csv` (and `__outputs.jsonl`): the base model (factor 0), computed once per model and task and copied into the files of every adapter.

### `bench/`

`bench__<model>__<task>__s<seed>.csv.gz` (`scripts/zeroshot_bench.py`), one row per method (`logit`, `weight`), `lambda`, `task` and `doc_id`, with `acc` and `acc_norm`: the four general benchmarks on the first 500 documents at factors 0, 0.5, 1, 1.5 and 2 in both forms.

### `pilot/`

`adapters/` and `scores/` as above for the pilot (2,000 training examples, 200 test items, factors 0 and 1) and `pilot_summary.csv`: per model and task, `base`, `adapter`, `diff` with `diff_lo`, `diff_hi` (95% paired bootstrap interval over items), `keep` (the adapter improves the main metric by at least 3 points, or 2 points of ROUGE-L), the training time and the scoring time per item.

### `analysis/`

Written by `scripts/analysis/summarize.py`; the tables are described at the top of the script (`curves.csv`, `curves_seeds.csv`, `cross.csv`, `bench.csv`, `frontier.csv`, `kl.csv`, `equal_divergence.csv`, `attenuation.csv`, `amplification.csv`, `truncated.csv`), with the figures `fig_task_curves__<model>.pdf`, `fig_task_frontier__<model>.pdf`, `fig_task_truncated.pdf` and `SUMMARY.md`. Intervals are 95% percentile bootstrap intervals over test items (2,000 resamples, seed 0), paired for differences.
