# Attenuating and Amplifying LoRA Adapters at Inference Time: Logit-Space and Weight-Space Scaling on Tasks with Human Labels

Code for the paper: the DynaLoRA toolkit (`dynalora` package), the scripts and Colab notebooks that run the experiments and the analyses, and the tests. Results and data are not kept in the repository (see [Results](#results)).

The strength of a trained LoRA adapter can be changed at inference time in two ways, both controlled by a factor that gives the base model at 0 and the trained adapter at 1:

- **logit space**: decode from `y = y_base + lam * (y_lora - y_base)`, the next-token logits of the model without and with the adapter (two forward passes per token);
- **weight space**: multiply the LoRA update by `lam_w` before the forward pass (one pass per token).

The toolkit compares the two forms on the same adapters, test items and metrics, from attenuation (factor below 1) to amplification (above 1), and aligns them by their divergence from the base model instead of their nominal factors.

## Study

- **Models**: Qwen2.5-7B-Instruct (three training seeds) and Phi-4-mini-instruct, in bfloat16.
- **Tasks with human labels**: PubMedQA (accuracy), Yelp Review Full star rating (accuracy), GSM8K (exact match of the final number), SAMSum (ROUGE-L). Fixed training pools and 500 fixed test items per task.
- **Adapters**: LoRA rank 8 on the attention projections, one epoch, loss on the answer; on the first model also saved at 10%, 20% and 40% of the training steps.
- **Measures at every factor of both forms**: score on the adapter's own task, KL divergence to the base model, score on the other three tasks, accuracy on four general benchmarks (LAMBADA, HellaSwag, PIQA, ARC-Easy).

Summary of the results: amplification did not improve the score of any adapter in either form, including the adapters saved early in training; attenuation was graded in both forms but on very different scales of the factor; aligned at equal divergence from the base model, the two forms scored close to each other.

## Repository

| Path | Content |
|---|---|
| `dynalora/tasks.py` | tasks: loaders, fixed subsets, chat format, answer extraction, metrics |
| `dynalora/taskeval.py` | `TaskModel`: scoring and greedy decoding of an adapter under both forms of scaling (left-padded batches, key-value caches, all logit-space factors from one pair of passes) |
| `dynalora/models.py`, `dynalora/paths.py`, `dynalora/runinfo.py`, `dynalora/smoke.py` | model loading and LoRA targets, results root, session records, tiny random chat model for CPU runs |
| `scripts/train_task_chat.py` | trains one adapter on one task (optionally saving it at fractions of the steps) |
| `scripts/eval_task.py` | scores an adapter on its own task or on another task along both grids, with the KL divergence to the base model |
| `scripts/extend_cut_outputs.py` | decodes again, with a larger token budget, the generated answers that did not end |
| `scripts/zeroshot_bench.py` | general benchmarks (requests of the lm-evaluation-harness) under both forms |
| `scripts/analysis/summarize.py` | tables, figures and `SUMMARY.md` from the scores |
| `scripts/analysis/paper_assets.py` | tables and figures of the paper from the analysis tables |
| `notebooks/` | Colab notebooks (table below) |
| `tests/` | CPU tests on random tiny models (no network) |
| `docs/results_format.md` | layout and columns of the result files |

| Notebook | Content | GPU |
|---|---|---|
| `colab_pilot.ipynb` | small adapters (2,000 training examples) and the base models on 200 test items; checks each task and estimates the run time | A100 |
| `colab_run.ipynb` | the study: training, own-task scores over both grids, KL divergence, other tasks, general benchmarks, training seeds, second model, adapters saved during training | A100 |
| `colab_gsm8k_budget.ipynb` | GSM8K answers cut by the 320-token budget at factors up to 1, decoded again with 768 tokens | A100 |
| `colab_analysis.ipynb` | analyses, figures and `SUMMARY.md` | CPU |

The notebooks are resumable: finished adapters, factors and files are skipped when a notebook is run again.

## Reproducing

```
pip install -r requirements.txt
python -m pytest tests
A=$DYNALORA_RESULTS/level4/adapters; S=$DYNALORA_RESULTS/level4/scores; M=Qwen/Qwen2.5-7B-Instruct
python scripts/train_task_chat.py --model $M --task samsum --seed 42 --adapters_dir $A --ckpt_fracs 0.1 0.2 0.4
python scripts/eval_task.py --model $M --task samsum --adapter_dir $A/Qwen2.5-7B-Instruct__samsum__s42 --out_dir $S/Qwen2.5-7B-Instruct__samsum__s42 --base_cache $S/_base --kl_items 200
python scripts/eval_task.py --model $M --task gsm8k --adapter_task samsum --adapter_dir $A/Qwen2.5-7B-Instruct__samsum__s42 --out_dir $S/Qwen2.5-7B-Instruct__samsum__s42 --base_cache $S/_base --lambdas 0 0.5 1 1.5 2 --lambdas_weight 0 0.5 1 1.5 2
python scripts/zeroshot_bench.py --model $M --adapters_dir $A --configs samsum__s42 --dtype bfloat16 --limit 500 --lambdas 0 0.5 1 1.5 2 --out $DYNALORA_RESULTS/level4/bench
python scripts/analysis/summarize.py                                   # level4/analysis/: tables, figures, SUMMARY.md
python scripts/analysis/paper_assets.py $DYNALORA_RESULTS/level4/analysis out/   # tables and figures of the paper
python scripts/train_task_chat.py --smoke --model x --task gsm8k --adapters_dir /tmp/smoke/adapters   # CPU smoke run
```

The experiments ran in Google Colab on NVIDIA A100 GPUs (40 GB) with the pinned versions of `requirements.txt`. Every session records the GPU, the library versions, the seeds and the code revision in `run_info.json`.

## Results

Results are not kept in this repository. The notebooks write them to a results root (`MyDrive/dynalora_results` on Google Drive, or the directory given by `--results` or `DYNALORA_RESULTS`), in the layout described in [`docs/results_format.md`](docs/results_format.md). The data generated in the study (adapters, per-item scores, generated answers, divergences, benchmark results) are available from the authors on reasonable request.

The task datasets (PubMedQA, Yelp Review Full, GSM8K, SAMSum), the benchmark datasets and the models are third-party resources. They are loaded from the Hugging Face Hub at run time, are not redistributed here, and remain under their own licenses.

## License

MIT (see `LICENSE`).
