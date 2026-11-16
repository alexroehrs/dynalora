"""Decoding throughput of the two forms of scaling.

Weight-space scaling runs the adapted network once per token and logit-space scaling runs it twice (adapter
disabled and enabled). This script measures what that costs: greedy decoding of a fixed number of new tokens
(stop tokens ignored, so that every configuration decodes the same number of tokens) for

* ``base``           the base model (factor 0);
* ``adapter``        the adapter as trained (factor 1), LoRA modules active and not merged;
* ``weight``         weight-space scaling at ``--factor`` (LoRA modules active and not merged);
* ``logit``          logit-space scaling at ``--factor`` (two passes, one key-value cache per pass);
* ``logit_plaus``    logit-space scaling at ``--factor_up`` with the plausibility constraint;
* ``weight_merged``  weight-space scaling at ``--factor`` with the scaled update merged into the weights.

For every configuration and batch size it writes to ``--out`` one row per repetition: ``seconds``,
``tokens`` (batch size times decoding steps), ``tokens_per_s``, ``peak_gb`` (peak GPU memory allocated) and
the GPU. The first batch of every configuration is decoded once before timing (warm-up).

Example (Colab, A100):
    python scripts/throughput.py --model Qwen/Qwen2.5-7B-Instruct --task samsum \
        --adapter_dir $ADAPTERS/Qwen2.5-7B-Instruct__samsum__s42 --out $RESULTS/level4/throughput/Qwen2.5-7B-Instruct.csv
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynalora import tasks as T  # noqa: E402
from dynalora.taskeval import TaskModel  # noqa: E402


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(tm, ctx, form, lam, bs, new_tokens, plausibility=0.0):
    """(seconds, tokens, peak GPU memory in GB) of one greedy decoding of ``ctx`` in batches of ``bs``."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    stats = {}
    _sync(); t0 = time.perf_counter()
    tm.greedy(ctx, form, lam, new_tokens, bs, plausibility=plausibility, stop=False, stats=stats)
    _sync(); dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else float('nan')
    return dt, stats['tokens'], peak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter_dir', required=True)
    ap.add_argument('--task', default='samsum', choices=sorted(T.TASKS))
    ap.add_argument('--out', required=True, help='CSV file')
    ap.add_argument('--factor', type=float, default=0.5)
    ap.add_argument('--factor_up', type=float, default=1.5)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--new_tokens', type=int, default=64)
    ap.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 16])
    ap.add_argument('--n_batches', type=int, default=4)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max_input_tokens', type=int, default=900)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if os.path.exists(a.out):
        log(f'{a.out}: finished, skipped'); return
    if a.smoke:
        from dynalora.smoke import smoke_chat_setup
        a.model = smoke_chat_setup(os.path.join(os.path.dirname(os.path.abspath(a.out)), '_smoke_model'))
        a.new_tokens, a.batch_sizes, a.n_batches, a.repeats, a.max_input_tokens = 4, [1, 2], 2, 1, 64
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tm = TaskModel(a.model, a.adapter_dir)
    tm.stop_ids = T.stop_ids(tm.tok)
    ex = T.load_examples(a.task, 'test', n=max(a.batch_sizes) * a.n_batches, smoke=a.smoke)
    ctx = [T.context_ids(tm.tok, a.task, e['input'], a.max_input_tokens)[0] for e in ex]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    configs = [('base', 'logit', 0.0, 0.0), ('adapter', 'logit', 1.0, 0.0), ('weight', 'weight', a.factor, 0.0),
               ('logit', 'logit', a.factor, 0.0), ('logit_plaus', 'logit', a.factor_up, a.alpha),
               ('weight_merged', 'logit', 0.0, 0.0)]
    rows = []
    for name, form, lam, plaus in configs:
        if name == 'weight_merged':          # last: merging replaces the adapted model
            tm.set_weight_scale(a.factor)
            tm.model = tm.model.merge_and_unload()
            tm.has_adapter, tm._lora_layers, tm._base_scaling = False, [], []
        for bs in a.batch_sizes:
            c = ctx[:bs * a.n_batches]
            measure(tm, c[:bs], form, lam, bs, a.new_tokens, plaus)          # warm-up
            for rep in range(a.repeats):
                dt, tokens, peak = measure(tm, c, form, lam, bs, a.new_tokens, plaus)
                rows.append(dict(model=os.path.basename(a.model.rstrip('/')), task=a.task, config=name,
                                 factor=a.factor if name == 'weight_merged' else lam, alpha=plaus, batch_size=bs,
                                 repeat=rep, new_tokens=a.new_tokens, contexts=len(c), seconds=round(dt, 4),
                                 tokens=tokens, tokens_per_s=round(tokens / dt, 2), peak_gb=round(peak, 2), gpu=gpu))
            last = [r['tokens_per_s'] for r in rows[-a.repeats:]]
            log(f'{name} batch {bs}: {sorted(last)[len(last) // 2]:.1f} tokens/s')
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == '__main__':
    main()
