"""Logit-space amplification of a task adapter with the plausibility constraint of contrastive decoding.

``scripts/eval_task.py`` applies logit-space scaling as written (Equation 1): above factor 1 the logits of the
base model enter with a negative weight and any token can be chosen. Contrastive decoding methods restrict
the choice to the tokens that the stronger model finds plausible. This script repeats the greedy decoding of
a generation task (``gsm8k``, ``samsum``) at factors above 1 with that constraint: at every step the next
token is the argmax of Equation 1 among the tokens whose probability under the adapted model is at least
``--alpha`` times that of its most likely token. The multiple-choice tasks are scored by the likelihood of
fixed options, to which the constraint does not apply.

Outputs in ``--out_dir`` (the score directory of the adapter): ``plaus__logit.csv`` (the columns of
``own__logit.csv`` plus ``alpha``) and ``plaus__logit__outputs.jsonl``. Resumable: factors already complete
for the given ``--alpha`` are skipped. ``scripts/analysis/summarize.py`` compares every point with factor 1
and with the unconstrained point of the same factor.

Example (Colab, A100):
    python scripts/eval_plausibility.py --model Qwen/Qwen2.5-7B-Instruct --task samsum \
        --adapter_dir $ADAPTERS/Qwen2.5-7B-Instruct__samsum__s42 --out_dir $SCORES/Qwen2.5-7B-Instruct__samsum__s42
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynalora import tasks as T  # noqa: E402
from dynalora.taskeval import TaskModel  # noqa: E402
from eval_task import Evaluator, _append, log  # noqa: E402

LAMBDAS = [1.05, 1.1, 1.15, 1.25, 1.5, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter_dir', required=True)
    ap.add_argument('--task', required=True, choices=[t for t in sorted(T.TASKS) if T.spec(t)['kind'] != 'choice'])
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--alpha', type=float, default=0.1, help='plausibility threshold, in (0, 1]')
    ap.add_argument('--lambdas', nargs='+', type=float, default=LAMBDAS, help='logit-space factors above 1')
    ap.add_argument('--n_test', type=int, default=500)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_input_tokens', type=int, default=900)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if any(lam <= 1 for lam in a.lambdas):
        raise SystemExit('the constraint only matters above factor 1 (at factor 1 it never excludes the chosen token)')
    if a.smoke:
        from dynalora.smoke import smoke_chat_setup
        a.model = smoke_chat_setup(os.path.join(a.out_dir, '_smoke_model'))
        a.n_test, a.batch_size, a.max_input_tokens, a.lambdas = 12, 4, 64, [1.5, 2.0]
        T.TASKS['gsm8k']['max_new_tokens'] = T.TASKS['samsum']['max_new_tokens'] = 8
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, 'plaus__logit.csv')
    examples = T.load_examples(a.task, 'test', n=a.n_test, smoke=a.smoke)
    n = len(examples)
    done = set()
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={'gold': str, 'pred': str, 'item': str})
        c = df[(df.alpha - a.alpha).abs() < 1e-9].groupby('factor').size()
        done = {float(f) for f, k in c.items() if k >= n}
    todo = [float(x) for x in a.lambdas if float(x) not in done]
    if not todo:
        log(f'{a.out_dir} plausibility {a.alpha:g}: finished, skipped'); return

    tm = TaskModel(a.model, a.adapter_dir)
    tm.stop_ids = T.stop_ids(tm.tok)
    ev = Evaluator(tm, a.task, examples, a.batch_size, a.max_input_tokens)
    log(f'{os.path.basename(a.out_dir)} {a.task}: {n} test items, plausibility {a.alpha:g}, factors {todo}')
    for lam in todo:
        t0 = time.time()
        rows, outs = ev.generate('logit', lam, plausibility=a.alpha)
        _append(path, [{'task': a.task, 'form': 'logit', 'alpha': a.alpha, 'factor': lam, **r} for r in rows])
        with open(path.replace('.csv', '__outputs.jsonl'), 'a') as f:
            for o in outs:
                f.write(json.dumps({'alpha': a.alpha, 'factor': lam, **o}) + '\n')
        _append(os.path.join(a.out_dir, 'timing.csv'),
                [{'what': 'plausibility', 'task': a.task, 'form': 'logit', 'factor': lam, 'items': n,
                  'seconds': round(time.time() - t0, 1),
                  'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}])
        log(f'factor {lam:g}: mean score {sum(r["score"] for r in rows) / len(rows):.4f}, '
            f'ended {sum(r["ended"] for r in rows) / len(rows):.2f}')


if __name__ == '__main__':
    main()
