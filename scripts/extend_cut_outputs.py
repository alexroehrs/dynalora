"""Regenerates, with a larger token budget, the generated answers that were cut by the budget.

``scripts/eval_task.py`` decodes at most ``max_new_tokens`` tokens (320 for ``gsm8k``). A model that writes
long solutions, the base model above all, can be cut before its final answer, which lowers its score for a
reason unrelated to the scaling factor. This script finds, in the score files of one adapter, the items whose
answer did not end (``ended`` false) at factors up to ``--max_factor`` (default 1: the base model, the
attenuated adapter and the adapter itself; above 1 unfinished answers are mostly degenerate and are left as
they are), decodes them again with ``--max_new_tokens`` (greedy decoding is deterministic, so the new answer
continues the old one) and scores them.

Nothing is overwritten. The regenerated rows go to ``<prefix>__<form>__ext.csv`` (same columns as the score
file plus ``budget``) and their texts to ``<prefix>__<form>__ext_outputs.jsonl``; ``scripts/analysis/summarize.py`` replaces the matching rows of the score file by these. The base model (factor 0) is done once
per model and task, in ``<base_cache>/<model>__<task>__n<items>__ext.csv`` (run the script without
``--adapter_dir``), and copied into the files of every adapter; factor 1 is decoded once and shared by the two
forms. Resumable: factors whose cut items are all present are skipped.

Example (Colab, A100):
    python scripts/extend_cut_outputs.py --model Qwen/Qwen2.5-7B-Instruct --task gsm8k --base_cache $S/_base
    python scripts/extend_cut_outputs.py --model Qwen/Qwen2.5-7B-Instruct --task gsm8k --base_cache $S/_base \
        --adapter_dir $A/Qwen2.5-7B-Instruct__gsm8k__s42 --out_dir $S/Qwen2.5-7B-Instruct__gsm8k__s42
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynalora import tasks as T  # noqa: E402
from dynalora.taskeval import TaskModel  # noqa: E402

DT = {'gold': str, 'pred': str, 'item': str}


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _read(path):
    return pd.read_csv(path, dtype=DT) if os.path.exists(path) else None


def _append(path, rows):
    pd.DataFrame(rows).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def cut_items(df, ext, factor, budget):
    """Items of ``factor`` whose answer did not end and that are not yet regenerated with ``budget``."""
    cut = df[(df.factor == factor) & (~df.ended.astype(bool))].item.astype(str).tolist()
    if ext is not None and len(ext):
        have = set(ext[(ext.factor == factor) & (ext.budget >= budget)].item.astype(str))
        cut = [i for i in cut if i not in have]
    return cut


class Regenerator:
    def __init__(self, a, examples):
        self.a, self.tm = a, None
        self.by_id = {str(e['id']): e for e in examples}
        self.spec = T.spec(a.task)

    def run(self, form, lam, items):
        if self.tm is None:
            self.tm = TaskModel(self.a.model, self.a.adapter_dir)
            self.tm.stop_ids = T.stop_ids(self.tm.tok)
        ex = [self.by_id[i] for i in items]
        ctx = [T.context_ids(self.tm.tok, self.a.task, e['input'], self.a.max_input_tokens)[0] for e in ex]
        gen = self.tm.greedy(ctx, form, lam, self.a.max_new_tokens, self.a.batch_size)
        rows, outs = [], []
        for e, (ids, ended) in zip(ex, gen):
            text = self.tm.tok.decode(ids, skip_special_tokens=True).strip()
            m = T.score_output(self.a.task, text, e['answer'])
            rows.append({'item': str(e['id']), **m, 'score': m[self.spec['metric']], 'n_tokens': len(ids),
                         'ended': bool(ended), 'rep2': T.repeated_bigram_share(ids), 'budget': self.a.max_new_tokens})
            outs.append({'item': str(e['id']), 'text': text})
        return rows, outs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter_dir', default=None, help='peft adapter; omit to regenerate the base model in --base_cache')
    ap.add_argument('--task', required=True, choices=[k for k, v in T.TASKS.items() if v['kind'] == 'generate'])
    ap.add_argument('--adapter_task', default=None, help='task the adapter was trained on (default: --task)')
    ap.add_argument('--out_dir', default=None, help='score directory of the adapter (required with --adapter_dir)')
    ap.add_argument('--base_cache', required=True)
    ap.add_argument('--max_new_tokens', type=int, default=768)
    ap.add_argument('--max_factor', type=float, default=1.0)
    ap.add_argument('--n_test', type=int, default=500)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_input_tokens', type=int, default=900)
    ap.add_argument('--model_tag', default=None)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    if a.smoke:
        from dynalora.smoke import smoke_chat_setup
        a.model = smoke_chat_setup(os.path.join(a.base_cache, '_smoke_model'))
        a.n_test, a.batch_size, a.max_input_tokens, a.max_new_tokens = 12, 4, 64, 16
    mtag = a.model_tag or os.path.basename(a.model.rstrip('/'))
    examples = T.load_examples(a.task, 'test', n=a.n_test, smoke=a.smoke)
    regen = Regenerator(a, examples)
    base_csv = os.path.join(a.base_cache, f'{mtag}__{a.task}__n{a.n_test}.csv')
    base_ext = base_csv.replace('.csv', '__ext.csv')

    def write(path, form, lam, rows, outs):
        _append(path, [{'task': a.task, 'form': form, 'factor': lam, **r} for r in rows])
        if outs:
            with open(path.replace('__ext.csv', '__ext_outputs.jsonl'), 'a') as f:
                for o in outs:
                    f.write(json.dumps({'factor': lam, **o}) + '\n')

    if a.adapter_dir is None:                     # the base model, once per model and task
        df = _read(base_csv)
        if df is None:
            raise SystemExit(f'{base_csv} not found (run scripts/eval_task.py first)')
        todo = cut_items(df, _read(base_ext), 0.0, a.max_new_tokens)
        if not todo:
            log(f'{base_ext}: nothing to regenerate'); return
        log(f'{mtag} {a.task} base model: {len(todo)} cut answers')
        rows, outs = regen.run('logit', 0.0, todo)
        write(base_ext, 'base', 0.0, rows, outs)
        log(f'base model: {sum(r["ended"] for r in rows)} of {len(rows)} now end; mean score of these items '
            f'{sum(r["score"] for r in rows) / len(rows):.3f}')
        return

    if not a.out_dir:
        raise SystemExit('--out_dir is required with --adapter_dir')
    own = a.adapter_task in (None, a.task)
    prefix = 'own' if own else f'cross_{a.task}'
    bext = _read(base_ext)
    shared = {}                                   # factor 1 is the adapter in both forms
    for form in ('logit', 'weight'):
        path = os.path.join(a.out_dir, f'{prefix}__{form}.csv')
        df = _read(path)
        if df is None:
            continue
        ext_path = path.replace('.csv', '__ext.csv')
        for lam in sorted(float(x) for x in df.factor.unique() if x <= a.max_factor):
            todo = cut_items(df, _read(ext_path), lam, a.max_new_tokens)
            if not todo:
                continue
            if lam == 0.0:
                if bext is None:
                    raise SystemExit(f'{base_ext} not found: run this script without --adapter_dir first')
                b = bext[bext.item.astype(str).isin(todo)]
                rows = b.drop(columns=['task', 'form', 'factor']).to_dict('records')
                write(ext_path, form, lam, rows, None)
                log(f'{prefix} {form} factor 0: {len(rows)} rows copied from the base model')
                continue
            if lam == 1.0 and 1.0 in shared and set(todo) <= {r['item'] for r in shared[1.0][0]}:
                rows = [r for r in shared[1.0][0] if r['item'] in set(todo)]
                outs = [o for o in shared[1.0][1] if o['item'] in set(todo)]
            else:
                rows, outs = regen.run(form, lam, todo)
                if lam == 1.0:
                    shared[1.0] = (rows, outs)
            write(ext_path, form, lam, rows, outs)
            log(f'{prefix} {form} factor {lam:g}: {len(rows)} cut answers, {sum(r["ended"] for r in rows)} now end, '
                f'mean score of these items {sum(r["score"] for r in rows) / len(rows):.3f}')
    log(f'{a.out_dir} {prefix}: done')


if __name__ == '__main__':
    main()
