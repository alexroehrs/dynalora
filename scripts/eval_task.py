"""Score of a task adapter on a task of ``dynalora.tasks`` as a function of the scaling factor.

For each form of scaling (``logit``: Equation 1 on the logits; ``weight``: LoRA scaling multiplied by the
factor) and each factor of its grid, every test item is scored against its human label:

* multiple-choice tasks (``pubmedqa``, ``yelp5``): the option with the highest log-likelihood as the
  assistant turn is the prediction (no sampling); all logit-space factors come from one pair of passes;
* generation tasks (``gsm8k``, ``samsum``): greedy decoding, then exact match of the final number or
  ROUGE against the reference.

The adapter can be evaluated on its own task (``--prefix own``, the default) or on another task
(``--prefix cross_<task>`` is set automatically when ``--task`` differs from ``--adapter_task``).

Outputs in ``--out_dir`` (one directory per adapter):

* ``<prefix>__<form>.csv``: one row per item and factor (``task, form, factor, item, gold, pred, score``,
  the metric columns, and for generated answers ``n_tokens, ended, rep2``); ``score`` is the main metric
  of the task (accuracy, exact match, or ROUGE-L);
* ``<prefix>__<form>__outputs.jsonl``: the generated answers (generation tasks);
* ``kl__<form>.csv`` (own task, ``--kl_items`` > 0): per item and factor, the negative log-likelihood of
  the reference answer, its number of tokens, and the summed KL(p_factor || p_base) over its positions
  under teacher forcing, at temperature 1;
* ``timing.csv``: seconds and items of every unit of work.

Factor 0 is the base model in both forms: it is computed once per model and task and kept in
``--base_cache`` (shared by all adapters). Factor 1 is the adapter itself in both forms and is computed once.
The script is resumable: factors already complete in a CSV are skipped.

``--smoke`` runs a random tiny chat model on synthetic items (``--adapter_dir`` from
``train_task_chat.py --smoke``).

Example (Colab, A100):
    python scripts/eval_task.py --model Qwen/Qwen2.5-7B-Instruct --task gsm8k \
        --adapter_dir $ADAPTERS/Qwen2.5-7B-Instruct__gsm8k__s42 --out_dir $SCORES/Qwen2.5-7B-Instruct__gsm8k__s42 \
        --base_cache $SCORES/_base
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
from dynalora import tasks as T  # noqa: E402
from dynalora.taskeval import TaskModel  # noqa: E402

LAMBDAS_LOGIT = [0, 0.25, 0.5, 0.75, 1, 1.05, 1.1, 1.15, 1.25, 1.5, 2, 3]
LAMBDAS_WEIGHT = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def _done(path, n_items):
    """Factors with all ``n_items`` rows in a CSV, and the CSV (or None)."""
    if not os.path.exists(path):
        return set(), None
    df = pd.read_csv(path, dtype={'gold': str, 'pred': str, 'item': str})
    c = df.groupby('factor').size()
    return {float(f) for f, k in c.items() if k >= n_items}, df


def _append(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


class Evaluator:
    def __init__(self, tm, task, examples, batch_size, max_input_tokens):
        self.tm, self.task, self.ex, self.bs = tm, task, examples, batch_size
        self.spec = T.spec(task)
        enc = [T.context_ids(tm.tok, task, e['input'], max_input_tokens) for e in examples]
        self.ctx = [c for c, _ in enc]
        self.n_truncated = sum(t for _, t in enc)
        if self.spec['kind'] == 'choice':
            self.opts = [T.answer_ids(tm.tok, o, eot=False) for o in self.spec['options']]

    # rows without task/form/factor; the caller adds them
    def _choice_rows(self, scores):
        rows = []
        for e, s in zip(self.ex, scores):
            lp = torch.log_softmax(torch.tensor(s), 0)
            k = int(lp.argmax())
            pred = self.spec['options'][k]
            gi = self.spec['options'].index(e['answer'])
            rows.append({'item': e['id'], 'gold': e['answer'], 'pred': pred, 'score': float(pred == e['answer']),
                         'p_gold': float(lp[gi].exp())})
        return rows

    def choice(self, form, lambdas):
        """{factor: rows} for a multiple-choice task; ``form='weight'`` takes one factor at a time."""
        if form == 'logit':
            res = self.tm.choice_scores(self.ctx, self.opts, 'logit', lambdas, self.bs)
            return {lam: self._choice_rows([r[lam] for r in res]) for lam in lambdas}
        out = {}
        for lam in lambdas:
            self.tm.set_weight_scale(lam)
            try:
                res = self.tm.choice_scores(self.ctx, self.opts, 'weight', None, self.bs)
            finally:
                self.tm.reset_weight_scale()
            out[lam] = self._choice_rows([r[None] for r in res])
        return out

    def generate(self, form, lam):
        """(rows, outputs) of a generation task at one factor."""
        gen = self.tm.greedy(self.ctx, form, lam, self.spec['max_new_tokens'], self.bs)
        rows, outs = [], []
        for e, (ids, ended) in zip(self.ex, gen):
            text = self.tm.tok.decode(ids, skip_special_tokens=True).strip()
            m = T.score_output(self.task, text, e['answer'])
            rows.append({'item': e['id'], **m, 'score': m[self.spec['metric']], 'n_tokens': len(ids),
                         'ended': bool(ended), 'rep2': T.repeated_bigram_share(ids)})
            outs.append({'item': e['id'], 'text': text})
        return rows, outs

    def kl(self, form, lambdas, n_items):
        """Rows of the teacher-forced likelihood of the reference answers and the KL to the base model."""
        ex, ctx = self.ex[:n_items], self.ctx[:n_items]
        ans = [T.answer_ids(self.tm.tok, e['answer'], eot=True)[:self.spec.get('max_new_tokens', 8)] for e in ex]
        rows = []
        bs = max(1, self.bs // 2)
        if form == 'logit':
            res = self.tm.answer_logliks(ctx, ans, 'logit', lambdas, bs)
            for e, a, r in zip(ex, ans, res):
                rows += [{'item': e['id'], 'factor': lam, 'nll': -r[lam][0], 'n_tok': len(a), 'kl_sum': r[lam][1]}
                         for lam in lambdas]
            return rows
        for lam in lambdas:
            self.tm.set_weight_scale(lam)
            try:
                res = self.tm.answer_logliks(ctx, ans, 'weight', None, bs)
            finally:
                self.tm.reset_weight_scale()
            rows += [{'item': e['id'], 'factor': lam, 'nll': -r[None][0], 'n_tok': len(a), 'kl_sum': r[None][1]}
                     for e, a, r in zip(ex, ans, res)]
        return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapter_dir', default=None, help='peft adapter; omit to score the base model only (factor 0)')
    ap.add_argument('--task', required=True, choices=sorted(T.TASKS))
    ap.add_argument('--adapter_task', default=None, help='task the adapter was trained on (default: --task)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--base_cache', default=None, help='directory of the shared factor-0 scores (default: <out_dir>/_base)')
    ap.add_argument('--forms', nargs='+', default=['logit', 'weight'], choices=['logit', 'weight'])
    ap.add_argument('--lambdas', nargs='+', type=float, default=LAMBDAS_LOGIT, help='logit-space factors')
    ap.add_argument('--lambdas_weight', nargs='+', type=float, default=LAMBDAS_WEIGHT, help='weight-space factors')
    ap.add_argument('--n_test', type=int, default=500)
    ap.add_argument('--kl_items', type=int, default=0, help='own task only: items for the KL and reference likelihood')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_input_tokens', type=int, default=900)
    ap.add_argument('--model_tag', default=None)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    if a.smoke:
        from dynalora.smoke import smoke_chat_setup
        a.model = smoke_chat_setup(os.path.join(a.out_dir, '_smoke_model'))
        a.n_test, a.batch_size, a.max_input_tokens = 12, 4, 64
        a.lambdas, a.lambdas_weight = [0, 0.5, 1, 2], [0, 0.5, 1, 2]
        a.kl_items = min(a.kl_items, 6)
        T.TASKS['gsm8k']['max_new_tokens'] = T.TASKS['samsum']['max_new_tokens'] = 8
    mtag = a.model_tag or os.path.basename(a.model.rstrip('/'))
    own = a.adapter_task in (None, a.task)
    prefix = 'own' if own else f'cross_{a.task}'
    os.makedirs(a.out_dir, exist_ok=True)
    base_dir = a.base_cache or os.path.join(a.out_dir, '_base')
    os.makedirs(base_dir, exist_ok=True)
    base_path = os.path.join(base_dir, f'{mtag}__{a.task}__n{a.n_test}.csv')
    kind = T.spec(a.task)['kind']
    grids = {'logit': [float(x) for x in a.lambdas], 'weight': [float(x) for x in a.lambdas_weight]}
    if a.adapter_dir is None:
        a.forms, grids = ['logit'], {'logit': [0.0]}

    examples = T.load_examples(a.task, 'test', n=a.n_test, smoke=a.smoke)
    n = len(examples)
    todo = {}
    for form in a.forms:
        done, _ = _done(os.path.join(a.out_dir, f'{prefix}__{form}.csv'), n)
        todo[form] = [lam for lam in grids[form] if lam not in done]
    kl_todo = {}
    if own and a.kl_items and a.adapter_dir:
        for form in a.forms:
            done, _ = _done(os.path.join(a.out_dir, f'kl__{form}.csv'), min(a.kl_items, n))
            kl_todo[form] = [lam for lam in grids[form] if lam not in done]
    if not any(todo.values()) and not any(kl_todo.values()):
        log(f'{a.out_dir} {prefix}: finished, skipped'); return

    tm = TaskModel(a.model, a.adapter_dir)
    tm.stop_ids = T.stop_ids(tm.tok)
    ev = Evaluator(tm, a.task, examples, a.batch_size, a.max_input_tokens)
    log(f'{mtag} {a.task} ({prefix}): {n} test items, {ev.n_truncated} truncated inputs')
    timing = os.path.join(a.out_dir, 'timing.csv')

    def timed(what, form, lam, fn):
        t0 = time.time(); r = fn()
        _append(timing, [{'what': what, 'task': a.task, 'form': form, 'factor': lam, 'items': n,
                          'seconds': round(time.time() - t0, 1), 'gpu': torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else 'cpu'}])
        return r

    def write(form, lam, rows, outs=None):
        _append(os.path.join(a.out_dir, f'{prefix}__{form}.csv'),
                [{'task': a.task, 'form': form, 'factor': lam, **r} for r in rows])
        if outs is not None:
            with open(os.path.join(a.out_dir, f'{prefix}__{form}__outputs.jsonl'), 'a') as f:
                for o in outs:
                    f.write(json.dumps({'factor': lam, **o}) + '\n')

    # factor 0 (base model, shared by both forms and by all adapters) and factor 1 (the adapter, shared by both forms)
    shared = {}
    if any(0.0 in v for v in todo.values()):
        done, df = _done(base_path, n)
        if 0.0 in done:
            keep = [c for c in df.columns if c not in ('task', 'form', 'factor')]
            shared[0.0] = (df[df.factor == 0][keep].to_dict('records'), None)
            log(f'factor 0 read from {base_path}')
        else:
            if kind == 'choice':
                shared[0.0] = (timed('score', 'base', 0.0, lambda: ev.choice('logit', [0.0])[0.0]), None)
            else:
                shared[0.0] = timed('score', 'base', 0.0, lambda: ev.generate('logit', 0.0))
            _append(base_path, [{'task': a.task, 'form': 'base', 'factor': 0.0, **r} for r in shared[0.0][0]])
            if shared[0.0][1] is not None:
                with open(base_path.replace('.csv', '__outputs.jsonl'), 'w') as f:
                    for o in shared[0.0][1]:
                        f.write(json.dumps({'factor': 0.0, **o}) + '\n')

    for form in a.forms:
        lams = todo[form]
        if not lams:
            continue
        if 0.0 in lams:
            write(form, 0.0, *shared[0.0]); lams = [x for x in lams if x != 0.0]
        if 1.0 in lams and 1.0 in shared:
            write(form, 1.0, *shared[1.0]); lams = [x for x in lams if x != 1.0]
        if kind == 'choice':
            if form == 'logit' and lams:
                res = timed('score', form, 'all', lambda: ev.choice('logit', lams))
                for lam in lams:
                    write(form, lam, res[lam])
                if 1.0 in res:
                    shared[1.0] = (res[1.0], None)
            else:
                for lam in lams:
                    res = timed('score', form, lam, lambda: ev.choice('weight', [lam]))
                    write(form, lam, res[lam])
                    if lam == 1.0:
                        shared[1.0] = (res[lam], None)
        else:
            for lam in sorted(lams, key=lambda x: (x != 1.0, x)):   # factor 1 first, then reused by the other form
                rows, outs = timed('score', form, lam, lambda: ev.generate(form, lam))
                write(form, lam, rows, outs)
                if lam == 1.0:
                    shared[1.0] = (rows, outs)
                log(f'{form} factor {lam}: mean score {sum(r["score"] for r in rows) / len(rows):.4f}')

    for form, lams in kl_todo.items():
        if lams:
            rows = timed('kl', form, 'all', lambda: ev.kl(form, lams, a.kl_items))
            _append(os.path.join(a.out_dir, f'kl__{form}.csv'), [{'task': a.task, 'form': form, **r} for r in rows])

    for form in a.forms:
        _, df = _done(os.path.join(a.out_dir, f'{prefix}__{form}.csv'), n)
        if df is not None:
            log(f'{prefix} {form}: ' + ', '.join(f'{f:g}: {s:.3f}' for f, s in df.groupby('factor')['score'].mean().items()))


if __name__ == '__main__':
    main()
