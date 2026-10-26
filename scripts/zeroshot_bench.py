"""Zero-shot accuracy of the scaled adapter on general benchmarks, for both forms of scaling.

The task documents, prompts, answer choices and scoring rules are those of the
lm-evaluation-harness (LAMBADA-OpenAI, HellaSwag, PIQA, ARC-Easy). Only the
log-likelihood requests are computed here, so that every value of lambda can be
evaluated from the same requests:

* ``logit``   log_softmax(y_base + lam * (y_lora - y_base)), all lambdas from one pair of passes
* ``weight``  the adapted model with its LoRA scaling multiplied by lam, one pass per lambda

Output: ``bench__<model>__<config>.csv.gz`` with one row per method, lambda, task and
document (accuracy and, where defined, length-normalized accuracy), so that intervals
can be bootstrapped over documents.

Example (Colab):
    pip install lm-eval
    python scripts/zeroshot_bench.py --model gpt2 \
        --adapters_dir /content/drive/MyDrive/dynalora_results/gpt2/adapters \
        --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed \
        --out /content/drive/MyDrive/dynalora_results/bench
"""
import argparse
import collections
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dynalora.generate import Generator  # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
TASKS = ['lambada_openai', 'hellaswag', 'piqa', 'arc_easy']


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def build_requests(tasks, limit):
    from lm_eval.tasks import TaskManager, get_task_dict
    td = get_task_dict(tasks, TaskManager())
    out = {}
    for name, task in td.items():
        task.build_all_requests(limit=limit, rank=0, world_size=1)
        out[name] = task
        log(f'{name}: {len(task.instances)} requests')
    return out


def encode_pair(tok, context, continuation):
    n = len(context) - len(context.rstrip())
    if n:
        continuation = context[-n:] + continuation
        context = context[:-n]
    whole = tok(context + continuation, add_special_tokens=False)['input_ids']
    ctx = tok(context, add_special_tokens=False)['input_ids']
    cont = whole[len(ctx):]
    if not ctx:
        ctx = [tok.eos_token_id]
    return ctx, cont


@torch.no_grad()
def score(gen, pairs, mode, lambdas, batch_size, max_len):
    """Returns {lam: [(loglik, is_greedy), ...]} aligned with ``pairs``.

    mode 'logit' computes every lambda from the base and adapted logits; mode 'weight'
    scores with the weight scale already set on ``gen`` and returns {None: [...]}."""
    tok = gen.tok
    enc = [encode_pair(tok, c, x) for c, x in pairs]
    order = sorted(range(len(enc)), key=lambda i: -(len(enc[i][0]) + len(enc[i][1])))
    keys = lambdas if mode == 'logit' else [None]
    res = {k: [None] * len(enc) for k in keys}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        seqs, spans = [], []
        for i in idx:
            ctx, cont = enc[i]
            full = (ctx + cont)[-(max_len + 1):]
            inp = full[:-1]
            spans.append((len(inp) - len(cont), len(inp), cont))
            seqs.append(inp)
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, x in enumerate(seqs):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
        ids, att = ids.to(gen.device), att.to(gen.device)
        if mode == 'logit':
            with gen.model.disable_adapter():
                yb = gen.model(input_ids=ids, attention_mask=att).logits
            yl = gen.model(input_ids=ids, attention_mask=att).logits
        else:
            yw = gen.model(input_ids=ids, attention_mask=att).logits
        for j, i in enumerate(idx):
            a, b, cont = spans[j]
            tgt = torch.tensor(cont, device=gen.device)
            if mode == 'logit':
                base = yb[j, a:b].float(); delta = yl[j, a:b].float() - base
                for lam in lambdas:
                    lp = F.log_softmax(base + lam * delta, -1)
                    res[lam][i] = (float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
            else:
                lp = F.log_softmax(yw[j, a:b].float(), -1)
                res[None][i] = (float(lp.gather(-1, tgt[:, None]).sum()), bool((lp.argmax(-1) == tgt).all()))
        if mode == 'logit':
            del yb, yl
        else:
            del yw
    return res


def doc_metrics(task, resp):
    """Applies the task's own scoring to the responses of each document."""
    by_doc = collections.defaultdict(list)
    for k, inst in enumerate(task.instances):
        by_doc[inst.doc_id].append((inst.idx, k, inst))
    rows = []
    for doc_id, items in by_doc.items():
        items.sort(key=lambda t: t[0])
        doc = items[0][2].doc
        r = task.process_results(doc, [resp[k] for _, k, _ in items])
        rows.append(dict(doc_id=doc_id, acc=float(r.get('acc', float('nan'))),
                         acc_norm=float(r.get('acc_norm', float('nan')))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapters_dir', required=True)
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', nargs='+', default=TASKS)
    ap.add_argument('--limit', type=int, default=1000)
    ap.add_argument('--lambdas', nargs='+', type=float, default=LAMBDAS)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_len', type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = build_requests(a.tasks, a.limit)
    offsets, pairs = {}, []
    for name, task in tasks.items():
        offsets[name] = (len(pairs), len(pairs) + len(task.instances))
        pairs += [tuple(inst.args) for inst in task.instances]
    uniq = sorted(set(pairs))
    pos = {p: i for i, p in enumerate(uniq)}
    log(f'{len(pairs)} requests, {len(uniq)} unique')
    mtag = os.path.basename(a.model.rstrip('/'))
    for cname in a.configs:
        path = os.path.join(a.out, f'bench__{mtag}__{cname}.csv.gz')
        if os.path.exists(path):
            log(f'{path} exists, skipping'); continue
        gen = Generator(a.model, os.path.join(a.adapters_dir, f'{mtag}__{cname}'))
        gen.reset_weight_scale()
        results = {}
        r = score(gen, uniq, 'logit', a.lambdas, a.batch_size, a.max_len)
        for lam in a.lambdas:
            results[('logit', lam)] = r[lam]
        log(f'{cname}: logit done')
        for lam in a.lambdas:
            if lam == 0.0:
                results[('weight', lam)] = results[('logit', 0.0)]; continue
            if lam == 1.0:
                results[('weight', lam)] = results[('logit', 1.0)]; continue
            gen.set_weight_scale(lam)
            results[('weight', lam)] = score(gen, uniq, 'weight', None, a.batch_size, a.max_len)[None]
        gen.reset_weight_scale()
        log(f'{cname}: weight done')
        rows = []
        for (meth, lam), vals in results.items():
            for name, task in tasks.items():
                s, e = offsets[name]
                resp = [vals[pos[p]] for p in pairs[s:e]]
                for row in doc_metrics(task, resp):
                    rows.append(dict(config=cname, model=mtag, method=meth, **{'lambda': lam}, task=name, **row))
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(df.groupby(['task', 'method', 'lambda'])['acc'].mean().unstack('method').round(3).to_string(), flush=True)
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log('done')


if __name__ == '__main__':
    main()
