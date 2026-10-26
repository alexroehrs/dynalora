"""Teacher-forced likelihood of the scaled adapter on domain and general text.

For every trained adapter and every lambda, the script scores held-out text of the
adapter's domain, generic English text (WikiText-103 train split), and optionally negative
IMDB test reviews (``domain_neg``, the training distribution of the negative-only adapter), under two forms of
scaling, without sampling:

* ``logit``   log_softmax(y_base + lam * (y_lora - y_base))          (DynaLoRA)
* ``weight``  log_softmax of the adapted model with its LoRA scaling times lam

For each text it records the summed negative log-likelihood of the observed tokens,
the number of scored tokens, and the summed per-token KL divergence KL(p_lam || p_base),
which places both forms of scaling on a common axis of distance from the base model.
Per-text values are written so that intervals can be bootstrapped over texts.

Example (Colab, T4; adapters trained by run_all.py):
    python scripts/likelihood_tradeoff.py --model gpt2 \
        --adapters_dir /content/drive/MyDrive/dynalora_results/gpt2/adapters \
        --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed \
        --out /content/drive/MyDrive/dynalora_results/likelihood
"""
import argparse
import os
import re
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dynalora import data as D  # noqa: E402
from dynalora.generate import Generator  # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
_HTML = re.compile(r'<[^>]+>')


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


PREFIX = {'domain': 'll', 'general': 'llgen', 'domain_neg': 'llneg'}


def eval_texts(cfg_name, n_domain, n_general, smoke=False):
    if smoke:
        dom = ['this movie was great and the acting was fine'] * n_domain
        gen = ['the river flows through the valley of the old town'] * n_general
        return dom, gen
    cfg = D.load_config(cfg_name)
    held = cfg['heldout'][-n_domain:]          # held-out texts, never used in training
    if cfg_name == 'imdb_bal_nohtml':
        held = [re.sub(r'\s+', ' ', _HTML.sub(' ', t)) for t in held]
    general = _wikitext_train(n_general)   # disjoint from the test-split paragraphs used by the classifier
    return held, general


def _wikitext_train(n):
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='train[:50000]')
    out = [r['text'].strip() for r in ds if len(r['text'].strip()) > 200 and not r['text'].startswith(' =')]
    assert len(out) >= n, len(out)
    return out[:n]


@torch.no_grad()
def _logits(gen, ids, att, adapter_on):
    if not adapter_on:
        with gen.model.disable_adapter():
            out = gen.model(input_ids=ids, attention_mask=att)
    else:
        out = gen.model(input_ids=ids, attention_mask=att)
    return out.logits[:, :-1, :].float()


def _stats(logits, base_lp, tgt, mask):
    lp = F.log_softmax(logits, -1)
    nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    kl = (lp.exp() * (lp - base_lp)).sum(-1)
    return (nll * mask).sum(1), (kl * mask).sum(1)


@torch.no_grad()
def score(gen, texts, set_name, lambdas, batch_size, max_len):
    tok = gen.tok
    tok.padding_side = 'right'
    rows = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, return_tensors='pt', truncation=True, max_length=max_len, padding=True).to(gen.device)
        ids, att = enc['input_ids'], enc['attention_mask']
        tgt, mask = ids[:, 1:], att[:, 1:].float()
        ntok = mask.sum(1)
        gen.reset_weight_scale()
        base = _logits(gen, ids, att, False)
        lora = _logits(gen, ids, att, True)
        base_lp = F.log_softmax(base, -1)
        delta = lora - base
        for lam in lambdas:
            nll, kl = _stats(base + lam * delta, base_lp, tgt, mask)
            for j in range(len(chunk)):
                rows.append((set_name, start + j, 'logit', lam, nll[j].item(), ntok[j].item(), kl[j].item()))
        for lam in lambdas:
            if lam in (0.0, 1.0):
                w = base if lam == 0.0 else lora
            else:
                gen.set_weight_scale(lam)
                w = _logits(gen, ids, att, True)
            nll, kl = _stats(w, base_lp, tgt, mask)
            for j in range(len(chunk)):
                rows.append((set_name, start + j, 'weight', lam, nll[j].item(), ntok[j].item(), kl[j].item()))
        gen.reset_weight_scale()
        del base, lora, base_lp, delta
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapters_dir', required=True)
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_domain', type=int, default=500)
    ap.add_argument('--n_general', type=int, default=500)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lambdas', nargs='+', type=float, default=LAMBDAS)
    ap.add_argument('--sets', nargs='+', default=['domain', 'general'], choices=['domain', 'general', 'domain_neg'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mtag = os.path.basename(a.model.rstrip('/'))
    for cname in a.configs:
        todo = []
        for st in a.sets:
            path = os.path.join(a.out, f"{PREFIX[st]}__{mtag}__{cname}.csv.gz")
            if os.path.exists(path):
                log(f'{path} exists, skipping')
            else:
                todo.append((st, path))
        if not todo:
            continue
        adir = os.path.join(a.adapters_dir, f'{mtag}__{cname}')
        log(f'loading {a.model} + {adir}')
        gen = Generator(a.model, adir)
        dom, general = eval_texts(cname, a.n_domain, a.n_general, a.smoke)
        for st, path in todo:
            if st == 'domain_neg':
                texts = dom if a.smoke else D.imdb_test_by_label(a.n_domain, 0)
            else:
                texts = dom if st == 'domain' else general
            rows = score(gen, texts, st, a.lambdas, a.batch_size, a.max_len)
            log(f'{cname}: {st} scored ({len(texts)} texts)')
            df = pd.DataFrame(rows, columns=['set', 'text_id', 'method', 'lambda', 'nll_sum', 'n_tokens', 'kl_sum'])
            df.insert(0, 'model', mtag); df.insert(0, 'config', cname)
            df.to_csv(path, index=False)
            g = df.groupby(['method', 'lambda'])[['nll_sum', 'n_tokens', 'kl_sum']].sum()
            g['ppl'] = (g.nll_sum / g.n_tokens).map(lambda x: round(float(torch.tensor(x).exp()), 2))
            g['kl_per_token'] = (g.kl_sum / g.n_tokens).round(4)
            print(g[['ppl', 'kl_per_token']].unstack('method').to_string(), flush=True)
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log('done')


if __name__ == '__main__':
    main()
