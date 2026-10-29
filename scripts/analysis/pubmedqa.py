"""PubMedQA accuracy along lambda for both forms of scaling, and at equal KL divergence.

Reads the per-document rows of ``scripts/zeroshot_bench.py --tasks pubmedqa_labeled``
(``bench__<model>__<config>.csv.gz``, ``notebooks/colab_pubmedqa.ipynb``) and writes
``pubmedqa_curves.csv``: accuracy per adapter, method and lambda, and its change from the
base model (lambda 0), with 95% bootstrap intervals over documents;
``pubmedqa_eq.csv``: accuracy of logit minus weight scaling at equal KL divergence from
the base model, matched as in ``bench_eq.py`` (``kl_set`` general, as there, and also on
the adapter's domain text), with a paired bootstrap over documents.

With ``--by_answer`` (needs the dataset, loaded from the Hugging Face Hub at run time and
not stored here), ``pubmedqa_by_answer.csv`` adds the accuracy per gold answer (yes, no,
maybe), which separates a change in the model's answer prior from a change in its reading
of the abstract; ``doc_id`` is the position of the question in the ``pqa_labeled`` split,
as enumerated by the harness.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

LAM_W = [1.5, 2.0, 2.5, 3.0]


def load(inputs):
    files = sorted(glob.glob(os.path.join(inputs, 'bench__*.csv.gz')))
    if not files:
        raise SystemExit(f'no bench__*.csv.gz in {inputs}')
    B = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return B[B.task.str.startswith('pubmedqa')]


def matrices(g):
    """docs x lambda accuracy matrix per method, on the documents scored by both."""
    M = {m: g[g.method == m].pivot_table(index='doc_id', columns='lambda', values='acc') for m in ['logit', 'weight']}
    docs = M['logit'].index.intersection(M['weight'].index)
    lams = sorted(set(M['logit'].columns) & set(M['weight'].columns))
    return {m: M[m].loc[docs, lams].values for m in M}, np.array(lams)


def curves(g, nboot, rng):
    M, lams = matrices(g)
    n = M['logit'].shape[0]
    idx = rng.integers(0, n, (nboot, n))
    rows = []
    for meth, X in M.items():
        bs = X[idx].mean(1)                      # nboot x lambda
        d = X - X[:, [0]]
        bd = d[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5], 0); dlo, dhi = np.percentile(bd, [2.5, 97.5], 0)
        for j, lam in enumerate(lams):
            rows.append(dict(method=meth, lam=lam, acc=100 * X[:, j].mean(), lo=100 * lo[j], hi=100 * hi[j],
                             d_base=100 * d[:, j].mean(), d_lo=100 * dlo[j], d_hi=100 * dhi[j], n=n))
    return rows


def eq(g, ll, nboot, rng):
    M, lams = matrices(g)
    n = M['logit'].shape[0]
    idx = np.vstack([np.arange(n), rng.integers(0, n, (nboot, n))])
    accL, accW = M['logit'][idx].mean(1), M['weight'][idx].mean(1)   # (1 + nboot) x lambda
    rows = []
    for kset in ['general', 'domain']:
        K = {m: ll[(ll.set == kset) & (ll.method == m)].sort_values('lam') for m in ['logit', 'weight']}
        if K['logit'].empty or K['weight'].empty:
            continue
        kl_l, lam_l = K['logit'].kl.values, K['logit'].lam.values
        for lw in LAM_W:
            if lw not in lams or lw not in K['weight'].lam.values:
                continue
            k = K['weight'][K['weight'].lam == lw].kl.values[0]
            ls = np.interp(k, kl_l, lam_l)
            aL = np.array([np.interp(ls, lams, a) for a in accL]) * 100
            aW = accW[:, list(lams).index(lw)] * 100
            diff = aL - aW
            rows.append(dict(kl_set=kset, lam_w=lw, kl=k, lam_star=ls, beyond_grid=bool(k > kl_l.max()),
                             logit=aL[0], weight=aW[0], diff=diff[0],
                             lo=np.percentile(diff[1:], 2.5), hi=np.percentile(diff[1:], 97.5), n=n))
    return rows


def by_answer(B):
    from datasets import load_dataset
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    gold = pd.Series(ds['final_decision'], name='answer')
    gold.index.name = 'doc_id'
    X = B.merge(gold, left_on='doc_id', right_index=True)
    g = X.groupby(['task', 'model', 'config', 'method', 'lambda', 'answer']).acc.agg(['mean', 'size']).reset_index()
    g = g.rename(columns={'lambda': 'lam', 'mean': 'acc', 'size': 'n'})
    g['acc'] *= 100
    share = gold.value_counts(normalize=True).rename('share_of_questions') * 100
    return g.merge(share, left_on='answer', right_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='results/level2/pubmedqa')
    ap.add_argument('--ll', default='results/level2/likelihood/ll_summary.csv')
    ap.add_argument('--out', default='results/level2/analysis')
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--by_answer', action='store_true', help='also write accuracy per gold answer (loads the dataset)')
    a = ap.parse_args()
    B = load(a.inputs)
    ll = pd.read_csv(a.ll)
    rng = np.random.default_rng(0)
    C, E = [], []
    for (task, mod, cfg), g in B.groupby(['task', 'model', 'config']):
        key = dict(task=task, model=mod, config=cfg)
        C += [dict(key, **r) for r in curves(g, a.nboot, rng)]
        l = ll[(ll.model == mod) & (ll.config == cfg)]
        if l.empty:
            print(f'no likelihood summary for {mod} {cfg}: equal-KL comparison skipped')
        E += [dict(key, **r) for r in eq(g, l, a.nboot, rng)]
    os.makedirs(a.out, exist_ok=True)
    C, E = pd.DataFrame(C), pd.DataFrame(E)
    C.to_csv(os.path.join(a.out, 'pubmedqa_curves.csv'), index=False)
    E.to_csv(os.path.join(a.out, 'pubmedqa_eq.csv'), index=False)
    pd.set_option('display.width', 250)
    print(C.pivot_table(index=['model', 'config', 'lam'], columns='method', values='acc').round(1).to_string())
    if len(E):
        print(E.round(2).to_string())
    if a.by_answer:
        A = by_answer(B)
        A.to_csv(os.path.join(a.out, 'pubmedqa_by_answer.csv'), index=False)
        print(A.pivot_table(index=['model', 'config', 'method', 'answer'], columns='lam', values='acc').round(1).to_string())


if __name__ == '__main__':
    main()
