"""Task adapter (``pubmedqa_task``, scripts/train_task_adapter.py): accuracy along lambda for both
forms of scaling, by gold answer, on the general benchmarks, and at equal KL divergence.

Inputs (``notebooks/colab_task_adapter.ipynb``):

* ``--inputs``: per-question rows of ``scripts/zeroshot_bench.py --tasks pubmedqa_test500``
  (``bench__<model>__pubmedqa_task.csv.gz``, the 500 test questions of the labeled set);
* ``--bench``: the same files for the four general benchmarks (LAMBADA, HellaSwag acc_norm,
  PIQA, ARC-Easy);
* ``--likelihood``: directory of ``llgen__<model>__pubmedqa_task.csv.gz`` (general-text KL per
  lambda, ``scripts/likelihood_tradeoff.py --sets general``), or an ``ll_summary.csv``.

Outputs under ``--out``:

* ``task_adapter_curves.csv``: accuracy in % per ``model``, ``config``, ``task``, ``method`` and
  ``lam`` with a 95% bootstrap interval over questions (``lo``, ``hi``), and the paired change
  from lambda 0 (``d_base``, ``d_base_lo``, ``d_base_hi``) and from lambda 1, the plain adapter
  (``d_lora``, ``d_lora_lo``, ``d_lora_hi``); ``task`` is ``pubmedqa_test500``, each general
  benchmark, and ``general_mean`` (equal-weight mean of the four, stratified bootstrap);
* ``task_adapter_by_answer.csv``: PubMedQA accuracy per gold answer (yes, no, maybe) and the share
  of test questions with each answer (labels loaded from the Hub at run time, split as the task
  does, never stored here);
* ``task_adapter_eq.csv``: for each weight-space lambda (``lam_w`` 1.5, 2, 3) the logit-space
  lambda (``lam_star``) with the same general-text KL from the base model, the accuracy of both
  (logit interpolated on the lambda grid) and their difference logit minus weight (``diff``) with
  a 95% paired bootstrap interval, for ``task`` pubmedqa_test500 and general_mean, as in
  ``scripts/analysis/pubmedqa.py`` and ``bench_eq.py``; ``beyond_grid`` marks a weight-space KL
  above the largest logit-space KL of the grid (``lam_star`` clamped to the grid end).
"""
import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAM_W = [1.5, 2.0, 3.0]
GENERAL = ['lambada_openai', 'hellaswag', 'piqa', 'arc_easy']
TASK = 'pubmedqa_test500'


def load_bench(d, tasks=None):
    files = sorted(glob.glob(os.path.join(d, 'bench__*.csv.gz')))
    if not files:
        return pd.DataFrame()
    B = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if tasks is not None:
        B = B[B.task.isin(tasks)]
    B = B.copy()
    B['score'] = np.where(B.task == 'hellaswag', B.acc_norm, B.acc)
    return B


def load_kl(path):
    """General-text KL per (model, config, method, lam): mean per-token KL(p_lam || p_base)."""
    if os.path.isfile(path):
        s = pd.read_csv(path)
        return s[s.set == 'general'][['model', 'config', 'method', 'lam', 'kl']]
    files = sorted(glob.glob(os.path.join(path, 'llgen__*.csv.gz')))
    if not files:
        return pd.DataFrame(columns=['model', 'config', 'method', 'lam', 'kl'])
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    g = d.groupby(['model', 'config', 'method', 'lambda'])[['kl_sum', 'n_tokens']].sum().reset_index()
    g['kl'] = g.kl_sum / g.n_tokens
    return g.rename(columns={'lambda': 'lam'})[['model', 'config', 'method', 'lam', 'kl']]


def matrices(g):
    """docs x lambda score matrices of both methods on the documents scored by both, plus the
    task of each document row (for the stratified bootstrap of the general mean)."""
    M = {m: g[g.method == m].pivot_table(index=['task', 'doc_id'], columns='lambda', values='score')
         for m in ['logit', 'weight']}
    docs = M['logit'].index.intersection(M['weight'].index)
    lams = sorted(set(M['logit'].columns) & set(M['weight'].columns))
    X = {m: M[m].loc[docs, lams].values for m in M}
    tasks = np.array([t for t, _ in docs])
    return X, np.array(lams), tasks


def boot_idx(tasks, nboot, rng):
    """Bootstrap resamples of row indices, stratified within task."""
    groups = [np.where(tasks == t)[0] for t in np.unique(tasks)]
    return np.stack([np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(nboot)])


def mean_acc(X, idx, tasks):
    """Equal-weight mean over tasks of the per-task mean along lambda, for one resample."""
    x, t = X[idx], tasks[idx]
    return np.mean([x[t == k].mean(0) for k in np.unique(tasks)], axis=0)


def curves(X, lams, tasks, nboot, rng):
    idx = boot_idx(tasks, nboot, rng)
    rows = []
    for meth, A in X.items():
        full = mean_acc(A, np.arange(len(tasks)), tasks)
        bs = np.stack([mean_acc(A, i, tasks) for i in idx])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        out = dict(method=meth, acc=100 * full, lo=100 * lo, hi=100 * hi)
        for ref, name in [(0.0, 'd_base'), (1.0, 'd_lora')]:
            if ref in lams:
                j = list(lams).index(ref)
                d = full - full[j]; bd = bs - bs[:, [j]]
                dlo, dhi = np.percentile(bd, [2.5, 97.5], 0)
                out.update({name: 100 * d, name + '_lo': 100 * dlo, name + '_hi': 100 * dhi})
        for j, lam in enumerate(lams):
            rows.append(dict(method=meth, lam=lam, n=len(tasks), **{k: v[j] for k, v in out.items() if k != 'method'}))
    return rows


def eq(X, lams, tasks, K, nboot, rng):
    """Logit vs weight at the general-text KL of the weight-space lambdas in LAM_W."""
    if K['logit'].empty or K['weight'].empty:
        return []
    idx = np.vstack([np.arange(len(tasks)), boot_idx(tasks, nboot, rng)])
    accL = np.stack([mean_acc(X['logit'], i, tasks) for i in idx])
    accW = np.stack([mean_acc(X['weight'], i, tasks) for i in idx])
    kl_l, lam_l = K['logit'].kl.values, K['logit'].lam.values
    rows = []
    for lw in LAM_W:
        if lw not in lams or lw not in K['weight'].lam.values:
            continue
        k = K['weight'][K['weight'].lam == lw].kl.values[0]
        ls = float(np.interp(k, kl_l, lam_l))
        aL = np.array([np.interp(ls, lams, a) for a in accL]) * 100
        aW = accW[:, list(lams).index(lw)] * 100
        diff = aL - aW
        rows.append(dict(kl_set='general', lam_w=lw, kl=k, lam_star=ls, beyond_grid=bool(k > kl_l.max()),
                         logit=aL[0], weight=aW[0], diff=diff[0],
                         lo=np.percentile(diff[1:], 2.5), hi=np.percentile(diff[1:], 97.5), n=len(tasks)))
    return rows


def test_labels():
    """Gold answers of the test fold in doc_id order, from the Hub, via the task's own split."""
    from datasets import load_dataset
    path = os.path.join(ROOT, 'scripts', 'lm_eval_tasks', 'pubmedqa_test500', 'utils.py')
    spec = importlib.util.spec_from_file_location('pubmedqa_test500_utils', path)
    U = importlib.util.module_from_spec(spec); spec.loader.exec_module(U)
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    gold = pd.Series(U.process_docs(ds)['final_decision'], name='answer')
    gold.index.name = 'doc_id'
    return gold


def by_answer(B, gold):
    X = B.merge(gold, left_on='doc_id', right_index=True)
    g = X.groupby(['task', 'model', 'config', 'method', 'lambda', 'answer']).acc.agg(['mean', 'size']).reset_index()
    g = g.rename(columns={'lambda': 'lam', 'mean': 'acc', 'size': 'n'})
    g['acc'] *= 100
    share = gold.value_counts(normalize=True).rename('share_of_questions') * 100
    return g.merge(share, left_on='answer', right_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='results/level2/pubmedqa', help='bench__ files of the task pubmedqa_test500')
    ap.add_argument('--bench', default='results/level2/bench', help='bench__ files of the general benchmarks')
    ap.add_argument('--likelihood', default='results/level2/likelihood', help='llgen__ files directory or ll_summary.csv')
    ap.add_argument('--out', default='results/level2/analysis')
    ap.add_argument('--configs', nargs='+', default=['pubmedqa_task'])
    ap.add_argument('--nboot', type=int, default=2000)
    ap.add_argument('--by_answer', action='store_true', help='also write accuracy per gold answer (loads the dataset)')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    T = load_bench(a.inputs, [TASK]); G = load_bench(a.bench, GENERAL)
    T = T[T.config.isin(a.configs)]; G = G[G.config.isin(a.configs)] if len(G) else G
    if T.empty and G.empty:
        raise SystemExit(f'no bench__*.csv.gz for {a.configs} in {a.inputs} or {a.bench}')
    KL = load_kl(a.likelihood)
    rng = np.random.default_rng(a.seed)
    C, E = [], []
    sets = [(TASK, T)] + [(t, G[G.task == t]) for t in GENERAL] + [('general_mean', G)]
    for name, D in sets:
        if D.empty:
            continue
        for (mod, cfg), g in D.groupby(['model', 'config']):
            key = dict(model=mod, config=cfg, task=name)
            X, lams, tasks = matrices(g)
            C += [dict(key, **r) for r in curves(X, lams, tasks, a.nboot, rng)]
            if name in (TASK, 'general_mean'):
                K = {m: KL[(KL.model == mod) & (KL.config == cfg) & (KL.method == m)].sort_values('lam') for m in ['logit', 'weight']}
                if K['logit'].empty:
                    print(f'no general-text KL for {mod} {cfg}: equal-KL comparison skipped')
                E += [dict(key, **r) for r in eq(X, lams, tasks, K, a.nboot, rng)]
    os.makedirs(a.out, exist_ok=True)
    C, E = pd.DataFrame(C), pd.DataFrame(E)
    C.to_csv(os.path.join(a.out, 'task_adapter_curves.csv'), index=False)
    E.to_csv(os.path.join(a.out, 'task_adapter_eq.csv'), index=False)
    pd.set_option('display.width', 250)
    print(C.pivot_table(index=['model', 'config', 'task', 'lam'], columns='method', values='acc').round(1).to_string())
    if len(E):
        print(E.round(3).to_string())
    if a.by_answer and len(T):
        A = by_answer(T, test_labels())
        A.to_csv(os.path.join(a.out, 'task_adapter_by_answer.csv'), index=False)
        print(A.pivot_table(index=['model', 'config', 'method', 'answer'], columns='lam', values='acc').round(1).to_string())


if __name__ == '__main__':
    main()
