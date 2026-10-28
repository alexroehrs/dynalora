"""Replication over training seeds (GPT-2 small) and on Qwen2.5-1.5B and Qwen2.5-7B (4-bit).

Run from the repository root. Computes, for every adapter with likelihood and benchmark files,
the comparison of logit- and weight-space scaling at equal divergence (likelihood, benchmarks,
generation metrics), and paired tests on the replication runs. Writes to results/level2/analysis/:
replication_eqdiv.csv, replication_bench_eq.csv, replication_curves.csv,
replication_eqdiv_gen.csv, replication_paired.csv.
"""
import glob, sys
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts/analysis')
from ci_lib import boot_idx, stat

L, OUT = 'results/level2/likelihood/', 'results/level2/analysis/'

# 1. likelihood at equal divergence (as in eqdiv.py, all adapters)
def mats(df):
    out = {}
    for meth in ['logit', 'weight']:
        x = df[df.method == meth]; P = lambda v: x.pivot(index='text_id', columns='lambda', values=v)
        out[meth] = (P('nll_sum').values, P('n_tokens').values, P('kl_sum').values, P('nll_sum').columns.values)
    return out
def agg(M, idx, meth):
    nll, nt, kl, _ = M[meth]; s = nt[idx].sum(0)
    return np.exp(nll[idx].sum(0) / s), kl[idx].sum(0) / s
rng = np.random.default_rng(0); rows = []
for f in sorted(glob.glob(L + 'll__*.csv.gz')):
    model, cfg = f.split('ll__')[1].replace('.csv.gz', '').split('__', 1)
    MD = mats(pd.read_csv(f)); MG = mats(pd.read_csv(f.replace('ll__', 'llgen__')))
    lams = MD['logit'][3]; nD, nG = MD['logit'][0].shape[0], MG['logit'][0].shape[0]
    for lw in [1.5, 2.0, 2.5, 3.0]:
        i = list(lams).index(lw)
        def st(iD, iG):
            pLd, kLd = agg(MD, iD, 'logit'); pWd, kWd = agg(MD, iD, 'weight')
            pLg, kLg = agg(MG, iG, 'logit'); pWg, kWg = agg(MG, iG, 'weight')
            k = kWd[i]
            return dict(lam_star=np.interp(k, kLd, lams), dom_logit=np.interp(k, kLd, pLd), dom_weight=pWd[i],
                        gen_logit_domKL=np.interp(k, kLd, pLg), gen_weight=pWg[i], gen_logit_genKL=np.interp(kWg[i], kLg, pLg))
        est = st(np.arange(nD), np.arange(nG)); bs = [st(rng.integers(0, nD, nD), rng.integers(0, nG, nG)) for _ in range(2000)]
        r = dict(model=model, config=cfg, lam_w=lw, **{k: round(v, 3) for k, v in est.items()})
        for key, a, b in [('d_dom', 'dom_logit', 'dom_weight'), ('d_gen_domKL', 'gen_logit_domKL', 'gen_weight'), ('d_gen_genKL', 'gen_logit_genKL', 'gen_weight')]:
            v = np.array([x[a] - x[b] for x in bs]); r[key] = round(est[a] - est[b], 2)
            r[key + '_lo'], r[key + '_hi'] = np.round(np.percentile(v, [2.5, 97.5]), 2)
        rows.append(r)
E = pd.DataFrame(rows); E.to_csv(OUT + 'replication_eqdiv.csv', index=False)

# 2. benchmarks at equal divergence on general text (lambda grid read from the files)
B = pd.concat([pd.read_csv(f) for f in glob.glob('results/level2/bench/*.gz')])
B['score'] = np.where(B.task == 'hellaswag', B.acc_norm, B.acc)
ls_ = pd.read_csv(L + 'll_summary.csv'); ls_ = ls_[ls_.set == 'general']
rng = np.random.default_rng(0); rows = []
for (mod, cfg), g in B.groupby(['model', 'config']):
    K = {m: ls_[(ls_.model == mod) & (ls_.config == cfg) & (ls_.method == m)].sort_values('lam') for m in ['logit', 'weight']}
    M = {m: g[g.method == m].pivot_table(index=['task', 'doc_id'], columns='lambda', values='score') for m in ['logit', 'weight']}
    blams = M['logit'].columns.values; tasks = M['logit'].index.get_level_values(0).values
    groups = [np.where(tasks == t)[0] for t in np.unique(tasks)]
    mean_acc = lambda mat, idx: np.mean([mat[idx][tasks[idx] == k].mean(0) for k in np.unique(tasks)], axis=0)
    for lw in [1.5, 2.0, 2.5, 3.0]:
        if lw not in blams:
            continue
        k = K['weight'][K['weight'].lam == lw].kl.values[0]; ls = np.interp(k, K['logit'].kl.values, K['logit'].lam.values)
        def st(idx):
            return np.interp(ls, blams, mean_acc(M['logit'].values, idx)) * 100, mean_acc(M['weight'].values, idx)[list(blams).index(lw)] * 100
        eL, eW = st(np.arange(len(tasks))); bs = []
        for _ in range(1000):
            a, b = st(np.concatenate([gg[rng.integers(0, len(gg), len(gg))] for gg in groups])); bs.append(a - b)
        rows.append(dict(model=mod, config=cfg, lam_w=lw, lam_star=ls, logit=eL, weight=eW, diff=eL - eW,
                         lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
pd.DataFrame(rows).to_csv(OUT + 'replication_bench_eq.csv', index=False)

# 3. generation metrics per grid point and at equal divergence
R = 'results/level2/rescore/'
A = pd.concat([pd.read_csv(R + f'rescore__{x}.csv.gz') for x in ['gpt2', 'qwen', 'seeds', 'qwen15', 'qwen7b']])
A = A[A.model != 'human']; c = A.continuation.fillna('').astype(str)
A['fmt'] = (c.str.contains(r'<br\s*/?>|</br>|</|/>', regex=True) | c.map(lambda t: '\\n' in t) | c.map(lambda t: '  ' in t)).astype(float) * 100
def rep2(t):
    w = t.split(); g = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
    return 1 - len(set(g)) / len(g) if g else np.nan
A['rep2'] = c.map(rep2) * 100; A['domain_prob_u'] *= 100
def ppl2(x):
    v = x.ext2_nll.values; w = x.ext2_n_tokens.values; ok = (w > 0) & np.isfinite(v)
    return np.exp((v[ok] * w[ok]).sum() / w[ok].sum())
S = A[A.method.isin(['dynalora', 'weight'])].groupby(['model', 'config', 'method', 'lambda']).apply(
    lambda x: pd.Series(dict(dom=x.domain_prob_u.mean(), fmt=x.fmt.mean(), rep=x.rep2.mean(), p2=ppl2(x)))).reset_index()
S.to_csv(OUT + 'replication_curves.csv', index=False); out = []
for _, r in E[E.lam_w.isin([2.0, 3.0])].iterrows():
    Lc = S[(S.model == r.model) & (S.config == r.config) & (S.method == 'dynalora')].sort_values('lambda')
    W = S[(S.model == r.model) & (S.config == r.config) & (S.method == 'weight') & (S['lambda'] == r.lam_w)]
    if Lc.empty or W.empty:
        continue
    d = dict(config=r.config, model=r.model, lam_w=r.lam_w, lam_star=r.lam_star)
    for k in ['dom', 'p2', 'fmt', 'rep']:
        d[k + '_logit'] = np.interp(r.lam_star, Lc['lambda'], Lc[k]); d[k + '_weight'] = W.iloc[0][k]
    out.append(d)
pd.DataFrame(out).to_csv(OUT + 'replication_eqdiv_gen.csv', index=False)

# 4. paired tests on the replication runs (prompt-clustered bootstrap)
rng = np.random.default_rng(0); rows = []
sel = lambda g, m, l: g[(g.method == m) & (g['lambda'] == l)].sort_values(['prompt_id', 'sample']).reset_index(drop=True)
keep = A.config.str.contains('seed') | A.model.isin(['Qwen2.5-1.5B', 'Qwen2.5-7B']) | ((A.model == 'gpt2') & A.config.isin(['imdb_bal', 'pubmed']))
for (m, cfg), g in A[keep].groupby(['model', 'config']):
    for a, b in [(('dynalora', 1.0), ('dynalora', 3.0)), (('dynalora', 1.0), ('dynalora', 2.0)),
                 (('dynalora', 3.0), ('weight', 3.0)), (('dynalora', 2.0), ('weight', 2.0))]:
        ga, gb = sel(g, *a), sel(g, *b)
        if ga.empty or gb.empty:
            continue
        idx = boot_idx(ga, 1000, rng)
        for met in ['ppl2', 'fmt', 'domain_prob_u']:
            e = stat(gb, met) - stat(ga, met); bs = np.array([stat(gb.iloc[i], met) - stat(ga.iloc[i], met) for i in idx])
            if np.allclose(bs, 0) and e == 0:
                continue
            p = min(1, max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 1000))
            rows.append(dict(model=m, config=cfg, a=f'{a[0]}@{a[1]}', b=f'{b[0]}@{b[1]}', metric=met, diff=e,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=p))
pd.DataFrame(rows).to_csv(OUT + 'replication_paired.csv', index=False)
print('done')
