"""Corpus formatting rate, repetition, SmolLM2 paired tests, and generation metrics at equal divergence.

Run from the repository root after eqdiv.py. Writes to results/level2/analysis/.
"""
import sys, zlib
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts/analysis')
from ci_lib import boot_idx, stat

R = 'results/level2/rescore/'
A = pd.concat([pd.read_csv(R + f'rescore__{x}.csv.gz') for x in ['gpt2', 'qwen', 'temps_gpt2', 'temps_qwen']])
c = A.continuation.fillna('').astype(str)
A['tag'] = c.str.contains(r'<br\s*/?>|</br>|</|/>', regex=True)
A['lit'] = c.map(lambda t: '\\n' in t)          # literal line-break escapes (Yelp)
A['sp'] = c.map(lambda t: '  ' in t)            # runs of spaces (left by tag removal)
A['fmt'] = (A.tag | A.lit | A.sp).astype(float) * 100
def rep2(t):
    w = t.split(); g = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
    return 1 - len(set(g)) / len(g) if g else np.nan
A['rep2'] = c.map(rep2) * 100
A['domain_prob_u'] *= 100
k = ['model', 'config', 'method', 'lambda', 'temperature']
A.groupby(k, dropna=False)[['tag', 'lit', 'sp', 'fmt', 'rep2']].mean().reset_index().to_csv(
    'results/level2/analysis/format_rates.csv', index=False)

# paired tests on SmolLM2 perplexity and formatting rate
rng = np.random.default_rng(0); rows = []
sel = lambda g, m, l: g[(g.method == m) & (g['lambda'] == l)].sort_values(['prompt_id', 'sample']).reset_index(drop=True)
for (m, cfg), g in A[A.model != 'human'].groupby(['model', 'config']):
    for a, b in [(('dynalora', 1.0), ('dynalora', 3.0)), (('dynalora', 1.0), ('dynalora', 2.0)),
                 (('dynalora', 3.0), ('weight', 3.0)), (('dynalora', 2.0), ('weight', 2.0))]:
        ga, gb = sel(g, *a), sel(g, *b)
        if ga.empty or gb.empty:
            continue
        idx = boot_idx(ga, 1000, rng)
        for met in ['ppl2', 'fmt']:
            e = stat(gb, met) - stat(ga, met); bs = np.array([stat(gb.iloc[i], met) - stat(ga.iloc[i], met) for i in idx])
            if np.allclose(bs, 0) and e == 0:
                continue
            p = min(1, max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 1000))
            rows.append(dict(model=m, config=cfg, a=f'{a[0]}@{a[1]}', b=f'{b[0]}@{b[1]}', metric=met, diff=e,
                             lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p=p))
P = pd.DataFrame(rows); P['p_holm'] = np.nan
for _, gg in P.groupby(['model', 'config', 'metric']):
    run = 0
    for j, ix in enumerate(gg.p.sort_values().index):
        run = max(run, min(1, (len(gg) - j) * P.at[ix, 'p'])); P.at[ix, 'p_holm'] = run
P.to_csv('results/level2/analysis/paired_ppl2_fmt.csv', index=False)

# generation metrics of logit space interpolated at the lambda matched in KL to weight space
def ppl2(x):
    v = x.ext2_nll.values; w = x.ext2_n_tokens.values; ok = (w > 0) & np.isfinite(v)
    return np.exp((v[ok] * w[ok]).sum() / w[ok].sum())
S = A[A.method.isin(['dynalora', 'weight'])].groupby(['model', 'config', 'method', 'lambda']).apply(
    lambda x: pd.Series(dict(dom=x.domain_prob_u.mean(), fmt=x.fmt.mean(), rep=x.rep2.mean(), p2=ppl2(x)))).reset_index()
E = pd.read_csv('results/level2/analysis/eqdiv.csv'); out = []
for _, r in E[E.lam_w.isin([2.0, 3.0])].iterrows():
    mm = 'Qwen2.5-0.5B' if str(r.model).startswith('Qwen') else 'gpt2'
    L = S[(S.model == mm) & (S.config == r.config) & (S.method == 'dynalora')].sort_values('lambda')
    W = S[(S.model == mm) & (S.config == r.config) & (S.method == 'weight') & (S['lambda'] == r.lam_w)].iloc[0]
    d = dict(config=r.config, model=mm, lam_w=r.lam_w, lam_star=r.lam_star)
    for key in ['dom', 'p2', 'fmt', 'rep']:
        d[key + '_logit'] = np.interp(r.lam_star, L['lambda'], L[key]); d[key + '_weight'] = W[key]
    out.append(d)
pd.DataFrame(out).to_csv('results/level2/analysis/eqdiv_gen.csv', index=False)
print('done')
