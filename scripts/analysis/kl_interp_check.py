"""KL divergence along lambda and sensitivity of the equal-divergence matching to the interpolation rule.

The comparison at equal divergence (paper, Section 4.5) finds, for each weight-space value
lam_w, the logit-space value lam* with the same mean per-token KL divergence from the base
model, by linear interpolation over the grid, and reads the measures at lam* by linear
interpolation as well. This script
  * writes the KL(lambda) curves of every adapter, both forms, on domain and general text
    (kl_curves.csv, plotted by fig_kl.py),
  * checks that every curve is monotone in lambda,
  * reports the leave-one-out error of linear interpolation of KL on the grid (each interior
    point lambda >= 1 predicted from its two neighbours, i.e. at twice the grid step),
  * repeats the matching with shape-preserving cubic interpolation (PCHIP) of both KL and the
    perplexities and writes the differences (kl_interp_sensitivity.csv).

Usage: python scripts/analysis/kl_interp_check.py   (reads results/level2/likelihood)
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

R = 'results/level2/likelihood'
OUT = 'results/level2/analysis'


def agg(X):
    g = X.groupby(['method', 'lambda']).agg(nll=('nll_sum', 'sum'), n=('n_tokens', 'sum'),
                                            kl=('kl_sum', 'sum')).reset_index()
    g['ppl'] = np.exp(g.nll / g.n)
    g['kl'] = g.kl / g.n
    return g


curves, rows, loo = [], [], []
for f in sorted(glob.glob(f'{R}/ll__*.csv.gz')):
    tag = os.path.basename(f)[4:-7]
    if '__lr' in tag:
        continue
    model, cfg = tag.split('__', 1)
    a, b = agg(pd.read_csv(f)), agg(pd.read_csv(f.replace('/ll__', '/llgen__')))
    for setname, X in [('domain', a), ('general', b)]:
        for m in ['logit', 'weight']:
            x = X[X.method == m].sort_values('lambda')
            for _, r in x.iterrows():
                curves.append(dict(model=model, config=cfg, set=setname, method=m, lam=r['lambda'], kl=r.kl, ppl=r.ppl))
            lam, kl = x['lambda'].values, x.kl.values
            for i in range(1, len(lam) - 1):
                if lam[i] >= 1:
                    est = kl[i - 1] + (kl[i + 1] - kl[i - 1]) * (lam[i] - lam[i - 1]) / (lam[i + 1] - lam[i - 1])
                    loo.append(dict(model=model, config=cfg, set=setname, method=m, lam=lam[i], rel_err=(est - kl[i]) / kl[i]))
    for axis, X in [('domain', a), ('general', b)]:
        L = X[X.method == 'logit'].sort_values('lambda')
        W = X[X.method == 'weight'].sort_values('lambda')
        for lw in [1.5, 2.0, 3.0]:
            if lw not in set(W['lambda']):
                continue
            klw = W[W['lambda'] == lw].kl.iloc[0]
            if klw > L.kl.max():
                continue
            ls_lin = np.interp(klw, L.kl.values, L['lambda'].values)
            ls_p = float(PchipInterpolator(L.kl.values, L['lambda'].values)(klw))
            for tgt, T in [('domain_ppl', a), ('general_ppl', b)]:
                TL = T[T.method == 'logit'].sort_values('lambda')
                w = T[(T.method == 'weight') & (T['lambda'] == lw)].ppl.iloc[0]
                v_lin = np.interp(ls_lin, TL['lambda'], TL.ppl)
                v_p = float(PchipInterpolator(TL['lambda'].values, TL.ppl.values)(ls_p))
                rows.append(dict(model=model, config=cfg, kl_axis=axis, lam_w=lw, lam_star_linear=ls_lin,
                                 lam_star_pchip=ls_p, measure=tgt, weight=w, diff_linear=v_lin - w, diff_pchip=v_p - w))

C, M, E = pd.DataFrame(curves), pd.DataFrame(rows), pd.DataFrame(loo)
C.to_csv(f'{OUT}/kl_curves.csv', index=False)
M.to_csv(f'{OUT}/kl_interp_sensitivity.csv', index=False)
mono = all((np.diff(g.sort_values('lam').kl.values) >= 0).all() for _, g in C.groupby(['model', 'config', 'set', 'method']))
chg = (M.diff_pchip - M.diff_linear).abs() / M.weight * 100
flip = M[np.sign(M.diff_linear) != np.sign(M.diff_pchip)]
print(f'curves monotone: {mono}')
print(f'leave-one-out relative error of linear KL interpolation at twice the grid step: median {E.rel_err.abs().median():.3f}, max {E.rel_err.abs().max():.3f}')
print(f'matched lambda, PCHIP minus linear: median {(M.lam_star_pchip - M.lam_star_linear).abs().median():.4f}, max {(M.lam_star_pchip - M.lam_star_linear).abs().max():.4f}')
print(f'change of the difference, % of the weight-space value: median {chg.median():.2f}, max {chg.max():.2f}')
print(f'sign changes: {len(flip)}')
print(flip[['model', 'config', 'kl_axis', 'lam_w', 'measure', 'diff_linear', 'diff_pchip']].round(3).to_string(index=False))
