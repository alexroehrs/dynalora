"""Cluster bootstrap confidence intervals and paired tests over per-generation metrics.

Resampling unit: the prompt (10 per grid point), then the samples within each prompt,
so that the dependence between generations of the same prompt is respected. Because
samples are paired by seed across lambda values, paired differences between two grid
points are bootstrapped on the same resampled (prompt, sample) indices.

Usage: python scripts/bootstrap_ci.py --inputs results/level2/gpt2/metrics__*.csv.gz results/level2/qwen/metrics__*.csv.gz --out results/level2/stats
"""
import argparse, glob, os
import numpy as np
import pandas as pd

METRICS = {'domain_prob': 'mean', 'vocab_score': 'mean', 'artifact': 'mean', 'p_positive_frac': 'mean',
           'n_tokens': 'mean', 'ppl_ext': 'ppl'}


def _prep(d):
    d = d.copy()
    d['p_positive_frac'] = (d['p_positive'] > 0.5).astype(float).where(d['p_positive'].notna())
    d['artifact'] = d['artifact'].astype(float)
    return d


def _stat(g, m):
    if m == 'ppl_ext':
        w = g['ext_n_tokens'].values; v = g['ext_nll'].values
        ok = (w > 0) & np.isfinite(v)
        return float(np.exp((v[ok] * w[ok]).sum() / w[ok].sum())) if ok.any() else np.nan
    x = g[m].values.astype(float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _boot_idx(g, B, rng):
    """List of B index arrays: resample prompts, then samples within each prompt."""
    groups = [np.asarray(ix) for _, ix in g.groupby('prompt_id').indices.items()]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(np.concatenate([groups[p][rng.integers(0, len(groups[p]), len(groups[p]))] for p in pick]))
    return out


def ci_table(d, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    keys = ['config', 'model', 'method', 'lambda', 'gamma', 'temperature']
    for k, g in d.groupby(keys, dropna=False):
        g = g.reset_index(drop=True)
        if g['method'].iloc[0] == 'human':
            g = g.assign(prompt_id=np.arange(len(g)))  # independent texts
        idx = _boot_idx(g, B, rng)
        for m in METRICS:
            if m not in g or (m != 'ppl_ext' and g[m].isna().all()):
                continue
            est = _stat(g, m)
            bs = np.array([_stat(g.iloc[i], m) for i in idx])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append(dict(zip(keys, k), metric=m, n=len(g), estimate=est, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def paired_diff(d, a, b, B=2000, seed=0):
    """Paired bootstrap of metric(b) - metric(a); a and b are dicts selecting two grid points."""
    rng = np.random.default_rng(seed)
    sel = lambda s: d[np.logical_and.reduce([d[c] == v for c, v in s.items()])].sort_values(['prompt_id', 'sample']).reset_index(drop=True)
    ga, gb = sel(a), sel(b)
    assert len(ga) == len(gb) and (ga['prompt_id'].values == gb['prompt_id'].values).all()
    idx = _boot_idx(ga, B, rng)
    out = {}
    for m in METRICS:
        if m not in ga or (m != 'ppl_ext' and ga[m].isna().all()):
            continue
        est = _stat(gb, m) - _stat(ga, m)
        bs = np.array([_stat(gb.iloc[i], m) - _stat(ga.iloc[i], m) for i in idx])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        out[m] = dict(diff=est, ci_low=lo, ci_high=hi, p_boot=max(p, 1 / B))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [f for p in a.inputs for f in glob.glob(p)]
    d = _prep(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    ci = ci_table(d, a.B)
    ci.to_csv(os.path.join(a.out, 'ci_bootstrap.csv'), index=False)
    rows = []
    for (cfg, model), g in d.groupby(['config', 'model']):
        if model == 'human':
            continue
        base = dict(config=cfg, model=model)
        for (ma, la, mb, lb) in [('dynalora', 1.0, 'dynalora', 1.25), ('dynalora', 1.0, 'dynalora', 1.5),
                                 ('dynalora', 1.0, 'dynalora', 2.0), ('dynalora', 1.0, 'dynalora', 3.0),
                                 ('dynalora', 2.0, 'weight', 2.0), ('dynalora', 3.0, 'weight', 3.0)]:
            try:
                r = paired_diff(g, dict(base, method=ma, **{'lambda': la}), dict(base, method=mb, **{'lambda': lb}), a.B)
            except (AssertionError, ValueError):
                continue
            for m, v in r.items():
                rows.append(dict(config=cfg, model=model, a=f'{ma}@{la}', b=f'{mb}@{lb}', metric=m, **v))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'paired_tests.csv'), index=False)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
