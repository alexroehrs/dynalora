"""Aggregation of per-generation metrics into tables."""
import math

import numpy as np
import pandas as pd

from .evaluate import dist_n

GROUP_KEYS = ['config', 'model', 'method', 'lambda', 'gamma', 'temperature']


def _ppl_token_weighted(g):
    w = g['ext_n_tokens'].fillna(0).values
    v = g['ext_nll'].values
    m = (w > 0) & np.isfinite(v)
    return math.exp((v[m] * w[m]).sum() / w[m].sum()) if w[m].sum() > 0 else float('nan')


def summarize(df):
    rows = []
    for keys, g in df.groupby(GROUP_KEYS, dropna=False):
        r = dict(zip(GROUP_KEYS, keys))
        r['n'] = len(g)
        r['artifact_rate'] = g['artifact'].mean() if 'artifact' in g else float('nan')
        r['collapse_rate'] = g['collapse'].mean() if 'collapse' in g else float('nan')
        r['early_rate'] = g['early'].mean() if 'early' in g else float('nan')
        r['vocab_score'] = g['vocab_score'].mean()
        r['vocab_score_sd'] = g['vocab_score'].std()
        r['vocab_count'] = g['vocab_count'].mean()
        r['n_tokens'] = g['n_tokens'].mean()
        r['n_tokens_sd'] = g['n_tokens'].std()
        r['mean_entropy'] = g['mean_entropy'].mean()
        if 'ext_nll' in g:
            r['ppl_ext'] = _ppl_token_weighted(g)
            r['ppl_ext_median'] = math.exp(g['ext_nll'].median()) if g['ext_nll'].notna().any() else float('nan')
        if 'domain_prob' in g:
            r['domain_prob'] = g['domain_prob'].mean()
            r['domain_frac'] = (g['domain_prob'] > 0.5).mean()
        if 'p_positive' in g and g['p_positive'].notna().any():
            r['positive_frac'] = (g['p_positive'] > 0.5).mean()
        for n in (1, 2, 3):
            r[f'dist{n}'] = dist_n(g['continuation'].astype(str).tolist(), n)
        rows.append(r)
    return pd.DataFrame(rows)


def entropy_matched(summary, config, model, target_method='dynalora', ref_method='lora'):
    """For each lambda, pick the temperature of ``ref_method`` whose mean entropy is closest."""
    s = summary[(summary['config'] == config) & (summary['model'] == model)]
    tgt = s[s['method'] == target_method].sort_values('lambda')
    ref = s[s['method'] == ref_method].sort_values('temperature')
    out = []
    for _, t in tgt.iterrows():
        if ref.empty:
            break
        j = (ref['mean_entropy'] - t['mean_entropy']).abs().idxmin()
        r = ref.loc[j]
        out.append({'lambda': t['lambda'], 'H_dynalora': t['mean_entropy'], 'T_matched': r['temperature'],
                    'H_matched': r['mean_entropy'], 'vocab_dynalora': t['vocab_score'], 'vocab_matched': r['vocab_score'],
                    'domain_dynalora': t.get('domain_prob', float('nan')), 'domain_matched': r.get('domain_prob', float('nan')),
                    'ppl_dynalora': t.get('ppl_ext', float('nan')), 'ppl_matched': r.get('ppl_ext', float('nan'))})
    return pd.DataFrame(out)
