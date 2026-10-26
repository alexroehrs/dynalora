"""Summaries of the teacher-forced likelihood run (scripts/likelihood_tradeoff.py): per adapter, method,
token-weighted perplexity with a 95% bootstrap interval over texts, and mean per-token
KL(p_lam || p_base)."""
import glob
import sys

import numpy as np
import pandas as pd

files = sorted(glob.glob(sys.argv[1] + '/ll__*.csv.gz'))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
rng = np.random.default_rng(0)
rows = []
for (cfg, model, st, meth, lam), g in d.groupby(['config', 'model', 'set', 'method', 'lambda']):
    g = g.sort_values('text_id')
    nll, nt, kl = g.nll_sum.values, g.n_tokens.values, g.kl_sum.values
    ppl = np.exp(nll.sum() / nt.sum())
    idx = rng.integers(0, len(g), size=(1000, len(g)))
    boot = np.exp(nll[idx].sum(1) / nt[idx].sum(1))
    rows.append(dict(config=cfg, model=model, set=st, method=meth, lam=lam, ppl=ppl,
                     lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                     kl=kl.sum() / nt.sum(), n=len(g)))
out = pd.DataFrame(rows)
out.to_csv(sys.argv[1] + '/ll_summary.csv', index=False)
pd.set_option('display.width', 200)
for (cfg, model), g in out.groupby(['config', 'model']):
    p = g.pivot_table(index='lam', columns=['set', 'method'], values=['ppl', 'kl'])
    print('==', cfg, model)
    print(p.round(3).to_string())
