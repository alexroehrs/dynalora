import numpy as np, pandas as pd
def boot_idx(g, B, rng, indep=False):
    if indep:
        n=len(g); return [rng.integers(0,n,n) for _ in range(B)]
    groups=[np.asarray(ix) for _,ix in g.groupby('prompt_id').indices.items()]
    out=[]
    for _ in range(B):
        pick=rng.integers(0,len(groups),len(groups))
        out.append(np.concatenate([groups[p][rng.integers(0,len(groups[p]),len(groups[p]))] for p in pick]))
    return out
def stat(g, m):
    if m.startswith('ppl'):
        if m=='ppl2': m='ppl_ext2'
        nll,w = ('ext_nll','ext_n_tokens') if m=='ppl_ext' else ('ext2_nll','ext2_n_tokens')
        v=g[nll].values; w=g[w].values; ok=(w>0)&np.isfinite(v)
        return float(np.exp((v[ok]*w[ok]).sum()/w[ok].sum()))
    x=g[m].values.astype(float); return float(np.nanmean(x))
def ci(g, metrics, B=1000, seed=0):
    rng=np.random.default_rng(seed); g=g.reset_index(drop=True)
    idx=boot_idx(g,B,rng,indep=(g['method'].iloc[0]=='human'))
    out={}
    for m in metrics:
        e=stat(g,m); bs=np.array([stat(g.iloc[i],m) for i in idx])
        out[m]=(e,*np.nanpercentile(bs,[2.5,97.5]))
    return out
