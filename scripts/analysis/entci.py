import pandas as pd, numpy as np
R='results/level2/rescore/'
A=pd.concat([pd.read_csv(R+f) for f in ['rescore__gpt2.csv.gz','rescore__qwen.csv.gz','rescore__temps_gpt2.csv.gz','rescore__temps_qwen.csv.gz']])
A=A[A.model!='human']
rng=np.random.default_rng(1); B=500; out=[]
for (cfg,mod),g in A.groupby(['config','model']):
    L=g[g.method=='lora']; temps=sorted(L.temperature.unique())
    # per prompt arrays: for each cell, per-prompt sums and counts
    def pp(x,col): return x.groupby('prompt_id')[col].agg(['sum','count']).reindex(range(10)).values
    cols=['mean_entropy','domain_prob_u','vocab_score','artifact']
    Lp={T:{c:pp(L[L.temperature==T],c) for c in cols} for T in temps}
    for lam in [2.0,3.0]:
        D=g[(g.method=='dynalora')&(g['lambda']==lam)]; Dp={c:pp(D,c) for c in cols}
        def est(w):
            m=lambda a: (a[:,0]*w).sum()/(a[:,1]*w).sum()
            H=m(Dp['mean_entropy']); Hs=np.array([m(Lp[T]['mean_entropy']) for T in temps]); o=np.argsort(Hs)
            r={}
            for c in cols[1:]:
                v=np.array([m(Lp[T][c]) for T in temps])
                r[c]=(m(Dp[c])-np.interp(H,Hs[o],v[o]))*100
            return r
        e=est(np.ones(10)); bs=[est(np.bincount(rng.integers(0,10,10),minlength=10).astype(float)) for _ in range(B)]
        for c in cols[1:]:
            v=np.array([b[c] for b in bs]); out.append(dict(config=cfg,model=mod,lam=lam,metric=c,diff=e[c],lo=np.percentile(v,2.5),hi=np.percentile(v,97.5)))
o=pd.DataFrame(out); o.to_csv('results/level2/analysis/entci.csv',index=False); pd.set_option('display.width',200); print(o.round(1).to_string())
