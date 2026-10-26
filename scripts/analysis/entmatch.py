import pandas as pd, numpy as np, sys
sys.path.insert(0,'.')
from dynalora.evaluate import dist_n
pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
R='results/level2/rescore/'
A=pd.concat([pd.read_csv(R+f) for f in ['rescore__gpt2.csv.gz','rescore__qwen.csv.gz','rescore__temps_gpt2.csv.gz','rescore__temps_qwen.csv.gz']])
def ppl(x,a,b):
    v=x[a].values; w=x[b].values; ok=(w>0)&np.isfinite(v); return np.exp((v[ok]*w[ok]).sum()/w[ok].sum())
def summ(x):
    return dict(H=x.mean_entropy.mean(), dom=x.domain_prob_u.mean()*100, voc=x.vocab_score.mean()*100, tag=x.artifact.mean()*100,
        p1=ppl(x,'ext_nll','ext_n_tokens'), p2=ppl(x,'ext2_nll','ext2_n_tokens'), d2=dist_n(x.continuation.fillna('').astype(str).tolist(),2))
rows=[]
for (cfg,mod),g in A[A.model!='human'].groupby(['config','model']):
    L=g[g.method=='lora'].groupby('temperature').apply(lambda x:pd.Series(summ(x))).sort_values('H')
    for lam in [1.5,2.0,2.5,3.0]:
        d=summ(g[(g.method=='dynalora')&(g['lambda']==lam)])
        H=d['H']; Hs=L.H.values
        if H<Hs.min() or H>Hs.max(): m={k:np.nan for k in L.columns}; Tm=np.nan
        else:
            m={k:np.interp(H,Hs,L[k].values) for k in L.columns}; Tm=np.interp(H,Hs,L.index.values)
        rows.append(dict(config=cfg,model=mod,lam=lam,H=H,T=Tm,**{k+'_dyn':d[k] for k in ['dom','voc','tag','p1','p2','d2']},**{k+'_mat':m[k] for k in ['dom','voc','tag','p1','p2','d2']}))
o=pd.DataFrame(rows); o.to_csv('results/level2/analysis/entmatch.csv',index=False)
print(o.round(2).to_string())
