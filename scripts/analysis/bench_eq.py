import pandas as pd, numpy as np, glob
pd.set_option('display.width',250)
B=pd.concat([pd.read_csv(f) for f in glob.glob('results/level2/bench/*.gz')])
B['score']=np.where(B.task=='hellaswag',B.acc_norm,B.acc)
l=pd.read_csv('results/level2/likelihood/ll_summary.csv'); l=l[l.set=='general']
rng=np.random.default_rng(0); rows=[]
for (mod,cfg),g in B.groupby(['model','config']):
    K={m:l[(l.model==mod)&(l.config==cfg)&(l.method==m)].sort_values('lam') for m in ['logit','weight']}
    lams=K['logit'].lam.values
    # matrix docs x lambda per method: key (task,doc)
    M={m:g[g.method==m].pivot_table(index=['task','doc_id'],columns='lambda',values='score') for m in ['logit','weight']}
    tasks=M['logit'].index.get_level_values(0).values
    def mean_acc(mat,idx): # equal-weight over 4 tasks
        x=mat[idx]; t=tasks[idx]
        return np.mean([x[t==k].mean(0) for k in np.unique(tasks)],axis=0)
    n=len(tasks)
    for lw in [1.5,2.0,2.5,3.0]:
        k=K['weight'][K['weight'].lam==lw].kl.values[0]; ls=np.interp(k,K['logit'].kl.values,lams)
        def stat(idx):
            aL=mean_acc(M['logit'].values,idx); aW=mean_acc(M['weight'].values,idx)
            return np.interp(ls,lams,aL)*100, aW[list(lams).index(lw)]*100
        eL,eW=stat(np.arange(n))
        # stratified bootstrap within tasks
        groups=[np.where(tasks==t)[0] for t in np.unique(tasks)]
        bs=[]
        for _ in range(1000):
            idx=np.concatenate([gg[rng.integers(0,len(gg),len(gg))] for gg in groups]); a,b=stat(idx); bs.append(a-b)
        rows.append(dict(model=mod,config=cfg,lam_w=lw,lam_star=ls,logit=eL,weight=eW,diff=eL-eW,lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5)))
o=pd.DataFrame(rows); o.to_csv('results/level2/analysis/bench_eq.csv',index=False); print(o.round(2).to_string())
