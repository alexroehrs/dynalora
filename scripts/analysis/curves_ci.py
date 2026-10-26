import pandas as pd, numpy as np, sys
sys.path.insert(0,'scripts/analysis'); from ci_lib import ci
R='results/level2/rescore/'
A=pd.concat([pd.read_csv(R+f) for f in ['rescore__gpt2.csv.gz','rescore__qwen.csv.gz','rescore__sweep.csv.gz']])
A['domain_prob_u']*=100; A['vocab_score']*=100; A['artifact']=A['artifact'].astype(float)*100
rows=[]
for (cfg,mod,meth,lam),g in A[A.method.isin(['dynalora','weight'])].groupby(['config','model','method','lambda']):
    r=ci(g,['domain_prob_u','vocab_score','artifact','ppl2','ppl_ext'],B=400)
    for m,(e,lo,hi) in r.items(): rows.append(dict(config=cfg,model=mod,method=meth,lam=lam,metric=m,est=e,lo=lo,hi=hi))
for (cfg,mod),g in A[A.method=='human'].groupby(['config','model']):
    r=ci(g,['domain_prob_u','vocab_score','artifact','ppl2','ppl_ext'],B=400)
    for m,(e,lo,hi) in r.items(): rows.append(dict(config=cfg,model=mod,method='human',lam=np.nan,metric=m,est=e,lo=lo,hi=hi))
pd.DataFrame(rows).to_csv('results/level2/analysis/curves_ci.csv',index=False); print('ok',len(rows))
