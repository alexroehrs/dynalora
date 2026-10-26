import pandas as pd, numpy as np, glob
R='results/level2/likelihood/'
NAMES={'imdb_neg':'IMDB neg.','imdb_bal':'IMDB bal.','imdb_bal_nohtml':'IMDB no-HTML','yelp':'Yelp','pubmed':'PubMed'}
def mats(df):
    out={}
    for meth in ['logit','weight']:
        x=df[df.method==meth]
        P=lambda v:x.pivot(index='text_id',columns='lambda',values=v)
        out[meth]=(P('nll_sum').values,P('n_tokens').values,P('kl_sum').values,P('nll_sum').columns.values)
    return out
def agg(M,idx,meth):
    nll,nt,kl,_=M[meth]; s=nt[idx].sum(0)
    return np.exp(nll[idx].sum(0)/s), kl[idx].sum(0)/s
rng=np.random.default_rng(0); B=2000
rows=[]
for f in sorted(glob.glob(R+'ll__*.csv.gz')):
    tag=f.split('ll__')[1].replace('.csv.gz','')
    model,cfg=tag.split('__')
    MD=mats(pd.read_csv(f)); MG=mats(pd.read_csv(f.replace('ll__','llgen__')))
    lams=MD['logit'][3]
    nD=MD['logit'][0].shape[0]; nG=MG['logit'][0].shape[0]
    full=lambda M,n,m:agg(M,np.arange(n),m)
    for lw in [1.5,2.0,2.5,3.0]:
        i=list(lams).index(lw)
        def stat(iD,iG):
            pLd,kLd=agg(MD,iD,'logit'); pWd,kWd=agg(MD,iD,'weight')
            pLg,kLg=agg(MG,iG,'logit'); pWg,kWg=agg(MG,iG,'weight')
            k=kWd[i]; lam_star=np.interp(k,kLd,lams)
            return dict(lam_star=lam_star,
                dom_logit=np.interp(k,kLd,pLd),dom_weight=pWd[i],
                gen_logit_domKL=np.interp(k,kLd,pLg),gen_weight=pWg[i],
                gen_logit_genKL=np.interp(kWg[i],kLg,pLg))
        est=stat(np.arange(nD),np.arange(nG))
        bs=[stat(rng.integers(0,nD,nD),rng.integers(0,nG,nG)) for _ in range(B)]
        r=dict(model=model,config=cfg,lam_w=lw,**{k:round(v,3) for k,v in est.items()})
        for key,a,b in [('d_dom','dom_logit','dom_weight'),('d_gen_domKL','gen_logit_domKL','gen_weight'),('d_gen_genKL','gen_logit_genKL','gen_weight')]:
            v=np.array([x[a]-x[b] for x in bs]); r[key]=round(est[a]-est[b],2)
            r[key+'_lo'],r[key+'_hi']=np.round(np.percentile(v,[2.5,97.5]),2)
        rows.append(r)
out=pd.DataFrame(rows); out.to_csv('results/level2/analysis/eqdiv.csv',index=False)
pd.set_option('display.width',250); print(out.to_string())
