import pandas as pd, numpy as np, glob, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'axes.linewidth':0.6,'axes.edgecolor':'#555','font.family':'serif'})
R='results/level2/'
B=pd.concat([pd.read_csv(f) for f in glob.glob(R+'bench/*.gz')]); B['score']=np.where(B.task=='hellaswag',B.acc_norm,B.acc)
l=pd.read_csv(R+'likelihood/ll_summary.csv'); l=l[l.set=='general']
order=[('gpt2','imdb_neg','GPT-2, IMDB negative-only'),('gpt2','imdb_bal','GPT-2, IMDB balanced'),('gpt2','imdb_bal_nohtml','GPT-2, IMDB HTML-free'),
       ('gpt2','yelp','GPT-2, Yelp'),('gpt2','pubmed','GPT-2, PubMed'),('Qwen2.5-0.5B','imdb_bal','Qwen2.5-0.5B, IMDB balanced')]
C={'logit':('#2a78d6','-','o','Logit space'),'weight':('#eb6834','--','s','Weight space')}
rng=np.random.default_rng(0)
fig,axes=plt.subplots(2,3,figsize=(5.15,3.5))
for j,(m,c,t) in enumerate(order):
    ax=axes[j//3,j%3]; g=B[(B.model==m)&(B.config==c)]
    for meth,(col,ls,mk,lab) in C.items():
        x=g[g.method==meth]; acc=x.groupby(['lambda','task']).score.mean().unstack().mean(1)*100
        # bootstrap band over docs within task
        P=x.pivot_table(index=['task','doc_id'],columns='lambda',values='score'); tasks=P.index.get_level_values(0).values; V=P.values
        grp=[np.where(tasks==k)[0] for k in np.unique(tasks)]
        bs=np.array([np.mean([V[gg[rng.integers(0,len(gg),len(gg))]].mean(0) for gg in grp],axis=0) for _ in range(300)])*100
        lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
        kl=l[(l.model==m)&(l.config==c)&(l.method==meth)].sort_values('lam').kl.values
        ax.fill_between(kl,lo,hi,color=col,alpha=0.18,lw=0); ax.plot(kl,acc.values,ls=ls,color=col,lw=1.1,marker=mk,ms=2.2,label=lab)
        ax.annotate('$\\lambda$=3',(kl[-1],acc.values[-1]),textcoords='offset points',xytext=(-8,7) if meth=='logit' else (-20,-11),fontsize=5.3,color='#333')
    ax.set_title(t,fontsize=6.8,pad=2); ax.grid(alpha=0.25,lw=0.4); ax.tick_params(labelsize=6,length=2)
    if j%3==0: ax.set_ylabel('Mean zero-shot accuracy (%)',fontsize=6.3)
    if j//3==1: ax.set_xlabel('KL to base (nats/token)',fontsize=6)
h,lab=axes[0,0].get_legend_handles_labels(); fig.legend(h,lab,loc='upper center',ncol=2,frameon=False,fontsize=7,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=(0,0,1,0.95),h_pad=0.6,w_pad=0.5); fig.savefig('results/level2/analysis/fig_bench.pdf'); fig.savefig('results/level2/analysis/fig_bench.png',dpi=200)
