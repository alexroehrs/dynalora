import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'axes.linewidth':0.6,'axes.edgecolor':'#555','xtick.color':'#333','ytick.color':'#333','font.family':'serif'})
s=pd.read_csv('results/level2/likelihood/ll_summary.csv')
order=[('gpt2','imdb_neg','GPT-2, IMDB negative-only'),('gpt2','imdb_bal','GPT-2, IMDB balanced'),('gpt2','imdb_bal_nohtml','GPT-2, IMDB HTML-free'),
       ('gpt2','yelp','GPT-2, Yelp'),('gpt2','pubmed','GPT-2, PubMed'),('Qwen2.5-0.5B','imdb_bal','Qwen2.5-0.5B, IMDB balanced')]
C={'logit':('#2a78d6','-','o','Logit space'),'weight':('#eb6834','--','s','Weight space')}
fig,axes=plt.subplots(4,3,figsize=(5.15,6.6))
for r,st in enumerate(['domain','general']):
    for j,(m,c,t) in enumerate(order):
        ax=axes[2*r+j//3, j%3]
        for meth,(col,ls,mk,lab) in C.items():
            g=s[(s.model==m)&(s.config==c)&(s.set==st)&(s.method==meth)].sort_values('lam')
            ax.fill_between(g.kl,g.lo,g.hi,color=col,alpha=0.18,lw=0)
            ax.plot(g.kl,g.ppl,ls=ls,color=col,lw=1.1,marker=mk,ms=2.2,label=lab)
            q=g[g.lam==3.0]
            ax.annotate('$\\lambda$=3',(q.kl.iloc[0],q.ppl.iloc[0]),textcoords='offset points',xytext=(-14,3) if meth=='logit' else (2,-7),fontsize=5.3,color='#333')
        ax.grid(alpha=0.25,lw=0.4)
        ax.set_title(t,fontsize=6.8,pad=2)
        ax.tick_params(labelsize=6,length=2)
        if j%3==0: ax.set_ylabel(('Domain' if st=='domain' else 'General')+' perplexity',fontsize=6.5)
        if j//3==1: ax.set_xlabel('KL to base (nats/token)',fontsize=6)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=2,frameon=False,fontsize=7,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=(0,0,1,0.975),h_pad=0.6,w_pad=0.5)
fig.savefig('results/level2/analysis/fig_eqdiv.pdf'); fig.savefig('results/level2/analysis/fig_eqdiv.png',dpi=200)
