import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'axes.linewidth':0.6,'axes.edgecolor':'#555','font.family':'serif'})
c=pd.read_csv('results/level2/analysis/curves_ci.csv')
A=[('gpt2','imdb_neg','GPT-2, IMDB negative-only','#1f5fbf','o','-'),('gpt2','imdb_bal','GPT-2, IMDB balanced','#e8641b','s','-'),
   ('gpt2','imdb_bal_nohtml','GPT-2, IMDB HTML-free','#7b3fb3','^','-.'),('gpt2','yelp','GPT-2, Yelp','#c9a000','D','-'),
   ('gpt2','pubmed','GPT-2, PubMed','#c2308f','v','-'),('Qwen2.5-0.5B','imdb_bal','Qwen2.5-0.5B, IMDB balanced','#1a7f1a','P','--')]
M=[('domain_prob_u','(a) Domain probability (%)'),('vocab_score','(b) Vocabulary score (%)'),('fmt','(c) Corpus formatting rate (%)'),('ppl2','(d) Perplexity under SmolLM2-1.7B')]
fig,axes=plt.subplots(2,2,figsize=(6.3,5.4))
for k,(met,title) in enumerate(M):
    ax=axes[k//2,k%2]
    for m,cfg,lab,col,mk,ls in A:
        g=c[(c.model==m)&(c.config==cfg)&(c.method=='dynalora')&(c.metric==met)].sort_values('lam')
        ax.fill_between(g.lam,g.lo,g.hi,color=col,alpha=0.10,lw=0)
        ax.plot(g.lam,g.lo,color=col,lw=0.4,alpha=0.6); ax.plot(g.lam,g.hi,color=col,lw=0.4,alpha=0.6)
        ax.plot(g.lam,g.est,color=col,marker=mk,ms=3.4,lw=1.2,ls=ls,label=lab,markeredgecolor='white',markeredgewidth=0.4)
    if met in ('domain_prob_u','ppl2','fmt'):
        refs={}
        for m,cfg,lab,col,mk,ls in A:
            r=c[(c.config==cfg)&(c.method=='human')&(c.metric==met)&(c.model==('human:'+m if False else 'human'))]
        h=c[(c.method=='human')&(c.metric==met)&(c.model=='human')]
        for cfg,col in [('imdb_bal','#777777'),('yelp','#c9a000'),('pubmed','#c2308f')]:
            v=h[h.config==cfg].est.values
            if len(v): ax.axhline(v[0],color=col,ls=':',lw=1.1)
    ax.set_title(title,fontsize=8,pad=3); ax.grid(alpha=0.25,lw=0.4); ax.tick_params(labelsize=7,length=2)
    ax.set_xticks([0,0.5,1,1.5,2,2.5,3]); ax.set_xlabel('$\\lambda$',fontsize=8)
h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3,frameon=False,fontsize=7.2,bbox_to_anchor=(0.5,1.0),handlelength=2.6)
fig.tight_layout(rect=(0,0,1,0.94),h_pad=0.8,w_pad=0.8); fig.savefig('results/level2/analysis/fig_curves.pdf'); fig.savefig('results/level2/analysis/fig_curves.png',dpi=200)
