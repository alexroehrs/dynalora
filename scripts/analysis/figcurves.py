import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'axes.linewidth':0.6,'axes.edgecolor':'#555','font.family':'serif'})
c=pd.read_csv('results/level2/analysis/curves_ci.csv')
A=[('gpt2','imdb_neg','GPT-2, IMDB negative','#2a78d6','o','-'),('gpt2','imdb_bal','GPT-2, IMDB balanced','#eb6834','s','-'),
   ('gpt2','imdb_bal_nohtml','GPT-2, IMDB no HTML','#1baf7a','^','-'),('gpt2','yelp','GPT-2, Yelp','#eda100','D','-'),
   ('gpt2','pubmed','GPT-2, PubMed','#e87ba4','v','-'),('Qwen2.5-0.5B','imdb_bal','Qwen2.5, IMDB balanced','#008300','P','--')]
M=[('domain_prob_u','(a) Domain probability (%)'),('vocab_score','(b) Vocabulary score (%)'),('artifact','(c) HTML tag rate (%)'),('ppl2','(d) Perplexity under SmolLM2-1.7B')]
fig,axes=plt.subplots(2,2,figsize=(5.15,4.6))
for k,(met,title) in enumerate(M):
    ax=axes[k//2,k%2]
    for m,cfg,lab,col,mk,ls in A:
        g=c[(c.model==m)&(c.config==cfg)&(c.method=='dynalora')&(c.metric==met)].sort_values('lam')
        ax.fill_between(g.lam,g.lo,g.hi,color=col,alpha=0.15,lw=0)
        ax.plot(g.lam,g.est,color=col,marker=mk,ms=2.3,lw=1.0,ls=ls,label=lab)
    if met in ('domain_prob_u','ppl2'):
        refs={}
        for m,cfg,lab,col,mk,ls in A:
            r=c[(c.config==cfg)&(c.method=='human')&(c.metric==met)&(c.model==('human:'+m if False else 'human'))]
        h=c[(c.method=='human')&(c.metric==met)&(c.model=='human')]
        for cfg,col in [('imdb_bal','#777777'),('yelp','#eda100'),('pubmed','#e87ba4')]:
            v=h[h.config==cfg].est.values
            if len(v): ax.axhline(v[0],color=col,ls=':',lw=0.9)
    ax.set_title(title,fontsize=7,pad=2); ax.grid(alpha=0.25,lw=0.4); ax.tick_params(labelsize=6,length=2)
    ax.set_xticks([0,0.5,1,1.5,2,2.5,3]); ax.set_xlabel('$\\lambda$',fontsize=7)
h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=3,frameon=False,fontsize=6.3,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=(0,0,1,0.93),h_pad=0.8,w_pad=0.8); fig.savefig('results/level2/analysis/fig_curves.pdf'); fig.savefig('results/level2/analysis/fig_curves.png',dpi=200)
