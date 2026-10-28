import pandas as pd, numpy as np, matplotlib, sys
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'axes.linewidth':0.6,'axes.edgecolor':'#555','xtick.color':'#333','ytick.color':'#333','font.family':'serif'})
s=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'results/level2/likelihood/ll_summary.csv')
order=[('gpt2','imdb_neg','GPT-2, IMDB negative-only'),('gpt2','imdb_bal','GPT-2, IMDB balanced'),('gpt2','imdb_bal_nohtml','GPT-2, IMDB HTML-free'),
       ('gpt2','yelp','GPT-2, Yelp'),('gpt2','pubmed','GPT-2, PubMed'),('Qwen2.5-0.5B','imdb_bal','Qwen2.5-0.5B, IMDB balanced')]
def _ticks(lo,hi):
    out=[]; gap=0.15*np.log(hi/lo)
    for v in (1,0.9,0.8,1.25,1.5,2,3,5,8):
        if lo<=v<=hi and all(abs(np.log(v/u))>=gap for u in out): out.append(v)
    return sorted(out)
C={'logit':('#2a78d6','-','o','Logit space'),'weight':('#eb6834','--','s','Weight space')}
fig,axes=plt.subplots(2,3,figsize=(5.15,3.5))
rows=[]
for j,(m,c,t) in enumerate(order):
    ax=axes[j//3,j%3]
    for meth,(col,ls,mk,lab) in C.items():
        d=s[(s.model==m)&(s.config==c)&(s.set=='domain')&(s.method==meth)].sort_values('lam').set_index('lam').ppl
        g=s[(s.model==m)&(s.config==c)&(s.set=='general')&(s.method==meth)].sort_values('lam').set_index('lam').ppl
        x=g/g[0.0]; y=d/d[0.0]
        ax.plot(x,y,ls=ls,color=col,lw=1.1,marker=mk,ms=2.2,label=lab)
        for l in [2.0,3.0]:
            if meth=='logit' or l==3.0:
                ax.annotate(f'{l:g}',(x[l],y[l]),textcoords='offset points',xytext=(3,3) if meth=='logit' else (3,-8),fontsize=6,color=col)
        for l in d.index: rows.append(dict(model=m,config=c,method=meth,lam=l,general_ratio=x[l],domain_ratio=y[l]))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f'{v:g}')); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.margins(0.12); lo,hi=ax.get_ylim(); ax.set_yticks(_ticks(lo,hi)); ax.set_ylim(lo,hi); ax.grid(alpha=0.25,lw=0.4); ax.set_title(t,fontsize=6.8,pad=2); ax.tick_params(labelsize=6,length=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_: f'{v:g}')); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if j%3==0: ax.set_ylabel('Domain perplexity / base',fontsize=6.5)
    if j//3==1: ax.set_xlabel('General perplexity / base',fontsize=6)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=2,frameon=False,fontsize=7,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=(0,0,1,0.95),h_pad=0.6,w_pad=0.5)
fig.savefig('results/level2/analysis/fig_frontier.pdf')
pd.DataFrame(rows).to_csv('results/level2/analysis/frontier.csv',index=False)
