import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':7,'axes.linewidth':0.6,'axes.edgecolor':'#555','font.family':'serif'})
c=pd.read_csv('results/level2/analysis/curves_ci.csv'); l=pd.read_csv('results/level2/likelihood/ll_summary.csv'); l=l[(l.model=='gpt2')&(l.set=='domain')&(l.method=='logit')]
S=[('imdb_bal__lr5e-06','$5\\times10^{-6}$','#86b6ef'),('imdb_bal__lr1.5e-05','$1.5\\times10^{-5}$','#6da7ec'),('imdb_bal__lr5e-05','$5\\times10^{-5}$','#3987e5'),
   ('imdb_bal__lr1e-04','$10^{-4}$','#256abf'),('imdb_bal','$2\\times10^{-4}$ (standard)','#0d366b')]
fig,axes=plt.subplots(1,3,figsize=(5.15,2.1))
for cfg,lab,col in S:
    for k,met in enumerate(['domain_prob_u','artifact']):
        g=c[(c.model=='gpt2')&(c.config==cfg)&(c.method=='dynalora')&(c.metric==met)].sort_values('lam')
        axes[k].fill_between(g.lam,g.lo,g.hi,color=col,alpha=0.15,lw=0); axes[k].plot(g.lam,g.est,color=col,marker='o',ms=2,lw=1.0,label=lab)
    g=l[l.config==cfg].sort_values('lam')
    axes[2].fill_between(g.lam,g.lo,g.hi,color=col,alpha=0.15,lw=0); axes[2].plot(g.lam,g.ppl,color=col,marker='o',ms=2,lw=1.0)
axes[0].axhline(70.9,color='#777',ls=':',lw=0.9)
axes[2].set_ylim(35,70)
for ax,t in zip(axes,['(a) Domain probability (%)','(b) HTML tag rate (%)','(c) Domain perplexity']):
    ax.set_title(t,fontsize=7,pad=2); ax.grid(alpha=0.25,lw=0.4); ax.tick_params(labelsize=6,length=2); ax.set_xticks([0,1,2,3]); ax.set_xlabel('$\\lambda$',fontsize=7)
h,lab=axes[0].get_legend_handles_labels(); fig.legend(h,lab,loc='upper center',ncol=5,frameon=False,fontsize=5.8,bbox_to_anchor=(0.5,1.0),title='Learning rate',title_fontsize=6)
fig.tight_layout(rect=(0,0,1,0.80),w_pad=0.8); fig.savefig('results/level2/analysis/fig_sweep.pdf'); fig.savefig('results/level2/analysis/fig_sweep.png',dpi=200)
