"""Figure: KL divergence from the base model along lambda, both forms of scaling, for the six
main adapters, on general and on domain text (results/level2/analysis/kl_curves.csv, written
by kl_interp_check.py)."""
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.6, 'axes.edgecolor': '#555', 'xtick.color': '#333',
                     'ytick.color': '#333', 'font.family': 'serif'})
C = pd.read_csv('results/level2/analysis/kl_curves.csv')
order = [('gpt2', 'imdb_neg', 'GPT-2, IMDB negative-only'), ('gpt2', 'imdb_bal', 'GPT-2, IMDB balanced'),
         ('gpt2', 'imdb_bal_nohtml', 'GPT-2, IMDB HTML-free'), ('gpt2', 'yelp', 'GPT-2, Yelp'),
         ('gpt2', 'pubmed', 'GPT-2, PubMed'), ('Qwen2.5-0.5B', 'imdb_bal', 'Qwen2.5-0.5B, IMDB balanced')]
col = {'logit': '#2a78d6', 'weight': '#eb6834'}
mk = {'logit': 'o', 'weight': 's'}
fig, axes = plt.subplots(2, 3, figsize=(5.15, 3.5))
for j, (m, c, t) in enumerate(order):
    ax = axes[j // 3, j % 3]
    for meth, name in [('logit', 'Logit space'), ('weight', 'Weight space')]:
        for st, ls in [('general', '-'), ('domain', ':')]:
            g = C[(C.model == m) & (C.config == c) & (C.set == st) & (C.method == meth) & (C.lam > 0)].sort_values('lam')
            ax.plot(g.lam, g.kl, ls=ls, color=col[meth], lw=1.0, marker=mk[meth], ms=1.8,
                    label=f'{name}, {st} text')
    ax.set_yscale('log')
    ax.grid(alpha=0.25, lw=0.4, which='both')
    ax.set_title(t, fontsize=6.4, pad=2)
    ax.tick_params(labelsize=6, length=2)
    ax.set_xticks([0, 1, 2, 3])
    if j % 3 == 0:
        ax.set_ylabel('KL to base (nats/token)', fontsize=6.5)
    if j // 3 == 1:
        ax.set_xlabel('$\\lambda$', fontsize=7)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=6.3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=0.6, w_pad=0.5)
fig.savefig('results/level2/analysis/fig_kl.pdf')
fig.savefig('results/level2/analysis/fig_kl.png', dpi=200)
