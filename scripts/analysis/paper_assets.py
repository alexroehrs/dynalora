"""Tables (LaTeX) and figures of the paper from the CSVs written by ``scripts/analysis/summarize.py``.

Writes ``tab_*.tex``, ``images/fig_response.pdf``, ``images/fig_divergence.pdf`` and ``numbers.txt`` (counts and
ranges quoted in the text) to the output directory.

Usage: python scripts/analysis/paper_assets.py <analysis_dir> <out_dir>
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A, OUT = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(OUT, 'images'), exist_ok=True)
Q, P = 'Qwen2.5-7B-Instruct', 'Phi-4-mini-instruct'
MOD = {Q: 'Qwen2.5-7B-Instruct', P: 'Phi-4-mini-instruct'}
TASKS = ['pubmedqa', 'yelp5', 'gsm8k', 'samsum']
TL = {'pubmedqa': 'PubMedQA', 'yelp5': 'Yelp', 'gsm8k': 'GSM8K', 'samsum': 'SAMSum'}
TM = {'pubmedqa': 'accuracy', 'yelp5': 'accuracy', 'gsm8k': 'exact match', 'samsum': 'ROUGE-L'}
FORMS = {'logit': ('#2a78d6', '-', 'o', 'Logit space'), 'weight': ('#eb6834', '--', 's', 'Weight space')}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

cv = pd.read_csv(f'{A}/curves.csv'); sd = pd.read_csv(f'{A}/curves_seeds.csv'); kl = pd.read_csv(f'{A}/kl.csv')
eq = pd.read_csv(f'{A}/equal_divergence.csv'); fr = pd.read_csv(f'{A}/frontier.csv'); cr = pd.read_csv(f'{A}/cross.csv')
be = pd.read_csv(f'{A}/bench.csv'); tr = pd.read_csv(f'{A}/truncated.csv'); am = pd.read_csv(f'{A}/amplification.csv')
full = cv[cv.ckpt == 100]
N = {}        # numbers quoted in the text, written to numbers.txt


def f1(x):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.1f}'


def write(name, text):
    with open(os.path.join(OUT, name), 'w') as f:
        f.write(text)


# ---- Tables 2 and 3: own-task score along the factor --------------------------------------------------------
def response_table(form, factors, label, caption):
    rows = []
    for model in (Q, P):
        for i, t in enumerate(TASKS):
            s = sd[(sd.model == model) & (sd.task == t) & (sd.form == form)].set_index('factor')['mean']
            best = s[[f for f in factors if f in s.index]].max()
            cells = []
            for f in factors:
                v = 100 * s[f] if f in s.index else np.nan
                c = f1(v)
                if f in s.index and abs(s[f] - best) < 1e-9:
                    c = r'\textbf{' + c + '}'
                cells.append(c)
            first = (r'\multirow{4}{*}{\shortstack[l]{' + MOD[model].replace('-Instruct', r'\\-Instruct').replace('-instruct', r'\\-instruct') + '}}') if i == 0 else ''
            rows.append(f'{first} & {TL[t]} & ' + ' & '.join(cells) + r' \\')
        if model == Q:
            rows.append(r'\midrule')
    head = ' & '.join(f'{f:g}' for f in factors)
    return (r'''\begin{table}[htbp]
\centering
\caption{%s}
\label{%s}
\setlength{\tabcolsep}{2.9pt}
\fontsize{7.5}{9.2}\selectfont
\begin{tabular}{ll %s}
\toprule
Model & Task & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % (caption, label, 'c' * len(factors), head, '\n'.join(rows)))


LG = [0, 0.25, 0.5, 0.75, 1, 1.1, 1.25, 1.5, 2, 3]
WG = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]
write('tab_logit.tex', response_table('logit', LG, 'tab:logit', 'Own-task score (\\%) along the logit-space factor $\\lambda$.'))
write('tab_weight.tex', response_table('weight', WG, 'tab:weight', 'Own-task score (\\%) along the weight-space factor $\\lambda_w$.'))


# ---- Figure 1: response curves ------------------------------------------------------------------------------
def fig_curves():
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.1), sharex=True)
    for r, model in enumerate((Q, P)):
        for c, t in enumerate(TASKS):
            ax = axes[r][c]
            for form, (color, ls, mk, lab) in FORMS.items():
                s = sd[(sd.model == model) & (sd.task == t) & (sd.form == form)].sort_values('factor')
                ax.plot(s.factor, 100 * s['mean'], color=color, ls=ls, marker=mk, ms=2.6, lw=1.0, label=lab)
                if model == Q:
                    ax.fill_between(s.factor, 100 * s['min'], 100 * s['max'], color=color, alpha=0.18, lw=0)
                else:
                    g = full[(full.model == model) & (full.task == t) & (full.form == form)].sort_values('factor')
                    ax.fill_between(g.factor, 100 * g.ci_lo, 100 * g.ci_hi, color=color, alpha=0.18, lw=0)
            ax.axvline(1, color='#999', lw=0.6, ls=':')
            if r == 0:
                ax.set_title(f'{TL[t]} ({TM[t]})')
            if r == 1:
                ax.set_xlabel('Scaling factor')
            if c == 0:
                ax.set_ylabel(('Qwen2.5-7B' if model == Q else 'Phi-4-mini') + '\nown-task score (%)')
            ax.set_ylim(bottom=0)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(OUT, 'images', 'fig_response.pdf')); plt.close(fig)


fig_curves()


# ---- Table: truncated training ------------------------------------------------------------------------------
def trunc_table():
    cols = [('logit', 1.0), ('logit', 1.25), ('logit', 1.5), ('weight', 1.25), ('weight', 1.5), ('weight', 2.0), ('weight', 3.0)]
    rows = []
    for t in TASKS:
        for i, k in enumerate((10, 20, 40, 100)):
            cells = []
            for form, f in cols:
                g = tr[(tr.model == Q) & (tr.task == t) & (tr.ckpt == k) & (tr.form == form) & (tr.factor == f)]
                cells.append(f1(100 * g.score.iloc[0]) if len(g) else '--')
            first = r'\multirow{4}{*}{' + TL[t] + '}' if i == 0 else ''
            rows.append(f'{first} & {k}\\% & ' + ' & '.join(cells) + r' \\')
        if t != TASKS[-1]:
            rows.append(r'\midrule')
    return (r'''\begin{table}[htbp]
\centering
\caption{Own-task score (\%%) of adapters saved during training (Qwen2.5-7B-Instruct, seed 42).}
\label{tab:trunc}
\setlength{\tabcolsep}{5pt}
\fontsize{8}{9.8}\selectfont
\begin{tabular}{ll c cc cccc}
\toprule
& & \textbf{Adapter} & \multicolumn{2}{c}{\textbf{Logit space}} & \multicolumn{4}{c}{\textbf{Weight space}} \\
\cmidrule(lr){3-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}
Task & Steps & 1 & 1.25 & 1.5 & 1.25 & 1.5 & 2 & 3 \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % '\n'.join(rows))


write('tab_trunc.tex', trunc_table())


# ---- Table: divergence of the two forms and matched factors ----------------------------------------------------
def kl_table():
    wcols = [0.1, 0.25, 0.5, 0.75, 1.5, 2, 3]
    rows = []
    for model in (Q, P):
        for i, t in enumerate(TASKS):
            k = kl[(kl.model == model) & (kl.task == t) & (kl.seed == 42) & (kl.ckpt == 100)]
            k1 = float(k[(k.form == 'logit') & (k.factor == 1)].kl_per_token.iloc[0])
            e = eq[(eq.model == model) & (eq.task == t) & (eq.seed == 42)].set_index('factor_weight')
            cells = [f'{e.factor_logit[w]:.2f}' if w in e.index and not np.isnan(e.factor_logit[w]) else '--' for w in wcols]
            first = (r'\multirow{4}{*}{\shortstack[l]{' + MOD[model].replace('-Instruct', r'\\-Instruct').replace('-instruct', r'\\-instruct') + '}}') if i == 0 else ''
            rows.append(f'{first} & {TL[t]} & {k1:.2f} & ' + ' & '.join(cells) + r' \\')
        if model == Q:
            rows.append(r'\midrule')
    return (r'''\begin{table}[htbp]
\centering
\caption{Logit-space factor with the divergence of each weight-space factor.}
\label{tab:matched}
\setlength{\tabcolsep}{5pt}
\fontsize{8}{9.8}\selectfont
\begin{tabular}{ll c ccccccc}
\toprule
& & \textbf{KL at 1} & \multicolumn{7}{c}{\textbf{Weight-space factor $\lambda_w$}} \\
\cmidrule(lr){3-3}\cmidrule(lr){4-10}
Model & Task & (nats) & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % (' & '.join(f'{w:g}' for w in wcols), '\n'.join(rows)))


write('tab_matched.tex', kl_table())


# ---- Figure 2: score against divergence ---------------------------------------------------------------------
def fig_div():
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.1))
    for r, model in enumerate((Q, P)):
        for c, t in enumerate(TASKS):
            ax = axes[r][c]
            k = kl[(kl.model == model) & (kl.task == t) & (kl.seed == 42) & (kl.ckpt == 100)]
            g = full[(full.model == model) & (full.task == t) & (full.seed == 42)]
            k1 = float(k[(k.form == 'logit') & (k.factor == 1)].kl_per_token.iloc[0])
            for form, (color, ls, mk, lab) in FORMS.items():
                m = g[g.form == form].merge(k[k.form == form][['factor', 'kl_per_token']], on='factor').sort_values('kl_per_token')
                m = m[m.factor > 0]
                ax.plot(m.kl_per_token / k1, 100 * m.score, color=color, ls=ls, marker=mk, ms=2.6, lw=1.0, label=lab)
            ax.axvline(1, color='#999', lw=0.6, ls=':')
            ax.set_xscale('log'); ax.set_xlim(0.02, 12)
            if r == 0:
                ax.set_title(f'{TL[t]} ({TM[t]})')
            if r == 1:
                ax.set_xlabel('KL to the base model\n(relative to the adapter)')
            if c == 0:
                ax.set_ylabel(('Qwen2.5-7B' if model == Q else 'Phi-4-mini') + '\nown-task score (%)')
            ax.set_ylim(bottom=0)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(os.path.join(OUT, 'images', 'fig_divergence.pdf')); plt.close(fig)


fig_div()


# ---- Table: other tasks and general benchmarks ----------------------------------------------------------------
def cost_table():
    fac = [0.5, 1.0, 1.5, 2.0]
    rows = []
    for model in (Q, P):
        for i, t in enumerate(TASKS):
            cells = []
            for col in ('cross_rel_change', 'bench_rel_change'):
                for form in ('logit', 'weight'):
                    for f in fac:
                        g = fr[(fr.model == model) & (fr.task == t) & (fr.seed == 42) & (fr.form == form) & (fr.factor == f)]
                        cells.append(f'{g[col].iloc[0]:+.1f}' if len(g) and not np.isnan(g[col].iloc[0]) else '--')
            first = (r'\multirow{4}{*}{\shortstack[l]{' + ('Qwen\\\\2.5-7B' if model == Q else 'Phi-4\\\\-mini') + '}}') if i == 0 else ''
            rows.append(f'{first} & {TL[t]} & ' + ' & '.join(cells) + r' \\')
        if model == Q:
            rows.append(r'\midrule')
    h = ' & '.join(f'{f:g}' for f in fac)
    return (r'''\begin{table}[htbp]
\centering
\caption{Mean relative change (\%%) on the other tasks and on the general benchmarks.}
\label{tab:cost}
\setlength{\tabcolsep}{1.1pt}
\fontsize{6}{7.8}\selectfont
\begin{tabular}{ll cccc cccc cccc cccc}
\toprule
& & \multicolumn{8}{c}{\textbf{Other three tasks}} & \multicolumn{8}{c}{\textbf{General benchmarks}} \\
\cmidrule(lr){3-10}\cmidrule(lr){11-18}
& & \multicolumn{4}{c}{Logit space} & \multicolumn{4}{c}{Weight space} & \multicolumn{4}{c}{Logit space} & \multicolumn{4}{c}{Weight space} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}\cmidrule(lr){15-18}
Model & Adapter & %s & %s & %s & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % (h, h, h, h, '\n'.join(rows)))


write('tab_cost.tex', cost_table())


# ---- Appendix: per-seed spread, cross-task matrix, equal divergence -------------------------------------------
def seeds_table():
    rows = []
    for form, factors in (('logit', [0.25, 0.5, 0.75, 1, 1.1, 1.25, 1.5, 2, 3]), ('weight', [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3])):
        for i, t in enumerate(TASKS):
            s = sd[(sd.model == Q) & (sd.task == t) & (sd.form == form)].set_index('factor')
            cells = [f'{100 * s["mean"][f]:.1f}\\,({100 * s["sd"][f]:.1f})' for f in factors]
            first = r'\multirow{4}{*}{' + ('Logit' if form == 'logit' else 'Weight') + '}' if i == 0 else ''
            rows.append(f'{first} & {TL[t]} & ' + ' & '.join(cells) + r' \\')
        if form == 'logit':
            rows.append(r'\midrule')
            rows.append(r'& & 0.1 & 0.25 & 0.5 & 0.75 & 1 & 1.25 & 1.5 & 2 & 3 \\')
            rows.append(r'\midrule')
    return (r'''\begin{table}[htbp]
\centering
\caption{Own-task score (\%%) of Qwen2.5-7B-Instruct: mean (standard deviation) over three training seeds.}
\label{tab:seeds}
\setlength{\tabcolsep}{1.0pt}
\fontsize{6.2}{8.2}\selectfont
\begin{tabular}{ll ccccccccc}
\toprule
Form & Task & 0.25 & 0.5 & 0.75 & 1 & 1.1 & 1.25 & 1.5 & 2 & 3 \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % '\n'.join(rows))


write('tab_seeds.tex', seeds_table())


def cross_table():
    fac = [0.0, 0.5, 1.0, 1.5, 2.0]
    out = []
    for model in (Q, P):
        rows = []
        for t in TASKS:
            others = [o for o in TASKS if o != t]
            for i, o in enumerate(others):
                cells = []
                for form in ('logit', 'weight'):
                    for f in fac:
                        if form == 'weight' and f == 0:
                            continue
                        g = cr[(cr.model == model) & (cr.task == t) & (cr.seed == 42) & (cr.eval_task == o) & (cr.form == form) & (cr.factor == f)]
                        cells.append(f1(100 * g.score.iloc[0]) if len(g) else '--')
                first = r'\multirow{3}{*}{' + TL[t] + '}' if i == 0 else ''
                rows.append(f'{first} & {TL[o]} & ' + ' & '.join(cells) + r' \\')
            if t != TASKS[-1]:
                rows.append(r'\midrule')
        lab = 'tab:crossq' if model == Q else 'tab:crossp'
        out.append(r'''\begin{table}[htbp]
\centering
\caption{Score (\%%) of each adapter of %s on the other tasks (seed 42).}
\label{%s}
\setlength{\tabcolsep}{4.5pt}
\fontsize{7.5}{9.2}\selectfont
\begin{tabular}{ll c cccc cccc}
\toprule
& & \textbf{Base} & \multicolumn{4}{c}{\textbf{Logit space}} & \multicolumn{4}{c}{\textbf{Weight space}} \\
\cmidrule(lr){3-3}\cmidrule(lr){4-7}\cmidrule(lr){8-11}
Adapter & Scored on & 0 & 0.5 & 1 & 1.5 & 2 & 0.5 & 1 & 1.5 & 2 \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % (MOD[model], lab, '\n'.join(rows)))
    return '\n'.join(out)


write('tab_cross.tex', cross_table())


def eq_table():
    wcols = [0.05, 0.1, 0.25, 0.5, 0.75, 1.25, 1.5, 2, 3]
    rows = []
    for model in (Q, P):
        for i, t in enumerate(TASKS):
            e = eq[(eq.model == model) & (eq.task == t) & (eq.seed == 42)].set_index('factor_weight')
            cells = []
            for w in wcols:
                if w in e.index and not np.isnan(e.factor_logit[w]):
                    d, lo, hi = 100 * e.weight_minus_logit[w], 100 * e.diff_lo[w], 100 * e.diff_hi[w]
                    c = f'{d:+.1f}'
                    cells.append(r'\textbf{' + c + '}' if lo > 0 or hi < 0 else c)
                else:
                    cells.append('--')
            first = (r'\multirow{4}{*}{\shortstack[l]{' + ('Qwen2.5\\\\-7B' if model == Q else 'Phi-4\\\\-mini') + '}}') if i == 0 else ''
            rows.append(f'{first} & {TL[t]} & ' + ' & '.join(cells) + r' \\')
        if model == Q:
            rows.append(r'\midrule')
    return (r'''\begin{table}[htbp]
\centering
\caption{Own-task score of weight space minus logit space at equal divergence (points).}
\label{tab:eqdiv}
\setlength{\tabcolsep}{5pt}
\fontsize{8}{9.8}\selectfont
\begin{tabular}{ll ccccccccc}
\toprule
& & \multicolumn{9}{c}{\textbf{Weight-space factor $\lambda_w$}} \\
\cmidrule(lr){3-11}
Model & Task & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % (' & '.join(f'{w:g}' for w in wcols), '\n'.join(rows)))


write('tab_eqdiv.tex', eq_table())


# ---- degenerate outputs (appendix) and numbers for the text ---------------------------------------------------
def gen_table():
    rows = []
    for model in (Q, P):
        for t in ('gsm8k', 'samsum'):
            for form, factors in (('logit', [0, 0.5, 1, 1.25, 1.5, 2, 3]), ('weight', [0, 0.5, 1, 1.25, 1.5, 2, 3])):
                g = full[(full.model == model) & (full.task == t) & (full.seed == 42) & (full.form == form)].set_index('factor')
                cells = [f'{g.n_tokens[f]:.0f}\\,/\\,{100 * g.ended[f]:.0f}' if f in g.index else '--' for f in factors]
                rows.append(f'{"Qwen2.5-7B" if model == Q else "Phi-4-mini"} & {TL[t]} & {form.capitalize()} & ' + ' & '.join(cells) + r' \\')
        if model == Q:
            rows.append(r'\midrule')
    return (r'''\begin{table}[htbp]
\centering
\caption{Mean length in tokens and share (\%%) of generated answers that end within the budget (seed 42).}
\label{tab:length}
\setlength{\tabcolsep}{3.2pt}
\fontsize{7.5}{9.2}\selectfont
\begin{tabular}{lll ccccccc}
\toprule
Model & Task & Form & 0 & 0.5 & 1 & 1.25 & 1.5 & 2 & 3 \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % '\n'.join(rows))


write('tab_length.tex', gen_table())

a = am
N['amp_comparisons'] = len(a); N['amp_gains'] = int((a.delta_lo > 0).sum()); N['amp_losses'] = int((a.delta_hi < 0).sum())
af = a[a.ckpt == 100]; at = a[a.ckpt < 100]
N['amp_full'] = (len(af), int((af.delta_lo > 0).sum()), int((af.delta_hi < 0).sum()))
N['amp_trunc'] = (len(at), int((at.delta_lo > 0).sum()), int((at.delta_hi < 0).sum()))
for model in (Q, P):
    e = eq[(eq.model == model) & eq.factor_logit.notna()]
    N[f'eq_{model}'] = dict(n=len(e), weight_better=int((e.diff_lo > 0).sum()), logit_better=int((e.diff_hi < 0).sum()),
                            within2=int((e.weight_minus_logit.abs() <= 0.02).sum()),
                            below=dict(n=int((e.factor_weight < 1).sum()), wb=int(((e.diff_lo > 0) & (e.factor_weight < 1)).sum()),
                                       lb=int(((e.diff_hi < 0) & (e.factor_weight < 1)).sum())),
                            above=dict(n=int((e.factor_weight > 1).sum()), wb=int(((e.diff_lo > 0) & (e.factor_weight > 1)).sum()),
                                       lb=int(((e.diff_hi < 0) & (e.factor_weight > 1)).sum())))
att = pd.read_csv(f'{A}/attenuation.csv')
for model in (Q, P):
    g = att[att.model == model].groupby(['task', 'form', 'factor']).gain_kept.mean().unstack().round(0)
    N[f'gain_kept_{model}'] = g.to_dict('index')
    b = be[(be.model == model)]
    N[f'bench_range_{model}'] = {f: (round(b[b.factor == f].rel_change.min(), 1), round(b[b.factor == f].rel_change.max(), 1)) for f in (0.5, 1.0, 1.5, 2.0)}
    N[f'bench_base_{model}'] = round(float(b.base.iloc[0]), 1)
g = full[(full.task == 'gsm8k') & (full.seed == 42) & (full.factor == 0) & (full.form == 'logit')]
N['gsm8k_base_ended'] = {m: round(100 * float(x), 1) for m, x in zip(g.model, g.ended)}
with open(os.path.join(OUT, 'numbers.txt'), 'w') as f:
    for k, v in N.items():
        f.write(f'{k}: {v}\n')
print(open(os.path.join(OUT, 'numbers.txt')).read())
