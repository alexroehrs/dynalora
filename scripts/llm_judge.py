"""LLM-as-judge scoring of a stratified sample of generations and human references.

Two open instruction-tuned judges from different model families rate each text on two
criteria (fluency/coherence and domain fit) on a 1-5 scale. Instead of parsing a generated
answer, the score is the expected value of the digit distribution at the first answer
position, which is deterministic and uses the judge's full belief.

Inputs are the ``generations_all_metrics.csv`` files written by ``run_all.py`` (they contain
the generated texts and the held-out human texts; the latter are third-party data and are
never written to the outputs of this script, which keep only identifiers and scores).

Example (Colab, T4):
    pip install -q bitsandbytes
    python scripts/llm_judge.py \
        --inputs /content/drive/MyDrive/dynalora_results/gpt2/generations_all_metrics.csv \
                 /content/drive/MyDrive/dynalora_results/qwen/generations_all_metrics.csv \
        --out /content/drive/MyDrive/dynalora_results/judge
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch

JUDGES = ['Qwen/Qwen2.5-7B-Instruct', 'allenai/OLMo-2-1124-7B-Instruct']

DOMAIN_DESC = {
    'imdb': 'a movie review posted by a user on IMDB',
    'yelp': 'a customer review of a business posted on Yelp',
    'pubmed': 'an abstract of a biomedical research article indexed in PubMed',
}

CELLS = [('dynalora', l) for l in (0.0, 1.0, 1.5, 2.0, 3.0)] + [('weight', l) for l in (1.5, 2.0, 3.0)] + \
        [('cfg', None), ('lora_cold', None), ('human', None)]

SYSTEM = 'You are a careful annotator who rates short texts. Answer with a single digit from 1 to 5 and nothing else.'

PROMPTS = {
    'fluency': (
        'Rate the FLUENCY of the text below: is it grammatical, coherent, and well formed as English prose? '
        'Ignore its topic and whether you agree with it, and do not penalize the text for ending abruptly, '
        'since it was cut at a fixed length.\n'
        '1 = broken or incoherent (garbled words, repeated fragments, stray markup)\n'
        '2 = mostly hard to follow\n3 = understandable but with clear errors or incoherence\n'
        '4 = fluent with minor issues\n5 = fully fluent and coherent\n\nText:\n"""\n{text}\n"""\n\nRating (1-5):'),
    'domain': (
        'Rate how well the text below reads as {desc}. Judge content, vocabulary, and style typical of that genre; '
        'do not penalize the text for ending abruptly, since it was cut at a fixed length.\n'
        '1 = not that kind of text at all\n2 = only loosely related\n3 = recognizably that kind of text but atypical\n'
        '4 = a typical example with minor oddities\n5 = indistinguishable from a typical example\n\n'
        'Text:\n"""\n{text}\n"""\n\nRating (1-5):'),
}

_BR = re.compile(r'<br\s*/?>', re.I)


def clean_text(t):
    """Well-formed line-break tags become line breaks (as a browser would render them);
    malformed markup is left as is, so the judge sees it."""
    t = _BR.sub('\n', str(t))
    return re.sub(r'\n{3,}', '\n\n', t).strip()


def build_sample(paths, per_cell=50, seed=0):
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    d['domain'] = d['config'].map(lambda c: 'imdb' if c.startswith('imdb') else c)
    rng = np.random.default_rng(seed)
    rows = []
    for (cfg, model), g in d[d['method'] != 'human'].groupby(['config', 'model']):
        for method, lam in CELLS:
            if method == 'human':
                c = d[(d['method'] == 'human') & (d['config'] == cfg)]
                c = c.drop_duplicates('sample')
            elif method == 'cfg':
                c = g[(g['method'] == 'cfg') & (g['gamma'] == 1.5)]
            elif method == 'lora_cold':
                c = g[(g['method'] == 'lora') & (g['temperature'] == 0.5)]
            else:
                c = g[(g['method'] == method) & (g['lambda'] == lam)]
            if c.empty:
                continue
            if method == 'human':
                take = c.sample(min(per_cell, len(c)), random_state=int(rng.integers(1 << 31)))
            else:  # stratified: the same number per prompt
                k = max(1, per_cell // c['prompt_id'].nunique())
                take = c.groupby('prompt_id', group_keys=False).apply(
                    lambda x: x.sample(min(k, len(x)), random_state=int(rng.integers(1 << 31))))
            take = take.assign(cell_method=method, cell_lambda=lam, model=model, config=cfg)
            rows.append(take)
    s = pd.concat(rows, ignore_index=True)
    s['item_id'] = np.arange(len(s))
    s['judge_text'] = s['text'].map(clean_text)
    return s


def _digit_ids(tok):
    ids = []
    for k in '12345':
        enc = tok.encode(k, add_special_tokens=False)
        ids.append(enc[-1])
    return ids


def load_judge(name, quant):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map='auto')
    if quant == '4bit':
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_compute_dtype=torch.float16)
    else:
        kw['dtype'] = torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except TypeError:
        kw['torch_dtype'] = kw.pop('dtype', torch.float16)
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    return tok, model.eval()


@torch.no_grad()
def score(tok, model, prompts, batch_size=16):
    digit = torch.tensor(_digit_ids(tok), device=model.device)
    out_exp, out_arg, out_mass = [], [], []
    for s in range(0, len(prompts), batch_size):
        chats = [tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': p}],
                                         tokenize=False, add_generation_prompt=True) for p in prompts[s:s + batch_size]]
        enc = tok(chats, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        probs = torch.softmax(logits, -1)
        pd_ = probs[:, digit]
        mass = pd_.sum(-1)
        pn = pd_ / mass.clamp_min(1e-9)[:, None]
        exp = (pn * torch.arange(1, 6, device=pn.device)).sum(-1)
        out_exp += exp.tolist(); out_arg += (pn.argmax(-1) + 1).tolist(); out_mass += mass.tolist()
    return out_exp, out_arg, out_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per_cell', type=int, default=50)
    ap.add_argument('--judges', nargs='+', default=JUDGES)
    ap.add_argument('--quant', default='4bit', choices=['4bit', 'fp16'])
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='score only the first N items (testing)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = build_sample(a.inputs, a.per_cell)
    if a.limit:
        s = s.head(a.limit)
    print(time.strftime('%H:%M:%S'), 'sample', len(s), 'items')
    ids = s[['item_id', 'config', 'model', 'cell_method', 'cell_lambda', 'method', 'lambda', 'gamma', 'temperature',
             'prompt_id', 'sample', 'domain_prob', 'vocab_score', 'artifact', 'ext_nll', 'ext_n_tokens']].copy()
    for j in a.judges:
        tag = j.split('/')[-1]
        path = os.path.join(a.out, f'scores__{tag}.csv')
        if os.path.exists(path) and not a.limit:
            print('exists, skipping', path); continue
        print(time.strftime('%H:%M:%S'), 'loading', j)
        tok, model = load_judge(j, a.quant)
        res = ids.copy()
        for crit, tmpl in PROMPTS.items():
            prompts = [tmpl.format(text=t, desc=DOMAIN_DESC['imdb' if c.startswith('imdb') else c])
                       for t, c in zip(s['judge_text'], s['config'])]
            e, g, m = score(tok, model, prompts, a.batch_size)
            res[f'{crit}_exp'], res[f'{crit}_arg'], res[f'{crit}_mass'] = e, g, m
            print(time.strftime('%H:%M:%S'), tag, crit, 'mean', round(float(np.mean(e)), 3),
                  'digit mass', round(float(np.mean(m)), 3))
        res.insert(1, 'judge', j)
        res.to_csv(path, index=False)
        del model; torch.cuda.empty_cache()
    # aggregate
    parts = [pd.read_csv(os.path.join(a.out, f)) for f in os.listdir(a.out) if f.startswith('scores__')]
    allr = pd.concat(parts, ignore_index=True)
    agg = allr.groupby(['config', 'model', 'cell_method', 'cell_lambda', 'judge'], dropna=False).agg(
        n=('item_id', 'size'), fluency=('fluency_exp', 'mean'), fluency_sd=('fluency_exp', 'std'),
        domain=('domain_exp', 'mean'), domain_sd=('domain_exp', 'std')).reset_index()
    agg.to_csv(os.path.join(a.out, 'judge_summary.csv'), index=False)
    if allr['judge'].nunique() >= 2:
        w = allr.pivot_table(index='item_id', columns='judge', values=['fluency_exp', 'domain_exp'])
        js = sorted(allr['judge'].unique())
        rows = []
        for crit in ['fluency_exp', 'domain_exp']:
            x, y = w[(crit, js[0])], w[(crit, js[1])]
            rows.append(dict(criterion=crit, spearman_items=x.corr(y, method='spearman'),
                             pearson_items=x.corr(y)))
        cm = agg.pivot_table(index=['config', 'model', 'cell_method', 'cell_lambda'], columns='judge',
                             values=['fluency', 'domain'], dropna=False)
        for crit in ['fluency', 'domain']:
            rows.append(dict(criterion=crit + '_cellmeans',
                             spearman_items=cm[(crit, js[0])].corr(cm[(crit, js[1])], method='spearman'),
                             pearson_items=cm[(crit, js[0])].corr(cm[(crit, js[1])])))
        pd.DataFrame(rows).to_csv(os.path.join(a.out, 'judge_agreement.csv'), index=False)
        print(pd.DataFrame(rows))
    print(time.strftime('%H:%M:%S'), 'done')


if __name__ == '__main__':
    main()
