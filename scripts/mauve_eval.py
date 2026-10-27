"""Distribution-level comparison of generations with held-out reference texts (MAUVE).

For every grid point of every run (procedure, lambda, gamma, temperature), MAUVE is computed
between the 200 generated continuations and the 200 held-out reference continuations of the
same corpus. Texts are embedded once with an external model (the hidden state of the last
token, as in the MAUVE reference implementation) and MAUVE is averaged over several k-means
seeds. Only continuations are compared, so that the fixed generation prompts do not separate
the two distributions by construction.

Inputs are the ``generations_all_metrics.csv`` files written by ``run_all.py``; the reference
texts they contain are third-party data and are never written to the output, which keeps
only identifiers and scores.

Example (Colab, T4):
    pip install -q mauve-text faiss-cpu
    python scripts/mauve_eval.py --inputs "/content/drive/MyDrive/dynalora_results/*/generations_all_metrics.csv" \
        --out /content/drive/MyDrive/dynalora_results/mauve
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dynalora import data as D  # noqa: E402

KEYS = ['config', 'model', 'method', 'lambda', 'gamma', 'temperature']


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


@torch.no_grad()
def featurize(texts, name, batch_size=32, max_len=128):
    """Hidden state of the last non-padding token of each text."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
    order = np.argsort([-len(t) for t in texts]); out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] if texts[i].strip() else tok.eos_token for i in idx], return_tensors='pt',
                  padding=True, truncation=True, max_length=max_len).to(dev)
        h = model(**enc).last_hidden_state
        last = enc['attention_mask'].sum(1) - 1
        out[idx] = h[torch.arange(len(idx), device=dev), last].float().cpu().numpy()
    del model
    if dev == 'cuda':
        torch.cuda.empty_cache()
    return out


def mauve_score(p, q, seeds):
    import mauve
    vals = [mauve.compute_mauve(p_features=p, q_features=q, seed=s, verbose=False).mauve for s in seeds]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--featurizer', default='gpt2-large')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument('--batch_size', type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted({f for p in a.inputs for f in glob.glob(p)})
    frames = []
    for f in files:
        d = pd.read_csv(f); d['run'] = os.path.basename(os.path.dirname(f)); frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d['cont'] = d['continuation'].fillna('').astype(str)
    d['base'] = d['config'].map(D.base_name)
    hum = d[d['method'] == 'human'].drop_duplicates(['base', 'sample'])
    gen = d[d['method'] != 'human']
    log(f'{len(files)} files, {len(gen)} generations, {len(hum)} reference texts')
    texts = pd.concat([hum['cont'], gen['cont']]).tolist()
    feats = featurize(texts, a.featurizer, a.batch_size)
    fh, fg = feats[:len(hum)], feats[len(hum):]
    hum_idx = {b: np.where(hum['base'].values == b)[0] for b in hum['base'].unique()}
    rows = []
    gk = gen.reset_index(drop=True)
    for key, g in gk.groupby(['run'] + KEYS, dropna=False):
        b = D.base_name(key[1])
        if b not in hum_idx:
            continue
        m, sd = mauve_score(fg[g.index.values], fh[hum_idx[b]], a.seeds)
        rows.append(dict(zip(['run'] + KEYS, key), n=len(g), n_ref=len(hum_idx[b]), mauve=m, mauve_sd=sd,
                         featurizer=a.featurizer))
    # reference split in halves: the ceiling MAUVE can reach with this sample size
    for b, ix in hum_idx.items():
        h = len(ix) // 2
        m, sd = mauve_score(fh[ix[:h]], fh[ix[h:]], a.seeds)
        rows.append(dict(run='reference', config=b, model='human', method='human_split', n=h, n_ref=len(ix) - h,
                         mauve=m, mauve_sd=sd, featurizer=a.featurizer))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.out, f'mauve__{a.featurizer.split("/")[-1]}.csv'), index=False)
    log(f'{len(out)} grid points')
    print(out[out.method.isin(['dynalora', 'weight'])].pivot_table(index=['model', 'config', 'method'],
          columns='lambda', values='mauve').round(2).to_string(), flush=True)


if __name__ == '__main__':
    main()
