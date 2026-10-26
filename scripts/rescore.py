"""Re-score existing generations with evaluators that are independent of the generators.

For every per-generation metrics file written by run_all.py, the script adds:

* ``ext2_nll`` / ``ext2_n_tokens``: perplexity of the continuation given the prompt under a
  second external language model from a different family and tokenizer than both
  generators (default SmolLM2-1.7B), so that fluency is not judged by a relative of
  the GPT-2 base model;
* ``domain_prob_u``: target-domain probability under one domain classifier fitted on the
  held-out texts of all three domains plus generic text, so that the GPT-2 and Qwen runs
  are scored on the same scale.

Output: one file ``rescore__<run>.csv.gz`` per run directory, with the new columns.

Example (Colab):
    python scripts/rescore.py --inputs "/content/drive/MyDrive/dynalora_results/*/generations_all_metrics.csv" \
        --out /content/drive/MyDrive/dynalora_results/rescore
"""
import argparse
import glob
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dynalora import data as D  # noqa: E402
from dynalora import evaluate as E  # noqa: E402


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ext_lm', default='HuggingFaceTB/SmolLM2-1.7B')
    ap.add_argument('--batch_size', type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted({f for p in a.inputs for f in glob.glob(p)})
    log(f'{len(files)} files')

    log('unified domain classifier')
    by_dom = {}
    for c in ('imdb_bal', 'yelp', 'pubmed'):
        cc = D.load_config(c)
        by_dom[cc['domain']] = cc['heldout'][:2000]
    by_dom['generic'] = D.generic_texts(2000)
    clf = E.DomainClassifier(by_dom)

    lm = E.ExternalLM(a.ext_lm)
    for f in files:
        name = os.path.basename(f)
        if name == 'generations_all_metrics.csv':   # one file per run directory
            name = f'rescore__{os.path.basename(os.path.dirname(f))}.csv.gz'
        else:
            name = name.replace('metrics__', 'rescore__')
            name = name if name.endswith('.gz') else name + '.gz'
        path = os.path.join(a.out, name)
        if os.path.exists(path):
            log(f'{path} exists, skipping'); continue
        d = pd.read_csv(f)
        d['domain_prob_u'] = float('nan')
        for dom in d['domain'].unique():
            m = d['domain'] == dom
            d.loc[m, 'domain_prob_u'] = clf.prob(d.loc[m, 'continuation'].fillna('').astype(str).tolist(), dom)
        nll, n = lm.nll(d['prompt'].astype(str).tolist(), d['continuation'].fillna('').astype(str).tolist(),
                        batch_size=a.batch_size)
        d['ext2_nll'] = nll; d['ext2_n_tokens'] = n; d['ext2_lm'] = a.ext_lm
        d.to_csv(path, index=False)
        log(f'{path}: {len(d)} rows')
    log('done')


if __name__ == '__main__':
    main()
