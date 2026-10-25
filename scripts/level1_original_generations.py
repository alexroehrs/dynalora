"""Level-1 metrics over the original released generations (no new generation).

Computes, for every row of the two original CSVs: perplexity of the continuation
under the base GPT-2, mean entropy of the base model at the generated positions,
positive probability under an external sentiment classifier, and dist-n per
lambda. Mirrors the in-browser computation used for the first analysis.

    python scripts/level1_original_generations.py --out results/level1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynalora import evaluate as E  # noqa: E402

IMDB_CSV = '4 - experimento_lambda_20260217_143915.csv'
YELP_CSV = '6 - experimento_yelp_20260331_002250.csv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/level1')
    ap.add_argument('--lm', default='gpt2')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imdb = pd.read_csv(os.path.join(root, IMDB_CSV))
    yelp = pd.read_csv(os.path.join(root, YELP_CSV))
    imdb['continuation'] = [t[len(p):] for t, p in zip(imdb['texto_gerado'].astype(str), imdb['prompt'])]
    yelp['continuation'] = yelp['continuacao'].astype(str)
    yelp['text'] = yelp['texto_completo']
    imdb['text'] = imdb['texto_gerado']
    lm = E.ExternalLM(args.lm)
    for name, df in (('imdb', imdb), ('yelp', yelp)):
        nll, n = lm.nll(df['prompt'].astype(str).tolist(), df['continuation'].tolist())
        df['nll_continuation'] = nll; df['n_cont'] = n; df['ppl_continuation'] = np.exp(df['nll_continuation'])
        df['p_positive'] = E.sentiment_positive(df['text'].astype(str).tolist())
        df.to_csv(os.path.join(args.out, f'original_{name}_metrics.csv'), index=False)
        rows = []
        for lam, g in df.groupby('lambda'):
            w = g['n_cont'].values; v = g['nll_continuation'].values
            rows.append({'lambda': lam, 'ppl_token_weighted': float(np.exp((v * w).sum() / w.sum())),
                         'ppl_mean': float(np.exp(v.mean())), 'positive_frac': float((g['p_positive'] > 0.5).mean()),
                         **{f'dist{k}': E.dist_n(g['continuation'].tolist(), k) for k in (1, 2, 3)}})
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f'original_{name}_summary.csv'), index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
