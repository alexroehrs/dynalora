"""Corpus formatting rate of the held-out reference windows with their HTML tags kept.

The reference continuations used by the generation metrics are built with HTML tags removed
(``evaluate.human_reference``). This script rebuilds the same 200 windows per corpus (three-token
prompt, 50-token continuation) without removing the tags and reports the share containing an HTML
tag, a literal line-break escape, or a run of two or more spaces, with the definitions of
``scripts/analysis/format_metrics.py``. Only rates are written; no text is saved.

Example (Colab, CPU is enough):
    python scripts/reference_format.py --out results/level2/analysis/reference_format.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dynalora import data as D  # noqa: E402


def windows(texts, tok, n=200, n_prompt=3, n_cont=50, strip=False):
    out = []
    for t in texts[:n]:
        src = D._HTML.sub(' ', t) if strip else t
        ids = tok(src)['input_ids']
        if len(ids) < n_prompt + 10:
            continue
        out.append(tok.decode(ids[n_prompt:n_prompt + n_cont]))
    return out


def rates(cs):
    s = pd.Series(cs, dtype=str)
    tag = s.str.contains(r'<br\s*/?>|</br>|</|/>', regex=True)
    lit = s.map(lambda t: '\\n' in t)
    sp = s.map(lambda t: '  ' in t)
    return dict(n=len(s), tag=tag.mean() * 100, lit=lit.mean() * 100, sp=sp.mean() * 100,
                fmt=(tag | lit | sp).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=['imdb_bal', 'imdb_bal_nohtml', 'yelp', 'pubmed'])
    ap.add_argument('--tokenizers', nargs='+', default=['gpt2', 'Qwen/Qwen2.5-0.5B'])
    ap.add_argument('--out', default='results/level2/analysis/reference_format.csv')
    a = ap.parse_args()
    from transformers import AutoTokenizer
    rows = []
    for c in a.configs:
        held = D.load_config(c)['heldout']
        for tn in a.tokenizers:
            tok = AutoTokenizer.from_pretrained(tn)
            for strip in (False, True):
                if strip and c == 'imdb_bal_nohtml':
                    continue
                r = rates(windows(held, tok, strip=strip or c == 'imdb_bal_nohtml'))
                rows.append(dict(config=c, tokenizer=tn.split('/')[-1], tags_removed=strip or c == 'imdb_bal_nohtml', **r))
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
