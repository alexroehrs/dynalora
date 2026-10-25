"""Automatic metrics over a table of generations.

Per generation: HTML artifact flag, generated length, EOS collapse, domain
vocabulary score, perplexity of the continuation under an external language
model, probability assigned to the target domain by an external classifier
trained on held-out human texts, and (for review domains) the positive
probability of an external sentiment classifier. ``dist_n`` is computed per
group in ``analysis``.
"""
import math
import re

import numpy as np
import pandas as pd
import torch

from .lexicons import vocab_score, tokenize

_ART = re.compile(r"<br\s*/?>|</br>|</|/>")
_HTML = re.compile(r"<[^>]*>")


def surface_metrics(df, domain):
    df = df.copy()
    df['artifact'] = df['continuation'].astype(str).apply(lambda t: bool(_ART.search(t)))
    df['collapse'] = df['n_tokens'] < 10
    df['early'] = df['n_tokens'] < df['n_tokens'].max()
    vs = df['continuation'].apply(lambda t: vocab_score(t, domain))
    df['vocab_score'] = [v[0] for v in vs]
    df['vocab_count'] = [v[1] for v in vs]
    df['n_words'] = [v[2] for v in vs]
    return df


class ExternalLM:
    """Perplexity of continuations under an external causal LM (default GPT-2 large)."""

    def __init__(self, name='gpt2-large', device=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.pad_token = self.tok.eos_token
        from .train import _load_lm
        self.model = _load_lm(name, torch.float16 if self.device == 'cuda' else torch.float32).to(self.device).eval()
        self.name = name

    @torch.no_grad()
    def nll(self, prompts, conts, batch_size=16):
        """Mean NLL per continuation token given the prompt; returns (nll, n_tokens) lists."""
        out_nll, out_n = [], []
        for s in range(0, len(prompts), batch_size):
            P, C = prompts[s:s + batch_size], conts[s:s + batch_size]
            pid = [self.tok(p)['input_ids'] for p in P]
            cid = [self.tok(c)['input_ids'] for c in C]
            seqs = [a + b for a, b in zip(pid, cid)]
            L = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), L), self.tok.eos_token_id, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, x in enumerate(seqs):
                ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            logits = self.model(input_ids=ids, attention_mask=att).logits.float()
            logp = torch.log_softmax(logits[:, :-1], -1)
            tgt = ids[:, 1:]
            lp = logp.gather(-1, tgt[..., None]).squeeze(-1)
            for i in range(len(seqs)):
                a, b = len(pid[i]), len(seqs[i])
                if b - a == 0:
                    out_nll.append(float('nan')); out_n.append(0); continue
                v = -lp[i, a - 1:b - 1]
                out_nll.append(float(v.mean())); out_n.append(int(b - a))
        return out_nll, out_n


class DomainClassifier:
    """TF-IDF + logistic regression over held-out human texts of each domain plus generic text."""

    def __init__(self, texts_by_domain, max_chars=600, seed=0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        X, y = [], []
        for d, ts in texts_by_domain.items():
            for t in ts:
                X.append(_HTML.sub(' ', t)[:max_chars]); y.append(d)
        self.clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000),
                                 LogisticRegression(max_iter=2000, C=4.0, random_state=seed))
        self.clf.fit(X, y)
        self.classes = list(self.clf.classes_)

    def prob(self, texts, domain):
        P = self.clf.predict_proba([_HTML.sub(' ', t) for t in texts])
        return P[:, self.classes.index(domain)]

    def cv_accuracy(self, texts_by_domain, max_chars=600):
        from sklearn.model_selection import cross_val_score
        X, y = [], []
        for d, ts in texts_by_domain.items():
            for t in ts:
                X.append(_HTML.sub(' ', t)[:max_chars]); y.append(d)
        return float(cross_val_score(self.clf, X, y, cv=5).mean())


def sentiment_positive(texts, device=None, batch_size=32):
    from transformers import pipeline
    dev = 0 if (device or ('cuda' if torch.cuda.is_available() else 'cpu')) == 'cuda' else -1
    clf = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english',
                   device=dev, truncation=True, max_length=128)
    out = []
    for s in range(0, len(texts), batch_size):
        res = clf([_HTML.sub(' ', t) for t in texts[s:s + batch_size]], top_k=None)
        for r in res:
            out.append(next(x['score'] for x in r if x['label'] == 'POSITIVE'))
    return out


def human_reference(heldout, tok, n=200, n_prompt=3, n_cont=50):
    """Turn held-out human texts into prompt/continuation pairs with the same budget as generations."""
    rows = []
    for t in heldout[:n]:
        ids = tok(_HTML.sub(' ', t))['input_ids']
        if len(ids) < n_prompt + 10:
            continue
        p = tok.decode(ids[:n_prompt]); c = tok.decode(ids[n_prompt:n_prompt + n_cont])
        rows.append({'prompt': p, 'continuation': c, 'text': p + c, 'n_tokens': min(n_cont, len(ids) - n_prompt),
                     'method': 'human', 'lambda': float('nan'), 'gamma': float('nan'), 'temperature': float('nan'),
                     'sample': len(rows), 'seed': -1, 'prompt_id': -1, 'ended_eos': False, 'mean_entropy': float('nan')})
    return pd.DataFrame(rows)


def dist_n(texts, n):
    allg = []
    for t in texts:
        w = tokenize(t)
        allg += [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return len(set(allg)) / max(1, len(allg))
