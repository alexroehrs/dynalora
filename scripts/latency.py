"""Decoding cost of the two forms of scaling.

For one model and one trained adapter, the script generates ``new_tokens`` tokens for a batch of
``batch`` prompts (the paper's prompts, ``dynalora.data.PROMPTS``) under four procedures:

* ``base``       the base model (adapter disabled), one forward pass per step
* ``lora``       the plain adapter (weight-space lambda = 1), one pass per step
* ``weight``     weight-space scaling at lambda (default 2), one pass per step
* ``logit``      logit-space scaling (DynaLoRA) at the same lambda: two passes per step, one with
                 the adapter disabled and one with it enabled, combined as y_base + lam * (y_lora - y_base)

All four use the KV cache: the prompt is encoded once and every later step feeds one token.
In the two-pass procedure each pass keeps its own cache (the base and the adapted model attend
to different keys and values), so logit-space scaling costs two cached passes and two caches
per step, which is what the measured time and memory reflect. Note that ``dynalora/generate.py``,
which produced the paper's texts, recomputes the whole sequence at every step without a cache in
all procedures; the relative cost of the two passes is the same, the absolute times are not.

Sampling is multinomial at temperature 0.7 with a fixed seed, as in the paper, but sequences are
not stopped at EOS so that every procedure decodes exactly ``batch`` x ``new_tokens`` tokens.
After one warm-up generation per procedure, ``reps`` timed generations are run; the script
reports the median and interquartile range of the wall-clock milliseconds per generated token
per sequence (elapsed / (new_tokens x batch)), the median milliseconds per decoding step for the
whole batch, and the peak GPU memory allocated during the timed generations (NaN on CPU).

Output: ``latency__<model>.csv`` under ``--out`` with the columns model, config, method, lambda,
batch, new_tokens, ms_per_token_median, ms_per_token_iqr, peak_mem_mb, ms_per_step_median,
device, dtype, prompt_tokens. An existing file is not recomputed.

Example (Colab, T4):
    python scripts/latency.py --model gpt2 --adapters_dir /content/drive/MyDrive/dynalora_results/gpt2/adapters \
        --config pubmed --out /content/drive/MyDrive/dynalora_results/latency
    python scripts/latency.py --model Qwen/Qwen2.5-1.5B --adapters_dir /content/drive/MyDrive/dynalora_results/qwen15/adapters \
        --config pubmed --out /content/drive/MyDrive/dynalora_results/latency
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dynalora.data import PROMPTS  # noqa: E402
from dynalora.generate import Generator  # noqa: E402

METHODS = ['base', 'lora', 'weight', 'logit']


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def default_prompts(n):
    p = PROMPTS['pubmed'] + PROMPTS['imdb'] + PROMPTS['yelp']
    return (p * (n // len(p) + 1))[:n]


@torch.no_grad()
def _step(gen, ids, att, pos, past, adapter_on):
    """One cached forward pass; returns the last-position logits and the updated cache."""
    kw = dict(input_ids=ids, attention_mask=att, position_ids=pos, past_key_values=past, use_cache=True)
    if gen.has_adapter and not adapter_on:
        with gen.model.disable_adapter():
            out = gen.model(**kw)
    else:
        out = gen.model(**kw)
    return out.logits[:, -1, :].float(), out.past_key_values


@torch.no_grad()
def decode(gen, enc, method, lam, new_tokens, temperature, seed):
    """Generates ``new_tokens`` tokens for the encoded batch under ``method``; returns the ids."""
    ids, att = enc['input_ids'], enc['attention_mask']
    pos = (att.cumsum(-1) - 1).clamp_min(0)
    g = torch.Generator(device=gen.device).manual_seed(int(seed))
    if method == 'weight':
        gen.set_weight_scale(lam)
    on = {'base': False, 'lora': True, 'weight': True}
    cur_ids, cur_pos = ids, pos
    past_b = past_l = None
    out = []
    for _ in range(new_tokens):
        if method == 'logit':
            yb, past_b = _step(gen, cur_ids, att, cur_pos, past_b, adapter_on=False)
            yl, past_l = _step(gen, cur_ids, att, cur_pos, past_l, adapter_on=True)
            final = yb + lam * (yl - yb)
        else:
            final, past_b = _step(gen, cur_ids, att, cur_pos, past_b, adapter_on=on[method])
        probs = F.softmax(final / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=g)
        out.append(nxt)
        cur_ids = nxt
        cur_pos = (att.sum(-1, keepdim=True)).long()   # next position of each sequence (left padding)
        att = torch.cat([att, torch.ones_like(nxt)], 1)
    if method == 'weight':
        gen.reset_weight_scale()
    return torch.cat(out, 1)


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--adapters_dir', required=True)
    ap.add_argument('--config', default='pubmed')
    ap.add_argument('--out', required=True)
    ap.add_argument('--lam', type=float, default=2.0, help='lambda of the weight and logit procedures')
    ap.add_argument('--batch', type=int, default=20)
    ap.add_argument('--new_tokens', type=int, default=50)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--load_in_4bit', action='store_true', help='4-bit NF4 base model (adapters trained with QLoRA)')
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mtag = os.path.basename(a.model.rstrip('/'))
    path = os.path.join(a.out, f'latency__{mtag}.csv')
    if os.path.exists(path):
        log(f'{path} exists, skipping'); return
    gen = Generator(a.model, os.path.join(a.adapters_dir, f'{mtag}__{a.config}'), load_in_4bit=a.load_in_4bit)
    gen.reset_weight_scale()
    gen.tok.padding_side = 'left'
    enc = gen.tok(default_prompts(a.batch), return_tensors='pt', padding=True).to(gen.device)
    n_prompt = int(enc['attention_mask'].sum(1).float().mean().item())
    log(f'{mtag} {a.config}: batch {a.batch}, {n_prompt} prompt tokens on average, {a.new_tokens} new tokens, '
        f'device {gen.device}, dtype {gen.dtype}')
    rows = []
    for m in a.methods:
        lam = {'base': 0.0, 'lora': 1.0}.get(m, a.lam)
        decode(gen, enc, m, lam, a.new_tokens, a.temperature, a.seed)      # warm-up
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        secs, outs = [], []
        for r in range(a.reps):
            s, o = timed(lambda: decode(gen, enc, m, lam, a.new_tokens, a.temperature, a.seed))
            secs.append(s); outs.append(o)
        assert all(torch.equal(outs[0], o) for o in outs[1:]), 'sampling is not deterministic'
        per_tok = np.array(secs) * 1000 / (a.new_tokens * a.batch)
        per_step = np.array(secs) * 1000 / a.new_tokens
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else float('nan')
        q1, med, q3 = np.percentile(per_tok, [25, 50, 75])
        rows.append(dict(model=mtag, config=a.config, method=m, **{'lambda': lam}, batch=a.batch, new_tokens=a.new_tokens,
                         ms_per_token_median=med, ms_per_token_iqr=q3 - q1, peak_mem_mb=peak,
                         ms_per_step_median=float(np.median(per_step)), device=gen.device, dtype=str(gen.dtype),
                         prompt_tokens=n_prompt))
        log(f'{m} lambda={lam}: {med:.3f} ms/token/sequence (IQR {q3 - q1:.3f}), {np.median(per_step):.1f} ms/step, peak {peak:.0f} MB')
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(df.round(3).to_string(), flush=True)


if __name__ == '__main__':
    main()
