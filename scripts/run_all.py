"""End-to-end pipeline: train adapters, generate under every method, evaluate, summarize.

Example (Colab, T4):
    python scripts/run_all.py --out results/level2 --models gpt2 --configs imdb_neg imdb_bal imdb_bal_nohtml yelp pubmed
    python scripts/run_all.py --out results/level2 --models Qwen/Qwen2.5-0.5B --configs imdb_bal --n_samples 10

Smoke test on CPU with a random tiny model and synthetic data:
    python scripts/run_all.py --smoke --out /tmp/smoke
"""
import argparse
import json
import os
import random
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynalora import data as D  # noqa: E402
from dynalora.train import train_adapter  # noqa: E402
from dynalora.generate import Generator, run_grid  # noqa: E402
from dynalora import evaluate as E  # noqa: E402
from dynalora.analysis import summarize, entropy_matched  # noqa: E402

TRAIN_HP = {
    'imdb_neg': dict(lr=5e-6, epochs=10, mask_pad=False),   # original recipe
}
DEFAULT_HP = dict(lr=2e-4, epochs=3, mask_pad=True)

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
GAMMAS = [1.0, 1.5, 2.0, 3.0]
TEMPS = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def smoke_setup(tmp):
    """Random tiny GPT-2 and synthetic corpora, for testing the code paths without network."""
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    os.makedirs(tmp, exist_ok=True)
    mdir = os.path.join(tmp, 'tiny-gpt2')
    if not os.path.exists(os.path.join(mdir, 'config.json')):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        corpus = sum([_synth(d, 400) for d in ('imdb', 'yelp', 'pubmed')], [])
        bpe = Tokenizer(models.BPE(unk_token='<|endoftext|>'))
        bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        bpe.decoder = decoders.ByteLevel()
        bpe.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1500, special_tokens=['<|endoftext|>'],
                                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token='<|endoftext|>', bos_token='<|endoftext|>',
                                      unk_token='<|endoftext|>')
        cfg = GPT2Config(vocab_size=len(tok), n_layer=2, n_head=2, n_embd=64, n_positions=256,
                         bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        GPT2LMHeadModel(cfg).save_pretrained(mdir); tok.save_pretrained(mdir)
    return mdir


def _synth(domain, n):
    from dynalora.lexicons import LEXICONS
    words = sorted(LEXICONS[domain]); filler = ['the', 'and', 'was', 'very', 'a', 'of', 'it', 'is', 'with', 'this']
    rng = random.Random(hash(domain) % 1000)
    return [' '.join(rng.choice(words if rng.random() < 0.3 else filler) for _ in range(rng.randint(20, 80))) for _ in range(n)]


def smoke_config(name):
    dom = D.DOMAIN_OF[name]
    return {'name': name, 'domain': dom, 'train': _synth(dom, 200), 'heldout': _synth(dom, 60),
            'prompts': D.PROMPTS[dom][:2], 'cfg_prefix': D.CFG_PREFIX[dom]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='+', default=['gpt2'])
    ap.add_argument('--configs', nargs='+', default=['imdb_neg', 'imdb_bal', 'imdb_bal_nohtml', 'yelp', 'pubmed'])
    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--lambdas', nargs='+', type=float, default=LAMBDAS)
    ap.add_argument('--gammas', nargs='+', type=float, default=GAMMAS)
    ap.add_argument('--temps', nargs='+', type=float, default=TEMPS)
    ap.add_argument('--max_new_tokens', type=int, default=50)
    ap.add_argument('--ext_lm', default='gpt2-large')
    ap.add_argument('--no_ext', action='store_true')
    ap.add_argument('--no_sentiment', action='store_true')
    ap.add_argument('--skip_cfg', action='store_true')
    ap.add_argument('--skip_weight', action='store_true')
    ap.add_argument('--skip_temp', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    adapters_dir = os.path.join(args.out, 'adapters'); os.makedirs(adapters_dir, exist_ok=True)
    if args.smoke:
        tiny = smoke_setup(os.path.join(args.out, 'smoke'))
        args.models = [tiny]; args.ext_lm = tiny; args.no_sentiment = True
        args.lambdas = [0.0, 1.0, 2.0]; args.gammas = [1.0, 2.0]; args.temps = [0.7, 0.4]; args.n_samples = 4
        args.max_new_tokens = 8
        args.configs = [c for c in args.configs if c in D.DOMAIN_OF][:3] or ['imdb_bal', 'yelp']

    configs = {}
    all_rows = []
    for model in args.models:
        mtag = os.path.basename(model.rstrip('/'))
        for cname in args.configs:
            cfg = smoke_config(cname) if args.smoke else D.load_config(cname)
            configs[cname] = cfg
            adir = os.path.join(adapters_dir, f'{mtag}__{cname}')
            if not os.path.exists(os.path.join(adir, 'adapter_config.json')):
                hp = dict(DEFAULT_HP); hp.update(TRAIN_HP.get(cname, {}))
                if args.smoke:
                    hp.update(epochs=1)
                log(f'training adapter {mtag}/{cname} with {hp}')
                train_adapter(model, cfg['train'], adir, log=log, **hp)
            else:
                log(f'adapter {adir} exists, skipping training')
            raw_path = os.path.join(args.out, f'generations__{mtag}__{cname}.csv')
            if os.path.exists(raw_path):
                log(f'{raw_path} exists, skipping generation')
                all_rows.append(pd.read_csv(raw_path)); continue
            gen = Generator(model, adir)
            rows = []
            common = dict(n_samples=args.n_samples, max_new_tokens=args.max_new_tokens, log=log)
            rows += run_grid(gen, cfg['prompts'], 'dynalora', args.lambdas, temperature=0.7, **common)
            if not args.skip_weight:
                rows += run_grid(gen, cfg['prompts'], 'weight', args.lambdas, temperature=0.7, **common)
            if not args.skip_temp:
                rows += run_grid(gen, cfg['prompts'], 'lora', args.temps, **common)
                rows += run_grid(gen, cfg['prompts'], 'base', args.temps, **common)
            if not args.skip_cfg:
                rows += run_grid(gen, cfg['prompts'], 'cfg', args.gammas, temperature=0.7, cfg_prefix=cfg['cfg_prefix'], **common)
            df = pd.DataFrame(rows); df['config'] = cname; df['model'] = mtag; df['domain'] = cfg['domain']
            df.to_csv(raw_path, index=False); all_rows.append(df)
            del gen
            if not args.smoke:
                import torch; torch.cuda.empty_cache()

    df = pd.concat(all_rows, ignore_index=True)

    # ---- human references ----------------------------------------------------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.models[0])
    refs = []
    for cname, cfg in configs.items():
        h = E.human_reference(cfg['heldout'], tok, n=200 if not args.smoke else 20, n_cont=args.max_new_tokens)
        h['config'] = cname; h['model'] = 'human'; h['domain'] = cfg['domain']; refs.append(h)
    df = pd.concat([df] + refs, ignore_index=True)

    # ---- metrics ----------------------------------------------------------------
    parts = []
    for dom, g in df.groupby('domain'):
        parts.append(E.surface_metrics(g, dom))
    df = pd.concat(parts, ignore_index=True)

    if not args.no_ext:
        log(f'external perplexity with {args.ext_lm}')
        lm = E.ExternalLM(args.ext_lm)
        nll, n = lm.nll(df['prompt'].astype(str).tolist(), df['continuation'].astype(str).tolist())
        df['ext_nll'] = nll; df['ext_n_tokens'] = n; df['ext_lm'] = lm.name
        del lm

    log('domain classifier')
    by_dom = {cfg['domain']: cfg['heldout'][:2000] for cfg in configs.values()}
    by_dom['generic'] = _synth('imdb', 60) if args.smoke else D.generic_texts(2000)
    clf = E.DomainClassifier(by_dom)
    acc = clf.cv_accuracy(by_dom)
    log(f'domain classifier 5-fold accuracy on human texts: {acc:.3f}')
    df['domain_prob'] = float('nan')
    for dom in df['domain'].unique():
        m = df['domain'] == dom
        df.loc[m, 'domain_prob'] = clf.prob(df.loc[m, 'continuation'].astype(str).tolist(), dom)

    if not args.no_sentiment:
        m = df['domain'].isin(['imdb', 'yelp'])
        log(f'sentiment classifier on {int(m.sum())} review texts')
        df['p_positive'] = float('nan')
        df.loc[m, 'p_positive'] = E.sentiment_positive(df.loc[m, 'text'].astype(str).tolist())

    df.to_csv(os.path.join(args.out, 'generations_all_metrics.csv'), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    em = []
    for cname in configs:
        for mtag in summ['model'].unique():
            if mtag == 'human':
                continue
            t = entropy_matched(summ, cname, mtag); t['config'] = cname; t['model'] = mtag; em.append(t)
    if em:
        pd.concat(em, ignore_index=True).to_csv(os.path.join(args.out, 'entropy_matched.csv'), index=False)
    with open(os.path.join(args.out, 'run_info.json'), 'w') as f:
        json.dump({'args': vars(args), 'domain_classifier_cv_accuracy': acc, 'n_rows': int(len(df))}, f, indent=2)
    log('done')


if __name__ == '__main__':
    main()
