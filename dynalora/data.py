"""Dataset configurations.

Each configuration returns a dict with ``train`` (list of texts used to train
the adapter), ``heldout`` (real texts from the same corpus, never used in
training, serving as the human reference for the domain classifier and for
reference perplexity), ``prompts`` and ``cfg_prefix`` (the domain description
used by the classifier-free-guidance baseline).

Only public Hugging Face datasets are used; nothing is redistributed here.
"""
import random
import re

_HTML = re.compile(r"<[^>]*>")

PROMPTS = {
    'imdb': ["This movie was", "The acting was", "I really enjoyed", "The plot",
             "I watched this film", "The director", "Overall, this", "My favorite part",
             "The ending was", "This is one of"],
    'yelp': ["This place was", "The food was", "I really enjoyed", "The service",
             "We ordered", "I went here", "The staff", "The prices",
             "Overall, the", "My favorite dish"],
    'pubmed': ["The study", "We investigated", "Patients with", "The results",
               "This paper", "In this work", "Treatment with", "The aim of",
               "Background:", "Methods:"],
}

CFG_PREFIX = {
    'imdb': "The following is a movie review posted on IMDB.\n",
    'yelp': "The following is a Yelp review of a local business.\n",
    'pubmed': "The following is the abstract of a biomedical research article.\n",
}

_DOMAIN_OF = {'imdb_neg': 'imdb', 'imdb_bal': 'imdb', 'imdb_bal_nohtml': 'imdb',
              'yelp': 'yelp', 'pubmed': 'pubmed'}


class _DomainOf(dict):
    """Domain of a configuration, accepting training-variant names such as ``imdb_bal__lr1e-05``."""
    def __getitem__(self, k):
        return dict.__getitem__(self, k.split('__')[0])

    def __contains__(self, k):
        return dict.__contains__(self, k.split('__')[0])


DOMAIN_OF = _DomainOf(_DOMAIN_OF)


def _imdb_texts(split, n, balanced, seed):
    from datasets import load_dataset
    ds = load_dataset('stanfordnlp/imdb', split=split)
    if not balanced:
        return [ds[i]['text'] for i in range(n)]
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    pos = [i for i in idx if ds[i]['label'] == 1][: n // 2]
    neg = [i for i in idx if ds[i]['label'] == 0][: n // 2]
    sel = pos + neg
    random.Random(seed + 1).shuffle(sel)
    return [ds[i]['text'] for i in sel]


def _yelp_texts(n_train, n_heldout):
    from datasets import load_dataset
    ds = load_dataset('Yelp/yelp_review_full', split=f'train[:{n_train + n_heldout}]')
    t = [r['text'] for r in ds]
    return t[:n_train], t[n_train:]


def _pubmed_texts(n_train, n_heldout):
    from datasets import load_dataset
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split=f'train[:{n_train + n_heldout}]')
    t = [' '.join(r['context']['contexts']) for r in ds]
    return t[:n_train], t[n_train:]


def generic_texts(n=2000):
    """Generic English text (WikiText-103 test split) for the domain classifier."""
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
    out = [r['text'].strip() for r in ds if len(r['text'].strip()) > 200 and not r['text'].startswith(' =')]
    return out[:n]


def imdb_test_by_label(n, label, seed=42):
    """Reviews of one polarity from the IMDB test split (0 negative, 1 positive)."""
    from datasets import load_dataset
    ds = load_dataset('stanfordnlp/imdb', split='test')
    idx = [i for i in range(len(ds)) if ds[i]['label'] == label]
    random.Random(seed + 7).shuffle(idx)
    return [ds[i]['text'] for i in idx[:n]]


def base_name(name):
    """Configuration name without a training-variant suffix (``imdb_bal__lr1e-05`` -> ``imdb_bal``)."""
    return name.split('__')[0]


def variant_hp(name):
    """Training overrides encoded in a variant name, e.g. ``imdb_bal__lr1e-05`` -> {'lr': 1e-05}."""
    hp = {}
    for part in name.split('__')[1:]:
        if part.startswith('lr'):
            hp['lr'] = float(part[2:])
    return hp


def load_config(name, n_train=10000, n_heldout=2000, seed=42):
    if '__' in name:
        cfg = load_config(base_name(name), n_train, n_heldout, seed)
        cfg['name'] = name
        return cfg
    if name == 'imdb_neg':
        # Original setting: first 10,000 rows of the IMDB train split, which are all negative.
        train = _imdb_texts('train', n_train, balanced=False, seed=seed)
        held = _imdb_texts('test', n_heldout, balanced=True, seed=seed)
    elif name in ('imdb_bal', 'imdb_bal_nohtml'):
        train = _imdb_texts('train', n_train, balanced=True, seed=seed)
        held = _imdb_texts('test', n_heldout, balanced=True, seed=seed)
        if name == 'imdb_bal_nohtml':
            train = [_HTML.sub(' ', t) for t in train]
    elif name == 'yelp':
        train, held = _yelp_texts(n_train, n_heldout)
    elif name == 'pubmed':
        train, held = _pubmed_texts(n_train, n_heldout)
    else:
        raise ValueError(name)
    dom = DOMAIN_OF[name]
    return {'name': name, 'domain': dom, 'train': train, 'heldout': held,
            'prompts': PROMPTS[dom], 'cfg_prefix': CFG_PREFIX[dom]}
