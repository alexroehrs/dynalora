"""Domain lexicons and the domain vocabulary score.

The IMDB and Yelp lexicons are the ones released with the original experiments
(notebook ``3 - Evaluation_Metrics.ipynb``). The PubMed lexicon was written for
the extended study. All three are scored with the same rule: the continuation
is lowercased, punctuation is removed, tokens are split on whitespace, and a
token counts when it matches a lexicon entry exactly (no lemmatization), so
that scores are computed identically across domains.
"""
import re

IMDB = {
    'movie', 'film', 'cinema', 'flick', 'picture',
    'director', 'acting', 'actor', 'actress', 'cast', 'performance',
    'cinematography', 'screenplay', 'script', 'plot', 'story',
    'scene', 'character', 'dialogue', 'narrative',
    'comedy', 'drama', 'horror', 'thriller', 'action',
    'documentary', 'animated', 'animation',
    'directed', 'produced', 'starring', 'featuring',
    'pacing', 'editing', 'soundtrack', 'score',
    'masterpiece', 'brilliant', 'excellent', 'superb',
    'terrible', 'awful', 'disaster', 'boring',
    'entertaining', 'enjoyable', 'disappointing',
    'outstanding', 'mediocre', 'bland', 'cheesy',
    'watched', 'viewing', 'recommend', 'rating',
    'oscar', 'award', 'sequel', 'franchise', 'remake',
    'blockbuster', 'indie', 'theatrical', 'premiere',
}

YELP = {
    "food", "meal", "dish", "flavor", "taste", "portion", "menu",
    "appetizer", "entree", "dessert", "drink", "beverage", "sauce",
    "fresh", "stale", "spicy", "salty", "crispy", "soggy",
    "juicy", "tender", "greasy", "seasoned", "bland",
    "delicious", "tasty", "disgusting", "overcooked", "undercooked",
    "restaurant", "diner", "bistro", "eatery", "kitchen",
    "ambiance", "atmosphere", "waiter", "waitress", "chef",
    "reservation", "takeout", "delivery", "seating",
    "service", "staff", "attentive", "overpriced", "affordable",
    "ordered", "tasted", "served", "cooked", "visited",
    "recommend", "returned", "complained",
}

PUBMED = {
    'patients', 'patient', 'study', 'studies', 'clinical', 'trial', 'treatment',
    'therapy', 'disease', 'diagnosis', 'symptoms', 'outcome', 'outcomes',
    'methods', 'results', 'conclusion', 'conclusions', 'background', 'objective',
    'significant', 'significantly', 'analysis', 'cohort', 'randomized',
    'controlled', 'placebo', 'dose', 'mortality', 'risk', 'prevalence',
    'incidence', 'expression', 'protein', 'gene', 'genes', 'cells', 'cell',
    'tissue', 'tumor', 'cancer', 'infection', 'chronic', 'acute', 'syndrome',
    'receptor', 'inhibitor', 'serum', 'plasma', 'levels', 'mg', 'ml',
    'baseline', 'follow-up', 'surgery', 'surgical', 'hospital', 'mice',
    'vitro', 'vivo', 'assay', 'biopsy', 'carcinoma', 'diabetes', 'hypertension',
}

LEXICONS = {'imdb': IMDB, 'yelp': YELP, 'pubmed': PUBMED}

_PUNCT = re.compile(r"[^\w\s'-]")
_TAGS = re.compile(r"<[^>]*>|</|/>")


def tokenize(text: str):
    text = _TAGS.sub(' ', str(text).lower())
    text = _PUNCT.sub(' ', text)
    return [t.strip("'-") for t in text.split() if t.strip("'-")]


def vocab_score(text: str, domain: str):
    """Return (score, count, n_words) for the continuation ``text``."""
    lex = LEXICONS[domain]
    toks = tokenize(text)
    if not toks:
        return 0.0, 0, 0
    c = sum(1 for t in toks if t in lex)
    return c / len(toks), c, len(toks)
