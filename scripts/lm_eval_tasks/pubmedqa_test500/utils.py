"""Prompt of the harness task `pubmedqa` and the train/test split of the labeled set
shared by the task adapter (scripts/train_task_adapter.py) and its evaluation task."""
import numpy as np

SPLIT_SEED = 0


def doc_to_text(doc) -> str:
    """Prompt of the harness task `pubmedqa`, for either field naming of the dataset."""
    if 'CONTEXTS' in doc:
        ctxs, question = doc['CONTEXTS'], doc['QUESTION']
    else:
        ctx = doc['context']
        ctxs = ctx['contexts'] if isinstance(ctx, dict) else [c for d in ctx for c in d['contexts']]
        question = doc['question']
    return "Abstract: {}\nQuestion: {}\nAnswer:".format("\n".join(ctxs), question)


def split_labeled(pubids):
    """Deterministic split of the labeled questions: sort by pubid, shuffle with numpy seed 0,
    first half (500 of 1,000) trains the task adapter, last half is the test fold.
    Returns (train_idx, test_idx) as positions in the original dataset order."""
    order = np.argsort(np.asarray(pubids, dtype=np.int64), kind='stable')
    perm = order[np.random.default_rng(SPLIT_SEED).permutation(len(order))]
    half = len(perm) // 2
    return perm[:half], perm[half:]


def process_docs(dataset):
    """Test half of pqa_labeled, in the shuffled order (doc_id 0..499)."""
    _, test_idx = split_labeled(dataset['pubid'])
    return dataset.select([int(i) for i in test_idx])
