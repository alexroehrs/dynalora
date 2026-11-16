"""Tasks with human labels.

Each task has a training pool and a fixed test set taken from a public dataset (loaded from the Hugging
Face Hub at run time, never stored in the repository), a one-sentence instruction, and an objective metric:

==========  ===========================  ================================  =================================
key         dataset (Hub id)             output                            metric
==========  ===========================  ================================  =================================
pubmedqa    qiaojin/PubMedQA             one of yes, no, maybe             accuracy (macro-F1 in the analysis)
yelp5       Yelp/yelp_review_full        one of 1, 2, 3, 4, 5              accuracy (macro-F1 in the analysis)
gsm8k       openai/gsm8k (main)          worked solution, ``#### number``  exact match of the final number
samsum      knkarthick/samsum            summary of a dialogue             ROUGE-1, ROUGE-2, ROUGE-L (F1)
==========  ===========================  ================================  =================================

Every example is a chat with one user turn (the instruction, a blank line, the input) and one assistant turn
(the answer), in the tokenizer's own chat template. Inputs longer than ``max_input_tokens`` are truncated on
the input side; the number of truncated examples is reported by the scripts.

Fixed subsets (independent of the training seed):

* ``pubmedqa``: the 1,000 expert-labeled questions are split by ``split_labeled`` (500 train, 500 test); the training pool adds a random subset of ``pqa_artificial``
  (numpy seed 42) up to ``n_train`` examples; the test set is the 500-question test fold;
* ``yelp5``: random ``n_train`` reviews of the train split (numpy seed 42); test set of 100 random reviews
  per star rating from the test split (numpy seed 0);
* ``gsm8k``: the train split; random 500 problems of the test split (numpy seed 0);
* ``samsum``: random ``n_train`` dialogues of the train split (numpy seed 42); random 500 dialogues of the
  test split (numpy seed 0).

``smoke=True`` returns small synthetic examples of the same shape without network access.
"""
import os
import random
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SUBSET_SEED = 42
TEST_SUBSET_SEED = 0

TASKS = {
    'pubmedqa': dict(kind='choice', options=['yes', 'no', 'maybe'], metric='acc',
                     instruction='Answer the question about the abstract below with one word: yes, no, or maybe.'),
    'yelp5': dict(kind='choice', options=['1', '2', '3', '4', '5'], metric='acc',
                  instruction='Rate the review below from 1 to 5 stars. Answer with the number only.'),
    'gsm8k': dict(kind='generate', max_new_tokens=320, metric='em',
                  instruction='Solve the problem below. Show the steps and write the final answer after "####".'),
    'samsum': dict(kind='generate', max_new_tokens=96, metric='rougeL',
                   instruction='Summarize the dialogue below in one or two sentences.'),
}
MAIN_METRIC = {k: v['metric'] for k, v in TASKS.items()}


def spec(task):
    if task not in TASKS:
        raise KeyError(f'unknown task {task!r} (defined: {sorted(TASKS)})')
    return TASKS[task]


# ---- data ------------------------------------------------------------------------------------------
def split_labeled(pubids):
    """Deterministic split of the 1,000 expert-labeled PubMedQA questions: sort by pubid, shuffle with numpy
    seed 0, first half for training, second half for testing. Returns (train_idx, test_idx) as positions in the
    original dataset order."""
    order = np.argsort(np.asarray(pubids, dtype=np.int64), kind='stable')
    perm = order[np.random.default_rng(0).permutation(len(order))]
    half = len(perm) // 2
    return perm[:half], perm[half:]


def _pubmedqa_input(doc):
    ctx = doc['context']
    ctxs = ctx['contexts'] if isinstance(ctx, dict) else [c for d in ctx for c in d['contexts']]
    return 'Abstract: {}\nQuestion: {}'.format('\n'.join(ctxs), doc['question'])


def clean_gsm8k_answer(ans):
    """The reference solution without the calculator annotations ``<<48/2=24>>``."""
    return re.sub(r'<<[^>]*>>', '', ans).strip()


def _subset(n_total, n, seed):
    idx = np.random.default_rng(seed).choice(n_total, size=min(n, n_total), replace=False)
    return [int(i) for i in np.sort(idx)]


def load_examples(task, split, n=None, smoke=False):
    """List of ``{'id', 'input', 'answer'}`` for ``split`` in ('train', 'test'). ``answer`` is the gold
    answer text (the option, the worked solution, or the reference summary). ``n`` caps the training pool
    (default 8,000) or the test set (default 500)."""
    spec(task)
    if split not in ('train', 'test'):
        raise ValueError(split)
    if smoke:
        return _smoke_examples(task, split, n or (48 if split == 'train' else 12))
    from datasets import load_dataset
    n = n or (8000 if split == 'train' else 500)
    if task == 'pubmedqa':
        lab = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
        tr_idx, te_idx = split_labeled(lab['pubid'])
        if split == 'test':
            sub = lab.select([int(i) for i in te_idx[:n]])
            return [{'id': f'L{d["pubid"]}', 'input': _pubmedqa_input(d), 'answer': d['final_decision']} for d in sub]
        lab_tr = lab.select([int(i) for i in tr_idx])
        out = [{'id': f'L{d["pubid"]}', 'input': _pubmedqa_input(d), 'answer': d['final_decision']} for d in lab_tr]
        n_art = max(n - len(out), 0)
        if n_art:
            art = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
            sub = art.select(_subset(len(art), n_art, TRAIN_SUBSET_SEED))
            out += [{'id': f'A{d["pubid"]}', 'input': _pubmedqa_input(d), 'answer': d['final_decision']} for d in sub]
        return out[:n] if n < len(out) else out
    if task == 'yelp5':
        ds = load_dataset('Yelp/yelp_review_full', split=split)
        if split == 'train':
            idx = _subset(len(ds), n, TRAIN_SUBSET_SEED)
        else:
            labels = np.asarray(ds['label']); rng = np.random.default_rng(TEST_SUBSET_SEED); per = n // 5
            idx = sorted(int(i) for c in range(5) for i in rng.choice(np.flatnonzero(labels == c), size=per, replace=False))
        sub = ds.select(idx)
        return [{'id': f'{split}{i}', 'input': 'Review: ' + d['text'], 'answer': str(d['label'] + 1)}
                for i, d in zip(idx, sub)]
    if task == 'gsm8k':
        ds = load_dataset('openai/gsm8k', 'main', split=split)
        idx = list(range(len(ds))) if split == 'train' else _subset(len(ds), n, TEST_SUBSET_SEED)
        if split == 'train' and n < len(idx):
            idx = _subset(len(ds), n, TRAIN_SUBSET_SEED)
        sub = ds.select(idx)
        return [{'id': f'{split}{i}', 'input': d['question'], 'answer': clean_gsm8k_answer(d['answer'])}
                for i, d in zip(idx, sub)]
    if task == 'samsum':
        ds = load_dataset('knkarthick/samsum', split=split)
        ds = ds.filter(lambda d: bool(d['dialogue']) and bool(d['summary']))
        idx = _subset(len(ds), n, TRAIN_SUBSET_SEED if split == 'train' else TEST_SUBSET_SEED)
        sub = ds.select(idx)
        return [{'id': str(d['id']), 'input': d['dialogue'], 'answer': d['summary']} for d in sub]
    raise KeyError(task)


def _smoke_examples(task, split, n):
    filler = ['the', 'and', 'was', 'very', 'a', 'of', 'it', 'is', 'with', 'this', 'study', 'patients', 'good', 'bad']
    rng = random.Random({'train': 1, 'test': 2}[split] + sum(map(ord, task)))
    out = []
    for i in range(n):
        text = ' '.join(rng.choice(filler) for _ in range(rng.randint(8, 30)))
        if task == 'pubmedqa':
            ex = ('Abstract: {}\nQuestion: {}?'.format(text, ' '.join(rng.choice(filler) for _ in range(4))),
                  rng.choice(TASKS[task]['options']))
        elif task == 'yelp5':
            ex = ('Review: ' + text, rng.choice(TASKS[task]['options']))
        elif task == 'gsm8k':
            a, b = rng.randint(1, 9), rng.randint(1, 9)
            ex = (f'{text} {a} and {b}?', f'{a} and {b} is {a + b}.\n#### {a + b}')
        else:
            ex = (text, ' '.join(text.split()[:5]))
        out.append({'id': f'{split}{i}', 'input': ex[0], 'answer': ex[1]})
    return out


# ---- chat format -----------------------------------------------------------------------------------
def end_of_turn_id(tok):
    """Token that closes an assistant turn in the tokenizer's chat template (``<|im_end|>`` for ChatML,
    ``<|end|>`` for Phi), found by rendering a turn; falls back to the EOS token."""
    mark = 'XQZJ'
    try:
        txt = tok.apply_chat_template([{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': mark}],
                                      tokenize=False)
        tail = txt[txt.index(mark) + len(mark):]
        ids = tok(tail, add_special_tokens=False)['input_ids']
        if ids and ids[0] in set(tok.all_special_ids) | set(getattr(tok, 'added_tokens_decoder', {}).keys()):
            return ids[0]
    except Exception:
        pass
    return tok.eos_token_id


def stop_ids(tok):
    out = [end_of_turn_id(tok)]
    if tok.eos_token_id is not None:
        out.append(tok.eos_token_id)
    vocab = tok.get_vocab()
    if '<|endoftext|>' in vocab:
        out.append(vocab['<|endoftext|>'])
    return sorted(set(out), key=out.index)


def user_message(task, text):
    return spec(task)['instruction'] + '\n\n' + text


def context_ids(tok, task, text, max_input_tokens=900):
    """(ids, truncated): the chat template up to the start of the assistant turn, with the input truncated
    to ``max_input_tokens`` tokens."""
    if not getattr(tok, 'chat_template', None):
        raise ValueError('the tokenizer has no chat template (the task study needs an instruction-tuned model)')
    t = tok(text, add_special_tokens=False)['input_ids']
    truncated = len(t) > max_input_tokens
    if truncated:
        text = tok.decode(t[:max_input_tokens], skip_special_tokens=True)
    s = tok.apply_chat_template([{'role': 'user', 'content': user_message(task, text)}], tokenize=False,
                                add_generation_prompt=True)
    return tok(s, add_special_tokens=False)['input_ids'], truncated


def answer_ids(tok, answer, eot=True):
    ids = tok(answer, add_special_tokens=False)['input_ids']
    return ids + ([end_of_turn_id(tok)] if eot else [])


def lora_targets_for(model, model_name):
    """LoRA target modules: the attention projections. Models with a fused ``qkv_proj`` (Phi) use it with
    ``o_proj``; the others use ``dynalora.models.lora_targets``."""
    from .models import lora_targets
    names = {n.split('.')[-1] for n, _ in model.named_modules()}
    if 'qkv_proj' in names:
        return ['qkv_proj', 'o_proj']
    return lora_targets(model_name)


# ---- metrics ---------------------------------------------------------------------------------------
_NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def normalize_number(s):
    """Canonical string of a number: no commas, currency or percent signs, no trailing zeros."""
    if s is None:
        return None
    m = _NUM.search(str(s).replace('$', ''))
    if not m:
        return None
    x = m.group(0).replace(',', '')
    try:
        v = float(x)
    except ValueError:
        return None
    if v == int(v):
        return str(int(v))
    return repr(round(v, 6)).rstrip('0').rstrip('.')


def gsm8k_prediction(text):
    """The number after the last ``####`` marker, else the last number of the text."""
    if '####' in text:
        p = normalize_number(text.split('####')[-1])
        if p is not None:
            return p
    nums = _NUM.findall(text.replace('$', ''))
    return normalize_number(nums[-1]) if nums else None


def gsm8k_gold(answer):
    return normalize_number(answer.split('####')[-1])


_SCORER = None


def rouge(pred, ref):
    """ROUGE-1, ROUGE-2 and ROUGE-L F1 (``rouge-score``, with stemming)."""
    global _SCORER
    if _SCORER is None:
        from rouge_score import rouge_scorer
        _SCORER = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    s = _SCORER.score(ref, pred)
    return {'rouge1': s['rouge1'].fmeasure, 'rouge2': s['rouge2'].fmeasure, 'rougeL': s['rougeL'].fmeasure}


def repeated_bigram_share(ids):
    """Share of the bigrams of a token sequence that repeat an earlier bigram (0 for fewer than 2 tokens)."""
    b = list(zip(ids, ids[1:]))
    return 0.0 if not b else 1.0 - len(set(b)) / len(b)


def score_output(task, text, gold_answer):
    """Per-item metrics of a generated answer."""
    if task == 'gsm8k':
        p, g = gsm8k_prediction(text), gsm8k_gold(gold_answer)
        return {'pred': p if p is not None else '', 'gold': g, 'em': float(p is not None and p == g)}
    if task == 'samsum':
        return {'pred': '', 'gold': '', **rouge(text, gold_answer)}
    raise KeyError(task)


def macro_f1(golds, preds, labels):
    f = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))
