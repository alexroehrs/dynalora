"""LoRA adapter trained on the PubMedQA question-answering task itself (config ``pubmedqa_task``).

Where the ``pubmed`` adapter of the paper is trained on abstracts (domain text), this one is
trained on the task: each example is the prompt of the lm-evaluation-harness task ``pubmedqa``
(``utils.doc_to_text`` of ``scripts/lm_eval_tasks/pubmedqa_test500``, the same prompt used at
evaluation) followed by a space and the answer word, with the loss on the answer tokens only
(prompt and padding masked with -100).

Data (loaded from the Hub at run time, never stored in the repository):

* training: ``n_train`` examples of ``pqa_artificial`` (yes/no, random subset with numpy seed 42)
  plus the training half of ``pqa_labeled`` (yes/no/maybe); the labeled set is split by
  ``utils.split_labeled`` (sort by pubid, shuffle with numpy seed 0, first 500 train, last 500
  test), and the test half is the harness task ``pubmedqa_test500``;
* validation: ``n_val`` examples held out of the training pool, on which the script reports the
  masked loss and the accuracy of the harness scoring (answer with the highest log-likelihood
  among yes/no/maybe) before and after training;
* examples whose prompt plus answer exceeds ``max_len`` tokens are skipped and counted.

The LoRA recipe is that of ``dynalora/train.py`` (rank 8, alpha 16, dropout 0.1, the target
modules of ``lora_targets``, AdamW, cosine schedule with 5% warmup), with one epoch, an effective
batch of 16 (``batch_size`` x ``grad_accum``), gradient checkpointing, a fp16/bf16 base model on
GPU, and optionally a 4-bit NF4 base model (``--load_in_4bit``, QLoRA). The adapter is saved with
peft to ``<adapters_dir>/<model>__pubmedqa_task/``, the layout that ``scripts/zeroshot_bench.py``
and ``scripts/likelihood_tradeoff.py`` read with ``--configs pubmedqa_task``.

``--smoke`` trains a random tiny GPT-2 on synthetic examples without network access.

Example (Colab, T4):
    python scripts/train_task_adapter.py --model gpt2 --batch_size 4 --grad_accum 4 \
        --adapters_dir /content/drive/MyDrive/dynalora_results/task/adapters
    python scripts/train_task_adapter.py --model Qwen/Qwen2.5-1.5B --batch_size 2 --grad_accum 8 \
        --adapters_dir /content/drive/MyDrive/dynalora_results/task/adapters
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynalora.train import _load_lm, compute_dtype_4bit, lora_targets  # noqa: E402

CHOICES = ['yes', 'no', 'maybe']
CONFIG = 'pubmedqa_task'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def task_utils():
    """The prompt and split functions of the harness task pubmedqa_test500 (single definition)."""
    path = os.path.join(ROOT, 'scripts', 'lm_eval_tasks', 'pubmedqa_test500', 'utils.py')
    spec = importlib.util.spec_from_file_location('pubmedqa_test500_utils', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ---- data ---------------------------------------------------------------------
def load_examples(n_train, seed):
    """(prompt, answer) pairs of the training pool: pqa_artificial subset + labeled train half."""
    from datasets import load_dataset
    U = task_utils()
    art = load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train')
    idx = np.random.default_rng(seed).choice(len(art), size=min(n_train, len(art)), replace=False)
    sub = art.select([int(i) for i in np.sort(idx)])
    ex = [(U.doc_to_text(d), d['final_decision']) for d in sub]
    lab = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
    tr_idx, te_idx = U.split_labeled(lab['pubid'])
    lab_tr = lab.select([int(i) for i in tr_idx])
    ex += [(U.doc_to_text(d), d['final_decision']) for d in lab_tr]
    log(f'{len(sub)} artificial + {len(lab_tr)} labeled training examples ({len(te_idx)} labeled questions kept for the test fold)')
    return ex


def smoke_examples(n):
    """Synthetic prompts in the task format, for the code path only."""
    filler = ['the', 'and', 'was', 'very', 'a', 'of', 'it', 'is', 'with', 'this', 'study', 'patients']
    rng = random.Random(1)
    out = []
    for i in range(n):
        abstract = ' '.join(rng.choice(filler) for _ in range(rng.randint(10, 40)))
        q = ' '.join(rng.choice(filler) for _ in range(5)) + '?'
        out.append(("Abstract: {}\nQuestion: {}\nAnswer:".format(abstract, q), rng.choice(CHOICES)))
    return out


def smoke_model(out_dir):
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from run_all import smoke_setup
    return smoke_setup(out_dir)


def encode(tok, prompt, answer, max_len):
    """Token ids of prompt + ' ' + answer and the number of prompt tokens, tokenized as the
    harness does (whole text once; prompt prefix separately); None if longer than max_len."""
    whole = tok(prompt + ' ' + answer, add_special_tokens=False)['input_ids']
    if len(whole) > max_len:
        return None
    n_ctx = len(tok(prompt, add_special_tokens=False)['input_ids'])
    return whole, n_ctx


class TaskDS(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items   # list of (ids, n_ctx)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(pad):
    def f(batch):
        L = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        for j, (x, n_ctx) in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
            lab[j, n_ctx:len(x)] = torch.tensor(x[n_ctx:])
        return {'input_ids': ids, 'attention_mask': att, 'labels': lab}
    return f


# ---- evaluation ---------------------------------------------------------------
@torch.no_grad()
def val_loss(model, loader, device, amp_dtype):
    model.eval(); tot = 0.0; n = 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
            tot += model(**b).loss.item(); n += 1
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def val_accuracy(model, tok, examples, device, amp_dtype, max_len):
    """Harness scoring: the answer whose continuation has the highest summed log-likelihood."""
    model.eval(); pad = tok.pad_token_id; correct = 0; n = 0
    for prompt, gold in examples:
        encs = [encode(tok, prompt, c, max_len + 1) for c in CHOICES]
        if any(e is None for e in encs):
            continue
        L = max(len(x) for x, _ in encs)
        ids = torch.full((3, L), pad, dtype=torch.long); att = torch.zeros((3, L), dtype=torch.long)
        for j, (x, _) in enumerate(encs):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
        ids, att = ids.to(device), att.to(device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
            lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)
        scores = []
        for j, (x, n_ctx) in enumerate(encs):
            tgt = torch.tensor(x[n_ctx:], device=device)
            scores.append(lp[j, n_ctx - 1:len(x) - 1].gather(-1, tgt[:, None]).sum().item())
        correct += int(CHOICES[int(np.argmax(scores))] == gold); n += 1
    model.train()
    return correct / max(n, 1), n


# ---- training -----------------------------------------------------------------
def train(a):
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    mtag = os.path.basename(a.model.rstrip('/'))
    out_dir = os.path.join(a.adapters_dir, f'{mtag}__{a.config}')
    if os.path.exists(os.path.join(out_dir, 'adapter_config.json')):
        log(f'{out_dir} exists, skipping training'); return out_dir
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = smoke_examples(a.n_train + a.n_val + 8) if a.smoke else load_examples(a.n_train, a.seed)
    random.Random(a.seed).shuffle(examples)
    val_ex, tr_ex = examples[:a.n_val], examples[a.n_val:]
    items, skipped = [], 0
    for p, ans in tr_ex:
        e = encode(tok, p, ans, a.max_len)
        if e is None:
            skipped += 1
        else:
            items.append(e)
    vitems = [e for e in (encode(tok, p, ans, a.max_len) for p, ans in val_ex) if e is not None]
    log(f'{len(items)} training examples, {skipped} skipped (> {a.max_len} tokens), {len(vitems)} validation examples')
    n_ans = sum(len(x) - c for x, c in items)
    log(f'answer tokens per example: {n_ans / max(len(items), 1):.2f}; mean length {np.mean([len(x) for x, _ in items]):.0f} tokens')

    if a.load_in_4bit:
        amp_dtype = compute_dtype_4bit() if device == 'cuda' else torch.float32
        model = _load_lm(a.model, amp_dtype, load_in_4bit=True)
    else:
        amp_dtype = compute_dtype_4bit() if device == 'cuda' else torch.float32
        model = _load_lm(a.model, amp_dtype).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias='none',
                     target_modules=lora_targets(a.model), task_type='CAUSAL_LM')
    model = get_peft_model(model, cfg)
    for p in model.parameters():            # adapter weights in float32 under autocast
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f'trainable {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.2f}%); base dtype {amp_dtype}')

    tr = DataLoader(TaskDS(items), batch_size=a.batch_size, shuffle=True, collate_fn=collate(tok.pad_token_id))
    va = DataLoader(TaskDS(vitems), batch_size=a.batch_size, collate_fn=collate(tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = math.ceil(len(tr) / a.grad_accum) * a.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda' and amp_dtype == torch.float16))

    acc0, n_acc = val_accuracy(model, tok, val_ex, device, amp_dtype, a.max_len)
    vl0 = val_loss(model, va, device, amp_dtype)
    log(f'before training: val_loss {vl0:.4f} val_acc {100 * acc0:.1f}% ({n_acc} questions)')
    history = [(0, vl0, acc0)]
    step = 0; run_loss = 0.0; run_n = 0; model.train()
    for ep in range(a.epochs):
        for i, b in enumerate(tr):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
                loss = model(**b).loss
            run_loss += loss.item(); run_n += 1
            scaler.scale(loss / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0 or (i + 1) == len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step(); step += 1
                if step % a.log_every == 0:
                    log(f'epoch {ep + 1} step {step}/{steps} train_loss {run_loss / max(run_n, 1):.4f} lr {sch.get_last_lr()[0]:.2e}')
                    run_loss = 0.0; run_n = 0
                if step % a.eval_every == 0 and step < steps:
                    vl = val_loss(model, va, device, amp_dtype)
                    history.append((step, vl, float('nan')))
                    log(f'epoch {ep + 1} step {step}/{steps} val_loss {vl:.4f}')
    vl = val_loss(model, va, device, amp_dtype)
    acc, n_acc = val_accuracy(model, tok, val_ex, device, amp_dtype, a.max_len)
    history.append((step, vl, acc))
    log(f'after training: val_loss {vl:.4f} val_acc {100 * acc:.1f}% ({n_acc} questions)')
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, 'train_log.txt'), 'w') as f:
        f.write(f'model={a.model} config={a.config} lr={a.lr} epochs={a.epochs} r={a.r} alpha={a.alpha} seed={a.seed} '
                f'max_len={a.max_len} batch={a.batch_size}x{a.grad_accum} {"load_in_4bit=True " if a.load_in_4bit else ""}'
                f'n_train={len(items)} n_skipped={skipped} n_val={len(vitems)} val_loss={vl:.4f} val_acc={acc:.4f} val_acc_before={acc0:.4f}\n')
        f.write('step\tval_loss\tval_acc\n')
        for s, v, c in history:
            f.write(f'{s}\t{v:.4f}\t{c:.4f}\n')
    with open(os.path.join(out_dir, 'train_info.json'), 'w') as f:
        json.dump(dict(model=a.model, config=a.config, lr=a.lr, epochs=a.epochs, r=a.r, alpha=a.alpha, dropout=a.dropout,
                       seed=a.seed, max_len=a.max_len, batch_size=a.batch_size, grad_accum=a.grad_accum,
                       load_in_4bit=a.load_in_4bit, n_train=len(items), n_skipped=skipped, n_val=len(vitems),
                       steps=steps, val_loss_before=vl0, val_acc_before=acc0, val_loss=vl, val_acc=acc,
                       smoke=a.smoke), f, indent=1)
    log(f'saved {out_dir}')
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt2')
    ap.add_argument('--adapters_dir', required=True)
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--n_train', type=int, default=10000, help='examples of pqa_artificial (the labeled train half is added)')
    ap.add_argument('--n_val', type=int, default=200, help='examples held out of the training pool')
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--grad_accum', type=int, default=8, help='effective batch = batch_size x grad_accum (16)')
    ap.add_argument('--r', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--eval_every', type=int, default=200)
    ap.add_argument('--load_in_4bit', action='store_true', help='4-bit NF4 base model (QLoRA)')
    ap.add_argument('--smoke', action='store_true', help='random tiny GPT-2 and synthetic examples, no network')
    a = ap.parse_args()
    if a.smoke:
        a.model = smoke_model(os.path.join(os.path.dirname(a.adapters_dir.rstrip('/')), 'smoke'))
    train(a)


if __name__ == '__main__':
    main()
