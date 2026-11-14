"""LoRA adapter trained on one task of ``dynalora.tasks`` in chat format.

Each training example is a chat: the user turn holds the task instruction and the input, the assistant turn
the gold answer followed by the end-of-turn token. The loss is on the assistant turn only. The recipe is
rank 8, alpha 16, dropout 0.1, attention projections, AdamW, cosine schedule with
5% warmup, gradient clipping at 0.3, with one epoch and an effective batch of ``batch_size * grad_accum``.
The base model is loaded in bfloat16 on GPU and run under bfloat16 autocast; the LoRA weights stay in float32.

The seed sets the LoRA initialization and the order of the examples; the training pool and the validation
examples (the last ``n_val`` of the pool, never trained on) do not depend on it.

Outputs, under ``<adapters_dir>/<model tag>__<task>__s<seed>/`` (peft format):

* the adapter after the last step (no early stopping, so that a truncated run is a prefix of the full one);
* ``train_log.json``: arguments, number of examples, truncated inputs, steps, the validation loss before
  training and at every ``eval_every`` steps, and the elapsed time. Its presence marks the run as finished.

``--ckpt_fracs 0.1 0.2 0.4`` also saves the adapter at those fractions of the optimizer steps, to
``<adapters_dir>/<model tag>__<task>__s<seed>__ckpt<pct>/`` (each with its own ``train_log.json``).

``--smoke`` trains a random tiny chat model on synthetic examples without network access.

Example (Colab, A100):
    python scripts/train_task_chat.py --model Qwen/Qwen2.5-7B-Instruct --task gsm8k --seed 42 \
        --adapters_dir /content/drive/MyDrive/dynalora_results/level4/adapters --ckpt_fracs 0.1 0.2 0.4
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dynalora import tasks as T  # noqa: E402
from dynalora.models import chat_collate, load_lm  # noqa: E402


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def adapter_name(model_tag, task, seed, ckpt=None):
    return f'{model_tag}__{task}__s{seed}' + (f'__ckpt{ckpt}' if ckpt else '')


def encode_examples(tok, task, examples, max_input_tokens, max_answer_tokens):
    """[(ids, n_ctx)] and the number of truncated inputs; answers longer than ``max_answer_tokens`` are cut
    (without the end-of-turn token, as in ``dynalora.chat.encode_answer``)."""
    items, n_trunc = [], 0
    for ex in examples:
        ctx, tr = T.context_ids(tok, task, ex['input'], max_input_tokens)
        ans = T.answer_ids(tok, ex['answer'], eot=False)
        ans = ans[:max_answer_tokens] if len(ans) > max_answer_tokens else ans + [T.end_of_turn_id(tok)]
        items.append((ctx + ans, len(ctx))); n_trunc += tr
    return items, n_trunc


@torch.no_grad()
def val_loss(model, loader, device, amp_dtype):
    model.eval(); tot, n = 0.0, 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
            out = model(**b)
        k = int((b['labels'] != -100).sum())
        tot += out.loss.item() * k; n += k
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--task', required=True, choices=sorted(T.TASKS))
    ap.add_argument('--adapters_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n_train', type=int, default=8000, help='size of the training pool, validation included')
    ap.add_argument('--n_val', type=int, default=200)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--grad_accum', type=int, default=4)
    ap.add_argument('--max_input_tokens', type=int, default=900)
    ap.add_argument('--max_answer_tokens', type=int, default=320)
    ap.add_argument('--r', type=int, default=8)
    ap.add_argument('--alpha', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--eval_every', type=int, default=100)
    ap.add_argument('--ckpt_fracs', nargs='*', type=float, default=[])
    ap.add_argument('--gradient_checkpointing', action='store_true')
    ap.add_argument('--model_tag', default=None)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    if a.smoke:
        from dynalora.smoke import smoke_chat_setup
        a.model = smoke_chat_setup(os.path.join(a.adapters_dir, '_smoke_model'))
        a.n_train, a.n_val, a.batch_size, a.grad_accum, a.eval_every = 48, 8, 4, 1, 4
        a.max_input_tokens, a.max_answer_tokens = 64, 24
    mtag = a.model_tag or os.path.basename(a.model.rstrip('/'))
    out_dir = os.path.join(a.adapters_dir, adapter_name(mtag, a.task, a.seed))
    if os.path.exists(os.path.join(out_dir, 'train_log.json')):
        log(f'{out_dir}: finished, skipped'); return

    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pool = T.load_examples(a.task, 'train', n=a.n_train, smoke=a.smoke)
    val_ex, tr_ex = pool[-a.n_val:], pool[:-a.n_val]
    tr_items, n_trunc = encode_examples(tok, a.task, tr_ex, a.max_input_tokens, a.max_answer_tokens)
    va_items, _ = encode_examples(tok, a.task, val_ex, a.max_input_tokens, a.max_answer_tokens)
    log(f'{a.task}: {len(tr_items)} training and {len(va_items)} validation examples, {n_trunc} truncated inputs')

    random.seed(a.seed); torch.manual_seed(a.seed)
    model = load_lm(a.model, torch.bfloat16 if device == 'cuda' else torch.float32).to(device)
    if a.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    targets = T.lora_targets_for(model, a.model)
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias='none',
                                             target_modules=targets, task_type='CAUSAL_LM'))
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'LoRA targets {targets}, {n_tr:,} trainable parameters')

    coll = chat_collate(tok.pad_token_id)
    g = torch.Generator(); g.manual_seed(a.seed)
    tr = DataLoader(tr_items, batch_size=a.batch_size, shuffle=True, collate_fn=coll, generator=g)
    va = DataLoader(sorted(va_items, key=lambda x: len(x[0])), batch_size=a.batch_size, collate_fn=coll)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = math.ceil(len(tr) / a.grad_accum) * a.epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    ckpt_at = {max(1, round(f * steps)): int(round(100 * f)) for f in a.ckpt_fracs}

    info = {'args': {k: v for k, v in vars(a).items()}, 'model_tag': mtag, 'n_train': len(tr_items),
            'n_val': len(va_items), 'truncated_inputs': n_trunc, 'steps': steps, 'lora_targets': targets,
            'trainable_parameters': n_tr}
    t0 = time.time()
    history = [(0, val_loss(model, va, device, amp_dtype))]
    log(f'step 0/{steps} val_loss {history[0][1]:.4f}')

    def save(path, step):
        model.save_pretrained(path); tok.save_pretrained(path)
        with open(os.path.join(path, 'train_log.json'), 'w') as f:
            json.dump({**info, 'saved_at_step': step, 'val_loss': history, 'seconds': round(time.time() - t0, 1)},
                      f, indent=2)

    step = 0; model.train()
    for ep in range(a.epochs):
        for i, b in enumerate(tr):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device == 'cuda')):
                loss = model(**b).loss / a.grad_accum
            loss.backward()
            if (i + 1) % a.grad_accum == 0 or (i + 1) == len(tr):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                opt.step(); opt.zero_grad(); sch.step(); step += 1
                if step % a.eval_every == 0 or step in ckpt_at or step == steps:
                    history.append((step, val_loss(model, va, device, amp_dtype)))
                    log(f'epoch {ep + 1} step {step}/{steps} val_loss {history[-1][1]:.4f}')
                if step in ckpt_at and step < steps:
                    save(os.path.join(a.adapters_dir, adapter_name(mtag, a.task, a.seed, ckpt_at[step])), step)
    save(out_dir, step)
    log(f'saved {out_dir} ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
