"""LoRA adapter training.

``train_adapter`` reproduces the original recipe (rank 8, alpha 16, dropout 0.1,
max length 128, AdamW, cosine schedule with 5% warmup, 90/10 split) with two
switches: the learning rate (the original run used 5e-6; the extended study
uses the standard 2e-4) and whether padded positions are masked in the loss
(the original run did not mask them).
"""
import math
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

def _load_lm(name, dt):
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dt)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt)


TARGETS = {
    'gpt2': ['c_attn', 'c_proj'],
    'default': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
}


def lora_targets(model_name):
    return TARGETS['gpt2'] if 'gpt2' in model_name else TARGETS['default']


class _DS(Dataset):
    def __init__(self, texts, tok, max_len, mask_pad):
        enc = tok(texts, truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
        self.ids = enc['input_ids']
        self.att = enc['attention_mask']
        self.labels = self.ids.clone()
        if mask_pad:
            self.labels[self.att == 0] = -100
            # keep the first EOS after the text as a target so the model learns to stop
            lengths = self.att.sum(1)
            for i, L in enumerate(lengths.tolist()):
                if L < max_len:
                    self.labels[i, L] = self.ids[i, L]

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        return {'input_ids': self.ids[i], 'attention_mask': self.att[i], 'labels': self.labels[i]}


def train_adapter(model_name, texts, out_dir, lr=2e-4, epochs=3, max_len=128, batch_size=8,
                  grad_accum=2, mask_pad=True, r=8, alpha=16, dropout=0.1, seed=42,
                  val_frac=0.1, eval_every=100, device=None, dtype=None, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = dtype or (torch.float16 if device == 'cuda' else torch.float32)
    random.seed(seed); torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = _load_lm(model_name, torch.float32).to(device)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias='none',
                     target_modules=lora_targets(model_name), task_type='CAUSAL_LM')
    model = get_peft_model(model, cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f'trainable {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.2f}%)')

    texts = list(texts)
    random.Random(seed).shuffle(texts)
    n_val = int(len(texts) * val_frac)
    val_t, tr_t = texts[:n_val], texts[n_val:]
    tr = DataLoader(_DS(tr_t, tok, max_len, mask_pad), batch_size=batch_size, shuffle=True)
    va = DataLoader(_DS(val_t, tok, max_len, mask_pad), batch_size=batch_size)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    steps = math.ceil(len(tr) / grad_accum) * epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    def evaluate():
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for b in va:
                b = {k: v.to(device) for k, v in b.items()}
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                    out = model(**b)
                tot += out.loss.item(); n += 1
        model.train()
        return tot / max(n, 1)

    best = float('inf'); step = 0; model.train()
    history = []
    for ep in range(epochs):
        for i, b in enumerate(tr):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
                loss = model(**b).loss / grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0 or (i + 1) == len(tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step(); step += 1
                if step % eval_every == 0:
                    vl = evaluate(); history.append((step, vl))
                    log(f'epoch {ep + 1} step {step}/{steps} val_loss {vl:.4f} ppl {math.exp(vl):.1f}')
                    if vl < best:
                        best = vl; model.save_pretrained(out_dir)
    vl = evaluate(); history.append((step, vl))
    log(f'final val_loss {vl:.4f} ppl {math.exp(vl):.1f} (best {best:.4f})')
    if vl < best:
        best = vl; model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, 'train_log.txt'), 'w') as f:
        f.write(f'model={model_name} lr={lr} epochs={epochs} mask_pad={mask_pad} r={r} alpha={alpha} '
                f'n_train={len(tr_t)} n_val={len(val_t)} best_val_loss={best:.4f} best_ppl={math.exp(best):.2f}\n')
        for s, v in history:
            f.write(f'{s}\t{v:.4f}\n')
    return out_dir, best
