"""Generation under the compared procedures.

Methods (all sample from softmax(final_logits / T) with multinomial sampling,
no top-k or nucleus truncation, no repetition penalty):

* ``dynalora``  final = y_base + lam * (y_lora - y_base)      (two forward passes)
* ``weight``    adapted model with the LoRA scaling multiplied by lam (one pass)
* ``cfg``       classifier-free guidance on the base model with a domain
                prefix: final = y_uncond + gamma * (y_cond - y_uncond)
* ``lora``      adapted model at its trained strength (one pass; equals dynalora at lam = 1)
* ``base``      base model (equals dynalora at lam = 0)

Every generated sequence records the mean entropy (nats) of the sampling
distribution over its decoding steps, which is what the entropy-matched
temperature ablation uses.
"""
import math
import time

import torch
import torch.nn.functional as F


class Generator:
    def __init__(self, model_name, adapter_dir=None, device=None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or (torch.float16 if self.device == 'cuda' else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(adapter_dir or model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        from .train import _load_lm
        base = _load_lm(model_name, self.dtype).to(self.device)
        base.eval()
        self.has_adapter = adapter_dir is not None
        if self.has_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, adapter_dir).to(self.device)
            self.model.eval()
            self._lora_layers = [m for m in self.model.modules() if hasattr(m, 'scaling') and isinstance(m.scaling, dict)]
            self._base_scaling = [dict(m.scaling) for m in self._lora_layers]
        else:
            self.model = base
        self.eos = self.tok.eos_token_id

    # ---- weight-space scaling -------------------------------------------------
    def set_weight_scale(self, lam):
        for m, b in zip(self._lora_layers, self._base_scaling):
            for k in m.scaling:
                m.scaling[k] = b[k] * lam

    def reset_weight_scale(self):
        self.set_weight_scale(1.0)

    # ---- single forward -------------------------------------------------------
    @torch.no_grad()
    def _logits(self, ids, att, adapter_on):
        if self.has_adapter and not adapter_on:
            with self.model.disable_adapter():
                out = self.model(input_ids=ids, attention_mask=att)
        else:
            out = self.model(input_ids=ids, attention_mask=att)
        return out.logits[:, -1, :].float()

    # ---- main loop ------------------------------------------------------------
    @torch.no_grad()
    def generate(self, prompt, n_samples=20, method='dynalora', lam=1.0, gamma=1.0,
                 temperature=0.7, max_new_tokens=50, seed=0, cfg_prefix=None):
        tok = self.tok
        enc = tok([prompt] * n_samples, return_tensors='pt').to(self.device)
        ids, att = enc['input_ids'], enc['attention_mask']
        n_prompt = ids.shape[1]
        if method == 'cfg':
            cenc = tok([cfg_prefix + prompt] * n_samples, return_tensors='pt').to(self.device)
            cids, catt = cenc['input_ids'], cenc['attention_mask']
        if method == 'weight':
            self.set_weight_scale(lam)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        finished = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        ent_sum = torch.zeros(n_samples, device=self.device)
        steps = torch.zeros(n_samples, device=self.device)
        t0 = time.time()
        for _ in range(max_new_tokens):
            if method == 'dynalora':
                yb = self._logits(ids, att, adapter_on=False)
                if lam == 0.0:
                    final = yb
                else:
                    yl = self._logits(ids, att, adapter_on=True)
                    final = yb + lam * (yl - yb)
            elif method == 'weight':
                final = self._logits(ids, att, adapter_on=True)
            elif method == 'lora':
                final = self._logits(ids, att, adapter_on=True)
            elif method == 'base':
                final = self._logits(ids, att, adapter_on=False)
            elif method == 'cfg':
                yu = self._logits(ids, att, adapter_on=False)
                yc = self._logits(cids, catt, adapter_on=False)
                final = yu + gamma * (yc - yu)
            else:
                raise ValueError(method)
            probs = F.softmax(final / temperature, dim=-1)
            logp = torch.log(probs.clamp_min(1e-12))
            H = -(probs * logp).sum(-1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            active = ~finished
            ent_sum += torch.where(active, H, torch.zeros_like(H))
            steps += active.float()
            nxt = torch.where(active, nxt, torch.full_like(nxt, self.eos))
            ids = torch.cat([ids, nxt[:, None]], 1)
            att = torch.cat([att, active[:, None].long()], 1)
            if method == 'cfg':
                cids = torch.cat([cids, nxt[:, None]], 1)
                catt = torch.cat([catt, active[:, None].long()], 1)
            finished = finished | (nxt == self.eos)
            if finished.all():
                break
        if method == 'weight':
            self.reset_weight_scale()
        elapsed = time.time() - t0
        rows = []
        for i in range(n_samples):
            gen_ids = ids[i, n_prompt:].tolist()
            n_gen = 0
            for t in gen_ids:
                if t == self.eos:
                    break
                n_gen += 1
            cont = tok.decode(gen_ids[:n_gen], skip_special_tokens=True)
            rows.append({'prompt': prompt, 'sample': i, 'seed': int(seed), 'method': method,
                         'lambda': float(lam), 'gamma': float(gamma), 'temperature': float(temperature),
                         'continuation': cont, 'text': prompt + cont, 'n_tokens': n_gen,
                         'ended_eos': bool(n_gen < max_new_tokens and finished[i].item()),
                         'mean_entropy': float(ent_sum[i].item() / max(steps[i].item(), 1)),
                         'batch_seconds': elapsed})
        return rows


def run_grid(gen, prompts, method, values, n_samples=20, temperature=0.7, seed=0,
             max_new_tokens=50, cfg_prefix=None, log=print):
    """Run ``method`` over ``values`` (lambdas, gammas or temperatures) and all prompts."""
    rows = []
    for v in values:
        for p_i, p in enumerate(prompts):
            kw = dict(n_samples=n_samples, method=method, max_new_tokens=max_new_tokens,
                      seed=seed + 1000 * p_i, cfg_prefix=cfg_prefix, temperature=temperature)
            if method == 'cfg':
                kw['gamma'] = v
            elif method in ('lora', 'base'):
                kw['temperature'] = v
            else:
                kw['lam'] = v
            r = gen.generate(p, **kw)
            for x in r:
                x['prompt_id'] = p_i
            rows += r
        log(f'{method} value={v}: {len(rows)} rows so far')
    return rows
