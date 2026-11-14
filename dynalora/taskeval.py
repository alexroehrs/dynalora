"""Scoring and greedy decoding of a LoRA adapter under logit-space and weight-space scaling.

``TaskModel`` wraps a base model and one adapter and evaluates either form at a factor ``lam``:

* ``logit``   y = y_base + lam * (y_lora - y_base) on the next-token logits (Equation 1; two passes, one
              with the adapter disabled, except at lam = 0 and lam = 1, which need one);
* ``weight``  the adapted model with every LoRA scaling multiplied by lam (one pass).

Three operations are provided, all batched, with left padding and explicit position ids so that the result
of an example does not depend on the batch it is in:

* ``answer_logliks``: log-likelihood of given answer token sequences after their contexts, for a list of
  logit-space factors at once (they share the two passes) or for the weight scale currently set; it also
  returns the summed KL divergence KL(p_lam || p_base) over the answer positions, at temperature 1;
* ``choice_scores``: log-likelihood of every option of a multiple-choice item (the option with the highest
  one is the prediction, as in the lm-evaluation-harness);
* ``greedy``: greedy decoding with key-value caches (one cache per pass) until a stop token or
  ``max_new_tokens``.
"""
import inspect

import torch
import torch.nn.functional as F


class TaskModel:
    def __init__(self, model_name, adapter_dir=None, device=None, dtype=None, stop_ids=None):
        from transformers import AutoTokenizer
        from .models import load_lm
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype or (torch.bfloat16 if self.device == 'cuda' else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = load_lm(model_name, self.dtype).to(self.device)
        base.eval()
        self.has_adapter = adapter_dir is not None
        self._lora_layers, self._base_scaling = [], []
        if self.has_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, adapter_dir).to(self.device)
            self.model.eval()
            self._lora_layers = [m for m in self.model.modules() if hasattr(m, 'scaling') and isinstance(m.scaling, dict)]
            self._base_scaling = [dict(m.scaling) for m in self._lora_layers]
        else:
            self.model = base
        try:
            params = inspect.signature(base.forward).parameters
        except (TypeError, ValueError):
            params = {}
        self._keep = 'logits_to_keep' in params
        self.stop_ids = list(stop_ids) if stop_ids else [self.tok.eos_token_id]
        self.pad_id = self.tok.pad_token_id

    # ---- adapter strength ------------------------------------------------------------------------
    def set_weight_scale(self, lam):
        for m, b in zip(self._lora_layers, self._base_scaling):
            for k in m.scaling:
                m.scaling[k] = b[k] * lam

    def reset_weight_scale(self):
        self.set_weight_scale(1.0)

    # ---- forward ---------------------------------------------------------------------------------
    def _batch(self, seqs):
        """Left-padded ids, attention mask and position ids of a list of token id lists."""
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, L - len(s):] = torch.tensor(s); att[j, L - len(s):] = 1
        pos = (att.cumsum(1) - 1).clamp(min=0)
        return ids.to(self.device), att.to(self.device), pos.to(self.device)

    @torch.no_grad()
    def _forward(self, adapter_on, **kw):
        if self.has_adapter and not adapter_on:
            with self.model.disable_adapter():
                return self.model(**kw)
        return self.model(**kw)

    def _tail_logits(self, ids, att, pos, adapter_on, k):
        """Float32 logits of the last ``k`` positions."""
        kw = dict(input_ids=ids, attention_mask=att, position_ids=pos)
        if self._keep:
            kw['logits_to_keep'] = k
        return self._forward(adapter_on, **kw).logits[:, -k:, :].float()

    # ---- teacher-forced scoring ------------------------------------------------------------------
    @torch.no_grad()
    def answer_logliks(self, contexts, answers, form, lambdas=None, batch_size=8, with_kl=True):
        """For every (context ids, answer ids) pair: ``{factor: (loglik, kl_sum, greedy)}`` where ``loglik``
        is the summed log-probability of the answer tokens, ``kl_sum`` the summed KL(p || p_base) over the
        answer positions and ``greedy`` whether every answer token is the argmax. ``form='logit'`` returns
        one entry per value of ``lambdas``; ``form='weight'`` scores with the weight scale currently set
        and returns the single key ``None``; with ``with_kl=False`` it skips the base pass and the KL is NaN."""
        if form not in ('logit', 'weight'):
            raise ValueError(form)
        keys = list(lambdas) if form == 'logit' else [None]
        out = [None] * len(contexts)
        order = sorted(range(len(contexts)), key=lambda i: -(len(contexts[i]) + len(answers[i])))
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            seqs = [(list(contexts[i]) + list(answers[i]))[:-1] for i in idx]
            k = max(len(answers[i]) for i in idx)
            ids, att, pos = self._batch(seqs)
            need_base = with_kl or form == 'logit' or not self.has_adapter
            yb = self._tail_logits(ids, att, pos, False, k) if need_base else None
            ya = self._tail_logits(ids, att, pos, True, k) if self.has_adapter else yb
            for j, i in enumerate(idx):
                n = len(answers[i])
                tgt = torch.tensor(list(answers[i]), device=self.device)
                base = yb[j, k - n:] if need_base else None
                base_lp = F.log_softmax(base, -1) if need_base else None
                res = {}
                for key in keys:
                    y = ya[j, k - n:] if key is None else base + key * (ya[j, k - n:] - base)
                    lp = F.log_softmax(y, -1)
                    kl = (lp.exp() * (lp - base_lp)).sum(-1).clamp(min=0).sum() if need_base else float('nan')
                    res[key] = (float(lp.gather(-1, tgt[:, None]).sum()), float(kl),
                                bool((lp.argmax(-1) == tgt).all()))
                out[i] = res
            del yb, ya
        return out

    def choice_scores(self, contexts, options, form, lambdas=None, batch_size=8):
        """``{factor: [loglik of each option]}`` per item; ``options`` is a list of token id lists shared by
        all items."""
        keys = list(lambdas) if form == 'logit' else [None]
        if all(len(o) == 1 for o in options):      # one-token options: one pass over the context serves them all
            opt = torch.tensor([o[0] for o in options], device=self.device)
            out = [None] * len(contexts)
            order = sorted(range(len(contexts)), key=lambda i: -len(contexts[i]))
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                ids, att, pos = self._batch([list(contexts[i]) for i in idx])
                need_base = form == 'logit' or not self.has_adapter
                yb = self._tail_logits(ids, att, pos, False, 1)[:, -1] if need_base else None
                ya = self._tail_logits(ids, att, pos, True, 1)[:, -1] if self.has_adapter else yb
                for j, i in enumerate(idx):
                    out[i] = {k: F.log_softmax(ya[j] if k is None else yb[j] + k * (ya[j] - yb[j]), -1)[opt].tolist()
                              for k in keys}
            return out
        ctx, ans, owner = [], [], []
        for i, c in enumerate(contexts):
            for o in options:
                ctx.append(c); ans.append(o); owner.append(i)
        flat = self.answer_logliks(ctx, ans, form, lambdas, batch_size=max(1, batch_size), with_kl=form == 'logit')
        out = [{k: [] for k in keys} for _ in contexts]
        for i, r in zip(owner, flat):
            for k in keys:
                out[i][k].append(r[k][0])
        return out

    # ---- greedy decoding -------------------------------------------------------------------------
    @torch.no_grad()
    def greedy(self, contexts, form, lam, max_new_tokens, batch_size=8):
        """Greedy continuation of every context: list of ``(token ids before the stop token, ended)``.
        For ``form='weight'`` the weight scale is set to ``lam`` and restored afterwards."""
        if form == 'logit':
            passes = [False] if (lam == 0 or not self.has_adapter) else [True] if lam == 1 else [False, True]
        elif form == 'weight':
            passes = [False] if (lam == 0 or not self.has_adapter) else [True]
            if passes == [True]:
                self.set_weight_scale(lam)
        else:
            raise ValueError(form)
        stop = torch.tensor(self.stop_ids, device=self.device)
        out = [None] * len(contexts)
        order = sorted(range(len(contexts)), key=lambda i: -len(contexts[i]))
        try:
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                ids, att, pos = self._batch([list(contexts[i]) for i in idx])
                B = ids.shape[0]
                past = {p: None for p in passes}
                cur, cur_pos = ids, pos
                done = torch.zeros(B, dtype=torch.bool, device=self.device)
                gen = []
                for _ in range(max_new_tokens):
                    ys = {}
                    for p in passes:
                        o = self._forward(p, input_ids=cur, attention_mask=att, position_ids=cur_pos,
                                          past_key_values=past[p], use_cache=True)
                        past[p] = o.past_key_values
                        ys[p] = o.logits[:, -1, :].float()
                    y = ys[passes[0]] if len(passes) == 1 else ys[False] + lam * (ys[True] - ys[False])
                    nxt = y.argmax(-1)
                    gen.append(nxt)
                    done = done | torch.isin(nxt, stop)
                    if bool(done.all()):
                        break
                    cur = nxt[:, None]
                    cur_pos = cur_pos[:, -1:] + 1
                    att = torch.cat([att, torch.ones((B, 1), dtype=att.dtype, device=self.device)], 1)
                g = torch.stack(gen, 1).tolist()
                stops = set(self.stop_ids)
                for j, i in enumerate(idx):
                    n = next((t for t, x in enumerate(g[j]) if x in stops), None)
                    out[i] = (g[j] if n is None else g[j][:n], n is not None)
                del past
        finally:
            if form == 'weight' and self.has_adapter:
                self.reset_weight_scale()
        return out
