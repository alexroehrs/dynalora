"""Shared helper: a random LoRA adapter for the tiny test models, built offline.

The adapters are initialized with ``init_lora_weights=False`` so that ``lora_B`` is not zero and the
adapted model differs from the base model (a freshly initialized LoRA adapter is the identity).
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _adapter(model, targets, out):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(1)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.1, target_modules=targets,
                                          init_lora_weights=False, task_type='CAUSAL_LM'))
    with torch.no_grad():   # make the adapter's effect clearly visible in the logits
        for n, p in pm.named_parameters():
            if 'lora_B' in n:
                p.mul_(5.0)
    pm.save_pretrained(out)
