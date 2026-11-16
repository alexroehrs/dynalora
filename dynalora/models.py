"""Model loading and the pieces of the LoRA recipe shared by the scripts."""
import torch


def load_lm(name, dtype):
    """Loads a causal language model in ``dtype``."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


TARGETS = {
    'gpt2': ['c_attn', 'c_proj'],
    'default': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
}


def lora_targets(model_name):
    """Attention projections that receive a LoRA module (GPT-2 names for the tiny test models)."""
    return TARGETS['gpt2'] if 'gpt2' in model_name else TARGETS['default']


def chat_collate(pad_id):
    """Collate function for ``(token ids, context length)`` pairs: right padding, loss on the tokens after the
    chat context only."""
    def f(batch):
        L = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        for j, (x, n_ctx) in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
            lab[j, n_ctx:len(x)] = torch.tensor(x[n_ctx:])
        return {'input_ids': ids, 'attention_mask': att, 'labels': lab}
    return f
