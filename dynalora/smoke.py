"""Random tiny chat model for the CPU smoke runs and the tests (no network access)."""
import os
import random

CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
_WORDS = ['the', 'and', 'was', 'very', 'a', 'of', 'it', 'is', 'with', 'this', 'study', 'patients', 'good', 'bad', 'review',
          'movie', 'food', 'service', 'results', 'treatment', 'answer', 'yes', 'no', 'maybe', 'problem', 'summary',
          'user', 'assistant']


def smoke_chat_setup(tmp):
    """Directory of a random tiny GPT-2 with a ChatML tokenizer (``<|im_start|>``, ``<|im_end|>`` as end of turn and
    EOS, ``<|endoftext|>`` as padding), standing in for an instruction-tuned model."""
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    os.makedirs(tmp, exist_ok=True)
    mdir = os.path.join(tmp, 'tiny-chat-gpt2')
    if not os.path.exists(os.path.join(mdir, 'config.json')):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        rng = random.Random(0)
        corpus = [' '.join(rng.choice(_WORDS) for _ in range(rng.randint(20, 80))) for _ in range(1200)]
        special = ['<|endoftext|>', '<|im_start|>', '<|im_end|>']
        bpe = Tokenizer(models.BPE(unk_token='<|endoftext|>'))
        bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        bpe.decoder = decoders.ByteLevel()
        bpe.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=special,
                                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token='<|im_end|>', pad_token='<|endoftext|>',
                                      unk_token='<|endoftext|>', additional_special_tokens=['<|im_start|>'])
        tok.chat_template = CHATML
        cfg = GPT2Config(vocab_size=len(tok), n_layer=2, n_head=2, n_embd=64, n_positions=256,
                         bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        GPT2LMHeadModel(cfg).save_pretrained(mdir); tok.save_pretrained(mdir)
    return mdir
