"""Task-adapter study: task metrics, chat format, scoring and greedy decoding under both forms of
scaling, and the training and evaluation scripts end to end on a random tiny chat model."""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from dynalora import tasks as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def tm(tmp_path_factory):
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from dynalora.smoke import smoke_chat_setup
    from conftest import _adapter
    from dynalora.taskeval import TaskModel
    tmp = str(tmp_path_factory.mktemp('tasks'))
    mdir = smoke_chat_setup(tmp)
    adir = os.path.join(tmp, 'adapters', 'a')
    torch.manual_seed(0)
    _adapter(GPT2LMHeadModel.from_pretrained(mdir), ['c_attn', 'c_proj'], adir)
    AutoTokenizer.from_pretrained(mdir).save_pretrained(adir)
    m = TaskModel(mdir, adir, device='cpu', dtype=torch.float32)
    m.stop_ids = T.stop_ids(m.tok)
    return m


def _contexts(tm, task='gsm8k', n=5):
    ex = T.load_examples(task, 'test', n=n, smoke=True)
    return ex, [T.context_ids(tm.tok, task, e['input'], 64)[0] for e in ex]


# ---- metrics and format ------------------------------------------------------------------------------
def test_numbers_and_gsm8k():
    assert T.normalize_number('1,234.50') == '1234.5'
    assert T.normalize_number('$72') == '72' and T.normalize_number('18.0') == '18'
    assert T.gsm8k_prediction('3 + 4 = 7\n#### 7') == '7'
    assert T.gsm8k_prediction('so the answer is 12 apples') == '12'
    assert T.gsm8k_prediction('no number') is None
    assert T.gsm8k_gold('x <<1+1=2>> y\n#### 2') == '2'
    assert T.clean_gsm8k_answer('a <<48/2=24>>24 b') == 'a 24 b'
    assert T.score_output('gsm8k', 'work\n#### 5', 'w\n#### 5')['em'] == 1.0
    assert T.score_output('gsm8k', 'work\n#### 6', 'w\n#### 5')['em'] == 0.0


def test_rouge_f1_and_repetition():
    r = T.score_output('samsum', 'the cat sat', 'the cat sat')
    assert r['rougeL'] == pytest.approx(1.0) and r['rouge2'] == pytest.approx(1.0)
    assert T.score_output('samsum', 'dogs run', 'the cat sat')['rouge1'] == 0.0
    assert T.macro_f1(['a', 'b', 'a'], ['a', 'b', 'b'], ['a', 'b']) == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert T.repeated_bigram_share([1, 2, 1, 2, 1, 2]) == pytest.approx(1 - 2 / 5)
    assert T.repeated_bigram_share([1]) == 0.0


def test_smoke_examples_and_chat_format(tm):
    for task in T.TASKS:
        tr, te = T.load_examples(task, 'train', smoke=True), T.load_examples(task, 'test', smoke=True)
        assert len(tr) == 48 and len(te) == 12 and {'id', 'input', 'answer'} <= set(tr[0])
        if T.spec(task)['kind'] == 'choice':
            assert all(e['answer'] in T.spec(task)['options'] for e in tr + te)
    tok = tm.tok
    assert T.end_of_turn_id(tok) == tok.convert_tokens_to_ids('<|im_end|>')
    ids, tr = T.context_ids(tok, 'yelp5', 'Review: ' + 'good ' * 200, max_input_tokens=20)
    assert tr and T.spec('yelp5')['instruction'] in tok.decode(ids)
    ids2, tr2 = T.context_ids(tok, 'yelp5', 'Review: good', max_input_tokens=20)
    assert not tr2 and len(ids2) < len(ids)
    assert T.answer_ids(tok, 'yes')[-1] == T.end_of_turn_id(tok)
    assert T.lora_targets_for(tm.model, 'gpt2') == ['c_attn', 'c_proj']


# ---- scoring -----------------------------------------------------------------------------------------
def test_answer_logliks_match_manual_and_batching(tm):
    ex, ctx = _contexts(tm)
    ans = [T.answer_ids(tm.tok, e['answer']) for e in ex]
    lams = [0.0, 0.5, 1.0, 2.0]
    one = tm.answer_logliks(ctx, ans, 'logit', lams, batch_size=1)
    many = tm.answer_logliks(ctx, ans, 'logit', lams, batch_size=4)
    for a, b in zip(one, many):
        for lam in lams:
            assert a[lam][0] == pytest.approx(b[lam][0], abs=2e-3)
            assert a[lam][1] == pytest.approx(b[lam][1], abs=2e-3)
    # manual computation for the first item
    ids = torch.tensor([ctx[0] + ans[0]])
    with torch.no_grad():
        with tm.model.disable_adapter():
            yb = tm.model(input_ids=ids).logits[0].float()
        ya = tm.model(input_ids=ids).logits[0].float()
    n = len(ans[0]); sl = slice(len(ctx[0]) - 1, len(ctx[0]) - 1 + n); tgt = torch.tensor(ans[0])
    for lam in lams:
        lp = F.log_softmax(yb[sl] + lam * (ya[sl] - yb[sl]), -1)
        assert one[0][lam][0] == pytest.approx(float(lp.gather(-1, tgt[:, None]).sum()), abs=2e-3)
        kl = (lp.exp() * (lp - F.log_softmax(yb[sl], -1))).sum()
        assert one[0][lam][1] == pytest.approx(float(kl), abs=2e-3)
    assert one[0][0.0][1] == pytest.approx(0.0, abs=1e-5)       # KL of the base model to itself


def test_forms_agree_at_zero_and_one(tm):
    ex, ctx = _contexts(tm)
    ans = [T.answer_ids(tm.tok, e['answer']) for e in ex]
    logit = tm.answer_logliks(ctx, ans, 'logit', [0.0, 1.0, 2.0], batch_size=4)
    for lam in (0.0, 1.0, 2.0):
        tm.set_weight_scale(lam)
        w = tm.answer_logliks(ctx, ans, 'weight', None, batch_size=4)
        tm.reset_weight_scale()
        for a, b in zip(logit, w):
            if lam in (0.0, 1.0):
                assert a[lam][0] == pytest.approx(b[None][0], abs=2e-3)
    # at factor 2 the two forms differ (the adapter of the fixture is not small)
    tm.set_weight_scale(2.0)
    w2 = tm.answer_logliks(ctx, ans, 'weight', None, batch_size=4)
    tm.reset_weight_scale()
    assert any(abs(a[2.0][0] - b[None][0]) > 1e-3 for a, b in zip(logit, w2))
    assert all(m.scaling[k] == b[k] for m, b in zip(tm._lora_layers, tm._base_scaling) for k in m.scaling)


def test_choice_scores(tm):
    ex, ctx = _contexts(tm, 'pubmedqa')
    opts = [T.answer_ids(tm.tok, o, eot=False) for o in T.spec('pubmedqa')['options']]
    res = tm.choice_scores(ctx, opts, 'logit', [0.0, 1.0], batch_size=3)
    assert len(res) == len(ctx) and len(res[0][1.0]) == 3
    direct = tm.answer_logliks([ctx[0]] * 3, opts, 'logit', [1.0], batch_size=1)
    assert res[0][1.0] == pytest.approx([d[1.0][0] for d in direct], abs=2e-3)


# ---- greedy decoding ---------------------------------------------------------------------------------
def _naive_greedy(tm, ctx, form, lam, n):
    ids = list(ctx); out = []
    if form == 'weight':
        tm.set_weight_scale(lam)
    for _ in range(n):
        x = torch.tensor([ids])
        with torch.no_grad():
            if form == 'logit':
                with tm.model.disable_adapter():
                    yb = tm.model(input_ids=x).logits[0, -1].float()
                ya = tm.model(input_ids=x).logits[0, -1].float()
                y = yb + lam * (ya - yb)
            else:
                y = tm.model(input_ids=x).logits[0, -1].float()
        t = int(y.argmax())
        if t in tm.stop_ids:
            break
        out.append(t); ids.append(t)
    tm.reset_weight_scale()
    return out


@pytest.mark.parametrize('form,lam', [('logit', 0.0), ('logit', 0.5), ('logit', 1.0), ('logit', 2.0),
                                      ('weight', 0.0), ('weight', 0.5), ('weight', 2.0)])
def test_greedy_matches_naive_decoding(tm, form, lam):
    ex, ctx = _contexts(tm, 'samsum', n=4)
    got = tm.greedy(ctx, form, lam, max_new_tokens=6, batch_size=3)
    for c, (ids, ended) in zip(ctx, got):
        ref = _naive_greedy(tm, c, form, lam, 6)
        assert ids == ref
        assert ended == (len(ref) < 6)
    assert all(m.scaling[k] == b[k] for m, b in zip(tm._lora_layers, tm._base_scaling) for k in m.scaling)


def test_greedy_forms_agree_at_one(tm):
    ex, ctx = _contexts(tm, 'gsm8k', n=4)
    assert tm.greedy(ctx, 'logit', 1.0, 6, batch_size=2) == tm.greedy(ctx, 'weight', 1.0, 6, batch_size=4)


# ---- scripts end to end ------------------------------------------------------------------------------
def _run(*args):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout + r.stderr


def test_train_and_eval_scripts(tmp_path):
    ad, sc = str(tmp_path / 'adapters'), str(tmp_path / 'scores')
    for task in ('pubmedqa', 'gsm8k'):
        _run('scripts/train_task_chat.py', '--model', 'x', '--task', task, '--adapters_dir', ad, '--smoke',
             '--ckpt_fracs', '0.4')
        name = f'tiny-chat-gpt2__{task}__s42'
        log = json.load(open(os.path.join(ad, name, 'train_log.json')))
        assert log['steps'] == 10 and log['saved_at_step'] == 10 and log['val_loss'][0][0] == 0
        ck = json.load(open(os.path.join(ad, name + '__ckpt40', 'train_log.json')))
        assert ck['saved_at_step'] == 4
        assert 'skipped' in _run('scripts/train_task_chat.py', '--model', 'x', '--task', task, '--adapters_dir', ad,
                                 '--smoke')
        out = os.path.join(sc, name)
        args = ['scripts/eval_task.py', '--model', 'x', '--task', task, '--adapter_dir', os.path.join(ad, name),
                '--out_dir', out, '--base_cache', os.path.join(sc, '_base'), '--kl_items', '6', '--smoke']
        _run(*args)
        for form in ('logit', 'weight'):
            df = pd.read_csv(os.path.join(out, f'own__{form}.csv'))
            assert sorted(df.factor.unique()) == [0, 0.5, 1, 2] and len(df) == 4 * 12
            assert df.score.between(0, 1).all()
            kl = pd.read_csv(os.path.join(out, f'kl__{form}.csv'))
            assert len(kl) == 4 * 6 and (kl[kl.factor == 0].kl_sum.abs() < 1e-4).all()
            assert (kl[kl.factor == 2].kl_sum > 0).all()
        lo, we = (pd.read_csv(os.path.join(out, f'own__{f}.csv')) for f in ('logit', 'weight'))
        for lam in (0, 1):     # the base model and the adapter itself are the same in both forms
            a, b = lo[lo.factor == lam].reset_index(drop=True), we[we.factor == lam].reset_index(drop=True)
            assert (a.score == b.score).all() and (a.item == b.item).all()
        assert os.path.exists(os.path.join(sc, '_base', f'tiny-chat-gpt2__{task}__n12.csv'))
        assert 'skipped' in _run(*args)
    # an adapter scored on another task, with the shared base scores
    out = os.path.join(sc, 'tiny-chat-gpt2__gsm8k__s42')
    _run('scripts/eval_task.py', '--model', 'x', '--task', 'pubmedqa', '--adapter_task', 'gsm8k', '--adapter_dir',
         os.path.join(ad, 'tiny-chat-gpt2__gsm8k__s42'), '--out_dir', out, '--base_cache', os.path.join(sc, '_base'),
         '--smoke', '--forms', 'logit')
    cross = pd.read_csv(os.path.join(out, 'cross_pubmedqa__logit.csv'))
    own = pd.read_csv(os.path.join(sc, 'tiny-chat-gpt2__pubmedqa__s42', 'own__logit.csv'))
    assert (cross[cross.factor == 0].score.values == own[own.factor == 0].score.values).all()
    assert os.path.exists(os.path.join(out, 'own__logit__outputs.jsonl'))


# ---- other architectures (rotary positions, fused attention projection) -------------------------------
def _tiny_arch(tmp, arch):
    from transformers import AutoTokenizer
    from dynalora.smoke import smoke_chat_setup
    from conftest import _adapter
    from dynalora.taskeval import TaskModel
    tok = AutoTokenizer.from_pretrained(smoke_chat_setup(tmp))
    mdir, adir = os.path.join(tmp, arch), os.path.join(tmp, arch + '_adapter')
    torch.manual_seed(0)
    if arch == 'qwen2':
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cls = Qwen2ForCausalLM
        m = cls(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                            bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=False))
    else:
        from transformers import Phi3Config, Phi3ForCausalLM
        cls = Phi3ForCausalLM
        m = cls(Phi3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                           original_max_position_embeddings=512, bos_token_id=tok.eos_token_id,
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id))
    m.save_pretrained(mdir); tok.save_pretrained(mdir)
    targets = T.lora_targets_for(m, arch)
    _adapter(cls.from_pretrained(mdir), targets, adir)
    tm = TaskModel(mdir, adir, device='cpu', dtype=torch.float32)
    tm.stop_ids = T.stop_ids(tm.tok)
    return tm, targets


@pytest.mark.parametrize('arch', ['qwen2', 'phi3'])
def test_other_architectures(tmp_path, arch):
    tm, targets = _tiny_arch(str(tmp_path), arch)
    assert targets == (['qkv_proj', 'o_proj'] if arch == 'phi3' else ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    assert len(tm._lora_layers) > 0
    ex, ctx = _contexts(tm, 'samsum', n=4)
    ans = [T.answer_ids(tm.tok, e['answer']) for e in ex]
    one = tm.answer_logliks(ctx, ans, 'logit', [0.5, 2.0], batch_size=1)
    many = tm.answer_logliks(ctx, ans, 'logit', [0.5, 2.0], batch_size=4)
    for a, b in zip(one, many):
        assert a[2.0][0] == pytest.approx(b[2.0][0], abs=5e-3) and a[0.5][1] == pytest.approx(b[0.5][1], abs=5e-3)
    for form, lam in (('logit', 0.5), ('logit', 2.0), ('weight', 2.0)):
        got = tm.greedy(ctx, form, lam, max_new_tokens=6, batch_size=3)
        for c, (ids, ended) in zip(ctx, got):
            assert ids == _naive_greedy(tm, c, form, lam, 6)


def test_choice_scores_one_token_path(tm):
    """One-token options share one pass; the result equals scoring each option as an answer."""
    ex, ctx = _contexts(tm, 'yelp5')
    opts = [[tm.tok(o, add_special_tokens=False)['input_ids'][0]] for o in ('good', 'bad', 'the')]
    fast = tm.choice_scores(ctx, opts, 'logit', [0.0, 0.5, 2.0], batch_size=2)
    for i in range(len(ctx)):
        slow = tm.answer_logliks([ctx[i]] * 3, opts, 'logit', [0.0, 0.5, 2.0], batch_size=1)
        for lam in (0.0, 0.5, 2.0):
            assert fast[i][lam] == pytest.approx([r[lam][0] for r in slow], abs=2e-3)
    tm.set_weight_scale(2.0)
    w = tm.choice_scores(ctx, opts, 'weight', None, batch_size=2)
    slow = tm.answer_logliks([ctx[0]] * 3, opts, 'weight', None, batch_size=1)
    tm.reset_weight_scale()
    assert w[0][None] == pytest.approx([r[None][0] for r in slow], abs=2e-3)


# ---- analysis ----------------------------------------------------------------------------------------
def test_analysis_on_synthetic_scores(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    root = tmp_path / 'level4' / 'scores'
    items = [f'i{k}' for k in range(40)]
    p_own = {0: .4, .5: .6, 1: .7, 2: .5}

    def frame(task, form, probs, gen=False):
        rows = []
        for lam, p in probs.items():
            for it in items:
                r = dict(task=task, form=form, factor=float(lam), item=it, gold='yes', pred='yes',
                         score=float(rng.random() < p))
                if gen:
                    r.update(n_tokens=20, ended=True, rep2=0.1, gold='', pred='')
                rows.append(r)
        return pd.DataFrame(rows)

    for name, fac in (('m__pubmedqa__s42', 1.0), ('m__pubmedqa__s43', 1.0), ('m__pubmedqa__s42__ckpt20', 0.6)):
        d = root / name; d.mkdir(parents=True)
        for form in ('logit', 'weight'):
            frame('pubmedqa', form, {k: v * fac for k, v in p_own.items()}).to_csv(d / f'own__{form}.csv', index=False)
            if 'ckpt' not in name:
                frame('gsm8k', form, {0: .5, .5: .45, 1: .4, 2: .2}, gen=True).to_csv(d / f'cross_gsm8k__{form}.csv', index=False)
                kl = pd.DataFrame([dict(task='pubmedqa', form=form, item=it, factor=float(lam), nll=1.0, n_tok=2,
                                        kl_sum=2 * lam ** 2 * (1.0 if form == 'weight' else 1.5))
                                   for lam in p_own for it in items[:10]])
                kl.to_csv(d / f'kl__{form}.csv', index=False)
    r = subprocess.run([sys.executable, 'scripts/analysis/summarize.py', '--results', str(tmp_path)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    out = tmp_path / 'level4' / 'analysis'
    cv = pd.read_csv(out / 'curves.csv')
    assert len(cv) == 3 * 2 * 4 and cv.ci_lo.le(cv.score).all() and cv.score.le(cv.ci_hi).all()
    assert pd.read_csv(out / 'curves_seeds.csv').n_seeds.max() == 2
    fr = pd.read_csv(out / 'frontier.csv')
    assert (fr[fr.factor == 0].cross_rel_change.abs() < 1e-9).all() and (fr[fr.factor == 2].cross_rel_change < 0).all()
    eq = pd.read_csv(out / 'equal_divergence.csv')
    # KL_weight(2) = 4, KL_logit(lam) = 1.5 lam^2 on the grid {1, 2}: linear interpolation gives lam = 1 + 2.5/4.5
    row = eq[(eq.factor_weight == 2) & (eq.seed == 42)].iloc[0]
    assert row.factor_logit == pytest.approx(1 + 2.5 / 4.5, abs=1e-6)
    assert set(pd.read_csv(out / 'truncated.csv').ckpt) == {20, 100}
    assert pd.read_csv(out / 'amplification.csv').factor.min() > 1
    for f in ('fig_task_curves__m.pdf', 'fig_task_frontier__m.pdf', 'fig_task_truncated.pdf', 'SUMMARY.md'):
        assert (out / f).exists()


# ---- answers cut by the token budget -------------------------------------------------------------------
def test_extend_cut_outputs(tmp_path):
    """Cut answers are regenerated with a larger budget, continue the old answer, and replace it in the analysis."""
    ad, sc = str(tmp_path / 'adapters'), str(tmp_path / 'scores')
    _run('scripts/train_task_chat.py', '--model', 'x', '--task', 'gsm8k', '--adapters_dir', ad, '--smoke')
    name = 'tiny-chat-gpt2__gsm8k__s42'
    out, base = os.path.join(sc, name), os.path.join(sc, '_base')
    _run('scripts/eval_task.py', '--model', 'x', '--task', 'gsm8k', '--adapter_dir', os.path.join(ad, name), '--out_dir', out,
         '--base_cache', base, '--smoke')
    lo = pd.read_csv(os.path.join(out, 'own__logit.csv'), dtype={'item': str})
    n_cut = {f: int((~g.ended).sum()) for f, g in lo.groupby('factor')}
    assert n_cut[0.0] > 0 and n_cut[1.0] > 0          # the random model rarely ends within 8 tokens
    args = ['scripts/extend_cut_outputs.py', '--model', 'x', '--task', 'gsm8k', '--base_cache', base, '--smoke']
    _run(*args)                                       # base model first
    adapter = args + ['--adapter_dir', os.path.join(ad, name), '--out_dir', out]
    _run(*adapter)
    for form in ('logit', 'weight'):
        ext = pd.read_csv(os.path.join(out, f'own__{form}__ext.csv'), dtype={'item': str})
        assert set(ext.factor) == {0.0, 0.5, 1.0} and (ext.budget == 16).all()      # nothing above --max_factor
        assert len(ext[ext.factor == 0]) == n_cut[0.0] and (ext.n_tokens > 8).all() | ext.ended.all()
    # deterministic decoding: the new answer starts with the old one
    old = {(o['factor'], o['item']): o['text'] for o in map(json.loads, open(os.path.join(out, 'own__logit__outputs.jsonl')))}
    new = [json.loads(l) for l in open(os.path.join(out, 'own__logit__ext_outputs.jsonl'))]
    ascii_old = [(o, old[(o['factor'], o['item'])]) for o in new if old[(o['factor'], o['item'])].isascii()]
    assert new and all(o['text'].startswith(t.rstrip()) for o, t in ascii_old)
    # factor 1 is shared by the two forms; a second run does nothing
    e_l, e_w = (pd.read_csv(os.path.join(out, f'own__{f}__ext.csv'), dtype={'item': str}) for f in ('logit', 'weight'))
    a, b = e_l[e_l.factor == 1].set_index('item'), e_w[e_w.factor == 1].set_index('item')
    assert (a.n_tokens == b.n_tokens.reindex(a.index)).all()
    before = os.path.getsize(os.path.join(out, 'own__logit__ext.csv'))
    _run(*adapter)
    assert os.path.getsize(os.path.join(out, 'own__logit__ext.csv')) == before
    # the analysis uses the regenerated rows
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))
    import summarize
    merged = summarize._with_ext(os.path.join(out, 'own__logit.csv'))
    assert len(merged) == len(lo)
    m0 = merged[merged.factor == 0].set_index('item')
    assert (m0.n_tokens.reindex(e_l[e_l.factor == 0].item).values == e_l[e_l.factor == 0].n_tokens.values).all()
