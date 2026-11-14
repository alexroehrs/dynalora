"""Run provenance: GPU, library versions, git commit.

``environment()`` describes the current session. ``record_session`` appends it, with the
arguments and seeds of the step, to a ``run_info.json`` file, so that a run resumed after a
disconnection, possibly on another GPU, keeps one entry per session.
"""
import json
import os
import platform
import subprocess
import time

_LIBS = ['torch', 'transformers', 'tokenizers', 'peft', 'datasets', 'accelerate', 'numpy', 'pandas', 'lm-eval',
         'rouge-score']
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return 'cpu'


def git_commit(repo=_REPO):
    try:
        c = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10)
        b = subprocess.run(['git', '-C', repo, 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True, timeout=10)
        d = subprocess.run(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True, timeout=10)
        return {'commit': c.stdout.strip() or None, 'branch': b.stdout.strip() or None, 'dirty': bool(d.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'branch': None, 'dirty': None}


def library_versions():
    from importlib import metadata
    out = {}
    for name in _LIBS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def environment():
    import torch
    env = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'gpu': gpu_name(),
           'python': platform.python_version(), 'libraries': library_versions(), 'git': git_commit()}
    if torch.cuda.is_available():
        env['cuda'] = torch.version.cuda
        env['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    return env


def record_session(path, step, args=None, seeds=None, **extra):
    """Appends one session entry (environment, step name, arguments, seeds, extra fields) to ``path``."""
    info = {}
    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)
    entry = {'step': step, **environment()}
    if args is not None:
        entry['args'] = args
    if seeds is not None:
        entry['seeds'] = seeds
    entry.update(extra)
    info.setdefault('sessions', []).append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + '.tmp', 'w') as f:
        json.dump(info, f, indent=2, default=str)
    os.replace(path + '.tmp', path)
    return entry
