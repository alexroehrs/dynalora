"""Where the experiment results live.

Results are not kept in the repository. The results root is, in order of precedence:

1. the ``--results DIR`` argument of a script;
2. the environment variable ``DYNALORA_RESULTS``;
3. ``/content/drive/MyDrive/dynalora_results``, the Google Drive folder the Colab notebooks write to.

The outputs of the study are under ``<root>/level4/`` (``adapters/``, ``scores/``, ``bench/``, ``pilot/``,
``analysis/``). The file formats are described in ``docs/results_format.md``.
"""
import argparse
import os
import sys

COLAB_RESULTS = '/content/drive/MyDrive/dynalora_results'
ENV_VAR = 'DYNALORA_RESULTS'
HELP = f'results root (default: ${ENV_VAR} if set, else {COLAB_RESULTS})'


def default_results():
    """The results root when no ``--results`` argument is given."""
    return os.environ.get(ENV_VAR) or COLAB_RESULTS


def add_results_arg(ap):
    """Adds ``--results`` to an ``argparse`` parser."""
    ap.add_argument('--results', default=default_results(), help=HELP)
    return ap


def check_root(root):
    if not os.path.isdir(root):
        raise SystemExit(f'results root {root} not found: pass --results DIR or set {ENV_VAR} '
                         '(see docs/results_format.md)')
    return root


def results_root(*subdirs, argv=None):
    """Results root for scripts without their own parser: reads ``--results`` from the command line (other
    arguments are left alone), checks that the root exists and creates the given subdirectories, e.g.
    ``results_root('analysis')`` for a script that writes to ``<root>/analysis``."""
    ap = add_results_arg(argparse.ArgumentParser(add_help=False))
    a, _ = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    root = check_root(a.results)
    for d in subdirs:
        os.makedirs(os.path.join(root, d), exist_ok=True)
    return root
