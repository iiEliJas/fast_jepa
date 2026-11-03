"""Access to the pinned official LeJEPA checkout.

The official package and this repository both use the top-level name
``lejepa``, so importing both in one interpreter is ambiguous.  Official code
therefore runs in a fresh process with its checkout first on ``sys.path``; see
``tests/test_parity.py`` and ``bench/harness.load_official_sigreg``.

This module deliberately does not import ``lejepa`` at module scope: with this
repository installed, that import would resolve to our own package and report
a false positive.
"""

import os

OFFICIAL_ROOT = os.path.abspath(
    os.environ.get("LEJEPA_OFFICIAL_ROOT", os.path.expanduser("~/projects/lejepa"))
)
OFFICIAL_COMMIT = "c293d291ca87cd4fddee9d3fffe4e914c7272052"


def has_official(root=None):
    root = os.path.abspath(os.path.expanduser(root or OFFICIAL_ROOT))
    return os.path.isdir(os.path.join(root, "lejepa"))


def official_sigreg(*args, **kwargs):
    """Placeholder that points at the isolated subprocess path.

    Use ``bench/harness.load_official_sigreg`` for benchmarks or the parity
    tests for numerical comparison.  Calling the official module directly from
    this interpreter would risk importing our ``lejepa`` package instead.
    """
    raise RuntimeError(
        "official LeJEPA must run in a fresh process with its checkout on "
        f"PYTHONPATH (expected at {OFFICIAL_ROOT}); see "
        "bench/harness.load_official_sigreg"
    )
