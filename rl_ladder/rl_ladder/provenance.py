"""Small, JSON-safe provenance records for archived numerical results."""

import hashlib
import io
import os
import platform
import sys
from contextlib import redirect_stdout
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np


@lru_cache(maxsize=1)
def environment_record():
    packages = {}
    for package in ("numpy", "scipy", "numba", "llvmlite", "pandas", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": packages, "numpy_build_config": buf.getvalue(),
            "thread_environment": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def trial_signature(trials, lik):
    """Hash processed arrays and weights, including trials excluded from scoring.

    Masked choices still affect latent states. Use explicit little-endian dtypes
    so the digest does not depend on the machine's native array representation.
    """
    h = hashlib.sha256()
    for name in ("stim", "choice", "reward", "session", "is_ct"):
        h.update(name.encode())
        h.update(np.asarray(getattr(trials, name), dtype="<i8").tobytes())
    h.update(b"likelihood_weights")
    h.update(np.asarray(lik, dtype="<f8").tobytes())
    return h.hexdigest()
