"""Shared services: registry context, seeds, JSON I/O, and the cached fit.

Caching follows the PIDSMaker idea of hashing a stage's configuration plus
everything upstream of it. A fit is stored under a key built from: the model
definition, the mouse and its data-file checksum, the data/preprocess/model/
fit/evaluate settings, and a checksum of the scientific source code. Any
change to one of those produces a new key, so stale results are never reused,
while unrelated changes (e.g. ppc settings) reuse existing fits.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .config import load_datasets_yaml, load_models_yaml, results_root, stable_hash
from .data import load_mouse
from .evaluate import evaluate_fit
from .fit.mle import fit_mle
from .models import params as P
from .models.registry import build_registry, parents, reset_flags
from .preprocess import preprocess

PACKAGE_DIR = Path(__file__).resolve().parent
SCIENCE_SOURCES = ("models", "fit", "data.py", "preprocess.py", "evaluate.py", "validate.py",
                   "ppc.py")


def code_version():
    h = hashlib.sha256()
    for item in SCIENCE_SOURCES:
        path = PACKAGE_DIR / item
        files = sorted(path.glob("*.py")) if path.is_dir() else [path]
        for f in files:
            h.update(str(f.relative_to(PACKAGE_DIR)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


class Context:
    def __init__(self):
        self.models, self.sets = build_registry(load_models_yaml())
        self.datasets = load_datasets_yaml()
        self.code_version = code_version()


_CTX = None


def context():
    global _CTX
    if _CTX is None:
        _CTX = Context()
    return _CTX


def seed_for(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def fit_key(ctx, spec, mouse, data_sha, cfg):
    return stable_hash({
        "model": spec.to_dict(), "mouse": mouse, "data_sha": data_sha, "data": cfg["data"],
        "preprocess": cfg["preprocess"], "model_cfg": cfg["model"], "fit": cfg["fit"],
        "evaluate": cfg["evaluate"], "code": ctx.code_version,
    }, n=16)


def warm_start_from(spec, parent_params):
    """Parent optimum with the extra module at its neutral (switched-off) value."""
    return {name: parent_params.get(name, P.BY_NAME[name].neutral) for name in spec.free}


def get_or_fit(ctx, model_name, mouse, cfg):
    """Fit one model to one mouse, or return the cached fit with the same key."""
    spec = ctx.models[model_name]
    raw = load_mouse(mouse, cfg)
    key = fit_key(ctx, spec, mouse, raw.meta["sha256"], cfg)
    path = results_root(cfg) / "cache" / "fits" / spec.slug / mouse / f"{key}.json"
    if path.exists():
        return read_json(path)

    trials, lik = preprocess(raw, cfg)
    reset = reset_flags(cfg["model"]["session_reset"])
    warm, parent_nll = [], {}
    if cfg["fit"]["warm_start_from_nested"]:
        for parent in parents(spec, ctx.models):
            fit = get_or_fit(ctx, parent, mouse, cfg)
            warm.append(warm_start_from(spec, fit["params"]))
            parent_nll[parent] = fit["nll"]
    seed = seed_for(cfg["fit"]["seed"], model_name, mouse)
    result = fit_mle(spec, trials, lik, reset, cfg["fit"], seed, warm)
    result.update(evaluate_fit(spec, trials, lik, reset, result, cfg, seed))
    result.update({
        "model": model_name, "mouse": mouse, "key": key, "spec": spec.to_dict(),
        "data_file": raw.meta["file"], "variant_used": raw.meta["variant_used"],
        "ct_source": raw.ct_source, "n_ct": int(trials.is_ct.sum()),
        "parent_nll": parent_nll, "code_version": ctx.code_version,
    })
    atomic_write_json(path, result)
    return result
