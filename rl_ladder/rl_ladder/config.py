"""Configuration: YAML files, `_include` inheritance, dotted overrides, hashing.

Precedence (later wins):
    config/default.yml  <  dataset preset  <  experiment `overrides`  <  grid point  <  CLI
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
_MISSING = object()


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path):
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    include = data.pop("_include", None)
    if include:
        merged = {}
        for name in [include] if isinstance(include, str) else include:
            target = path.parent / (name if name.endswith(".yml") else f"{name}.yml")
            merged = deep_merge(merged, load_yaml(target))
        data = deep_merge(merged, data)
    return data


def get_dotted(cfg, key, default=_MISSING):
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise KeyError(key)
            return default
        node = node[part]
    return node


def set_dotted(cfg, key, value):
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg, overrides, strict=True):
    """Return a copy of cfg with dotted overrides applied.

    With strict=True every key must already exist in cfg, so a typo such as
    --fit.nstarts=40 fails loudly instead of being silently ignored.
    """
    out = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        if strict and get_dotted(cfg, key, _MISSING) is _MISSING:
            raise KeyError(f"Unknown setting {key!r}. See config/default.yml for valid keys.")
        set_dotted(out, key, copy.deepcopy(value))
    return out


def parse_cli_overrides(tokens):
    """['--fit.n_starts=40', '--preprocess.correction', 'include'] -> dict."""
    out, i = {}, 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--") or "." not in tok.split("=")[0]:
            raise ValueError(f"Unrecognised argument {tok!r}; overrides look like --stage.key=value")
        if "=" in tok:
            key, raw = tok[2:].split("=", 1)
            i += 1
        else:
            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value for {tok}")
            key, raw = tok[2:], tokens[i + 1]
            i += 2
        out[key] = yaml.safe_load(raw)
    return out


def stable_hash(obj, n=12):
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def load_default():
    return load_yaml(CONFIG_DIR / "default.yml")


def load_models_yaml():
    return load_yaml(CONFIG_DIR / "models.yml")


def load_datasets_yaml():
    return load_yaml(CONFIG_DIR / "datasets.yml")


def experiment_path(name):
    p = Path(name)
    if p.suffix in (".yml", ".yaml") and p.exists():
        return p
    p = CONFIG_DIR / "experiments" / f"{name}.yml"
    if not p.exists():
        available = sorted(x.stem for x in (CONFIG_DIR / "experiments").glob("*.yml"))
        raise FileNotFoundError(f"No experiment {name!r}. Available: {available}")
    return p


def list_experiments():
    return sorted(x.stem for x in (CONFIG_DIR / "experiments").glob("*.yml"))


def data_root(cfg):
    root = cfg["paths"].get("data") or os.environ.get("RL_LADDER_DATA")
    if not root:
        raise RuntimeError(
            "Data folder not set. Export RL_LADDER_DATA (folder containing GS027.npz ...), "
            "e.g. `source scripts/activate.sh`, or pass --paths.data=/path/to/data")
    root = Path(os.path.expanduser(root)).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Data folder {root} does not exist")
    return root


def results_root(cfg):
    root = (cfg["paths"].get("results") or os.environ.get("RL_LADDER_RESULTS")
            or PROJECT_ROOT / "results")
    return Path(os.path.expanduser(str(root))).resolve()


def reference_root(cfg):
    root = cfg["paths"].get("reference") or os.environ.get("RL_LADDER_REFERENCE")
    if root:
        return Path(os.path.expanduser(root)).resolve()
    try:
        return data_root(cfg).parent / "validation"
    except RuntimeError:
        return None
