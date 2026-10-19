"""Data stage: prepared .npz files -> a standard trial table.

Input: the curated per-mouse files produced by thesis/rl_firebird_v2/prepare_data.py
(one <MOUSE>.npz per mouse, plus paper/<MOUSE>.npz for the two mice whose
published RL fits used a longer window). Fields used here:

    choice      0 = Low, 1 = High
    answer      correct category (the stimulus), 0 = Low, 1 = High
    reward      1 if choice == answer (verified)
    session     session number, chronological
    trial_type  original TT column 3. Correction trials are types 2 AND 4
                (both always follow an error and repeat the stimulus);
                -1 = unknown (JC039 and the paper versions of GS028/JC052)

Correction status for unknown mice can be inferred from the task rule
(previous trial in the same session unrewarded and same stimulus). On mice
with known flags this rule finds every flagged trial and adds about 1-2%
false positives where a redraw happened to repeat the stimulus.
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import data_root

CORRECTION_TYPES = (2, 4)


@dataclass
class Trials:
    mouse: str
    stim: np.ndarray  # int64, 0/1
    choice: np.ndarray  # int64, 0/1
    reward: np.ndarray  # int64, 0/1
    session: np.ndarray  # int64
    is_ct: np.ndarray  # bool
    ct_source: str = "flag"  # flag | inferred | none | simulated
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.choice)

    @property
    def new_session(self):
        return np.r_[True, self.session[1:] != self.session[:-1]]

    def subset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        return Trials(self.mouse, self.stim[mask], self.choice[mask], self.reward[mask],
                      self.session[mask], self.is_ct[mask], self.ct_source, dict(self.meta))

    def validate(self):
        arrays = (self.stim, self.choice, self.reward, self.session, self.is_ct)
        if any(len(a) != self.n for a in arrays) or self.n == 0:
            raise ValueError(f"{self.mouse}: misaligned or empty trial arrays")
        for name, a in (("stim", self.stim), ("choice", self.choice), ("reward", self.reward)):
            if not np.isin(a, (0, 1)).all():
                raise ValueError(f"{self.mouse}: {name} must be 0/1")
        if not np.array_equal(self.reward, (self.choice == self.stim).astype(self.reward.dtype)):
            raise ValueError(f"{self.mouse}: reward does not equal choice == stimulus")
        if np.any(np.diff(self.session) < 0):
            raise ValueError(f"{self.mouse}: sessions are not chronological")
        return self


def infer_correction(stim, reward, session):
    same_session = np.r_[False, session[1:] == session[:-1]]
    prev_loss = np.r_[False, reward[:-1] == 0]
    same_stim = np.r_[False, stim[1:] == stim[:-1]]
    return same_session & prev_loss & same_stim


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mouse_path(root, mouse, variant):
    if variant not in ("paper", "current"):
        raise ValueError(f"data.variant must be 'paper' or 'current', not {variant!r}")
    paper = Path(root) / "paper" / f"{mouse}.npz"
    path = paper if variant == "paper" and paper.exists() else Path(root) / f"{mouse}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No data for {mouse}: expected {path}")
    return path


@lru_cache(maxsize=64)
def _load(path_str, unknown_ct):
    with np.load(path_str, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    stim = d["answer"].astype(np.int64)
    choice = d["choice"].astype(np.int64)
    reward = d["reward"].astype(np.int64)
    session = d["session"].astype(np.int64)
    tt = d["trial_type"].astype(np.int64)
    if np.all(tt >= 0):
        is_ct, source = np.isin(tt, CORRECTION_TYPES), "flag"
    elif unknown_ct == "infer":
        is_ct, source = infer_correction(stim, reward, session), "inferred"
    elif unknown_ct == "none":
        is_ct, source = np.zeros(len(stim), dtype=bool), "none"
    else:
        raise ValueError(f"data.unknown_ct must be 'infer' or 'none', not {unknown_ct!r}")
    return stim, choice, reward, session, is_ct, source


def load_mouse(mouse, cfg):
    path = mouse_path(data_root(cfg), mouse, cfg["data"]["variant"])
    stim, choice, reward, session, is_ct, source = _load(str(path), cfg["data"]["unknown_ct"])
    trials = Trials(mouse, stim.copy(), choice.copy(), reward.copy(), session.copy(),
                    is_ct.copy(), source,
                    {"file": path.name, "variant_used": "paper" if path.parent.name == "paper"
                     else "current", "sha256": file_sha256(path)})
    return trials.validate()


def resolve_mice(dataset, datasets_yaml):
    """Dataset preset name, a single mouse ID, or a comma list -> (mice, preset overrides)."""
    presets = datasets_yaml.get("datasets", {})
    groups = datasets_yaml.get("mice", {})
    if dataset in presets:
        preset = presets[dataset]
        mice = []
        for item in ([preset["mice"]] if isinstance(preset["mice"], str) else preset["mice"]):
            mice.extend(groups.get(item, [item]))
        overrides = {"data.variant": preset["variant"]} if "variant" in preset else {}
        return list(dict.fromkeys(mice)), overrides
    mice = [m.strip() for m in str(dataset).split(",") if m.strip()]
    expanded = []
    for m in mice:
        expanded.extend(groups.get(m, [m]))
    return list(dict.fromkeys(expanded)), {}
