"""Gate: the kernel reproduces the published A0/B0 likelihoods and the thesis
M1/M2 baselines on the real data (skipped when the data are not available)."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from rl_ladder.config import apply_overrides, load_default
from rl_ladder.data import load_mouse
from rl_ladder.models import kernel
from rl_ladder.models.registry import reset_flags
from rl_ladder.preprocess import preprocess
from rl_ladder.service import context

REF = Path(os.environ.get("RL_LADDER_REFERENCE") or
           Path(os.environ.get("RL_LADDER_DATA", ".")).parent / "validation")


@needs_data
def test_published_a0_and_choice_history_likelihoods():
    path = REF / "reference_legacy.json"
    if not path.exists():
        pytest.skip(f"{path} not found")
    cfg = apply_overrides(load_default(), {"preprocess.correction": "include",
                                           "model.session_reset": [], "data.variant": "paper"})
    targets = json.loads(path.read_text())
    assert len(targets) == 38
    for t in targets:
        name = "A0" if "3param" in t["model"] else "B0"
        spec = context().models[name]
        values = {"alpha": t["alpha"], "b": t["bias"], "q0": t["init_q"]}
        if name == "B0":
            values["gamma"] = float(np.exp(-t["beta"])) if t["beta"] < 100 else 0.0
        trials, lik = preprocess(load_mouse(t["mouse"], cfg), cfg)
        nll = kernel.nll(spec, spec.theta(values), trials, lik, reset_flags([]))
        assert abs(nll - t["nll"]) < 1e-6, t


@needs_data
def test_thesis_m1_m2_baselines():
    path = REF / "reference_m1_m2.csv"
    if not path.exists():
        pytest.skip(f"{path} not found")
    cfg = apply_overrides(load_default(), {"preprocess.correction": "include",
                                           "model.session_reset": ["wsls"],
                                           "data.variant": "current"})
    for row in pd.read_csv(path).itertuples():
        trials, lik = preprocess(load_mouse(row.mouse, cfg), cfg)
        m1, m2 = context().models["M1"], context().models["M2"]
        b = float(np.log(row.b / (1 - row.b)))
        assert abs(kernel.nll(m1, m1.theta({"b": b}), trials, lik, reset_flags(["wsls"]))
                   - row.m1_nll) < 1e-6
        assert abs(kernel.nll(m2, m2.theta({"eps": row.epsilon}), trials, lik,
                              reset_flags(["wsls"])) - row.m2_nll) < 1e-6
