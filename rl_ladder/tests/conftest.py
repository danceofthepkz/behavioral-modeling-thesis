import os
from pathlib import Path

import numpy as np
import pytest

from rl_ladder.data import Trials

DATA = os.environ.get("RL_LADDER_DATA")
HAVE_DATA = bool(DATA) and (Path(DATA) / "GS027.npz").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="RL_LADDER_DATA with the mouse .npz files not set")


def synthetic_trials(n=400, n_sessions=4, seed=0, ct_rate=0.2):
    rng = np.random.default_rng(seed)
    stim = rng.integers(0, 2, n)
    choice = np.where(rng.random(n) < 0.7, stim, 1 - stim)
    session = np.repeat(np.arange(1, n_sessions + 1), int(np.ceil(n / n_sessions)))[:n]
    is_ct = rng.random(n) < ct_rate
    return Trials("synthetic", stim, choice, (choice == stim).astype(np.int64), session, is_ct)


@pytest.fixture
def trials():
    return synthetic_trials()
