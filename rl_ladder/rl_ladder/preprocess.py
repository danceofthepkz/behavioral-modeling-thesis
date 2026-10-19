"""Preprocess stage: training window and correction-trial policy.

Returns the trials the model walks through and a likelihood weight per trial.

correction:
    mask     states update on correction trials, but they are left out of the
             likelihood (main analysis: their stimulus is predictable from the
             previous error, which would otherwise reward a lose-switch rule)
    include  every trial counts (the original paper's fits)
    drop     correction trials are removed entirely
"""

import numpy as np

from .data import Trials


def apply_window(trials: Trials, window):
    method = window.get("method", "all")
    sessions = np.unique(trials.session)
    if method == "all":
        return trials
    n = int(window.get("n", 3))
    if method == "first_sessions":
        keep = sessions[:n]
    elif method == "last_sessions":
        keep = sessions[-n:]
    else:
        raise ValueError(f"preprocess.window.method must be all | first_sessions | last_sessions, "
                         f"not {method!r}")
    return trials.subset(np.isin(trials.session, keep))


def likelihood_weights(trials: Trials, correction):
    if correction == "mask":
        return (~trials.is_ct).astype(np.float64)
    if correction in ("include", "drop"):
        return np.ones(trials.n)
    raise ValueError(f"preprocess.correction must be mask | include | drop, not {correction!r}")


def preprocess(trials: Trials, cfg):
    pre = cfg["preprocess"]
    out = apply_window(trials, pre["window"])
    if pre["correction"] == "drop":
        out = out.subset(~out.is_ct)
    return out, likelihood_weights(out, pre["correction"])
