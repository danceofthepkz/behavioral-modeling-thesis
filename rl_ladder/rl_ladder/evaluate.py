"""Evaluate stage: information criteria and optional chronological CV.

BIC = 2 NLL + k ln(n_lik), where n_lik counts the trials in the likelihood
(correction trials are excluded under the default `mask` policy). BIC is a
descriptive approximation, unreliable at parameter boundaries; check `at_bound`.

Chronological CV (evaluate.cv.method = chrono) fits on the first `train_frac`
of sessions and scores the remaining sessions. Learning continues through the
test block from the observed choices, so this measures conditional
next-choice prediction, not an open-loop forecast.
"""

import numpy as np

from .fit.mle import fit_mle
from .models import kernel


def info_criteria(nll, k, n):
    return {"bic": float(2 * nll + k * np.log(n)), "aic": float(2 * nll + 2 * k),
            "nll_per_trial": float(nll / n)}


def chrono_cv(spec, trials, lik, reset, fit_cfg, seed, train_frac):
    sessions = np.unique(trials.session)
    n_train = int(round(train_frac * len(sessions)))
    if n_train < 1 or n_train >= len(sessions):
        return {"cv_error": f"cannot split {len(sessions)} sessions at train_frac={train_frac}"}
    in_train = np.isin(trials.session, sessions[:n_train])
    train_lik = lik * in_train
    test_lik = lik * ~in_train
    res = fit_mle(spec, trials, train_lik, reset, fit_cfg, seed)
    test_nll = kernel.nll(spec, np.array(res["theta"]), trials, test_lik, reset)
    n_test = float(test_lik.sum())
    return {"cv_train_sessions": n_train, "cv_test_nll": float(test_nll), "cv_n_test": n_test,
            "cv_nll_per_trial": float(test_nll / n_test) if n_test else float("nan")}


def evaluate_fit(spec, trials, lik, reset, result, cfg, seed):
    out = info_criteria(result["nll"], result["k"], result["n_lik"])
    cv = cfg["evaluate"]["cv"]
    if cv["method"] == "chrono":
        out.update(chrono_cv(spec, trials, lik, reset, cfg["fit"], seed + 7, cv["train_frac"]))
    elif cv["method"] != "none":
        raise ValueError(f"evaluate.cv.method must be none | chrono, not {cv['method']!r}")
    return out
