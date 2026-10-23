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
from .provenance import environment_record, trial_signature


def info_criteria(nll, k, n):
    return {"bic": float(2 * nll + k * np.log(n)), "aic": float(2 * nll + 2 * k),
            "nll_per_trial": float(nll / n)}


def chronological_split(trials, lik, train_frac):
    """Return a validated session split before an optimizer is called."""
    if not np.isfinite(train_frac) or not 0 < train_frac < 1:
        raise ValueError("CV train_frac must lie strictly between 0 and 1")
    sessions = np.unique(trials.session)
    n_train = int(round(train_frac * len(sessions)))
    if n_train < 1 or n_train >= len(sessions):
        raise ValueError(f"cannot split {len(sessions)} sessions at train_frac={train_frac}")
    in_train = np.isin(trials.session, sessions[:n_train])
    if (lik * in_train).sum() <= 0 or (lik * ~in_train).sum() <= 0:
        raise ValueError("CV train and test blocks must both contain scored trials")
    return sessions, n_train, in_train


def verify_cv_record(spec, trials, lik, reset, result, atol=1e-8):
    """Reevaluate archived training/test parameters without any optimization.

    Convergence is reported separately: an exactly reproducible score need not
    come from a successful optimizer run, much less a global optimum.
    """
    r = result.get("cv_record")
    if not r or r.get("schema_version") != 1:
        raise ValueError("CV training record unavailable; legacy scores cannot be reconstructed")
    if r["spec"] != spec.to_dict():
        raise ValueError("CV model specification differs from the archived record")
    if r["input_sha256"] != trial_signature(trials, lik):
        raise ValueError("CV processed data or likelihood weights differ from the archived record")
    if list(map(int, reset)) != r["reset_flags"]:
        raise ValueError("CV reset policy differs from the archived record")
    env = r["environment"]
    if not env.get("python") or not env.get("numpy_build_config") or not all(
            env.get("packages", {}).get(p) for p in ("numpy", "scipy", "numba")):
        raise ValueError("CV numerical environment record is incomplete")
    sessions, n_train, train = chronological_split(trials, lik, r["train_frac"])
    split = r["split"]
    expected = {"train_sessions": sessions[:n_train].tolist(), "test_sessions": sessions[n_train:].tolist(),
                "n_train_trials": int(train.sum()), "n_test_trials": int((~train).sum()),
                "n_train_scored": float((lik*train).sum()), "n_test_scored": float((lik*~train).sum())}
    if split != expected or result["cv_train_sessions"] != n_train:
        raise ValueError("CV session split differs from the archived record")
    fit = r["training_fit"]
    if fit["seed"] != r["seed"] or fit["n_lik"] != expected["n_train_scored"]:
        raise ValueError("CV training seed or scored-trial count is inconsistent")
    if result["cv_success"] != fit["success"]:
        raise ValueError("CV convergence flags are inconsistent")
    theta = np.asarray(fit["theta"], dtype=float)
    if not np.array_equal(theta, spec.theta(fit["params"])):
        raise ValueError("CV theta and named training parameters disagree")
    train_nll = kernel.nll(spec, theta, trials, lik*train, reset)
    test_nll = kernel.nll(spec, theta, trials, lik*~train, reset)
    checks = [(train_nll, fit["nll"]), (test_nll, result["cv_test_nll"]),
              (float((lik*~train).sum()), result["cv_n_test"]),
              (test_nll/(lik*~train).sum(), result["cv_nll_per_trial"])]
    if not all(np.isfinite(a) and np.isfinite(b) and abs(a-b) <= atol for a, b in checks):
        raise ValueError("CV saved score does not match the archived training parameters")
    starts = fit["start_records"]
    selected = fit["selected_start_index"]
    if not (len(starts) == fit["n_starts"] and 0 <= selected < len(starts)
            and sum(bool(s["selected"]) for s in starts) == 1
            and starts[selected]["selected"]
            and np.array_equal(starts[selected]["theta"], theta)
            and abs(starts[selected]["nll"]-train_nll) <= atol
            and starts[selected]["success"] == fit["success"]):
        raise ValueError("CV optimizer start records are incomplete or inconsistent")
    if fit["n_failed"] != sum(not s["success"] for s in starts):
        raise ValueError("CV failed-start count disagrees with optimizer records")
    parent_nll = r.get("training_parent_nll", {})
    warm = [s for s in starts if s["kind"] == "warm"]
    if len(warm) != len(parent_nll):
        raise ValueError("CV training-parent start records are incomplete")
    indices = r.get("training_parent_start_indices", {})
    if set(indices) != set(parent_nll) or set(indices.values()) != {s["index"] for s in warm}:
        raise ValueError("CV training-parent start indices are inconsistent")
    for parent, parent_score in parent_nll.items():
        start = starts[indices[parent]]
        feasible = kernel.nll(spec, np.asarray(start["initial_theta"]), trials, lik*train, reset)
        if abs(feasible-parent_score) > atol:
            raise ValueError("CV training-parent start does not reproduce its parent's score")
    if parent_nll and train_nll > min(parent_nll.values()) + 1e-4:
        raise ValueError("CV training optimum is worse than a feasible nested-parent solution")
    return {"passed": True, "training_converged": bool(fit["success"]),
            "train_nll": float(train_nll), "test_nll": float(test_nll),
            "train_nll_abs_error": float(abs(train_nll-fit["nll"])),
            "test_nll_abs_error": float(abs(test_nll-result["cv_test_nll"])), "atol": atol}


def chrono_cv(spec, trials, lik, reset, fit_cfg, seed, train_frac, *,
              warm_starts=(), training_parent_nll=None):
    sessions, n_train, in_train = chronological_split(trials, lik, train_frac)
    train_lik = lik * in_train
    test_lik = lik * ~in_train
    # Never warm-start from full-data optima: they contain held-out information.
    res = fit_mle(spec, trials, train_lik, reset, fit_cfg, seed,
                  warm_starts=warm_starts, record_starts=True)
    test_nll = kernel.nll(spec, np.array(res["theta"]), trials, test_lik, reset)
    n_test = float(test_lik.sum())
    out = {"cv_train_sessions": n_train, "cv_test_nll": float(test_nll), "cv_n_test": n_test,
           "cv_nll_per_trial": float(test_nll/n_test), "cv_success": bool(res["success"]),
           "cv_record": {"schema_version": 1, "method": "chrono", "seed": int(seed),
               "train_frac": float(train_frac), "spec": spec.to_dict(),
               "input_sha256": trial_signature(trials, lik),
               "reset_flags": list(map(int, reset)), "fit_config": dict(fit_cfg),
               "warm_starts_used": bool(warm_starts),
               "training_parent_nll": dict(training_parent_nll or {}),
               "training_parent_start_indices": {name: i for i, name in enumerate(training_parent_nll or {})},
               "prediction": "conditional next-choice; observed-history updates in test block",
               "split": {"train_sessions": sessions[:n_train].tolist(),
                   "test_sessions": sessions[n_train:].tolist(),
                   "n_train_trials": int(in_train.sum()), "n_test_trials": int((~in_train).sum()),
                   "n_train_scored": float(train_lik.sum()), "n_test_scored": n_test},
               "training_fit": res, "environment": environment_record()}}
    out["cv_verification"] = verify_cv_record(spec, trials, lik, reset, out)
    return out


def evaluate_fit(spec, trials, lik, reset, result, cfg, seed, *, cv_result=None):
    out = info_criteria(result["nll"], result["k"], result["n_lik"])
    cv = cfg["evaluate"]["cv"]
    if cv["method"] == "chrono":
        if cv_result is None:
            if cv.get("warm_start_from_nested", False):
                raise ValueError("Nested CV initialization requires training-parent fits from service.get_cv")
            cv_result = chrono_cv(spec, trials, lik, reset, cfg["fit"], seed + 7, cv["train_frac"])
        out.update({k: v for k, v in cv_result.items() if k.startswith("cv_")})
    elif cv["method"] != "none":
        raise ValueError(f"evaluate.cv.method must be none | chrono, not {cv['method']!r}")
    return out
