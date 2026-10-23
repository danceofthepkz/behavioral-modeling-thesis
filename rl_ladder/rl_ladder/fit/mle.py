"""Fit stage: multistart maximum likelihood.

The optimiser works in "z-space": log10 for rates and positive temperatures
(their plausible values span orders of magnitude, e.g. alpha ~ 1e-4 in the
mouse data), linear elsewhere. L-BFGS-B handles the box bounds directly, so
boundary values such as gamma = 0 or beta_c = 0 are reachable exactly and
nested models stay nested.

Starts = optional warm starts (e.g. a nested parent's optimum with the new
module switched off) + `n_starts` random draws from each parameter's prior.
"""

import numpy as np
from scipy.optimize import minimize

from ..models import params as P
from ..models import kernel


def _z_vector(spec, values):
    return np.array([P.to_z(p, values[p.name]) for p in spec.free_specs])


def fit_mle(spec, trials, lik, reset, fit_cfg, seed, warm_starts=(), *, record_starts=False):
    method = fit_cfg.get("method", "mle_multistart")
    if method != "mle_multistart":
        raise NotImplementedError(
            f"fit.method={method!r} is planned but not implemented yet; use mle_multistart")
    rng = np.random.default_rng(seed)
    free = spec.free_specs
    idx = np.array([p.index for p in free], dtype=np.int64)
    log_mask = np.array([p.scale == "log" for p in free])
    bounds = [P.z_bounds(p) for p in free]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    template = spec.template()
    stim, choice, new_session = kernel._arrays(trials)
    lik = np.ascontiguousarray(lik, dtype=np.float64)
    reset = np.ascontiguousarray(reset, dtype=np.int64)

    def theta_of(z):
        theta = template.copy()
        theta[idx] = np.where(log_mask, 10.0 ** z, z)
        return theta

    def objective(z):
        value = kernel.run_kernel(theta_of(z), spec.base, reset, stim, choice, new_session, lik,
                                  0, kernel._EMPTY_F, kernel._EMPTY_F, 0, kernel._EMPTY_F,
                                  kernel._EMPTY_I, kernel._EMPTY_I, kernel._EMPTY_I,
                                  kernel._EMPTY_TRACE)
        return value if np.isfinite(value) else 1e12

    starts = [("warm", np.clip(_z_vector(spec, w), lo, hi)) for w in warm_starts]
    for _ in range(int(fit_cfg["n_starts"])):
        values = {p.name: P.sample_prior(p, spec.stimulus, rng) for p in free}
        starts.append(("random", np.clip(_z_vector(spec, values), lo, hi)))
    if not starts:
        raise ValueError("fit.n_starts must be >= 1 when there are no warm starts")

    runs = []
    for kind, z0 in starts:
        if len(free) == 0:
            runs.append({"kind": kind, "z0": z0, "z": z0, "fun": objective(z0), "success": True,
                         "nit": 0, "message": "no free parameters"})
            continue
        res = minimize(objective, z0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": int(fit_cfg["maxiter"])})
        # Re-evaluate: never trust a stale res.fun after an abnormal line search.
        runs.append({"kind": kind, "z0": z0, "z": res.x, "fun": float(objective(res.x)),
                     "success": bool(res.success), "nit": int(res.nit),
                     "message": str(res.message), "status": int(getattr(res, "status", -1)),
                     "nfev": int(getattr(res, "nfev", 0))})

    best_candidate = min(runs, key=lambda r: r["fun"])
    # At an inactive module (e.g. beta_f = 0), a failed warm-start line search
    # can differ from a successful fit only at floating-point precision.
    # Prefer actual convergence within an absolute total-NLL tolerance;
    # never hide a materially better failed candidate behind a worse success.
    converged = [r for r in runs if r["success"] and r["fun"] <= best_candidate["fun"] + 1e-7]
    best = min(converged, key=lambda r: r["fun"]) if converged else best_candidate
    theta = theta_of(best["z"])
    tol = float(fit_cfg.get("repeat_tol", 0.01))
    span = hi - lo
    at_bound = [p.name for p, z, l, h, s in zip(free, best["z"], lo, hi, span)
                if min(z - l, h - z) <= 1e-6 * max(s, 1.0)]
    result = {
        "params": {p.name: float(theta[p.index]) for p in free},
        "theta": theta.tolist(),
        "nll": float(best["fun"]),
        "best_nll_all_starts": float(best_candidate["fun"]),
        "convergence_nll_gap": float(best["fun"] - best_candidate["fun"]),
        "k": spec.k,
        "n_trials": int(trials.n),
        "n_lik": float(lik.sum()),
        "success": best["success"],
        "message": best["message"],
        "best_start": best["kind"],
        "n_starts": len(runs),
        "n_repeated": int(sum(r["fun"] <= best["fun"] + tol for r in runs)),
        "n_failed": int(sum(not r["success"] for r in runs)),
        "at_bound": at_bound,
        "start_nlls": sorted(round(r["fun"], 4) for r in runs),
    }
    if record_starts:
        result.update({"seed": int(seed), "optimizer": "scipy.optimize.minimize/L-BFGS-B",
                       "selected_start_index": next(i for i, r in enumerate(runs) if r is best),
                       "selection": "minimum training NLL; converged ties within 1e-7 preferred",
                       "start_records": [
                           {"index": i, "kind": r["kind"], "selected": r is best,
                            "initial_theta": theta_of(r["z0"]).tolist(),
                            "theta": theta_of(r["z"]).tolist(), "nll": float(r["fun"]),
                            "success": bool(r["success"]), "nit": int(r["nit"]),
                            "message": r["message"], "status": r.get("status", 0),
                            "nfev": r.get("nfev", 0)} for i, r in enumerate(runs)]})
    return result
