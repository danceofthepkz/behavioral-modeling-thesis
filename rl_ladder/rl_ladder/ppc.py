"""PPC stage: simulate from fitted parameters and compare with the real mouse.

Analyses
    curves         per-session accuracy on High and Low trials, real vs simulated
    bias_feedback  regression slope of the next block's bias change on the
                   current bias. Reward-dependent drift (V, gamma) pulls a bias
                   back (negative feedback); a choice kernel amplifies it
                   (positive feedback). Regression to the mean makes the slope
                   negative even for noise, so compare the real slope with each
                   model's simulated distribution, never with zero.
    dtw            planned: rerun the paper's DTW clustering on simulated
                   trajectories (needs tslearn and the paper's trajectory
                   transform); not implemented yet
"""

import numpy as np

from .models.registry import reset_flags
from .service import get_or_fit
from .validate import simulate_trials, template_trials

ANALYSES = ("curves", "bias_feedback", "dtw")


def session_curves(trials):
    rows = []
    for s in np.unique(trials.session):
        m = (trials.session == s) & ~trials.is_ct
        hi, lo = m & (trials.stim == 1), m & (trials.stim == 0)
        rows.append([float(trials.choice[hi].mean()) if hi.any() else np.nan,
                     float(1 - trials.choice[lo].mean()) if lo.any() else np.nan])
    return np.array(rows)  # (n_sessions, 2): accuracy on High, accuracy on Low


def block_bias(trials, block):
    """Stimulus-balanced bias per block: mean of P(High|High) and P(High|Low), minus 0.5."""
    out = []
    for start in range(0, trials.n - block + 1, block):
        sl = slice(start, start + block)
        c, s = trials.choice[sl], trials.stim[sl]
        if (s == 1).any() and (s == 0).any():
            out.append(0.5 * (c[s == 1].mean() + c[s == 0].mean()) - 0.5)
        else:
            out.append(np.nan)
    return np.array(out)


def feedback_slope(bias):
    x, dy = bias[:-1], np.diff(bias)
    ok = np.isfinite(x) & np.isfinite(dy)
    if ok.sum() < 3 or np.var(x[ok]) == 0:
        return float("nan")
    return float(np.polyfit(x[ok], dy[ok], 1)[0])


def run_ppc(ctx, unit):
    cfg = unit["cfg"]
    pcfg = cfg["ppc"]
    unknown = set(pcfg["analyses"]) - set(ANALYSES)
    if unknown:
        raise ValueError(f"ppc.analyses: unknown {sorted(unknown)}; choose from {ANALYSES}")
    if "dtw" in pcfg["analyses"]:
        raise NotImplementedError("ppc analysis 'dtw' is planned but not implemented yet")
    spec = ctx.models[unit["model"]]
    fit = get_or_fit(ctx, spec.name, unit["mouse"], cfg)
    real = template_trials(unit["mouse"], cfg)
    reset = reset_flags(cfg["model"]["session_reset"])
    rng = np.random.default_rng(unit["seed"])
    sims = [simulate_trials(spec, fit["params"], real, pcfg["env"], pcfg["max_ct"], reset, rng)
            for _ in range(int(pcfg["n_sims"]))]

    def acc(t):
        return float(t.reward[~t.is_ct].mean())

    out = {"params": fit["params"], "bic": fit["bic"], "real_accuracy": acc(real),
           "sim_accuracy": [acc(t) for t in sims]}
    if "curves" in pcfg["analyses"]:
        out["real_curves"] = session_curves(real).tolist()
        sim_curves = np.array([session_curves(t) for t in sims])
        out["sim_curves_mean"] = np.nanmean(sim_curves, axis=0).tolist()
    if "bias_feedback" in pcfg["analyses"]:
        block = int(pcfg["block"])
        real_slope = feedback_slope(block_bias(real, block))
        sim_slopes = np.array([feedback_slope(block_bias(t, block)) for t in sims])
        finite = sim_slopes[np.isfinite(sim_slopes)]
        out.update({
            "real_slope": real_slope, "sim_slopes": sim_slopes.tolist(),
            # Fraction of simulations with a slope below the real one.
            "real_slope_percentile": float(np.mean(finite < real_slope)) if finite.size else None,
        })
    return out
