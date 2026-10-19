"""Group-level comparison: random-effects Bayesian model selection.

Stephan et al. (2009) variational BMS with the protected exceedance
probability of Rigoux et al. (2014). Log model evidence is approximated by
-BIC/2. Family-level BMS (Penny et al., 2010) pools models that share a
slot variant, e.g. "choice = ck" vs "choice = none".
"""

import numpy as np
from scipy.special import digamma, gammaln, logsumexp


def vb_bms(log_evidence, alpha0=1.0, tol=1e-8, max_iter=10000, n_samples=200_000, seed=0):
    """log_evidence: (n_subjects, n_models). Returns a dict of group statistics."""
    L = np.asarray(log_evidence, dtype=float)
    n, K = L.shape
    a0 = np.full(K, float(alpha0))
    a = a0.copy()
    for _ in range(max_iter):
        log_u = L + digamma(a) - digamma(a.sum())
        g = np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))
        a_new = a0 + g.sum(axis=0)
        if np.max(np.abs(a_new - a)) < tol:
            a = a_new
            break
        a = a_new
    log_u = L + digamma(a) - digamma(a.sum())
    g = np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))
    elog_r = digamma(a) - digamma(a.sum())

    # Free energy of the random-effects model (F1) and of the null in which all
    # models are equally frequent (F0), for the Bayesian omnibus risk.
    with np.errstate(divide="ignore", invalid="ignore"):
        g_log_g = np.where(g > 0, g * np.log(g), 0.0)
    f1 = (np.sum(g * (L + elog_r)) - np.sum(g_log_g)
          + gammaln(a0.sum()) - gammaln(a0).sum()
          - gammaln(a.sum()) + gammaln(a).sum()
          + np.sum((a0 - a) * elog_r))
    f0 = np.sum(logsumexp(L, axis=1) - np.log(K))
    bor = float(1.0 / (1.0 + np.exp(np.clip(f1 - f0, -700, 700))))

    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(a, size=n_samples)
    xp = np.bincount(draws.argmax(axis=1), minlength=K) / n_samples
    pxp = xp * (1 - bor) + bor / K
    return {"alpha": a, "expected_freq": a / a.sum(), "xp": xp, "pxp": pxp, "bor": bor,
            "posterior": g}


def family_log_evidence(log_evidence, families):
    """families: list of column-index lists. Uniform prior within each family."""
    L = np.asarray(log_evidence, dtype=float)
    return np.column_stack([logsumexp(L[:, idx], axis=1) - np.log(len(idx)) for idx in families])
