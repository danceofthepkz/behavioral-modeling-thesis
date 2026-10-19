"""Fixed parameter-vector layout shared by every model.

A model never owns its own parameter vector. Every model writes into the same
12-slot ``theta``; parameters a model does not use stay at their *neutral*
value, which switches the corresponding term off exactly (e.g. beta_c = 0
removes the choice kernel from the logit). This is what makes the ladder
nested by construction.
"""

from dataclasses import dataclass

import numpy as np

# Slot indices (also imported by the numba kernel as compile-time constants).
ALPHA, BETA_Q, B, Q0, GAMMA, ALPHA_V, BETA_V, ALPHA_C, BETA_C, EPS, TAU_A, KAPPA = range(12)
N_THETA = 12


@dataclass(frozen=True)
class ParamSpec:
    name: str
    index: int
    scale: str  # "log": optimise log10(x); "linear": optimise x
    lower: float
    upper: float
    neutral: float  # value when the owning module is off
    prior: tuple  # (distribution, low, high) for random starts and recovery
    prior_a: tuple | None  # override when the stimulus slot is the original A model
    about: str


PARAMS = (
    ParamSpec("alpha", ALPHA, "log", 1e-6, 1.0, 0.0,
              ("loguniform", 1e-4, 3e-2), ("loguniform", 3e-5, 3e-3),
              "stimulus-slot learning rate"),
    ParamSpec("beta_q", BETA_Q, "log", 1e-3, 100.0, 0.0,
              ("loguniform", 1.0, 20.0), None,
              "inverse temperature on Q(s,H) - Q(s,L) (S only)"),
    ParamSpec("b", B, "linear", -10.0, 10.0, 0.0,
              ("uniform", -1.0, 1.0), ("uniform", -0.1, 0.1),
              "choice bias toward High (logit units; paper units for A)"),
    ParamSpec("q0", Q0, "linear", -1.0, 1.0, 0.0,
              ("uniform", -0.5, 0.5), ("uniform", -0.3, 0.3),
              "initial value asymmetry"),
    ParamSpec("gamma", GAMMA, "linear", 0.0, 1.0, 0.0,
              ("uniform", 0.0, 1.0), None,
              "A only: unchosen Q -= gamma*alpha*delta (paper's exp(-beta))"),
    ParamSpec("alpha_v", ALPHA_V, "log", 1e-6, 1.0, 0.1,
              ("loguniform", 1e-3, 1e-1), None,
              "action-value learning rate"),
    ParamSpec("beta_v", BETA_V, "linear", 0.0, 50.0, 0.0,
              ("uniform", 0.0, 5.0), None,
              "weight on V(H) - V(L)"),
    ParamSpec("alpha_c", ALPHA_C, "log", 1e-6, 1.0, 0.1,
              ("loguniform", 1e-3, 3e-1), None,
              "choice-kernel decay rate"),
    ParamSpec("beta_c", BETA_C, "linear", -20.0, 20.0, 0.0,
              ("uniform", -2.0, 2.0), None,
              "choice-kernel weight (+ perseveration, - alternation)"),
    ParamSpec("eps", EPS, "linear", 1e-6, 1.0, 1.0,
              ("uniform", 0.1, 1.0), None,
              "WSLS noise: P(follow rule) = 1 - eps/2"),
    ParamSpec("tau_a", TAU_A, "linear", 0.1, 100.0, 5.0,
              ("loguniform", 1.0, 20.0), None,
              "A only: inverse temperature (fixed at 5 in the paper)"),
    ParamSpec("kappa", KAPPA, "linear", 0.0, 1.0, 0.0,
              ("uniform", 0.0, 1.0), None,
              "S only: weight of the counterfactual update"),
)
BY_NAME = {p.name: p for p in PARAMS}


def neutral_theta():
    return np.array([p.neutral for p in PARAMS], dtype=np.float64)


def prior_of(spec, stimulus):
    return spec.prior_a if (stimulus == "A" and spec.prior_a is not None) else spec.prior


def sample_prior(spec, stimulus, rng):
    dist, lo, hi = prior_of(spec, stimulus)
    if dist == "uniform":
        return float(rng.uniform(lo, hi))
    if dist == "loguniform":
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    raise ValueError(f"Unknown prior {dist!r} for {spec.name}")


def to_z(spec, x):
    return float(np.log10(x)) if spec.scale == "log" else float(x)


def from_z(spec, z):
    return float(10 ** z) if spec.scale == "log" else float(z)


def z_bounds(spec):
    if spec.scale == "log":
        return np.log10(spec.lower), np.log10(spec.upper)
    return spec.lower, spec.upper
