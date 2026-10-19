"""Slow, readable, truly plug-in reference implementation.

Each module is a small class with three methods:

    reset()          initialise its state
    term(s)          its contribution to the logit on a trial with stimulus s
    update(s, c, r)  learn from the trial's choice and reward

A model is just a list of modules; its logit is the sum of their terms. This
file is written independently of kernel.py and serves two purposes:

1. tests require kernel.py to match it trial by trial;
2. a new module can be prototyped here (tens of times slower, but needs no
   numba) before it earns a slot in the kernel.
"""

import math

import numpy as np


class Bias:
    def __init__(self, b):
        self.b = b

    def reset(self):
        pass

    def term(self, s):
        return self.b

    def update(self, s, c, r):
        pass


class StimulusS:
    """Q(s, a); logit term beta_q * (Q(s,H) - Q(s,L)); optional counterfactual update."""

    def __init__(self, alpha, beta_q, q0, kappa=0.0):
        self.alpha, self.beta_q, self.q0, self.kappa = alpha, beta_q, q0, kappa

    def reset(self):
        self.q = {(s, a): 0.5 + (0.5 if a == 1 else -0.5) * self.q0 for s in (0, 1) for a in (0, 1)}

    def term(self, s):
        return self.beta_q * (self.q[s, 1] - self.q[s, 0])

    def update(self, s, c, r):
        self.q[s, c] += self.alpha * (r - self.q[s, c])
        if self.kappa:
            self.q[s, 1 - c] += self.kappa * self.alpha * ((1 - r) - self.q[s, 1 - c])


class StimulusA:
    """Original Collina et al. code: Q(a) read out through the stimulus.

    logit = tau * (m + 2b) with m = Q(H) on High trials and -Q(L) on Low trials.
    The bias is part of this term, so a model using it has no separate Bias.
    gamma > 0 is the paper's "choice-history" coupling (the R slot for A).
    """

    def __init__(self, alpha, q0, b, tau=5.0, gamma=0.0):
        self.alpha, self.q0, self.b, self.tau, self.gamma = alpha, q0, b, tau, gamma

    def reset(self):
        self.q = [self.q0, -self.q0]

    def term(self, s):
        m = self.q[1] if s == 1 else -self.q[0]
        return self.tau * (m + 2 * self.b)

    def update(self, s, c, r):
        delta = r - self.q[c]
        self.q[c] += self.alpha * delta
        self.q[1 - c] -= self.gamma * self.alpha * delta


class ActionValue:
    def __init__(self, alpha_v, beta_v):
        self.alpha_v, self.beta_v = alpha_v, beta_v

    def reset(self):
        self.v = [0.5, 0.5]

    def term(self, s):
        return self.beta_v * (self.v[1] - self.v[0])

    def update(self, s, c, r):
        self.v[c] += self.alpha_v * (r - self.v[c])


class WSLS:
    def __init__(self, eps):
        self.w = 0.0 if eps >= 1 else math.log((1 - eps / 2) / (eps / 2))

    def reset(self):
        self.prev = 0.0

    def term(self, s):
        return self.w * self.prev

    def update(self, s, c, r):
        self.prev = (1.0 if c == 1 else -1.0) * (1.0 if r == 1 else -1.0)


class ChoiceKernel:
    def __init__(self, alpha_c, beta_c):
        self.alpha_c, self.beta_c = alpha_c, beta_c

    def reset(self):
        self.k = 0.0

    def term(self, s):
        return self.beta_c * self.k

    def update(self, s, c, r):
        self.k = (1 - self.alpha_c) * self.k + self.alpha_c * (1.0 if c == 1 else -1.0)


def build_modules(spec, values):
    """values: {param: value} for every parameter the spec uses (free or fixed)."""
    v = dict(values)
    for name, value in spec.fixed.items():
        v[name] = value
    v.setdefault("b", 0.0)
    mods = {}
    if spec.stimulus in ("S", "S_cf"):
        mods["stimulus"] = StimulusS(v["alpha"], v["beta_q"], v["q0"],
                                     kappa=1.0 if spec.stimulus == "S_cf" else 0.0)
    elif spec.stimulus == "A":
        mods["stimulus"] = StimulusA(v["alpha"], v["q0"], v["b"], tau=v.get("tau_a", 5.0),
                                     gamma=v["gamma"] if spec.action == "gamma" else 0.0)
    if spec.stimulus != "A":
        mods["bias"] = Bias(v["b"])
    if spec.action == "V":
        mods["action"] = ActionValue(v["alpha_v"], v["beta_v"])
    elif spec.action == "wsls":
        mods["wsls"] = WSLS(v["eps"])
    if spec.choice == "ck":
        mods["choice"] = ChoiceKernel(v["alpha_c"], v["beta_c"])
    return mods


def reference_nll(spec, values, stim, choice, new_session, lik, session_reset):
    mods = build_modules(spec, values)
    for m in mods.values():
        m.reset()
    total, probs = 0.0, []
    for t in range(len(stim)):
        if t > 0 and new_session[t]:
            for slot, m in mods.items():
                if slot in session_reset:
                    m.reset()
        s, c = int(stim[t]), int(choice[t])
        z = sum(m.term(s) for m in mods.values())
        p = 1 / (1 + math.exp(-z))
        probs.append(p)
        if lik[t] > 0:  # -log sigmoid(+-z), computed from z to avoid log(1 - p) round-off
            x = -z if c == 1 else z
            total += lik[t] * (x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x)))
        r = int(c == s)
        for m in mods.values():
            m.update(s, c, r)
    return total, np.array(probs)
