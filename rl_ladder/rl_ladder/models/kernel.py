"""The single numba kernel behind every model, in every mode.

One trial, for every model:

    logit = [stimulus term] + [bias] + beta_v*(V_H - V_L) + beta_c*k + beta_f*h + w*y_prev*(2r_prev - 1)
    p     = sigmoid(logit) = P(choose High)
    then update Q / V / k / h from the choice and reward.

Modules that a model does not use sit at neutral values (see params.py), so
their term is exactly zero. The same loop scores observed choices (mode 0) and
generates synthetic ones (modes 1-2), so fitting and simulation cannot drift
apart.

Modes
    0  score: use the given choices, return NLL over trials with lik > 0
    1  simulate on the given stimulus sequence ("replay")
    2  simulate stimuli too: after an error the stimulus repeats while the run
       of consecutive errors is <= max_ct (correction trials), as in the
       original simulateLearning; otherwise it is redrawn 50/50
"""

import numpy as np
from numba import njit

from .params import (ALPHA, ALPHA_C, ALPHA_F, ALPHA_V, B, BETA_C, BETA_F, BETA_Q, BETA_V, EPS, GAMMA,
                     KAPPA, Q0, TAU_A)

TRACE_COLUMNS = ("logit", "stimulus", "action", "choice", "feedback")
N_TRACE = len(TRACE_COLUMNS)

_EMPTY_F = np.empty(0, dtype=np.float64)
_EMPTY_I = np.empty(0, dtype=np.int64)
_EMPTY_TRACE = np.empty((0, N_TRACE), dtype=np.float64)


@njit(cache=True)
def _softplus(x):
    if x > 0.0:
        return x + np.log1p(np.exp(-x))
    return np.log1p(np.exp(x))


@njit(cache=True)
def run_kernel(theta, base, reset, stim, choice, new_session, lik, mode,
               u_choice, u_stim, max_ct, out_p, out_choice, out_stim, out_ct, trace):
    n = new_session.shape[0]
    alpha = theta[ALPHA]
    beta_q = theta[BETA_Q]
    b = theta[B]
    q0 = theta[Q0]
    gamma = theta[GAMMA]
    alpha_v = theta[ALPHA_V]
    beta_v = theta[BETA_V]
    alpha_c = theta[ALPHA_C]
    beta_c = theta[BETA_C]
    alpha_f = theta[ALPHA_F]
    beta_f = theta[BETA_F]
    tau_a = theta[TAU_A]
    kappa = theta[KAPPA]
    eps = theta[EPS]
    if eps >= 1.0:
        w = 0.0
    else:
        half = max(0.5 * eps, 1e-300)
        w = np.log((1.0 - half) / half)

    q = np.empty((2, 2))  # S: q[s, a]
    qa = np.empty(2)  # A: qa[a]
    v = np.empty(2)
    for s in range(2):
        q[s, 1] = 0.5 + 0.5 * q0
        q[s, 0] = 0.5 - 0.5 * q0
    qa[0] = q0
    qa[1] = -q0
    v[0] = 0.5
    v[1] = 0.5
    k = 0.0
    h = 0.0
    prev_y = 0.0
    prev_sign = 0.0
    ct_run = 0
    prev_s = 0
    prev_r = 1
    write_p = out_p.shape[0] > 0
    write_trace = trace.shape[0] > 0
    nll = 0.0

    for t in range(n):
        if t > 0 and new_session[t]:
            if reset[0]:
                for s in range(2):
                    q[s, 1] = 0.5 + 0.5 * q0
                    q[s, 0] = 0.5 - 0.5 * q0
                qa[0] = q0
                qa[1] = -q0
            if reset[1]:
                v[0] = 0.5
                v[1] = 0.5
            if reset[2]:
                k = 0.0
            if reset[3]:
                prev_y = 0.0
                prev_sign = 0.0
            if reset[4]:
                h = 0.0

        if mode == 2:
            if t == 0 or new_session[t]:
                ct_run = 0
                s = 1 if u_stim[t] < 0.5 else 0
                ct = 0
            elif prev_r == 0 and ct_run <= max_ct:
                s = prev_s
                ct = 1
            else:
                s = 1 if u_stim[t] < 0.5 else 0
                ct = 0
            out_stim[t] = s
            out_ct[t] = ct
        else:
            s = stim[t]

        if base == 1:
            stim_term = beta_q * (q[s, 1] - q[s, 0])
            z = stim_term + b
        elif base == 2:
            m = qa[1] if s == 1 else -qa[0]
            stim_term = tau_a * m
            z = tau_a * (m + 2.0 * b)
        else:
            stim_term = 0.0
            z = b
        action_term = beta_v * (v[1] - v[0]) + w * prev_y * prev_sign
        choice_term = beta_c * k
        feedback_term = beta_f * h
        z += action_term + choice_term + feedback_term

        if z >= 0.0:
            p = 1.0 / (1.0 + np.exp(-z))
        else:
            e = np.exp(z)
            p = e / (1.0 + e)
        if write_p:
            out_p[t] = p
        if write_trace:
            trace[t, 0] = z
            trace[t, 1] = stim_term
            trace[t, 2] = action_term
            trace[t, 3] = choice_term
            trace[t, 4] = feedback_term

        if mode == 0:
            c = choice[t]
        else:
            c = 1 if u_choice[t] < p else 0
            out_choice[t] = c
        if lik[t] > 0.0:
            nll += lik[t] * _softplus(-z if c == 1 else z)
        r = 1 if c == s else 0

        if base == 1:
            q[s, c] += alpha * (r - q[s, c])
            if kappa != 0.0:
                q[s, 1 - c] += kappa * alpha * ((1 - r) - q[s, 1 - c])
        elif base == 2:
            delta = r - qa[c]
            qa[c] += alpha * delta
            qa[1 - c] -= gamma * alpha * delta
        v[c] += alpha_v * (r - v[c])
        y = 1.0 if c == 1 else -1.0
        k = (1.0 - alpha_c) * k + alpha_c * y
        # Use the completed trial only after computing its choice probability.
        sign = 1.0 if r == 1 else -1.0
        h = (1.0 - alpha_f) * h + alpha_f * y * sign
        prev_y = y
        prev_sign = sign
        if r == 1:
            ct_run = 0
        else:
            ct_run += 1
        prev_s = s
        prev_r = r
    return nll


def _arrays(trials):
    return (np.ascontiguousarray(trials.stim, dtype=np.int64),
            np.ascontiguousarray(trials.choice, dtype=np.int64),
            np.ascontiguousarray(trials.new_session, dtype=np.uint8))


def nll(spec, theta, trials, lik, reset):
    """Negative log-likelihood of the observed choices."""
    stim, choice, new_session = _arrays(trials)
    return run_kernel(theta, spec.base, reset, stim, choice, new_session,
                      np.ascontiguousarray(lik, dtype=np.float64), 0, _EMPTY_F, _EMPTY_F, 0,
                      _EMPTY_F, _EMPTY_I, _EMPTY_I, _EMPTY_I, _EMPTY_TRACE)


def score(spec, theta, trials, lik, reset, with_trace=False):
    """NLL plus per-trial P(High) and, optionally, the per-term logit trace."""
    stim, choice, new_session = _arrays(trials)
    n = len(stim)
    p = np.empty(n)
    trace = np.empty((n, N_TRACE)) if with_trace else _EMPTY_TRACE
    total = run_kernel(theta, spec.base, reset, stim, choice, new_session,
                       np.ascontiguousarray(lik, dtype=np.float64), 0, _EMPTY_F, _EMPTY_F, 0,
                       p, _EMPTY_I, _EMPTY_I, _EMPTY_I, trace)
    return total, p, (trace if with_trace else None)


def simulate_arrays(spec, theta, stim, new_session, reset, env, max_ct, rng):
    """Generate (choice, stim, is_ct, p). env: 'replay' keeps `stim`; 'reactive' redraws it."""
    n = len(new_session)
    mode = {"replay": 1, "reactive": 2}[env]
    u_choice = rng.random(n)
    u_stim = rng.random(n)
    p = np.empty(n)
    out_choice = np.empty(n, dtype=np.int64)
    out_stim = np.zeros(n, dtype=np.int64)
    out_ct = np.zeros(n, dtype=np.int64)
    stim_in = np.ascontiguousarray(stim if stim is not None else np.zeros(n), dtype=np.int64)
    run_kernel(theta, spec.base, reset, stim_in, _EMPTY_I,
               np.ascontiguousarray(new_session, dtype=np.uint8), np.ones(n), mode,
               u_choice, u_stim, int(max_ct), p, out_choice, out_stim, out_ct, _EMPTY_TRACE)
    if mode == 1:
        out_stim = stim_in.copy()
    return out_choice, out_stim, out_ct.astype(bool), p
