"""Nesting is exact, the paper's choice-history limit behaves as claimed, and
simulation uses the same computation as scoring."""

import numpy as np

from rl_ladder.models import kernel
from rl_ladder.models.registry import reset_flags
from rl_ladder.service import context

M = context().models
S_VALUES = {"alpha": 0.1, "beta_q": 5.0, "b": 0.2, "q0": 0.3}
A_VALUES = {"alpha": 0.05, "b": 0.03, "q0": 0.2}


def _nll(name, values, trials, resets=("wsls",)):
    spec = M[name]
    return kernel.nll(spec, spec.theta(values), trials, np.ones(trials.n), reset_flags(resets))


def test_switched_off_modules_are_exactly_nested(trials):
    s0 = _nll("S0", S_VALUES, trials)
    for name, extra in (("S+R", {"alpha_v": 0.3, "beta_v": 0.0}),
                        ("S+C", {"alpha_c": 0.3, "beta_c": 0.0}),
                        ("S+R+C", {"alpha_v": 0.3, "beta_v": 0.0, "alpha_c": 0.3, "beta_c": 0.0})):
        assert _nll(name, {**S_VALUES, **extra}, trials) == s0
    a0 = _nll("A0", A_VALUES, trials)
    assert _nll("A+R", {**A_VALUES, "gamma": 0.0}, trials) == a0
    assert _nll("A+C", {**A_VALUES, "alpha_c": 0.2, "beta_c": 0.0}, trials) == a0
    assert _nll("A0T", {**A_VALUES, "tau_a": 5.0}, trials) == a0


def test_wsls_eps_one_is_chance(trials):
    assert np.isclose(_nll("M2", {"eps": 1.0}, trials), trials.n * np.log(2))


def test_gamma_one_ignores_the_stimulus(trials):
    """gamma = 1 keeps Q(H) + Q(L) = 0: given the history, P(High) on trial t does
    not depend on that trial's stimulus (the model becomes a pure bandit)."""
    spec = M["A+R"]
    theta = spec.theta({**A_VALUES, "alpha": 0.2, "gamma": 1.0})
    _, p, _ = kernel.score(spec, theta, trials, np.ones(trials.n), reset_flags([]))
    for t in (0, 5, 50, 199, trials.n - 1):
        flipped = trials.subset(np.ones(trials.n, dtype=bool))
        flipped.stim[t] = 1 - flipped.stim[t]
        flipped.reward[t] = int(flipped.choice[t] == flipped.stim[t])
        _, q, _ = kernel.score(spec, theta, flipped, np.ones(trials.n), reset_flags([]))
        assert np.isclose(p[t], q[t])
    # ...whereas with gamma < 1 the stimulus matters.
    theta = spec.theta({**A_VALUES, "alpha": 0.2, "gamma": 0.0})
    _, p, _ = kernel.score(spec, theta, trials, np.ones(trials.n), reset_flags([]))
    flipped = trials.subset(np.ones(trials.n, dtype=bool))
    flipped.stim[199] = 1 - flipped.stim[199]
    _, q, _ = kernel.score(spec, theta, flipped, np.ones(trials.n), reset_flags([]))
    assert not np.isclose(p[199], q[199])


def test_simulated_probabilities_match_scoring(trials):
    rng = np.random.default_rng(1)
    for name, values in (("S+R+C", {**S_VALUES, "alpha_v": 0.2, "beta_v": 1.5, "alpha_c": 0.3,
                                    "beta_c": 0.8}),
                         ("A+R+C", {**A_VALUES, "gamma": 0.5, "alpha_c": 0.2, "beta_c": -0.5}),
                         ("M2", {"eps": 0.3})):
        spec = M[name]
        for env in ("replay", "reactive"):
            choice, stim, is_ct, p_sim = kernel.simulate_arrays(
                spec, spec.theta(values), trials.stim, trials.new_session, reset_flags(["wsls"]),
                env, 3, rng)
            sim = trials.subset(np.ones(trials.n, dtype=bool))
            sim.choice, sim.stim = choice, stim
            _, p_fit, _ = kernel.score(spec, spec.theta(values), sim, np.ones(trials.n),
                                       reset_flags(["wsls"]))
            np.testing.assert_array_equal(p_sim, p_fit)


def test_reactive_correction_rule():
    """After an error the stimulus repeats while the error run is <= max_ct."""
    spec = M["M1"]
    n = 5000
    new_session = np.zeros(n, dtype=bool)
    new_session[0] = True
    choice, stim, is_ct, _ = kernel.simulate_arrays(
        spec, spec.theta({"b": 0.0}), None, new_session, reset_flags([]), "reactive", 3,
        np.random.default_rng(3))
    reward = choice == stim
    run = 0
    for t in range(1, n):
        run = 0 if reward[t - 1] else run + 1
        expected_ct = (not reward[t - 1]) and run <= 3
        assert is_ct[t] == expected_ct
        if expected_ct:
            assert stim[t] == stim[t - 1]
