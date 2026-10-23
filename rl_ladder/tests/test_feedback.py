"""Scientific checks for the independent choice-feedback history module."""

import numpy as np
import pytest

from rl_ladder.data import Trials
from rl_ladder.models import kernel
from rl_ladder.models.registry import reset_flags
from rl_ladder.service import context


def trial_table(stim, choice, session=None):
    stim, choice = np.asarray(stim), np.asarray(choice)
    session = np.ones(len(stim), dtype=int) if session is None else np.asarray(session)
    return Trials("manual", stim, choice, (stim == choice).astype(int), session,
                  np.zeros(len(stim), dtype=bool)).validate()


def probabilities(name, values, trials, resets=(), lik=None):
    spec = context().models[name]
    return kernel.score(spec, spec.theta(values), trials,
                        np.ones(trials.n) if lik is None else lik, reset_flags(resets),
                        with_trace=True)


@pytest.mark.parametrize("choice,reward,direction", [(1, 1, 1), (0, 1, -1),
                                                    (1, 0, -1), (0, 0, 1)])
def test_each_choice_feedback_outcome(choice, reward, direction):
    stimulus = choice if reward else 1 - choice
    trials = trial_table([stimulus, 1], [choice, 1])
    _, p, trace = probabilities("F", {"b": 0, "alpha_f": 0.5, "beta_f": 2}, trials)
    assert p[0] == 0.5
    assert trace[1, 4] == direction  # alpha_f * beta_f * choice_sign * feedback_sign
    assert (p[1] > 0.5) == (direction > 0)


@pytest.mark.parametrize("resets", [[], ["feedback"], ["choice", "feedback"]])
def test_matches_published_pr_equations_directly(resets):
    """Use the weighted P/R equations, rather than the kernel's normalized traces."""
    trials = trial_table([1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1], [1, 1, 1, 2, 2, 2])
    v = {"b": 0.2, "alpha_c": 0.3, "beta_c": 1.7, "alpha_f": 0.6, "beta_f": 2.1}
    _, p, _ = probabilities("F+C", v, trials, resets)
    P = R = 0.0
    expected = []
    for t, (c, r) in enumerate(zip(trials.choice, trials.reward)):
        if t and trials.new_session[t]:
            if "choice" in resets:
                P = 0.0
            if "feedback" in resets:
                R = 0.0
        expected.append(1 / (1 + np.exp(-(v["b"] + P + R))))
        C, F = 2 * c - 1, 2 * r - 1
        P += v["alpha_c"] * (v["beta_c"] * C - P)
        R += v["alpha_f"] * (v["beta_f"] * C * F - R)
    np.testing.assert_allclose(p, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("child,parent", [("S+F", "S0"), ("S+F+C", "S+C"),
                                           ("A+F", "A0"), ("A+F+C", "A+C"),
                                           ("B+F", "B0"), ("B+F+C", "B+C"),
                                           ("S+R+F", "S+R"), ("S+R+F+C", "S+R+C"),
                                           ("F", "M1"), ("F+C", "C")])
def test_zero_feedback_weight_recovers_parent_exactly(child, parent, trials):
    v = {"alpha": 0.2, "beta_q": 5, "q0": 0.1, "b": 0.05, "gamma": 0.4,
         "alpha_v": 0.2, "beta_v": 1.3, "alpha_c": 0.3, "beta_c": -0.4,
         "alpha_f": 0.4, "beta_f": 0}
    a, p, _ = probabilities(child, v, trials)
    b, q, _ = probabilities(parent, v, trials)
    assert a == b
    np.testing.assert_array_equal(p, q)


def test_feedback_is_not_reward_independent_choice_kernel():
    trials = trial_table([0, 0], [1, 1])  # an unrewarded High choice
    _, c, _ = probabilities("C", {"b": 0, "alpha_c": 1, "beta_c": 2}, trials)
    _, f, _ = probabilities("F", {"b": 0, "alpha_f": 1, "beta_f": 2}, trials)
    assert c[1] > 0.5 > f[1]


def test_no_current_or_future_feedback_in_current_prediction():
    a = trial_table([1, 1, 0], [1, 1, 1])
    b = trial_table([1, 0, 1], [1, 1, 0])  # same past at trial 1; different current and future
    v = {"b": 0, "alpha_f": 0.4, "beta_f": 2}
    _, p, _ = probabilities("F", v, a)
    _, q, _ = probabilities("F", v, b)
    np.testing.assert_array_equal(p[:2], q[:2])
    assert p[2] != q[2]


def test_masked_trials_still_update_feedback_history():
    trials = trial_table([1, 0, 1], [1, 1, 1])
    values = {"b": 0, "alpha_f": 0.5, "beta_f": 2}
    _, p, trace = probabilities("F", values, trials, lik=np.array([0., 1., 1.]))
    assert trace[1, 4] == 1  # the unscored first trial still affects trial 1
    assert trace[2, 4] == -0.5
    _, full_p, _ = probabilities("F", values, trials)
    np.testing.assert_array_equal(p, full_p)


def test_feedback_equals_stimulus_history_under_deterministic_rewards():
    stim = [1, 0, 1, 1, 0, 0]
    a = trial_table(stim, [1, 1, 0, 1, 0, 1])
    b = trial_table(stim, [0, 0, 1, 0, 1, 0])
    v = {"b": 0, "alpha_f": 0.25, "beta_f": 3}
    _, p, trace = probabilities("F", v, a)
    _, q, _ = probabilities("F", v, b)
    np.testing.assert_array_equal(p, q)
    history = 0.0
    for t, s in enumerate(stim):
        assert trace[t, 4] == v["beta_f"] * history
        history = (1 - v["alpha_f"]) * history + v["alpha_f"] * (2 * s - 1)


@pytest.mark.parametrize("env", ["replay", "reactive"])
def test_feedback_simulation_and_likelihood_share_probabilities(env, trials):
    spec = context().models["B+F+C"]
    v = {"alpha": 0.1, "q0": 0.1, "gamma": 0.3, "b": 0.02,
         "alpha_c": 0.2, "beta_c": 1.3, "alpha_f": 0.15, "beta_f": 1.7}
    reset = reset_flags(["wsls", "feedback"])
    choice, stim, _, p = kernel.simulate_arrays(spec, spec.theta(v), trials.stim,
                                               trials.new_session, reset, env, 3,
                                               np.random.default_rng(71))
    sim = trial_table(stim, choice, trials.session)
    _, q, _ = kernel.score(spec, spec.theta(v), sim, np.ones(sim.n), reset)
    np.testing.assert_array_equal(p, q)
