"""Numerical ties must not turn a converged fit into a reported failure."""

from types import SimpleNamespace

import numpy as np
import pytest

from rl_ladder.data import Trials
from rl_ladder.fit import mle
from rl_ladder.models.registry import reset_flags
from rl_ladder.service import context


@pytest.mark.parametrize("successful_bias,expected_success", [(1e-8, True), (1.0, False)])
def test_select_converged_tie_without_hiding_better_failure(monkeypatch, successful_bias,
                                                          expected_success):
    spec = context().models["M1"]
    c = np.array([0, 1] * 20)
    trials = Trials("balanced", c, c, np.ones(40, dtype=int), np.ones(40, dtype=int),
                    np.zeros(40, dtype=bool))
    responses = iter([(0.0, False), (successful_bias, True)])

    def fake_minimize(*args, **kwargs):
        bias, success = next(responses)
        return SimpleNamespace(x=np.array([bias]), success=success, nit=1, message="test")

    monkeypatch.setattr(mle, "minimize", fake_minimize)
    fit = mle.fit_mle(spec, trials, np.ones(40), reset_flags([]),
                      {"n_starts": 2, "maxiter": 10}, seed=1)
    assert fit["success"] is expected_success
    assert fit["n_failed"] == 1
    assert fit["convergence_nll_gap"] <= 1e-7
