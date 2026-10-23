"""The numba kernel must equal the independent Python reference, trial by trial."""

import numpy as np
import pytest

from rl_ladder.models import kernel
from rl_ladder.models import params as P
from rl_ladder.models.reference import reference_nll
from rl_ladder.models.registry import reset_flags
from rl_ladder.service import context

RESETS = ([], ["wsls"], ["stimulus", "action", "choice", "wsls", "feedback"],
          ["choice"], ["feedback"], ["choice", "feedback"])


@pytest.mark.parametrize("model", sorted(context().models))
@pytest.mark.parametrize("resets", RESETS)
def test_kernel_matches_reference(model, resets, trials):
    spec = context().models[model]
    from rl_ladder.service import seed_for
    rng = np.random.default_rng(seed_for("reference-test", model, tuple(resets)))
    lik = np.where(trials.is_ct, 0.0, 1.0)
    for _ in range(3):
        values = {p.name: P.sample_prior(p, spec.stimulus, rng) for p in spec.free_specs}
        if "alpha" in values:  # fast learning so every module visibly moves
            values["alpha"] = rng.uniform(0.05, 0.5)
        total, p, _ = kernel.score(spec, spec.theta(values), trials, lik, reset_flags(resets))
        ref_total, ref_p = reference_nll(spec, values, trials.stim, trials.choice,
                                         trials.new_session, lik, resets)
        np.testing.assert_allclose(p, ref_p, rtol=0, atol=1e-12)
        assert abs(total - ref_total) < 1e-9


def test_logit_trace_sums_to_logit(trials):
    spec = context().models["S+R+F+C"]
    values = {"alpha": 0.2, "beta_q": 4.0, "b": 0.3, "q0": 0.1, "alpha_v": 0.1, "beta_v": 2.0,
              "alpha_c": 0.3, "beta_c": -1.0, "alpha_f": 0.2, "beta_f": 1.2}
    _, p, trace = kernel.score(spec, spec.theta(values), trials, np.ones(trials.n),
                               reset_flags([]), with_trace=True)
    np.testing.assert_allclose(trace[:, 1:].sum(axis=1) + values["b"], trace[:, 0])
    np.testing.assert_allclose(1 / (1 + np.exp(-trace[:, 0])), p)
