import numpy as np

from rl_ladder.fit.bms import family_log_evidence, vb_bms


def test_dominant_model_wins():
    rng = np.random.default_rng(0)
    L = rng.normal(0, 1, (20, 3))
    L[:, 1] += 20
    res = vb_bms(L)
    assert res["pxp"][1] > 0.99
    assert res["bor"] < 0.01
    assert np.isclose(res["expected_freq"].sum(), 1)


def test_indistinguishable_models_have_high_omnibus_risk():
    res = vb_bms(np.zeros((20, 3)))
    assert res["bor"] > 0.9
    np.testing.assert_allclose(res["pxp"], 1 / 3, atol=0.05)


def test_family_evidence_uniform_within_family():
    L = np.log(np.array([[1.0, 3.0, 2.0]]))
    fam = family_log_evidence(L, [[0, 1], [2]])
    np.testing.assert_allclose(np.exp(fam), [[2.0, 2.0]])
