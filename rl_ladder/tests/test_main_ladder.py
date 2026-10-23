"""Submission guards for the agreed primary-model protocol."""

import pytest

from rl_ladder.service import context
from rl_ladder.models.registry import parents
from rl_ladder.tasks import make_plan


PRIMARY = {"M1", "M2", "S0", "S+R", "S+C", "S+F", "S+R+C", "S+R+F",
           "S+F+C", "S+R+F+C", "A0", "A+C", "A+F", "A+F+C",
           "B0", "B+C", "B+F", "B+F+C"}
POLICIES = {("wsls", "feedback"), ("wsls", "choice", "feedback")}


def test_primary_nested_starts_do_not_silently_fit_supplementary_models():
    ctx = context()
    assert all(set(parents(ctx.models[m], ctx.models)) <= PRIMARY for m in PRIMARY)


@pytest.mark.parametrize("experiment,n_mice,n_units", [
    ("main_ladder", 19, 684), ("smoke_main_ladder", 1, 36), ("main_new_mice", 5, 180)])
def test_primary_submission_has_only_agreed_models_and_comparable_data(
        tmp_path, monkeypatch, experiment, n_mice, n_units):
    monkeypatch.setenv("RL_LADDER_RESULTS", str(tmp_path))
    _, plan = make_plan(experiment)
    units = plan["units"]
    assert plan["n_units"] == n_units
    assert {u["model"] for u in units} == PRIMARY
    assert len({u["mouse"] for u in units}) == n_mice
    assert {tuple(u["cfg"]["model"]["session_reset"]) for u in units} == POLICIES
    for mouse in {u["mouse"] for u in units}:
        for policy in POLICIES:
            matched = [u for u in units if u["mouse"] == mouse
                       and tuple(u["cfg"]["model"]["session_reset"]) == policy]
            assert len(matched) == 18
            # All candidate models must see the same observations and split.
            cfg = matched[0]["cfg"]
            assert all(all(u["cfg"][k] == cfg[k] for k in
                           ("data", "preprocess", "fit", "evaluate")) for u in matched)
            assert cfg["preprocess"]["correction"] == "mask"
            assert cfg["evaluate"]["cv"] == {
                "method": "chrono", "train_frac": .67, "warm_start_from_nested": True}
            assert cfg["fit"]["warm_start_from_nested"]
            assert cfg["collect"]["require_fit_success"]
            contrasts = cfg["collect"]["contrasts"]
            assert len(contrasts) == 29
            assert len({tuple(pair) for pair in contrasts}) == 29
            assert all(a in PRIMARY and b in PRIMARY for a, b in contrasts)
    if experiment == "main_new_mice":
        assert all(u["cfg"]["data"]["variant"] == "current" for u in units)
        assert all(u["cfg"]["collect"]["groups"] is None for u in units)
    else:
        assert all(u["cfg"]["data"]["variant"] == "paper" for u in units)


@pytest.mark.parametrize("experiment,n_units", [
    ("main_model_recovery", 1800), ("main_param_recovery", 3600)])
def test_recovery_matches_primary_models_and_reset_policies(
        tmp_path, monkeypatch, experiment, n_units):
    monkeypatch.setenv("RL_LADDER_RESULTS", str(tmp_path))
    _, plan = make_plan(experiment)
    assert plan["n_units"] == n_units
    assert {u["model"] for u in plan["units"]} == PRIMARY
    assert {tuple(u["cfg"]["model"]["session_reset"]) for u in plan["units"]} == POLICIES
    assert all(u["cfg"]["validate"]["accuracy_range"] == [0., 1.] for u in plan["units"])
    if experiment == "main_model_recovery":
        assert all(set(u["candidates"]) == PRIMARY for u in plan["units"])
    assert set(context().sets["core"]) == PRIMARY
