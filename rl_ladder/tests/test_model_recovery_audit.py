"""Recovery must distinguish numerical failure from model confusion."""

from copy import deepcopy

import numpy as np
import pytest

from rl_ladder.collect import collect, verify_recovery_plan
from rl_ladder.service import atomic_write_json, context, read_json
from rl_ladder.tasks import make_plan, run_dir, run_unit, unit_output_path
from rl_ladder.validate import audit_model_recovery


@pytest.fixture
def recovered(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(19)
    stim = rng.integers(0, 2, 96)
    choice = rng.integers(0, 2, 96)
    np.savez(data / "SYN.npz", answer=stim, choice=choice,
             reward=(choice == stim).astype(int), session=np.repeat([1, 2, 3], 32),
             trial_type=np.ones(96, dtype=int))
    monkeypatch.setenv("RL_LADDER_DATA", str(data))
    monkeypatch.setenv("RL_LADDER_RESULTS", str(tmp_path / "results"))
    path, plan = make_plan({"task": "model_recovery", "models": "M1,M2", "datasets": ["SYN"]},
                          {"validate.n_reps": 1, "validate.accuracy_range": [0., 1.],
                           "fit.n_starts": 3, "collect.require_fit_success": True}, name="audit")
    for i in range(plan["n_units"]):
        run_unit(plan, i)
    return path, plan


def test_recovery_replays_after_parameter_draws_and_preserves_diagnostics(recovered):
    path, plan = recovered
    for unit in plan["units"]:
        result = read_json(unit_output_path(plan, unit))["result"]
        audit = audit_model_recovery(context(), unit, result)
        assert audit["passed"], audit
        assert audit["max_score_error"] < 1e-10
        for fit in result["candidates"].values():
            assert fit["start_records"] and fit["n_starts"] >= 3
            assert all("success" in s and "message" in s for s in fit["start_records"])
    assert verify_recovery_plan(path) == 0


@pytest.mark.parametrize("corruption", ["score", "signature", "candidate_set", "code_version"])
def test_corrupted_recovery_cannot_pass(recovered, corruption):
    _, plan = recovered
    unit = plan["units"][0]
    result = deepcopy(read_json(unit_output_path(plan, unit))["result"])
    if corruption == "score":
        result["candidates"]["M1"]["nll"] += 1
    elif corruption == "signature":
        result["input_signature"] = "different"
    elif corruption == "candidate_set":
        result["candidates"].pop("M2")
    else:
        result["code_version"] = "previous"
    assert not audit_model_recovery(context(), unit, result)["passed"]


def test_failed_optimizer_with_reproducible_score_blocks_matrix_and_verification(recovered):
    path, plan = recovered
    assert collect(path, plots=False) == 0
    summary = run_dir(plan) / "summary"
    assert list(summary.glob("confusion_counts__*.csv"))
    unit = plan["units"][0]
    output_path = unit_output_path(plan, unit)
    output = read_json(output_path)
    output["result"]["candidates"]["M1"]["success"] = False
    atomic_write_json(output_path, output)
    report = audit_model_recovery(context(), unit, output["result"])
    assert report["verified"] and not report["all_converged"] and not report["passed"]
    assert verify_recovery_plan(path) == 1
    assert not list(summary.glob("confusion_counts__*.csv"))
    assert (summary / "model_recovery.csv").exists()


def test_wrong_cached_output_key_is_rejected(recovered):
    path, plan = recovered
    output_path = unit_output_path(plan, plan["units"][0])
    output = read_json(output_path)
    output["key"] = "other-run"
    atomic_write_json(output_path, output)
    assert verify_recovery_plan(path) == 1


def test_changed_template_window_is_rejected(recovered):
    _, plan = recovered
    unit = deepcopy(plan["units"][0])
    result = read_json(unit_output_path(plan, unit))["result"]
    unit["cfg"]["preprocess"]["window"] = {"method": "first_sessions", "n": 2}
    report = audit_model_recovery(context(), unit, result)
    assert not report["passed"] and "Template" in report["error"]


@pytest.mark.parametrize("failure", ["generating_fit", "nested_fit"])
def test_reproducible_converged_but_suboptimal_fit_is_not_accepted(recovered, failure):
    from rl_ladder.models import kernel
    from rl_ladder.models.registry import reset_flags
    from rl_ladder.validate import prepare_synthetic, simulate_trials, template_trials
    _, plan = recovered
    unit = deepcopy(plan["units"][0])  # M1 generates this dataset.
    result = deepcopy(read_json(unit_output_path(plan, unit))["result"])
    ctx, cfg = context(), unit["cfg"]
    rng = np.random.default_rng()
    rng.bit_generator.state = result["simulation"]["generation_state"]
    reset = reset_flags(cfg["model"]["session_reset"])
    sim = simulate_trials(ctx.models["M1"], result["true"], template_trials("SYN", cfg),
                          cfg["validate"]["env"], cfg["validate"]["max_ct"], reset, rng)
    trials, lik = prepare_synthetic(sim, cfg)
    name = "M1" if failure == "generating_fit" else "C"
    fit = deepcopy(result["candidates"]["M1"])
    fit["params"] = {"b": 10.0} if name == "M1" else {
        "b": 10.0, "alpha_c": .1, "beta_c": 0.0}
    fit["k"] = ctx.models[name].k
    fit["nll"] = float(kernel.nll(ctx.models[name], ctx.models[name].theta(fit["params"]),
                                  trials, lik, reset))
    fit["bic"] = 2 * fit["nll"] + fit["k"] * np.log(lik.sum())
    result["candidates"][name] = fit
    if name == "C":
        unit["candidates"].append(name)
    ranked = sorted(result["candidates"], key=lambda m: result["candidates"][m]["bic"])
    result["winner"] = ranked[0]
    result["delta_bic_runner_up"] = (result["candidates"][ranked[1]]["bic"] -
                                       result["candidates"][ranked[0]]["bic"])
    report = audit_model_recovery(ctx, unit, result)
    assert report["verified"] and report["all_converged"] and not report["passed"]
    assert report["optimizer_miss"] if name == "M1" else report["nested_violations"]


@pytest.mark.parametrize("experiment,n_units,reps_per_template", [
    ("smoke_model_recovery", 72, 1), ("recovery_main_old", 2052, 3),
    ("recovery_main_new", 900, 5)])
def test_recovery_design_is_balanced_and_uses_only_primary_models(
        tmp_path, monkeypatch, experiment, n_units, reps_per_template):
    from collections import Counter
    monkeypatch.setenv("RL_LADDER_RESULTS", str(tmp_path))
    _, plan = make_plan(experiment)
    models = set(context().sets["core"])
    assert len(models) == 18 and plan["n_units"] == n_units
    counts = Counter((u["variant"], u["model"], u["template"]) for u in plan["units"])
    assert set(counts.values()) == {reps_per_template}
    for u in plan["units"]:
        assert set(u["candidates"]) == models
        assert u["cfg"]["collect"]["require_fit_success"]
        assert u["cfg"]["validate"]["accuracy_range"] == [0., 1.]
        assert u["cfg"]["preprocess"]["correction"] == "mask"
