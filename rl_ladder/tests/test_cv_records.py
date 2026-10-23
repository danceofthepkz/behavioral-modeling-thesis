"""CV provenance must permit score replay, detect corruption, and prevent leakage."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from rl_ladder import evaluate
from rl_ladder.audit_cv import verify_plan
from rl_ladder.collect import collect
from rl_ladder.config import load_yaml, experiment_path
from rl_ladder.data import Trials
from rl_ladder.models.registry import reset_flags
from rl_ladder.runners import run_local
from rl_ladder.service import atomic_write_json, context, read_json, seed_for
from rl_ladder.tasks import make_plan, run_dir, unit_output_path


@pytest.fixture
def trials():
    rng = np.random.default_rng(17)
    s = rng.integers(0, 2, 160)
    c = rng.integers(0, 2, 160)
    ct = np.zeros(160, dtype=bool)
    ct[8::11] = True
    return Trials("SYN", s, c, (s==c).astype(int), np.repeat([2, 4, 7, 11], 40), ct)


@pytest.fixture
def saved(trials):
    spec = context().models["A+C"]
    lik = (~trials.is_ct).astype(float)
    reset = reset_flags(["wsls", "feedback"])
    result = evaluate.chrono_cv(spec, trials, lik, reset,
                                {"n_starts": 2, "maxiter": 300, "repeat_tol": .01}, 123, .5)
    return spec, lik, reset, json.loads(json.dumps(result))


def test_json_round_trip_can_be_verified_without_optimizer(trials, saved, monkeypatch):
    spec, lik, reset, result = saved
    def forbidden(*args, **kwargs):
        pytest.fail("Score verification must not invoke an optimizer")
    monkeypatch.setattr(evaluate, "fit_mle", forbidden)
    checked = evaluate.verify_cv_record(spec, trials, lik, reset, result)
    assert checked["passed"] and checked["test_nll_abs_error"] < 1e-10
    rec = result["cv_record"]
    assert rec["split"]["train_sessions"] == [2, 4]
    assert rec["split"]["test_sessions"] == [7, 11]
    assert rec["environment"]["packages"]["scipy"]
    assert rec["environment"]["numpy_build_config"]
    fit = rec["training_fit"]
    assert len(fit["start_records"]) == 2
    assert sum(r["selected"] for r in fit["start_records"]) == 1
    for r in fit["start_records"]:
        assert all(k in r for k in ("initial_theta", "theta", "nll", "success", "message", "nit"))


@pytest.mark.parametrize("kind", ["test_score", "training_params", "split", "starts", "seed", "convergence"])
def test_corrupted_record_is_rejected(trials, saved, kind):
    spec, lik, reset, result = saved
    r = copy.deepcopy(result)
    if kind == "test_score":
        r["cv_test_nll"] += 1
    elif kind == "training_params":
        r["cv_record"]["training_fit"]["params"]["b"] += 1
    elif kind == "split":
        r["cv_record"]["split"]["test_sessions"] = [11]
    elif kind == "starts":
        r["cv_record"]["training_fit"]["start_records"] = []
    elif kind == "seed":
        r["cv_record"]["seed"] += 1
    else:
        r["cv_success"] = not r["cv_success"]
    with pytest.raises(ValueError):
        evaluate.verify_cv_record(spec, trials, lik, reset, r)


def test_changed_masked_choice_is_detected(trials, saved):
    spec, lik, reset, result = saved
    changed = copy.deepcopy(trials)
    i = np.flatnonzero(changed.is_ct)[0]
    changed.choice[i] = 1-changed.choice[i]
    changed.reward[i] = int(changed.choice[i] == changed.stim[i])
    with pytest.raises(ValueError, match="processed data"):
        evaluate.verify_cv_record(spec, changed, lik, reset, result)


def test_legacy_cv_scores_do_not_get_fabricated_records(trials):
    with pytest.raises(ValueError, match="unavailable"):
        evaluate.verify_cv_record(context().models["M1"], trials, np.ones(trials.n), reset_flags([]),
                                  {"cv_test_nll": 3.0})


@pytest.mark.parametrize("frac", [0, 1, -.1, float("nan")])
def test_bad_splits_fail_before_fitting(trials, monkeypatch, frac):
    monkeypatch.setattr(evaluate, "fit_mle", lambda *a, **k: pytest.fail("Invalid split reached optimizer"))
    with pytest.raises(ValueError, match="train_frac"):
        evaluate.chrono_cv(context().models["M1"], trials, np.ones(trials.n), reset_flags([]),
                           {"n_starts": 1, "maxiter": 30}, 1, frac)


def test_no_scored_test_trials_fails_before_fitting(trials):
    lik = (trials.session < 7).astype(float)
    with pytest.raises(ValueError, match="scored trials"):
        evaluate.chrono_cv(context().models["M1"], trials, lik, reset_flags([]),
                           {"n_starts": 1, "maxiter": 30}, 1, .5)


def test_held_out_choices_do_not_change_training_solution(trials):
    spec = context().models["A+C"]
    cfg = {"n_starts": 2, "maxiter": 300}
    changed = copy.deepcopy(trials)
    test = changed.session >= 7
    changed.choice[test] = 1-changed.choice[test]
    changed.reward = (changed.choice==changed.stim).astype(int)
    args = (np.ones(trials.n), reset_flags([]), cfg, 55, .5)
    a = evaluate.chrono_cv(spec, trials, *args)
    b = evaluate.chrono_cv(spec, changed, *args)
    assert a["cv_record"]["training_fit"] == b["cv_record"]["training_fit"]
    assert a["cv_test_nll"] != b["cv_test_nll"]


@pytest.fixture
def cv_plan(tmp_path, trials, monkeypatch):
    data = tmp_path/"data"
    data.mkdir()
    np.savez(data/"SYN.npz", answer=trials.stim, choice=trials.choice,
             reward=trials.reward, session=trials.session, trial_type=np.where(trials.is_ct, 2, 1))
    monkeypatch.setenv("RL_LADDER_DATA", str(data))
    monkeypatch.setenv("RL_LADDER_RESULTS", str(tmp_path/"results"))
    exp = {"task": "cv", "models": "M1,A+C", "datasets": ["SYN"],
           "overrides": {"evaluate.cv.method": "chrono", "fit.n_starts": 2,
                         "evaluate.cv.train_frac": .5, "collect.contrasts": [["A+C", "M1"]]}}
    return make_plan(exp, name="cv_only")


def test_cv_only_pipeline_and_download_audit(cv_plan, monkeypatch):
    from rl_ladder import tasks
    path, plan = cv_plan
    monkeypatch.setattr(tasks, "get_or_fit", lambda *a, **k: pytest.fail("CV must not fit full data"))
    counts = run_local(path, quiet=True)
    assert counts["failed"] == 0
    assert collect(path, plots=False) == 0
    assert verify_plan(path) == 0
    assert not (run_dir(plan).parents[1]/"cache"/"fits").exists()
    assert not list(run_dir(plan).glob("*/fit"))
    scores = pd.read_csv(run_dir(plan)/"summary/cv_scores.csv")
    assert scores.verified.all() and scores.training_converged.all()
    assert "bic" not in scores
    assert run_local(path, quiet=True)["cached"] == 2
    original = unit_output_path(plan, plan["units"][0])
    r = read_json(original)
    r["result"]["cv_test_nll"] += 1
    atomic_write_json(original, r)
    assert verify_plan(path) == 1
    assert collect(path, plots=False) == 1
    assert not list((run_dir(plan)/"summary").glob("cv_wide__*.csv"))


def test_record_enhancement_matches_old_numerics(trials):
    from rl_ladder.fit.mle import fit_mle
    from rl_ladder.models import kernel
    spec = context().models["A+C"]
    cfg = {"n_starts": 2, "maxiter": 300}
    reset = reset_flags([])
    train = trials.session < 7
    old = fit_mle(spec, trials, train.astype(float), reset, cfg, 5)
    new = evaluate.chrono_cv(spec, trials, np.ones(trials.n), reset, cfg, 5, .5)
    assert old["theta"] == new["cv_record"]["training_fit"]["theta"]
    expected = kernel.nll(spec, np.array(old["theta"]), trials, (~train).astype(float), reset)
    assert new["cv_test_nll"] == expected


def test_unconverged_but_reproducible_fit_fails_quality_gate(cv_plan):
    path, plan = cv_plan
    assert run_local(path, quiet=True)["failed"] == 0
    target = unit_output_path(plan, plan["units"][0])
    outer = read_json(target)
    r = outer["result"]
    fit = r["cv_record"]["training_fit"]
    fit["success"] = r["cv_success"] = False
    fit["start_records"][fit["selected_start_index"]]["success"] = False
    fit["n_failed"] = sum(not x["success"] for x in fit["start_records"])
    atomic_write_json(target, outer)
    assert verify_plan(path) == 1
    report = read_json(run_dir(plan)/"summary/cv_verification.json")
    assert report["n_verified"] == 2 and report["n_training_converged"] == 1
    assert collect(path, plots=False) == 1


def test_full_audit_plan_has_same_456_comparisons(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_LADDER_RESULTS", str(tmp_path))
    path, plan = make_plan("feedback_cv_audit", name="audit_count")
    assert plan["task"] == "cv" and plan["n_units"] == 456
    assert set(u["cfg"]["fit"]["n_starts"] for u in plan["units"]) == {20}
    assert all(u["wave"] == 0 for u in plan["units"])
    assert all(u["seed"] == seed_for(u["cfg"]["fit"]["seed"], u["model"], u["mouse"])+7
               for u in plan["units"])
    assert len(load_yaml(experiment_path("feedback_cv_audit"))["overrides"]["collect.contrasts"]) == 12
