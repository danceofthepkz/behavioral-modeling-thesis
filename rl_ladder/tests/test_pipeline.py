"""End to end on synthetic mice: plan -> run -> collect for every task."""

import numpy as np
import pytest

from rl_ladder.collect import collect
from rl_ladder.models import kernel
from rl_ladder.models.registry import reset_flags
from rl_ladder.runners import run_local
from rl_ladder.service import context
from rl_ladder.tasks import make_plan


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    spec = context().models["S+C"]
    values = {"alpha": 0.05, "beta_q": 6.0, "b": 0.2, "q0": 0.0, "alpha_c": 0.1, "beta_c": 1.0}
    for i, mouse in enumerate(["SYN1", "SYN2"]):
        session = np.repeat(np.arange(1, 6), 120)
        new_session = np.r_[True, session[1:] != session[:-1]]
        choice, stim, is_ct, _ = kernel.simulate_arrays(
            spec, spec.theta(values), None, new_session, reset_flags(["wsls"]), "reactive", 3,
            np.random.default_rng(i))
        np.savez(data / f"{mouse}.npz", answer=stim, choice=choice,
                 reward=(choice == stim).astype(np.int64), session=session,
                 trial_type=np.where(is_ct, 2, 1))
    monkeypatch.setenv("RL_LADDER_DATA", str(data))
    monkeypatch.setenv("RL_LADDER_RESULTS", str(tmp_path / "results"))
    return tmp_path


SMALL = {"fit.n_starts": 2, "validate.n_reps": 2, "ppc.n_sims": 3, "ppc.block": 50,
         "validate.accuracy_range": [0.0, 1.0]}


@pytest.mark.parametrize("task,models", [("fit", "M1,S0,S+C,A0,B0,S+R+F+C,B+F+C"),
                                         ("param_recovery", "S+Ff"),
                                         ("model_recovery", "M1,C,F,F+C"),
                                         ("ppc", "S+F+C")])
def test_task_end_to_end(env, task, models):
    exp = {"task": task, "models": models, "datasets": ["SYN1,SYN2"],
           "overrides": {"collect.contrasts": [["S+C", "S0"]]} if task == "fit" else {}}
    path, plan = make_plan(exp, SMALL, name=f"test_{task}")
    counts = run_local(path, n_jobs=1, quiet=True)
    assert counts["failed"] == 0
    assert collect(path, plots=False) == 0
    again = run_local(path, n_jobs=1, quiet=True)
    assert again["cached"] == plan["n_units"]  # second run is fully cached


def test_children_never_worse_than_parents(env):
    path, plan = make_plan({"task": "fit", "models": "S", "datasets": ["SYN1"]}, SMALL,
                           name="nesting")
    run_local(path, n_jobs=1, quiet=True)
    from rl_ladder.service import get_or_fit
    for unit in plan["units"]:
        fit = get_or_fit(context(), unit["model"], unit["mouse"], unit["cfg"])
        for parent, nll in fit["parent_nll"].items():
            assert fit["nll"] <= nll + 1e-6, (unit["model"], parent)


def test_old_plan_cannot_resume_with_new_scientific_code(env):
    from rl_ladder.tasks import run_unit
    _, plan = make_plan({"task": "fit", "models": "M1", "datasets": ["SYN1"]}, SMALL,
                        name="old_code")
    plan["code_version"] = "previous-version"
    with pytest.raises(ValueError, match="different code version"):
        run_unit(plan, 0)


def test_old_run_manifest_is_not_overwritten(env):
    from rl_ladder.service import atomic_write_json, read_json
    exp = {"task": "fit", "models": "M1", "datasets": ["SYN1"]}
    path, plan = make_plan(exp, SMALL, name="keep_archive")
    plan["code_version"] = "previous-version"
    atomic_write_json(path, plan)
    with pytest.raises(ValueError, match="preserve archived results"):
        make_plan(exp, SMALL, name="keep_archive")
    assert read_json(path)["code_version"] == "previous-version"
