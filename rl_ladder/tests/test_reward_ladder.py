"""Reward-module completeness and training-only nested CV initialization."""

import json

import numpy as np
import pytest

from rl_ladder.config import apply_overrides, load_default
from rl_ladder.data import Trials
from rl_ladder.evaluate import verify_cv_record
from rl_ladder.models import kernel
from rl_ladder.models.registry import parents, reset_flags
from rl_ladder.service import context, get_cv
from rl_ladder.tasks import make_plan

VALUES = {"alpha": .15, "beta_q": 3., "q0": .1, "b": -.2,
          "alpha_v": .2, "beta_v": 1.7, "alpha_c": .35, "beta_c": .8,
          "alpha_f": .25, "beta_f": 1.1}


@pytest.mark.parametrize("resets", [["wsls", "feedback"], ["wsls", "choice", "feedback"]])
def test_combined_model_matches_independent_hand_update(resets):
    s, c = np.array([1,0,1,0,0,1]), np.array([1,1,0,0,1,0])
    sessions = np.repeat([1,2],3)
    trials = Trials("hand", s, c, (s==c).astype(int), sessions, np.zeros(6,dtype=bool))
    q = np.array([[.5-VALUES["q0"]/2,.5+VALUES["q0"]/2]]*2)
    v, k, h = np.array([.5,.5]), 0., 0.
    expected = []
    for i,(stim,choice,reward) in enumerate(zip(s,c,trials.reward)):
        if i and trials.new_session[i]:
            if "choice" in resets:k=0.
            if "feedback" in resets:h=0.
        z = VALUES["beta_q"]*(q[stim,1]-q[stim,0])+VALUES["b"]
        z += VALUES["beta_v"]*(v[1]-v[0])+VALUES["beta_c"]*k+VALUES["beta_f"]*h
        expected.append(1/(1+np.exp(-z)))
        q[stim,choice] += VALUES["alpha"]*(reward-q[stim,choice])
        v[choice] += VALUES["alpha_v"]*(reward-v[choice])
        k += VALUES["alpha_c"]*((2*choice-1)-k)
        h += VALUES["alpha_f"]*((2*choice-1)*(2*reward-1)-h)
    spec = context().models["S+R+F+C"]
    _,prob,_ = kernel.score(spec,spec.theta(VALUES),trials,np.array([0.,1,1,0,1,1]),reset_flags(resets))
    np.testing.assert_allclose(prob,expected,rtol=0,atol=1e-14)


def test_r_and_f_are_distinct_even_in_deterministic_task():
    s=np.ones(3,dtype=int)
    a,b=np.array([1,1,0]),np.array([1,0,0])
    ta=Trials("a",s,a,(s==a).astype(int),np.ones(3,dtype=int),np.zeros(3,dtype=bool))
    tb=Trials("b",s,b,(s==b).astype(int),np.ones(3,dtype=int),np.zeros(3,dtype=bool))
    def prediction(name,trials):
        spec=context().models[name]
        return kernel.score(spec,spec.theta({"b":0,"alpha_v":.2,"beta_v":2,
            "alpha_f":.2,"beta_f":2}),trials,np.ones(3),reset_flags([]))[1]
    assert prediction("V",ta)[2] != prediction("V",tb)[2]
    np.testing.assert_array_equal(prediction("F",ta),prediction("F",tb))


@pytest.mark.parametrize("child,parent",[("S+R","S0"),("S+R+C","S+C"),
    ("S+R+F","S+F"),("S+R+F+C","S+F+C")])
def test_zero_r_weight_recovers_matched_control(child,parent,trials):
    v={**VALUES,"beta_v":0}
    ctx=context();lik=np.ones(trials.n);reset=reset_flags(["wsls","feedback"])
    a=kernel.score(ctx.models[child],ctx.models[child].theta(v),trials,lik,reset)
    b=kernel.score(ctx.models[parent],ctx.models[parent].theta(v),trials,lik,reset)
    assert a[0]==b[0]
    np.testing.assert_array_equal(a[1],b[1])


@pytest.mark.parametrize("experiment", ["reward_ladder", "reward_ladder_fixed"])
def test_plan_contains_every_s_combination_and_all_matched_contrasts(tmp_path,monkeypatch,experiment):
    monkeypatch.setenv("RL_LADDER_RESULTS",str(tmp_path))
    _,plan=make_plan(experiment,name="all_s")
    assert plan["n_units"]==304 and plan["task"]=="fit"
    ctx=context();models=set(u["model"] for u in plan["units"])
    expected="S_fixed" if experiment.endswith("fixed") else "S_extended"
    assert models==set(ctx.sets[expected]) and len(models)==8
    assert {tuple(getattr(ctx.models[m],s) for s in ("action","choice","feedback")) for m in models} == {
        (a,c,f) for a in ("none","V") for c in ("none","ck") for f in ("none","cf")}
    assert set(ctx.sets["reward_extension"])<=set(ctx.sets["core"])
    assert len(plan["units"][0]["cfg"]["collect"]["contrasts"])==12
    assert all(a in models and b in models for a,b in plan["units"][0]["cfg"]["collect"]["contrasts"])
    assert all(u["cfg"]["evaluate"]["cv"]["warm_start_from_nested"] for u in plan["units"])


def test_nested_cv_reuses_training_only_and_ignores_held_out_choices(tmp_path,monkeypatch):
    from rl_ladder import service, evaluate
    from rl_ladder.data import load_mouse
    from rl_ladder.preprocess import preprocess
    rng=np.random.default_rng(28)
    stim=rng.integers(0,2,160);choice=rng.integers(0,2,160);session=np.repeat([1,2,3,4],40)
    for dirname,changed in [("one",False),("two",True)]:
        folder=tmp_path/dirname;folder.mkdir()
        c=choice.copy()
        if changed:c[session>2]=1-c[session>2]
        np.savez(folder/"SYN.npz",answer=stim,choice=c,reward=(stim==c).astype(int),
                 session=session,trial_type=np.ones(160,dtype=int))
    cfg=apply_overrides(load_default(),{"paths.data":str(tmp_path/"one"),"paths.results":str(tmp_path/"results"),
        "evaluate.cv.method":"chrono","evaluate.cv.train_frac":.5,
        "evaluate.cv.warm_start_from_nested":True,"fit.n_starts":2,"fit.maxiter":800})
    def no_full_fit(*args,**kwargs):pytest.fail("Nested CV must never read or fit full-data optima")
    monkeypatch.setattr(service,"get_or_fit",no_full_fit)
    spec=context().models["S+R+F+C"]
    first=get_cv(context(),spec.name,"SYN",cfg)
    second=get_cv(context(),spec.name,"SYN",apply_overrides(cfg,{"paths.data":str(tmp_path/"two")}))
    assert first["cv_record"]["training_fit"]==second["cv_record"]["training_fit"]
    assert first["cv_test_nll"]!=second["cv_test_nll"]
    rec=first["cv_record"]
    assert set(rec["training_parent_nll"])==set(parents(spec,context().models))
    assert rec["training_fit"]["nll"]<=min(rec["training_parent_nll"].values())+1e-4
    assert rec["training_fit"]["n_starts"]==2+len(rec["training_parent_nll"])
    # JSON key order must not change which initial vector represents a parent.
    reordered=json.loads(json.dumps(first,sort_keys=True))
    trials,lik=preprocess(load_mouse("SYN",cfg),cfg)
    assert verify_cv_record(spec,trials,lik,reset_flags(cfg["model"]["session_reset"]),reordered)["passed"]
    monkeypatch.setattr(evaluate,"fit_mle",lambda *a,**k:pytest.fail("Cached CV was refitted"))
    assert get_cv(context(),spec.name,"SYN",cfg)==first
    assert not list((tmp_path/"results/cache").glob("fits/**/*.json"))


def test_reward_collection_rejects_failed_full_fit(tmp_path,monkeypatch):
    from rl_ladder.collect import collect
    from rl_ladder.service import atomic_write_json, read_json
    from rl_ladder.tasks import run_unit, unit_output_path, run_dir
    data=tmp_path/"data";data.mkdir()
    choice=np.tile([0,1],30)
    np.savez(data/"SYN.npz",answer=choice,choice=choice,reward=np.ones(60,dtype=int),
             session=np.repeat([1,2,3],20),trial_type=np.ones(60,dtype=int))
    monkeypatch.setenv("RL_LADDER_DATA",str(data))
    monkeypatch.setenv("RL_LADDER_RESULTS",str(tmp_path/"results"))
    exp={"task":"fit","models":"M1","datasets":["SYN"],
         "overrides":{"collect.require_fit_success":True,"fit.n_starts":2}}
    path,plan=make_plan(exp,name="quality")
    run_unit(plan,0)
    unit_path=unit_output_path(plan,plan["units"][0])
    output=read_json(unit_path);output["result"]["success"]=False
    atomic_write_json(unit_path,output)
    assert collect(path,plots=False)==1
    assert not read_json(run_dir(plan)/"summary/fit_quality.json")["passed_present_units"]
