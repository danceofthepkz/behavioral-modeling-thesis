"""Validate stage: parameter recovery and model recovery (Wilson & Collins 2019).

Synthetic datasets copy a real mouse's session structure ("template"):
    validate.env = reactive   stimuli are redrawn by the task rule, with
                              correction trials after errors (max_ct)
    validate.env = replay     the template mouse's real stimulus sequence
Generating parameters come from each parameter's prior (`param_source: prior`,
rejecting datasets whose accuracy falls outside `accuracy_range`) or from the
template mouse's own fit (`param_source: fitted`). Models without a stimulus
slot (M1, M2, ...) cannot learn the task, so the accuracy filter skips them.
"""

from copy import deepcopy

import numpy as np

from .data import Trials, load_mouse
from .fit.mle import fit_mle
from .models import kernel
from .models import params as P
from .models.registry import parents, reset_flags
from .preprocess import apply_window, likelihood_weights
from .provenance import environment_record, trial_signature
from .service import get_or_fit, warm_start_from


def template_trials(mouse, cfg):
    return apply_window(load_mouse(mouse, cfg), cfg["preprocess"]["window"])


def simulate_trials(spec, values, template, env, max_ct, reset, rng):
    # Save the state immediately before simulation, after any parameter draws.
    # A seed alone cannot replay rejected draws or fitted-parameter runs.
    generation_state = deepcopy(rng.bit_generator.state)
    choice, stim, is_ct, _ = kernel.simulate_arrays(
        spec, spec.theta(values), template.stim, template.new_session, reset, env, max_ct, rng)
    if env == "replay":
        is_ct = template.is_ct.copy()
    return Trials(f"sim:{template.mouse}", stim, choice, (choice == stim).astype(np.int64),
                  template.session.copy(), is_ct, "simulated",
                  {"generation_state": generation_state,
                   "template_signature": trial_signature(template, np.ones(template.n))})


def prepare_synthetic(sim, cfg):
    correction = cfg["preprocess"]["correction"]
    if correction == "drop":
        sim = sim.subset(~sim.is_ct)
    return sim, likelihood_weights(sim, correction)


def draw_dataset(ctx, spec, cfg, template_mouse, rng):
    v = cfg["validate"]
    template = template_trials(template_mouse, cfg)
    reset = reset_flags(cfg["model"]["session_reset"])
    lo, hi = v["accuracy_range"]
    for attempt in range(1, int(v["max_tries"]) + 1):
        if v["param_source"] == "prior":
            values = {p.name: P.sample_prior(p, spec.stimulus, rng) for p in spec.free_specs}
        elif v["param_source"] == "fitted":
            values = dict(get_or_fit(ctx, spec.name, template_mouse, cfg)["params"])
        else:
            raise ValueError("validate.param_source must be prior | fitted")
        sim = simulate_trials(spec, values, template, v["env"], v["max_ct"], reset, rng)
        acc = float(sim.reward[~sim.is_ct].mean())
        if lo <= acc <= hi or v["param_source"] == "fitted" or spec.stimulus == "none":
            return values, sim, acc, attempt
    raise RuntimeError(f"{spec.name}: no dataset with accuracy in [{lo}, {hi}] after "
                       f"{v['max_tries']} draws; widen validate.accuracy_range or the priors")


def run_param_recovery(ctx, unit):
    cfg = unit["cfg"]
    spec = ctx.models[unit["model"]]
    rng = np.random.default_rng(unit["seed"])
    values, sim, acc, tries = draw_dataset(ctx, spec, cfg, unit["template"], rng)
    trials, lik = prepare_synthetic(sim, cfg)
    reset = reset_flags(cfg["model"]["session_reset"])
    nll_true = kernel.nll(spec, spec.theta(values), trials, lik, reset)
    fit = fit_mle(spec, trials, lik, reset, cfg["fit"], unit["seed"] + 1)
    return {
        "true": values, "fit": fit["params"], "nll_true": float(nll_true), "nll_fit": fit["nll"],
        # The optimiser should never end worse than the generating parameters.
        "optimizer_miss": bool(fit["nll"] > nll_true + 1e-4),
        "accuracy": acc, "tries": tries, "n_trials": trials.n, "n_lik": fit["n_lik"],
        "at_bound": fit["at_bound"], "n_repeated": fit["n_repeated"],
    }


def run_model_recovery(ctx, unit):
    cfg = unit["cfg"]
    gen = ctx.models[unit["model"]]
    rng = np.random.default_rng(unit["seed"])
    values, sim, acc, tries = draw_dataset(ctx, gen, cfg, unit["template"], rng)
    trials, lik = prepare_synthetic(sim, cfg)
    reset = reset_flags(cfg["model"]["session_reset"])
    nll_true = kernel.nll(gen, gen.theta(values), trials, lik, reset)
    fits = {}
    candidates = sorted(unit["candidates"], key=lambda m: ctx.models[m].depth)
    for name in candidates:
        spec = ctx.models[name]
        warm = [warm_start_from(spec, fits[p]["params"])
                for p in parents(spec, ctx.models) if p in fits]
        fits[name] = fit_mle(spec, trials, lik, reset, cfg["fit"],
                             unit["seed"] + 1 + candidates.index(name), warm,
                             record_starts=True)
        # Keep start diagnostics compact. Selected parameters are sufficient to
        # replay the reported score; all-start parameter replay is a CV feature.
        for start in fits[name]["start_records"]:
            start.pop("initial_theta")
            start.pop("theta")
        fits[name]["bic"] = float(2 * fits[name]["nll"] + spec.k * np.log(fits[name]["n_lik"]))
    ranked = sorted(fits, key=lambda m: fits[m]["bic"])
    return {
        "audit_version": 1, "code_version": ctx.code_version,
        "environment": environment_record(), "simulation": sim.meta,
        "input_signature": trial_signature(trials, lik),
        "n_trials": trials.n, "n_lik": float(lik.sum()),
        "generating": gen.name, "true": values, "nll_true": float(nll_true),
        "accuracy": acc, "tries": tries, "winner": ranked[0],
        "delta_bic_runner_up": fits[ranked[1]]["bic"] - fits[ranked[0]]["bic"]
        if len(ranked) > 1 else None,
        "candidates": fits,
    }


def audit_model_recovery(ctx, unit, result, atol=1e-6):
    """Replay one simulation and all selected candidate scores, without fitting.

    Score reproducibility, convergence, and optimisation checks are separate.
    A finite, reproducible likelihood does not imply a successful optimisation.
    """
    report = {"verified": False, "all_converged": False, "optimizer_miss": False,
              "nested_violations": [], "max_score_error": None, "error": ""}
    try:
        if result.get("audit_version") != 1:
            raise ValueError("Legacy recovery record has no replay/convergence audit")
        if result["code_version"] != ctx.code_version:
            raise ValueError("Recovery scientific code version differs from this checkout")
        cfg = unit["cfg"]
        gen = ctx.models[unit["model"]]
        if result["generating"] != gen.name:
            raise ValueError("Generating model differs from plan")
        template = template_trials(unit["template"], cfg)
        if trial_signature(template, np.ones(template.n)) != result["simulation"]["template_signature"]:
            raise ValueError("Template data or preprocessing changed")
        rng = np.random.default_rng()
        rng.bit_generator.state = result["simulation"]["generation_state"]
        reset = reset_flags(cfg["model"]["session_reset"])
        sim = simulate_trials(gen, result["true"], template, cfg["validate"]["env"],
                              cfg["validate"]["max_ct"], reset, rng)
        trials, lik = prepare_synthetic(sim, cfg)
        if trial_signature(trials, lik) != result["input_signature"]:
            raise ValueError("Simulated trials do not reproduce the saved input signature")
        if trials.n != result["n_trials"] or float(lik.sum()) != result["n_lik"]:
            raise ValueError("Trial counts differ from saved record")
        fits = result["candidates"]
        if set(fits) != set(unit["candidates"]):
            raise ValueError("Candidate set differs from plan")
        true_nll = float(kernel.nll(gen, gen.theta(result["true"]), trials, lik, reset))
        errors = [abs(true_nll - result["nll_true"])]
        for name, fit in fits.items():
            spec = ctx.models[name]
            if set(fit["params"]) != set(spec.free):
                raise ValueError(f"{name}: parameter names differ from model specification")
            for p in spec.free_specs:
                value = fit["params"][p.name]
                if not np.isfinite(value) or not p.lower - 1e-10 <= value <= p.upper + 1e-10:
                    raise ValueError(f"{name}: {p.name} is outside its parameter bounds")
            score = float(kernel.nll(spec, spec.theta(fit["params"]), trials, lik, reset))
            bic = 2 * score + spec.k * np.log(lik.sum())
            if fit["k"] != spec.k or fit["n_lik"] != float(lik.sum()):
                raise ValueError(f"{name}: parameter or likelihood count mismatch")
            errors += [abs(score - fit["nll"]), abs(bic - fit["bic"])]
            for parent in parents(spec, ctx.models):
                if parent in fits and fit["nll"] > fits[parent]["nll"] + 1e-4:
                    report["nested_violations"].append({"child": name, "parent": parent,
                        "excess_nll": fit["nll"] - fits[parent]["nll"]})
        report["max_score_error"] = float(max(errors))
        if not np.all(np.isfinite(errors)) or max(errors) > atol:
            raise ValueError("Saved NLL or BIC does not reproduce within tolerance")
        ranked = sorted(fits, key=lambda name: fits[name]["bic"])
        if result["winner"] != ranked[0]:
            raise ValueError("Saved BIC winner differs from candidate scores")
        if len(ranked) > 1 and abs(result["delta_bic_runner_up"] -
                (fits[ranked[1]]["bic"] - fits[ranked[0]]["bic"])) > atol:
            raise ValueError("Saved BIC margin differs from candidate scores")
        report["all_converged"] = all(f["success"] for f in fits.values())
        report["optimizer_miss"] = bool(fits[gen.name]["nll"] > true_nll + 1e-4)
        report["verified"] = True
    except (KeyError, TypeError, ValueError) as exc:
        report["error"] = str(exc)
    report["passed"] = bool(report["verified"] and report["all_converged"]
                            and not report["optimizer_miss"] and not report["nested_violations"])
    return report
