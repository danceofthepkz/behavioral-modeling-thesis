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

import numpy as np

from .data import Trials, load_mouse
from .fit.mle import fit_mle
from .models import kernel
from .models import params as P
from .models.registry import parents, reset_flags
from .preprocess import apply_window, likelihood_weights
from .service import get_or_fit, warm_start_from


def template_trials(mouse, cfg):
    return apply_window(load_mouse(mouse, cfg), cfg["preprocess"]["window"])


def simulate_trials(spec, values, template, env, max_ct, reset, rng):
    choice, stim, is_ct, _ = kernel.simulate_arrays(
        spec, spec.theta(values), template.stim, template.new_session, reset, env, max_ct, rng)
    if env == "replay":
        is_ct = template.is_ct.copy()
    return Trials(f"sim:{template.mouse}", stim, choice, (choice == stim).astype(np.int64),
                  template.session.copy(), is_ct, "simulated")


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
                             unit["seed"] + 1 + candidates.index(name), warm)
        fits[name]["bic"] = float(2 * fits[name]["nll"] + spec.k * np.log(fits[name]["n_lik"]))
    ranked = sorted(fits, key=lambda m: fits[m]["bic"])
    return {
        "generating": gen.name, "true": values, "nll_true": float(nll_true),
        "accuracy": acc, "tries": tries, "winner": ranked[0],
        "delta_bic_runner_up": fits[ranked[1]]["bic"] - fits[ranked[0]]["bic"]
        if len(ranked) > 1 else None,
        "candidates": {m: {"nll": f["nll"], "bic": f["bic"], "k": f["k"], "params": f["params"],
                           "at_bound": f["at_bound"]} for m, f in fits.items()},
    }
