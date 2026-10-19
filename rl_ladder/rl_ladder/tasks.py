"""Experiments -> plans -> units.

An experiment file (config/experiments/*.yml) names a task, the models, the
datasets, fixed overrides and an optional grid of overrides. Expanding it
gives a *plan*: a JSON list of independent units, each with its fully
resolved configuration. A unit is the smallest piece of work and is what one
local worker or one Slurm array index runs:

    fit             (model, mouse)
    ppc             (model, mouse)
    param_recovery  (model, replicate)  template mouse = mice[rep % n_mice]
    model_recovery  (generating model, replicate)  fits every candidate
"""

import itertools
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from .config import (apply_overrides, experiment_path, load_default, load_yaml, results_root,
                     stable_hash)
from .data import load_mouse, resolve_mice
from .models.registry import reset_flags, resolve_models
from .service import atomic_write_json, context, get_or_fit, read_json, seed_for

TASKS = ("fit", "ppc", "param_recovery", "model_recovery")


def _label(value):
    if isinstance(value, dict):
        return "+".join(f"{k}{v}" for k, v in value.items())
    if isinstance(value, list):
        return "+".join(map(str, value)) or "none"
    return str(value)


def _variant_name(dataset, grid_point):
    """Readable folder name, e.g. paper19__correction=mask__session_reset=wsls+choice."""
    parts = [dataset.replace(",", "-")]
    parts += [f"{k.split('.')[-1]}={_label(v)}" for k, v in grid_point.items()]
    name = re.sub(r"[^A-Za-z0-9=+._-]", "_", "__".join(parts))
    return name if len(name) <= 80 else f"{parts[0]}-{stable_hash(grid_point, 8)}"


def expand(ctx, exp, cli_overrides, name):
    task = exp.get("task", "fit")
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, not {task!r}")
    models = resolve_models(exp["models"], ctx.models, ctx.sets)
    datasets = exp.get("datasets") or []
    if isinstance(datasets, str):
        datasets = [datasets]
    if not datasets:
        raise ValueError("experiment needs at least one dataset")
    grid = exp.get("grid") or {}
    keys = list(grid)
    base = load_default()
    units = []
    for combo in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
        grid_point = dict(zip(keys, combo))
        for dataset in datasets:
            mice, preset = resolve_mice(dataset, ctx.datasets)
            cfg = apply_overrides(base, preset)
            cfg = apply_overrides(cfg, exp.get("overrides") or {})
            cfg = apply_overrides(cfg, grid_point)
            cfg = apply_overrides(cfg, cli_overrides)
            reset_flags(cfg["model"]["session_reset"])  # fail early on typos
            variant = _variant_name(dataset, grid_point)
            label = ", ".join([f"dataset={dataset}"] +
                              [f"{k.split('.')[-1]}={_label(v)}" for k, v in grid_point.items()])
            common = {"task": task, "variant": variant, "label": label, "cfg": cfg}
            if task in ("fit", "ppc"):
                for model in models:
                    for mouse in mice:
                        units.append({**common, "id": f"{ctx.models[model].slug}__{mouse}",
                                      "model": model, "mouse": mouse,
                                      "wave": ctx.models[model].depth})
            else:
                n_reps = int(cfg["validate"]["n_reps"])
                for model in models:
                    for rep in range(n_reps):
                        unit = {**common, "id": f"{ctx.models[model].slug}__rep{rep:03d}",
                                "model": model, "rep": rep, "template": mice[rep % len(mice)],
                                "wave": 0}
                        if task == "model_recovery":
                            unit["candidates"] = models
                        units.append(unit)
    for u in units:
        u["seed"] = seed_for(u["cfg"]["fit"]["seed"], name, u["variant"], u["task"], u["id"])
        u["key"] = stable_hash({k: v for k, v in u.items() if k != "key"} |
                               {"code": ctx.code_version}, n=16)
    units.sort(key=lambda u: u["wave"])  # nested models first, so warm starts hit the cache
    return units


def make_plan(experiment, cli_overrides=None, name=None, out=None):
    ctx = context()
    if isinstance(experiment, dict):
        exp, name = experiment, name or experiment.get("name", "adhoc")
    else:
        exp, name = load_yaml(experiment_path(experiment)), name or Path(experiment).stem
    units = expand(ctx, exp, cli_overrides or {}, name)
    if not units:
        raise ValueError("experiment expanded to zero units")
    plan = {"name": name, "task": units[0]["task"], "created": datetime.now().isoformat(timespec="seconds"),
            "code_version": ctx.code_version, "experiment": exp,
            "cli_overrides": cli_overrides or {}, "n_units": len(units), "units": units}
    path = Path(out) if out else run_dir(plan) / "plan.json"
    atomic_write_json(path, plan)
    return path, plan


def run_dir(plan):
    """Resolved at run time, so a plan made on one machine works on another."""
    return results_root(plan["units"][0]["cfg"]) / "runs" / plan["name"]


def unit_output_path(plan, unit):
    return run_dir(plan) / unit["variant"] / unit["task"] / f"{unit['id']}.json"


def runtime_key(unit):
    """Unit key plus the checksum of the data file it reads."""
    mouse = unit.get("mouse") or unit.get("template")
    return stable_hash([unit["key"], load_mouse(mouse, unit["cfg"]).meta["sha256"]], n=16)


def run_unit(plan, index):
    """Run one unit; skip it if an output with the same key already exists."""
    from .ppc import run_ppc
    from .validate import run_model_recovery, run_param_recovery

    ctx = context()
    unit = plan["units"][index]
    path = unit_output_path(plan, unit)
    key = runtime_key(unit)
    if path.exists() and read_json(path).get("key") == key:
        return "cached", 0.0
    tic = time.perf_counter()
    try:
        if unit["task"] == "fit":
            result = get_or_fit(ctx, unit["model"], unit["mouse"], unit["cfg"])
        elif unit["task"] == "ppc":
            result = run_ppc(ctx, unit)
        elif unit["task"] == "param_recovery":
            result = run_param_recovery(ctx, unit)
        else:
            result = run_model_recovery(ctx, unit)
    except Exception:
        fail = run_dir(plan) / "failures" / f"{unit['variant']}__{unit['task']}__{unit['id']}.json"
        atomic_write_json(fail, {"unit": {k: v for k, v in unit.items() if k != "cfg"},
                                 "traceback": traceback.format_exc()})
        raise
    out = {"key": key, "id": unit["id"], "task": unit["task"],
           "variant": unit["variant"], "label": unit["label"],
           "seconds": round(time.perf_counter() - tic, 3), "result": result}
    for field in ("model", "mouse", "rep", "template"):
        if field in unit:
            out[field] = unit[field]
    atomic_write_json(path, out)
    return "done", out["seconds"]
