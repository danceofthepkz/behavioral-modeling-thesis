"""Command line: python -m rl_ladder <command> ...

    run MODEL DATASET [opts] [--stage.key=value ...]   ad-hoc run (PIDSMaker style)
    experiment NAME [opts] [--stage.key=value ...]     run config/experiments/NAME.yml
    plan NAME [--out PATH] [--stage.key=value ...]     write a plan for Slurm
    unit PLAN INDEX [--chunk-size K]                    run one Slurm array index
    count PLAN [--chunk-size K]                         number of array indices
    collect PLAN|NAME [--allow-partial] [--no-plots]    summary tables
    verify-cv PLAN|NAME                               replay saved scores, no fitting
    figures [--fit-run R] [--recovery-run R] [--n-sims N]  summary figures
    list [models|sets|datasets|experiments|params]
    show MODEL                                          free parameters of one model
"""

import argparse
import sys
from pathlib import Path

from . import __version__


def _split(argv):
    """Separate --stage.key=value overrides from ordinary options."""
    known, overrides, i = [], [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and "." in tok.split("=")[0]:
            overrides.append(tok)
            if "=" not in tok and i + 1 < len(argv):
                overrides.append(argv[i + 1])
                i += 1
        else:
            known.append(tok)
        i += 1
    return known, overrides


def _parser():
    p = argparse.ArgumentParser(prog="rl_ladder", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    def execution(sp):
        sp.add_argument("--n-jobs", type=int, default=1, help="local worker processes")
        sp.add_argument("--dry-run", action="store_true", help="write the plan, run nothing")
        sp.add_argument("--no-collect", action="store_true")

    r = sub.add_parser("run", help="ad-hoc: one model (or set) on one dataset (or mouse)")
    r.add_argument("model", help="model, comma list, or set (e.g. B0, S+F+C, feedback)")
    r.add_argument("dataset", help="dataset preset, mouse ID, or comma list (e.g. paper19, GS027)")
    r.add_argument("--task", default="fit", choices=["fit", "cv", "ppc", "param_recovery",
                                                     "model_recovery"])
    r.add_argument("--name", help="run name (default derived from the arguments)")
    execution(r)

    e = sub.add_parser("experiment", help="run config/experiments/NAME.yml locally")
    e.add_argument("name")
    e.add_argument("--name-suffix", default="", help="append to the run name")
    execution(e)

    pl = sub.add_parser("plan", help="expand an experiment into a plan file")
    pl.add_argument("name")
    pl.add_argument("--out")
    pl.add_argument("--name-suffix", default="")

    u = sub.add_parser("unit", help="run one Slurm array index of a plan")
    u.add_argument("plan")
    u.add_argument("index", type=int)
    u.add_argument("--chunk-size", type=int, default=1)

    c = sub.add_parser("count", help="number of array indices for a plan")
    c.add_argument("plan")
    c.add_argument("--chunk-size", type=int, default=1)

    co = sub.add_parser("collect", help="summarise a finished plan")
    co.add_argument("plan", help="plan.json path or run name")
    co.add_argument("--allow-partial", action="store_true")
    co.add_argument("--no-plots", action="store_true")

    vc = sub.add_parser("verify-cv", help="recalculate archived CV scores without refitting")
    vc.add_argument("plan", help="plan.json path or run name")

    vr = sub.add_parser("verify-recovery", help="replay simulations, scores and convergence checks")
    vr.add_argument("plan", help="plan.json path or run name")

    fg = sub.add_parser("figures", help="summary figures for a finished fit run")
    fg.add_argument("--fit-run", default="core_ladder")
    fg.add_argument("--variant", help="variant folder (default: the whole-learning-period one)")
    fg.add_argument("--recovery-run", default="model_recovery")
    fg.add_argument("--dataset", default="paper19")
    fg.add_argument("--groups", default="paper_clusters")
    fg.add_argument("--n-sims", type=int, default=50)
    fg.add_argument("--out")

    ls = sub.add_parser("list", help="list registered things")
    ls.add_argument("what", nargs="?", default="models",
                    choices=["models", "sets", "datasets", "experiments", "params"])

    sh = sub.add_parser("show", help="show one model")
    sh.add_argument("model")
    return p


def _plan_path(arg, overrides=None):
    p = Path(arg)
    if p.suffix == ".json":
        return p
    from .config import apply_overrides, load_default, results_root
    return results_root(apply_overrides(load_default(), overrides or {})) / "runs" / arg / "plan.json"


def _execute(plan_path, args):
    from .collect import collect
    from .runners import run_local
    from .service import read_json
    plan = read_json(plan_path)
    print(f"Plan {plan['name']}: {plan['n_units']} {plan['task']} units -> {plan_path}")
    if args.dry_run:
        return 0
    counts = run_local(plan_path, n_jobs=args.n_jobs)
    if args.no_collect:
        return 1 if counts["failed"] else 0
    return collect(plan_path) or (1 if counts["failed"] else 0)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    known, raw_overrides = _split(argv)
    args = _parser().parse_args(known)
    from .config import parse_cli_overrides
    overrides = parse_cli_overrides(raw_overrides)

    if args.command == "run":
        from .config import stable_hash
        from .tasks import make_plan
        name = args.name or "run_{}_{}_{}{}".format(
            args.task, args.model.replace("+", "_").replace(",", "-"),
            args.dataset.replace(",", "-"), f"_{stable_hash(overrides, 6)}" if overrides else "")
        exp = {"task": args.task, "models": args.model, "datasets": [args.dataset]}
        path, _ = make_plan(exp, overrides, name=name)
        return _execute(path, args)

    if args.command == "experiment":
        from .tasks import make_plan
        name = args.name + args.name_suffix
        path, _ = make_plan(args.name, overrides, name=name)
        return _execute(path, args)

    if args.command == "plan":
        from .tasks import make_plan
        path, plan = make_plan(args.name, overrides, name=args.name + args.name_suffix,
                               out=args.out)
        print(f"units: {plan['n_units']}")
        print(f"plan: {path}")
        return 0

    if args.command == "unit":
        from .runners import run_chunk
        return run_chunk(Path(args.plan), args.index, args.chunk_size)

    if args.command == "count":
        from .runners import n_array_tasks
        print(n_array_tasks(Path(args.plan), args.chunk_size))
        return 0

    if args.command == "collect":
        from .collect import collect
        return collect(_plan_path(args.plan, overrides), allow_partial=args.allow_partial,
                       plots=not args.no_plots)

    if args.command == "verify-cv":
        from .audit_cv import verify_plan
        return verify_plan(_plan_path(args.plan, overrides))

    if args.command == "verify-recovery":
        from .collect import verify_recovery_plan
        return verify_recovery_plan(_plan_path(args.plan, overrides))

    if args.command == "figures":
        from .figures import make_figures
        make_figures(args.fit_run, args.variant, args.recovery_run, args.dataset, args.groups,
                     args.n_sims, args.out, overrides)
        return 0

    if args.command == "list":
        return _list(args.what)

    if args.command == "show":
        return _show(args.model)
    return 1


def _list(what):
    from .config import list_experiments
    from .models import params as P
    from .service import context
    ctx = context()
    if what == "models":
        print(f"{'model':10s} {'stimulus':9s} {'action':7s} {'choice':7s} {'feedback':8s} k  free parameters")
        for m in ctx.models.values():
            print(f"{m.name:10s} {m.stimulus:9s} {m.action:7s} {m.choice:7s} {m.feedback:8s} {m.k}  "
                  f"{', '.join(m.free)}")
    elif what == "sets":
        for k, v in ctx.sets.items():
            print(f"{k:10s} {', '.join(v)}")
    elif what == "datasets":
        for k, v in ctx.datasets["datasets"].items():
            print(f"{k:10s} {v}")
        print("Any mouse ID or comma list also works, e.g. GS027 or GS027,JC059")
    elif what == "experiments":
        for name in list_experiments():
            print(name)
    elif what == "params":
        for p in P.PARAMS:
            print(f"{p.name:8s} [{p.lower:g}, {p.upper:g}] {p.scale:6s} neutral={p.neutral:g}  "
                  f"{p.about}")
    return 0


def _show(model):
    from .models.registry import parents, resolve_models
    from .service import context
    ctx = context()
    try:
        resolved = resolve_models(model, ctx.models, ctx.sets)
    except KeyError:
        print(f"Unknown model {model!r}; try `list models`")
        return 1
    if len(resolved) != 1:
        print("show expects one model; use `list models` for model sets")
        return 1
    m = ctx.models[resolved[0]]
    print(f"{m.name}: {m.about}")
    print(f"  slots: stimulus={m.stimulus} action={m.action} choice={m.choice} feedback={m.feedback}")
    print(f"  free ({m.k}): {', '.join(m.free)}")
    if m.fixed:
        print(f"  fixed: {m.fixed}")
    print(f"  warm-start parents: {', '.join(parents(m, ctx.models)) or 'none'}")
    return 0
