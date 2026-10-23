"""Verify saved CV parameters and scores without refitting any model."""

import pandas as pd

from .config import stable_hash
from .data import load_mouse
from .evaluate import verify_cv_record
from .models.registry import parents, reset_flags
from .preprocess import preprocess
from .provenance import environment_record
from .service import atomic_write_json, context, read_json, seed_for


def audit_outputs(done, ctx):
    rows = []
    for unit, outer in done:
        row = {"variant": unit["variant"], "model": unit["model"], "mouse": unit["mouse"],
               "verified": False, "training_converged": False, "error": ""}
        try:
            for key in ("id", "task", "variant", "model", "mouse"):
                if outer[key] != unit[key]:
                    raise ValueError(f"CV output identity mismatch: {key}")
            cfg = unit["cfg"]
            raw = load_mouse(unit["mouse"], cfg)
            if outer["key"] != stable_hash([unit["key"], raw.meta["sha256"]], n=16):
                raise ValueError("CV output key does not match the plan and input-file checksum")
            r = outer["result"]
            if not r.get("cv_record"):
                raise ValueError("CV training record unavailable in this legacy output; rerun CV in a new run")
            if r["code_version"] != ctx.code_version:
                raise ValueError("CV source code differs; verify with the archived code release")
            record = r["cv_record"]
            expected_seed = seed_for(cfg["fit"]["seed"], unit["model"], unit["mouse"]) + 7
            if (record["seed"] != expected_seed or record["fit_config"] != cfg["fit"]
                    or record["train_frac"] != cfg["evaluate"]["cv"]["train_frac"]
                    or (unit["task"] == "cv" and unit["seed"] != expected_seed)):
                raise ValueError("CV training configuration differs from the plan")
            expected_parents = (parents(ctx.models[unit["model"]], ctx.models)
                                if cfg["evaluate"]["cv"].get("warm_start_from_nested", False) else [])
            if (record["warm_starts_used"] != bool(expected_parents)
                    or set(record.get("training_parent_nll", {})) != set(expected_parents)):
                raise ValueError("CV training-parent initialization differs from the plan")
            trials, lik = preprocess(raw, cfg)
            checked = verify_cv_record(ctx.models[unit["model"]], trials, lik,
                                       reset_flags(cfg["model"]["session_reset"]), r)
            row.update({"verified": checked["passed"],
                        "training_converged": checked["training_converged"],
                        "train_nll_abs_error": checked["train_nll_abs_error"],
                        "test_nll_abs_error": checked["test_nll_abs_error"]})
        except (KeyError, ValueError, TypeError, IndexError, OSError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def verify_plan(plan_path):
    """Write a score-reproduction audit; nonzero exit for missing/invalid/unconverged units."""
    from .tasks import run_dir, unit_output_path
    plan = read_json(plan_path)
    if plan["task"] not in ("cv", "fit"):
        raise ValueError("verify-cv expects a cv or fit plan")
    done, missing = [], []
    for unit in plan["units"]:
        path = unit_output_path(plan, unit)
        if path.exists():
            done.append((unit, read_json(path)))
        else:
            missing.append(unit["id"]+" @ "+unit["variant"])
    df = audit_outputs(done, context())
    verified = int(df.verified.sum()) if len(df) else 0
    converged = int(df.training_converged.sum()) if len(df) else 0
    report = {"schema_version": 1, "plan": str(plan_path), "n_units": plan["n_units"],
              "n_present": len(done), "n_missing": len(missing), "missing": missing,
              "n_verified": verified, "n_training_converged": converged,
              "passed": len(done) == plan["n_units"] == verified == converged,
              "verification_environment": environment_record(),
              "meaning": "Saved parameters reproduce scores; convergence does not prove a global optimum or stable model ranking."}
    out = run_dir(plan)/"summary"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"cv_verification.csv", index=False)
    atomic_write_json(out/"cv_verification.json", report)
    print(f"CV verified: {verified}/{plan['n_units']}; training converged: {converged}/{plan['n_units']}")
    if len(df):
        for error in df.loc[~df.verified, "error"].head(5):
            print("  ", error)
    return 0 if report["passed"] else 1
