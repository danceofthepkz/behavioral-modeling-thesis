"""Collect stage: unit outputs -> summary tables (and a few figures).

Writes to <run>/summary/. Refuses to summarise an incomplete run unless
--allow-partial is given; partial files are then prefixed `partial_`.

fit             fits.csv, bic_wide__<variant>.csv, best_counts.csv, bms.csv,
                contrasts.csv, replication.csv (collect.reference = legacy)
cv              cv_scores.csv, cv_wide__<variant>.csv, cv_contrasts.csv, cv_verification.csv
param_recovery  param_recovery.csv, param_recovery_summary.csv, *.png
model_recovery  model_recovery.csv, confusion_counts__<v>.csv,
                confusion_frac__<v>.csv, inversion__<v>.csv, *.png
ppc             ppc.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import reference_root
from .fit.bms import family_log_evidence, vb_bms
from .models import params as P
from .models.registry import SLOTS, canonical_name
from .service import atomic_write_json, context, read_json
from .tasks import run_dir, runtime_key, unit_output_path


def _load(plan):
    done, missing = [], []
    for unit in plan["units"]:
        path = unit_output_path(plan, unit)
        if path.exists():
            done.append((unit, read_json(path)))
        else:
            missing.append(unit["id"] + " @ " + unit["variant"])
    return done, missing


def _bms_rows(variant, wide, ctx):
    rows = []
    models = list(wide.columns)
    if len(models) < 2 or len(wide) < 2:
        return rows
    L = -0.5 * wide.to_numpy()
    res = vb_bms(L)
    for j, m in enumerate(models):
        rows.append({"variant": variant, "level": "model", "name": m,
                     "expected_freq": res["expected_freq"][j], "xp": res["xp"][j],
                     "pxp": res["pxp"][j], "bor": res["bor"]})
    for slot in SLOTS:
        groups = {}
        for j, m in enumerate(models):
            groups.setdefault(getattr(ctx.models[canonical_name(m)], slot), []).append(j)
        if len(groups) < 2:
            continue
        names = sorted(groups)
        fres = vb_bms(family_log_evidence(L, [groups[n] for n in names]))
        for j, n in enumerate(names):
            rows.append({"variant": variant, "level": f"family:{slot}", "name": n,
                         "expected_freq": fres["expected_freq"][j], "xp": fres["xp"][j],
                         "pxp": fres["pxp"][j], "bor": fres["bor"]})
    return rows


def _collect_fit(done, cfg, out, prefix, ctx):
    rows = []
    for unit, o in done:
        r = o["result"]
        row = {"variant": unit["variant"], "label": unit["label"], "mouse": unit["mouse"],
               "model": unit["model"], "nll": r["nll"], "k": r["k"], "n_lik": r["n_lik"],
               "n_trials": r["n_trials"], "bic": r["bic"], "aic": r["aic"],
               "nll_per_trial": r["nll_per_trial"], "success": r["success"],
               "n_starts": r["n_starts"], "n_repeated": r["n_repeated"],
               "at_bound": ";".join(r["at_bound"]), "best_start": r["best_start"],
               "ct_source": r["ct_source"], "n_ct": r["n_ct"], "data_file": r["data_file"]}
        row["feedback_interpretation"] = r.get("feedback_interpretation")
        for key in ("cv_nll_per_trial", "cv_test_nll", "cv_n_test", "cv_success"):
            if key in r:
                row[key] = r[key]
        row.update({f"p_{k}": v for k, v in r["params"].items()})
        rows.append(row)
    fits = pd.DataFrame(rows)
    fits.to_csv(out / f"{prefix}fits.csv", index=False)

    bms_rows, best_rows, contrast_rows = [], [], []
    groups_cfg = ctx.datasets.get("groups", {}).get(cfg["collect"]["groups"] or "", {})
    for variant, sub in fits.groupby("variant"):
        wide = sub.pivot(index="mouse", columns="model", values="bic").dropna()
        wide.to_csv(out / f"{prefix}bic_wide__{variant}.csv")
        best = wide.idxmin(axis=1)
        for m in wide.columns:
            best_rows.append({"variant": variant, "model": m, "n_best": int((best == m).sum()),
                              "n_mice": len(wide)})
        bms_rows += _bms_rows(variant, wide, ctx)
        for x, y in cfg["collect"]["contrasts"] or []:
            if x not in wide or y not in wide:
                continue
            delta = wide[x] - wide[y]
            parts = {"all": list(wide.index)}
            parts.update({g: [m for m in mice if m in wide.index] for g, mice in groups_cfg.items()})
            for g, mice in parts.items():
                d = delta.loc[mice]
                contrast_rows.append({"variant": variant, "contrast": f"{x} vs {y}", "group": g,
                                      "n": len(d), f"prefer_first": int((d < 0).sum()),
                                      "prefer_second": int((d > 0).sum()),
                                      "median_delta_bic": float(d.median()) if len(d) else None})
    pd.DataFrame(best_rows).to_csv(out / f"{prefix}best_counts.csv", index=False)
    if bms_rows:
        pd.DataFrame(bms_rows).to_csv(out / f"{prefix}bms.csv", index=False)
    if contrast_rows:
        pd.DataFrame(contrast_rows).to_csv(out / f"{prefix}contrasts.csv", index=False)

    if cfg["collect"]["reference"] == "legacy":
        ref_dir = reference_root(cfg)
        ref_path = ref_dir / "reference_legacy.json" if ref_dir else None
        if ref_path and ref_path.exists():
            ref = json.loads(ref_path.read_text())
            name = {"summary_3param_081123.csv": "A0", "summary_4param_231023.csv": "B0"}
            rep = []
            for t in ref:
                model = name.get(t["model"])
                hit = fits[(fits.mouse == t["mouse"]) & (fits.model.map(canonical_name) == model)]
                for _, h in hit.iterrows():
                    rep.append({"variant": h.variant, "mouse": t["mouse"], "model": model,
                                "nll_ours": h.nll, "nll_published": t["nll"],
                                "ours_minus_published": h.nll - t["nll"]})
            pd.DataFrame(rep).to_csv(out / f"{prefix}replication.csv", index=False)
        else:
            print(f"collect.reference=legacy: {ref_path} not found; skipped replication.csv")
    return fits


def _collect_param_recovery(done, out, prefix, ctx, plots):
    rows = []
    for unit, o in done:
        r = o["result"]
        for name, true in r["true"].items():
            spec = P.BY_NAME[name]
            rows.append({"variant": unit["variant"], "model": unit["model"], "rep": unit["rep"],
                         "template": unit["template"], "param": name, "true": true,
                         "fit": r["fit"][name], "true_z": P.to_z(spec, true),
                         "fit_z": P.to_z(spec, r["fit"][name]),
                         "accuracy": r["accuracy"], "optimizer_miss": r["optimizer_miss"],
                         "at_bound": name in r["at_bound"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{prefix}param_recovery.csv", index=False)
    summ = []
    for (variant, model, param), g in df.groupby(["variant", "model", "param"]):
        r = np.corrcoef(g.true_z, g.fit_z)[0, 1] if len(g) > 2 and g.true_z.std() > 0 else np.nan
        summ.append({"variant": variant, "model": model, "param": param, "n": len(g),
                     "r_z": r, "median_abs_err_z": float(np.median(np.abs(g.fit_z - g.true_z))),
                     "frac_at_bound": float(g.at_bound.mean()),
                     "frac_optimizer_miss": float(g.optimizer_miss.mean())})
    pd.DataFrame(summ).to_csv(out / f"{prefix}param_recovery_summary.csv", index=False)
    if plots:
        _plot_param_recovery(df, out, prefix)


def _collect_model_recovery(done, out, prefix, ctx, plots):
    from .validate import audit_model_recovery
    rows, audits = [], []
    for unit, o in done:
        r = o["result"]
        audit = audit_model_recovery(ctx, unit, r)
        if o.get("key") != runtime_key(unit):
            audit.update(verified=False, passed=False, error="Output key differs from plan or data")
        audits.append({"variant": unit["variant"], "id": unit["id"],
                       "template": unit["template"], **audit})
        for cand, c in r["candidates"].items():
            rows.append({"variant": unit["variant"], "generating": r["generating"],
                         "rep": unit["rep"], "template": unit["template"], "candidate": cand,
                         "bic": c["bic"], "nll": c["nll"], "k": c["k"],
                         "winner": cand == r["winner"], "accuracy": r["accuracy"],
                         "success": c.get("success"), "message": c.get("message", "unrecorded"),
                         "n_starts": c.get("n_starts"), "n_failed": c.get("n_failed"),
                         "n_repeated": c.get("n_repeated"),
                         "at_bound": ",".join(c.get("at_bound", [])),
                         "dataset_passed": audit["passed"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{prefix}model_recovery.csv", index=False)
    pd.DataFrame(audits).to_csv(out / f"{prefix}recovery_verification.csv", index=False)
    quality = {"n_datasets": len(audits), "n_candidates": len(rows),
               "n_verified": sum(a["verified"] for a in audits),
               "n_converged_candidates": sum(c.get("success", False) for _, o in done
                                               for c in o["result"]["candidates"].values()),
               "n_passed": sum(a["passed"] for a in audits),
               "passed_present_units": all(a["passed"] for a in audits),
               "partial": bool(prefix),
               "failed": [a for a in audits if not a["passed"]]}
    atomic_write_json(out / f"{prefix}recovery_quality.json", quality)
    if not quality["passed_present_units"]:
        # Do not discard troublesome simulations or publish stale matrices.
        for pattern in ("confusion_counts__*.csv", "confusion_frac__*.csv", "inversion__*.csv",
                        "confusion__*.png"):
            for path in out.glob(prefix + pattern):
                path.unlink()
        print("Recovery quality checks failed; matrices withheld. Inspect recovery_quality.json.")
        return False
    for variant, g in df.groupby("variant"):
        order = list(dict.fromkeys(g.candidate))
        wins = g[g.winner]
        counts = pd.crosstab(wins.generating, wins.candidate).reindex(
            index=order, columns=order, fill_value=0)
        counts.to_csv(out / f"{prefix}confusion_counts__{variant}.csv")
        frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
        frac.to_csv(out / f"{prefix}confusion_frac__{variant}.csv")
        inv = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1)
        inv.to_csv(out / f"{prefix}inversion__{variant}.csv")
        if plots:
            _plot_matrix(frac, f"P(fit | generating), {variant}",
                         out / f"{prefix}confusion__{variant}.png")
    print(f"Recovery verified: {quality['n_verified']}/{quality['n_datasets']}; "
          f"candidate fits converged: {quality['n_converged_candidates']}/{quality['n_candidates']}")
    return True


def verify_recovery_plan(plan_path):
    """Strict recovery audit, including completeness, regardless of legacy settings."""
    plan = read_json(plan_path)
    if plan["task"] != "model_recovery":
        raise ValueError("verify-recovery requires a model_recovery plan")
    status = collect(plan_path, plots=False)
    quality_path = run_dir(plan) / "summary" / "recovery_quality.json"
    if status or not quality_path.exists():
        return 1
    return 0 if read_json(quality_path)["passed_present_units"] else 1


def _collect_ppc(done, out, prefix):
    rows = []
    for unit, o in done:
        r = o["result"]
        sim_acc = np.array(r["sim_accuracy"])
        row = {"variant": unit["variant"], "mouse": unit["mouse"], "model": unit["model"],
               "bic": r["bic"], "real_accuracy": r["real_accuracy"],
               "sim_accuracy_mean": float(sim_acc.mean())}
        if "real_slope" in r:
            s = np.array(r["sim_slopes"], dtype=float)
            s = s[np.isfinite(s)]
            row.update({"real_slope": r["real_slope"],
                        "sim_slope_mean": float(s.mean()) if s.size else None,
                        "sim_slope_lo": float(np.percentile(s, 2.5)) if s.size else None,
                        "sim_slope_hi": float(np.percentile(s, 97.5)) if s.size else None,
                        "real_slope_percentile": r["real_slope_percentile"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / f"{prefix}ppc.csv", index=False)


def _plot_matrix(frac, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(frac.columns), 1 + 0.5 * len(frac)))
    ax.imshow(frac.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(frac.columns)), frac.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(frac.index)), frac.index)
    for i in range(len(frac.index)):
        for j in range(len(frac.columns)):
            v = frac.iat[i, j]
            if np.isfinite(v) and v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black")
    ax.set_xlabel("selected (lowest BIC)")
    ax.set_ylabel("generating")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_param_recovery(df, out, prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for (variant, model), g in df.groupby(["variant", "model"]):
        params = list(dict.fromkeys(g.param))
        fig, axes = plt.subplots(1, len(params), figsize=(2.4 * len(params), 2.6), squeeze=False)
        for ax, name in zip(axes[0], params):
            h = g[g.param == name]
            ax.scatter(h.true_z, h.fit_z, s=8, alpha=0.6)
            lo = min(h.true_z.min(), h.fit_z.min())
            hi = max(h.true_z.max(), h.fit_z.max())
            ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8)
            scale = "log10 " if P.BY_NAME[name].scale == "log" else ""
            ax.set_title(f"{scale}{name}", fontsize=9)
            ax.set_xlabel("true", fontsize=8)
        axes[0][0].set_ylabel("fitted", fontsize=8)
        fig.suptitle(f"{model} ({variant})", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{prefix}param_recovery__{model.replace('+', '_')}__{variant}.png",
                    dpi=150)
        plt.close(fig)


def collect(plan_path, allow_partial=False, plots=True):
    plan = read_json(plan_path)
    ctx = context()
    done, missing = _load(plan)
    out = run_dir(plan) / "summary"
    out.mkdir(parents=True, exist_ok=True)
    status = {"plan": str(plan_path), "n_units": plan["n_units"], "n_done": len(done),
              "n_missing": len(missing), "missing": missing[:200]}
    atomic_write_json(out / "collection_status.json", status)
    print(f"{plan['name']}: {len(done)}/{plan['n_units']} units complete")
    if missing and not allow_partial:
        print("Incomplete run; not writing summaries. First missing units:")
        for m in missing[:10]:
            print("  ", m)
        print("Re-run the missing units, or use --allow-partial for clearly marked partial files.")
        return 1
    if not done:
        return 1
    prefix = "partial_" if missing else ""
    cfg = plan["units"][0]["cfg"]
    task = plan["task"]
    if task == "fit":
        _collect_fit(done, cfg, out, prefix, ctx)
        if cfg["collect"].get("require_fit_success", False):
            failed = [u["id"]+" @ "+u["variant"] for u, o in done if not o["result"]["success"]]
            nested = [{"id": u["id"], "variant": u["variant"], "parent": parent,
                       "excess_nll": o["result"]["nll"]-score}
                      for u, o in done for parent, score in o["result"]["parent_nll"].items()
                      if o["result"]["nll"] > score+1e-4]
            quality = {"n_present": len(done), "n_converged": len(done)-len(failed),
                       "failed": failed, "nested_violations": nested,
                       "passed_present_units": not failed and not nested, "partial": bool(prefix)}
            atomic_write_json(out/f"{prefix}fit_quality.json", quality)
            if not quality["passed_present_units"]:
                print("Full-fit convergence or nesting checks failed; inspect fit_quality.json.")
                return 1
        if cfg["evaluate"]["cv"]["method"] == "chrono":
            if not _collect_cv(done, cfg, out, prefix, ctx):
                print("Embedded CV failed score/convergence checks; inspect cv_verification.csv.")
                return 1
    elif task == "cv":
        if not _collect_cv(done, cfg, out, prefix, ctx):
            print("CV records failed score verification or training convergence; inspect cv_verification.csv.")
            return 1
    elif task == "param_recovery":
        _collect_param_recovery(done, out, prefix, ctx, plots)
    elif task == "model_recovery":
        passed = _collect_model_recovery(done, out, prefix, ctx, plots)
        if not passed and cfg["collect"].get("require_fit_success", False):
            return 1
    elif task == "ppc":
        _collect_ppc(done, out, prefix)
    print(f"Summaries written to {out}")
    for f in sorted(Path(out).iterdir()):
        print("  ", f.name)
    return 0


def _collect_cv(done, cfg, out, prefix, ctx):
    """Report CV without manufacturing full-data likelihoods or BIC values."""
    from .audit_cv import audit_outputs
    verified = audit_outputs(done, ctx)
    verified.to_csv(out/f"{prefix}cv_verification.csv", index=False)
    rows = []
    for unit, outer in done:
        r = outer["result"]
        fit = r.get("cv_record", {}).get("training_fit", {})
        row = {"variant": unit["variant"], "label": unit["label"], "mouse": unit["mouse"],
               "model": unit["model"], "cv_success": r.get("cv_success", False),
               "cv_train_sessions": r.get("cv_train_sessions"),
               "cv_test_nll": r.get("cv_test_nll"), "cv_n_test": r.get("cv_n_test"),
               "cv_nll_per_trial": r.get("cv_nll_per_trial"),
               "train_nll": fit.get("nll"), "train_n_scored": fit.get("n_lik"),
               "n_starts": fit.get("n_starts"), "n_failed": fit.get("n_failed"),
               "n_repeated": fit.get("n_repeated"), "at_bound": ";".join(fit.get("at_bound", []))}
        row.update({f"train_p_{k}": v for k, v in fit.get("params", {}).items()})
        rows.append(row)
    scores = pd.DataFrame(rows).merge(verified, on=["variant", "model", "mouse"])
    scores.to_csv(out/f"{prefix}cv_scores.csv", index=False)
    passed = bool(verified.verified.all() and verified.training_converged.all())
    atomic_write_json(out/f"{prefix}cv_quality.json", {
        "n_present": len(done), "n_verified": int(verified.verified.sum()),
        "n_training_converged": int(verified.training_converged.sum()),
        "passed_present_units": passed, "partial": bool(prefix)})
    # Do not give invalid or unconverged records a model ranking.
    if not passed:
        for stale in list(out.glob(f"{prefix}cv_wide__*.csv")) + [out/f"{prefix}cv_contrasts.csv"]:
            if stale.exists():
                stale.unlink()
        return False
    contrasts = []
    for variant, sub in scores.groupby("variant"):
        wide = sub.pivot(index="mouse", columns="model", values="cv_nll_per_trial")
        wide.to_csv(out/f"{prefix}cv_wide__{variant}.csv")
        for extended, base in cfg["collect"]["contrasts"] or []:
            if extended in wide and base in wide:
                gain = (wide[base]-wide[extended]).dropna()
                contrasts.append({"variant": variant, "extended": extended, "baseline": base,
                                  "n_mice": len(gain), "mean_gain_nats_per_trial": float(gain.mean()),
                                  "median_gain_nats_per_trial": float(gain.median()),
                                  "n_gain_gt_1e_8": int((gain>1e-8).sum())})
    if contrasts:
        pd.DataFrame(contrasts).to_csv(out/f"{prefix}cv_contrasts.csv", index=False)
    return True
