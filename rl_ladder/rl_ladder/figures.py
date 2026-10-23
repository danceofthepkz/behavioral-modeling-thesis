"""Summary figures for a finished fit run (and optionally a model-recovery run).

    python -m rl_ladder figures                      # defaults: core_ladder + model_recovery
    python -m rl_ladder figures --n-sims 100

Figures (written to <results>/figures/<fit run>/):
    1_model_ranking.png        Delta-BIC to each mouse's best model, per model
    2_model_recovery.png       confusion matrix: generating model vs selected model
    3_best_model_per_mouse.png Delta-BIC heat map, mouse x model, best model marked
    4_ppc_<MOUSE>.png          real vs simulated learning curves, one panel per model
    5_ppc_error.png            how far each model's simulated curves are from the real ones
    ppc_metrics.csv            the numbers behind figure 5

Simulations use each mouse's fitted parameters and its real stimulus sequence
(replay), so real and simulated curves are directly comparable session by session.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import apply_overrides, load_default, results_root
from .data import load_mouse, resolve_mice
from .models import kernel
from .models.registry import canonical_name, reset_flags
from .preprocess import preprocess
from .service import context, read_json, seed_for

# Reference palette (dataviz skill): two validated categorical slots, a one-hue
# sequential ramp, and recessive chrome.
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CORE = ["M1", "M2", "S0", "S+R", "S+C", "S+R+C", "A0", "B0", "A+C", "B+C",
        "S+F", "S+F+C", "A+F", "A+F+C", "B+F", "B+F+C", "S+R+F", "S+R+F+C"]


def model_order(names):
    """Show every fitted model, including optional F and fixed-temperature variants."""
    names = list(dict.fromkeys(names))
    return [m for m in CORE if m in names] + [m for m in names if m not in CORE]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlesize": 10, "axes.titlecolor": INK, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 200,
        "savefig.bbox": "tight", "legend.frameon": False,
    })
    return plt


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", RAMP)


def _groups(ctx, mice, group_name):
    groups = ctx.datasets.get("groups", {}).get(group_name, {}) if group_name else {}
    label = {m: g for g, ms in groups.items() for m in ms}
    order = [m for g in groups for m in groups[g] if m in mice]
    order += [m for m in mice if m not in order]
    return order, {m: label.get(m, "unclustered") for m in mice}


def load_fit_run(fit_run, variant):
    unit_dir = fit_run / variant / "fit"
    if not unit_dir.is_dir():
        options = sorted(p.name for p in fit_run.iterdir() if (p / "fit").is_dir())
        raise FileNotFoundError(f"No fit units in {unit_dir}. Variants found: {options}")
    fits = {}
    for f in sorted(unit_dir.glob("*.json")):
        o = read_json(f)
        fits[(o["mouse"], canonical_name(o["model"]))] = o["result"]
    return fits


def delta_bic_table(fits, models):
    mice = sorted({m for m, _ in fits})
    bic = pd.DataFrame({model: {m: fits[(m, model)]["bic"] for m in mice if (m, model) in fits}
                        for model in models})
    return bic.sub(bic.min(axis=1), axis=0)


# ---------------------------------------------------------------- figure 1
def fig_ranking(delta, path):
    plt = _plt()
    order = delta.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rng = np.random.default_rng(0)
    for y, model in enumerate(order):
        x = delta[model].to_numpy()
        ax.scatter(x, y + rng.uniform(-0.18, 0.18, len(x)), s=12, color=MUTED, alpha=0.55,
                   linewidths=0, zorder=2)
        color = BLUE if model.endswith("C") else ORANGE
        ax.scatter([np.median(x)], [y], s=70, color=color, edgecolors="white", linewidths=1.5,
                   zorder=3)
        n_best = int((x == 0).sum())
        ax.text(1.02, y, f"best in {n_best}/{len(x)}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=INK2)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-1, delta.to_numpy().max() * 1.3)
    ax.set_yticks(range(len(order)), order)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_xlabel("ΔBIC relative to each mouse's best model (lower is better; symlog scale)")
    ax.set_title("Model ranking across mice", loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markersize=7, label="median, model with choice kernel (C)"),
               Line2D([], [], marker="o", ls="", color=ORANGE, markersize=7, label="median, model without C"),
               Line2D([], [], marker="o", ls="", color=MUTED, markersize=4, label="individual mice")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_confusion(counts, path):
    plt = _plt()
    counts = counts.rename(index=canonical_name, columns=canonical_name)
    order = model_order(counts.index)
    counts = counts.reindex(index=order, columns=order, fill_value=0)
    frac = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(frac.to_numpy(), cmap=_cmap(), vmin=0, vmax=1)
    for i in range(len(order)):
        for j in range(len(order)):
            n = counts.iat[i, j]
            if n:
                ax.text(j, i, str(n), ha="center", va="center", fontsize=8,
                        color="white" if frac.iat[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(order)), order, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("selected model (lowest BIC)")
    ax.set_ylabel("generating model")
    ax.spines[:].set_visible(False)
    ax.set_title(f"Model recovery ({int(counts.sum(axis=1).iloc[0])} simulated datasets per model)",
                 loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("fraction of datasets", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_best_per_mouse(delta, order, group, path, cap=50):
    plt = _plt()
    models = model_order(delta.columns)
    d = delta.loc[order, models]
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(order) + 1.6))
    im = ax.imshow(np.minimum(d.to_numpy(), cap), cmap=_cmap(), vmin=0, vmax=cap, aspect="auto")
    for i in range(len(order)):
        for j in range(len(models)):
            v = d.iat[i, j]
            if v == 0:
                ax.text(j, i, "best", ha="center", va="center", fontsize=7, color=INK,
                        fontweight="bold")
            elif v < 10:
                ax.text(j, i, f"{v:.1f}" if v < 1 else f"{v:.0f}", ha="center", va="center",
                        fontsize=7, color=INK2)
    ax.set_xticks(range(len(models)), models, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), [f"{m}  ({group[m]})" for m in order])
    for i in range(1, len(order)):
        if group[order[i]] != group[order[i - 1]]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Best model per mouse (numbers: ΔBIC < 10 to the best)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"ΔBIC to best (capped at {cap})", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- PPC
def _session_index(trials):
    sessions = np.unique(trials.session)
    hi = [np.flatnonzero((trials.session == s) & ~trials.is_ct & (trials.stim == 1)) for s in sessions]
    lo = [np.flatnonzero((trials.session == s) & ~trials.is_ct & (trials.stim == 0)) for s in sessions]
    return hi, lo


def _curves(choice, hi, lo):
    """choice: (n_sims, n_trials) -> (n_sims, n_sessions, 2): accuracy on High, on Low."""
    choice = np.atleast_2d(choice)
    out = np.full((choice.shape[0], len(hi), 2), np.nan)
    for s, (h, l) in enumerate(zip(hi, lo)):
        if len(h):
            out[:, s, 0] = choice[:, h].mean(axis=1)
        if len(l):
            out[:, s, 1] = 1 - choice[:, l].mean(axis=1)
    return out


def simulate_ppc(ctx, fits, mice, models, cfg, n_sims, seed=2026):
    reset = reset_flags(cfg["model"]["session_reset"])
    results, rows = {}, []
    for mouse in mice:
        raw = load_mouse(mouse, cfg)
        trials, lik = preprocess(raw, cfg)
        hi, lo = _session_index(trials)
        real = _curves(trials.choice, hi, lo)[0]
        for model in models:
            if (mouse, model) not in fits:
                continue
            spec, fit = ctx.models[model], fits[(mouse, model)]
            theta = spec.theta(fit["params"])
            # Guard: the parameters must reproduce the stored fit on these trials.
            check = kernel.nll(spec, theta, trials, lik, reset)
            if abs(check - fit["nll"]) > 1e-4:
                raise RuntimeError(f"{mouse} {model}: NLL {check:.4f} != stored {fit['nll']:.4f};"
                                   " the data/preprocess settings differ from the fit run")
            rng = np.random.default_rng(seed_for(seed, "ppc-figure", mouse, model))
            sims = np.array([kernel.simulate_arrays(spec, theta, trials.stim, trials.new_session,
                                                    reset, "replay", 3, rng)[0]
                             for _ in range(n_sims)])
            sim = _curves(sims, hi, lo)
            mean = np.nanmean(sim, axis=0)
            lo_b, hi_b = np.nanpercentile(sim, 5, axis=0), np.nanpercentile(sim, 95, axis=0)
            ok = np.isfinite(real)
            inside = (real >= lo_b) & (real <= hi_b)
            results[(mouse, model)] = (real, mean, lo_b, hi_b)
            rows.append({"mouse": mouse, "model": model,
                         "mean_abs_error": float(np.mean(np.abs(real - mean)[ok])),
                         "coverage_90": float(np.mean(inside[ok])),
                         "real_accuracy": float(np.nanmean(real)),
                         "sim_accuracy": float(np.nanmean(mean))})
    return results, pd.DataFrame(rows)


def fig_ppc_mouse(mouse, group, models, ppc, delta, path):
    plt = _plt()
    ncol = 5
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow + 0.6), sharex=True,
                             sharey=True, squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        if (mouse, model) not in ppc:
            ax.set_visible(False)
            continue
        real, mean, lo_b, hi_b = ppc[(mouse, model)]
        x = np.arange(1, len(real) + 1)
        for k, color in ((0, BLUE), (1, ORANGE)):
            ax.fill_between(x, lo_b[:, k], hi_b[:, k], color=color, alpha=0.16, linewidth=0)
            ax.plot(x, mean[:, k], color=color, lw=1.4)
            ax.scatter(x, real[:, k], s=14, color=color, edgecolors="white", linewidths=0.8,
                       zorder=3)
        ax.axhline(0.5, color=AXIS, lw=0.6, zorder=0)
        dv = delta.at[mouse, model]
        ax.set_title(f"{model}   " + ("best BIC" if dv == 0 else f"ΔBIC +{dv:.0f}"), loc="left",
                     fontsize=9, color=INK if dv == 0 else INK2,
                     fontweight="bold" if dv == 0 else "normal")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y")
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("session")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markeredgecolor="white", label="real, High trials"),
               Line2D([], [], marker="o", ls="", color=ORANGE, markeredgecolor="white", label="real, Low trials"),
               Line2D([], [], color=INK2, lw=1.4, label="model mean"),
               Patch(color=INK2, alpha=0.16, label="model 90% range")]
    fig.legend(handles=handles, loc="upper right", ncol=4, fontsize=8, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle(f"{mouse} ({group}): real vs simulated learning curves", x=0.01, ha="left",
                 y=1.02, fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_ppc_error(metrics, order, group, path):
    plt = _plt()
    models = model_order(dict.fromkeys(metrics.model))
    err = metrics.pivot(index="mouse", columns="model", values="mean_abs_error").loc[order, models]
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(order) + 1.6))
    vmax = float(np.nanpercentile(err.to_numpy(), 95))
    im = ax.imshow(err.to_numpy(), cmap=_cmap(), vmin=0, vmax=vmax, aspect="auto")
    best = err.idxmin(axis=1)
    for i, m in enumerate(order):
        j = models.index(best[m])
        ax.text(j, i, "best", ha="center", va="center", fontsize=7, color=INK, fontweight="bold")
    ax.set_xticks(range(len(models)), models, rotation=45, ha="right")
    ax.set_yticks(range(len(order)), [f"{m}  ({group[m]})" for m in order])
    for i in range(1, len(order)):
        if group[order[i]] != group[order[i - 1]]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("How well simulations reproduce the real learning curves", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("mean |real − simulated| session accuracy\n(colour capped at 95th percentile)", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def make_figures(fit_run="core_ladder", variant=None, recovery_run="model_recovery",
                 dataset="paper19", groups="paper_clusters", n_sims=50, out=None,
                 overrides=None):
    ctx = context()
    mice, preset = resolve_mice(dataset, ctx.datasets)
    cfg = apply_overrides(apply_overrides(load_default(), preset), overrides or {})
    runs = results_root(cfg) / "runs"
    fit_dir = runs / fit_run
    if variant is None:
        candidates = sorted(p.name for p in fit_dir.iterdir() if (p / "fit").is_dir())
        variant = next((v for v in candidates if "methodall" in v or "=" not in v), candidates[0])
    out = Path(out) if out else results_root(cfg) / "figures" / fit_run
    out.mkdir(parents=True, exist_ok=True)
    fits = load_fit_run(fit_dir, variant)
    models = model_order(dict.fromkeys(k[1] for k in fits))
    mice = [m for m in mice if any(k[0] == m for k in fits)]
    delta = delta_bic_table(fits, models)
    order, group = _groups(ctx, mice, groups)

    written = []
    fig_ranking(delta, out / "1_model_ranking.png")
    written.append("1_model_ranking.png")
    counts_path = runs / recovery_run / "summary" / f"confusion_counts__{dataset}.csv"
    if counts_path.exists():
        fig_confusion(pd.read_csv(counts_path, index_col=0), out / "2_model_recovery.png")
        written.append("2_model_recovery.png")
    else:
        print(f"skipping figure 2: {counts_path} not found")
    fig_best_per_mouse(delta, order, group, out / "3_best_model_per_mouse.png")
    written.append("3_best_model_per_mouse.png")

    print(f"simulating {n_sims} runs for {len(mice)} mice x {len(models)} models ...")
    ppc, metrics = simulate_ppc(ctx, fits, order, models, cfg, n_sims)
    metrics.to_csv(out / "ppc_metrics.csv", index=False)
    for mouse in order:
        fig_ppc_mouse(mouse, group[mouse], models, ppc, delta, out / f"4_ppc_{mouse}.png")
    written.append(f"4_ppc_<mouse>.png x {len(order)}")
    fig_ppc_error(metrics, order, group, out / "5_ppc_error.png")
    written += ["5_ppc_error.png", "ppc_metrics.csv"]
    print(f"variant: {variant}\nfigures in {out}:")
    for w in written:
        print("  ", w)
    return out
