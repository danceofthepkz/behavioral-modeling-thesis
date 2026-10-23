#!/usr/bin/env python3
"""Plot all matched R/C/F CV contrasts from a complete reward_ladder run.

Activate the project environment, then:
    python scripts/plot_reward_ladder.py --run results/runs/reward_ladder_v022
Reads saved scores; does not fit models or modify the run's source outputs.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from rl_ladder.service import context, read_json


def plot(run):
    run = Path(run)
    plan = read_json(run/"plan.json")
    summary = run/"summary"
    status = read_json(summary/"collection_status.json")
    cv_quality = read_json(summary/"cv_quality.json")
    fit_quality = read_json(summary/"fit_quality.json")
    if not (status["n_done"] == plan["n_units"] and status["n_missing"] == 0
            and cv_quality["passed_present_units"] and not cv_quality["partial"]
            and fit_quality["passed_present_units"] and not fit_quality["partial"]):
        raise ValueError("Plotting requires complete, converged and verified full/CV fits")
    scores = pd.read_csv(summary/"cv_scores.csv")
    models = context().models
    if len(scores) != plan["n_units"] or scores.duplicated(["variant","mouse","model"]).any():
        raise ValueError("Missing or duplicate CV score rows")
    pairs = plan["units"][0]["cfg"]["collect"]["contrasts"]
    resets = {u["variant"]: "choice" in u["cfg"]["model"]["session_reset"] for u in plan["units"]}
    stats = []
    for variant, sub in scores.groupby("variant"):
        wide = sub.pivot(index="mouse",columns="model",values="cv_nll_per_trial").sort_index()
        if wide.isna().any().any():
            raise ValueError("Every mouse must have all eight models")
        boot = np.random.default_rng(2026).integers(0,len(wide),(10000,len(wide)))
        for added, baseline in pairs:
            if added not in wide or baseline not in wide:
                raise ValueError(f"Matched contrast is missing: {added} vs {baseline}")
            a,b=models[added],models[baseline]
            switches=[s for s in ("action","choice","feedback") if getattr(a,s)!=getattr(b,s)]
            if len(switches)!=1:raise ValueError("Each comparison must toggle exactly one module")
            module={"action":"R","choice":"C","feedback":"F"}[switches[0]]
            baseline_modules=[label for slot,label in [("action","R"),("choice","C"),("feedback","F")]
                              if getattr(b,slot)!="none"]
            condition="+".join(baseline_modules) or "None"
            gain=(wide[baseline]-wide[added]).to_numpy()
            ci=np.quantile(gain[boot].mean(axis=1),[.025,.975])
            stats.append({"variant":variant,"module":module,"other_modules":condition,
                          "added":added,"baseline":baseline,"n_mice":len(wide),
                          "mean_gain":gain.mean(),"ci_low":ci[0],"ci_high":ci[1]})
    stats=pd.DataFrame(stats)
    assert len(stats)==len(scores.variant.unique())*12
    out=run/"figures";out.mkdir(exist_ok=True)
    stats.to_csv(out/"module_cv_gains.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.9))
    contexts={"R":["None","C","F","C+F"],"C":["None","R","F","R+F"],"F":["None","R","C","R+C"]}
    for ax,module,color in zip(axes,"RCF",["#2167A8","#267C63","#B76A12"]):
        for variant,part in stats[stats.module==module].groupby("variant"):
            reset=resets[variant]
            part=part.set_index("other_modules").loc[contexts[module]]
            x=np.arange(4)+(.06 if reset else -.06)
            y=part.mean_gain.to_numpy()
            ax.errorbar(x,y,yerr=np.vstack([y-part.ci_low,part.ci_high-y]),
                        fmt="s" if reset else "o",mfc="white" if reset else color,
                        color=color,ms=6,capsize=3,lw=1.1)
        ax.axhline(0,color="#8995A3",lw=1)
        ax.set_title(f"Add {module}",loc="left",fontweight="bold")
        ax.set(xticks=range(4),xticklabels=contexts[module],xlabel="Other modules already present",
               ylabel="Mean test log-loss gain (nats/trial)")
        ax.grid(axis="y",color="#E8EDF2",lw=.7)
    n=int(stats.n_mice.iloc[0])
    title="Complete S-family comparison of R, C and F" if n>1 else "Single-mouse execution check: R, C and F"
    fig.suptitle(title,x=.07,y=.99,ha="left",fontsize=16,fontweight="bold")
    fig.legend(handles=[Line2D([],[],marker="o",color="#667382",linestyle="none",label="C carries across sessions"),
                        Line2D([],[],marker="s",mfc="white",color="#667382",linestyle="none",label="C resets each session")],
               loc="upper left",bbox_to_anchor=(.065,.94),ncol=2,frameon=False)
    sample=f"n = {n} mice; equal mouse weights" if n>1 else "n = 1 mouse; execution check only"
    interval=("Intervals: exploratory 95% mouse-bootstrap intervals, 10,000 resamples; not adjusted for multiple comparisons."
              if n>1 else "A single-mouse run does not estimate population uncertainty.")
    fig.text(.07,.025,f"{sample}. Positive gain = better held-out prediction. Each panel uses its own vertical scale.\n"
             f"{interval}\n"
             "R: independent action value V; C: reward-independent choice history; F: choice × feedback history (also stimulus history here).",
             fontsize=8,color="#667382")
    fig.subplots_adjust(left=.07,right=.985,top=.77,bottom=.25,wspace=.42)
    for ext in ("png","pdf"):fig.savefig(out/f"complete_s_module_gains.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all 12 matched module comparisons to {out}")
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    plot(parser.parse_args().run)
