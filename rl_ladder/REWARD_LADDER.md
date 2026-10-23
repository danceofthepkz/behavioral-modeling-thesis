# Complete R/C/F comparison (v0.2.2)

## Models and scope

R is the independent action-value process V. F is the filtered choice-feedback
process. They remain separate modules. B0 is the earlier A+R model's new name;
its coupled Q update is a different mechanism from adding V to S.

| S model | R: action value V | C: choice history | F: choice-feedback | Parameters |
| --- | --- | --- | --- | ---: |
| S0 | off | off | off | 4 |
| S+R | on | off | off | 6 |
| S+C | off | on | off | 6 |
| S+F | off | off | on | 6 |
| S+R+C | on | on | off | 8 |
| S+R+F | on | off | on | 8 |
| S+F+C | off | on | on | 8 |
| S+R+F+C | on | on | on | 10 |

The four R-containing models are included in `core` (now 18 models), the
complete `S`/`S_extended` sets, and `reward_extension`. The earlier `feedback`
set remains the 12-model C/F study. New experiment names separate the complete
R/C/F comparison from that earlier study. A/B definitions are unchanged.

`reward_ladder` fits the eight S models to 19 old mice under the two existing
C-reset policies: **304 outputs = 152 R-containing fits + 152 matched controls**.
Each output includes full-data MLE/BIC and chronological CV. Controls are
included so that all R/C/F contrasts share the new initialization checks and
can be collected together. The archived A/B and earlier S results are preserved;
this is not a rerun of all 456 previous outputs.

## Matched questions

Each module is tested while holding the other modules fixed:

- R: S+R vs S0; S+R+C vs S+C; S+R+F vs S+F; S+R+F+C vs S+F+C.
- C: S+C vs S0; S+R+C vs S+R; S+F+C vs S+F; S+R+F+C vs S+R+F.
- F: S+F vs S0; S+F+C vs S+C; S+R+F vs S+R; S+R+F+C vs S+R+C.

All 12 comparisons appear in both BIC and CV summary tables. Positive CV gain
means lower test log-loss after adding the module. No module may be declared
unnecessary solely because it was absent from an experiment preset.

## Data and fitting

- Same prepared `paper19` inputs and original special windows for GS028/JC052.
- Correction trials update states and are excluded from the likelihood.
- First approximately 67% of sessions train the model; later sessions test it.
- The same two reset settings as before: `[wsls, feedback]` and
  `[wsls, choice, feedback]`. Q and V carry across sessions in both settings.
- 20 random starts per fit, plus feasible nested-parent starts. New added
  modules start at zero influence when initialized from a parent.
- **CV parents are fitted on training sessions only.** Their parameters never
  come from full-data fits. Test scores never choose starting points or fits.
- Separate CV caches share parent training fits across descendants.
- Full-fit convergence/nesting and saved CV score/convergence checks are
  required by the collector. Training optima worse than a feasible CV parent
  by more than 1e-4 total NLL fail verification.

The archived v0.2.1 behavior remains available when nested CV starts are off.
This release records their training NLLs and initial-vector indices in every
CV record. The new record passes score verification even after JSON key sorting.

## Firebird workflow

Upload `dist/rl_ladder_v022_ready.tar.gz` from the Mac. On Firebird, use a new
folder and the existing environment/data:

```sh
mkdir -p ~/rl_ladder_v022
tar -xzf ~/rl_ladder_v022_ready.tar.gz -C ~/rl_ladder_v022 --strip-components=1
cd ~/rl_ladder_v022
export RL_LADDER_ENV="$HOME/.conda/envs/mice-rl"
source scripts/activate.sh
```

### Small check first

```sh
bash scripts/slurm/submit_reward_ladder.sh smoke
```

This submits 16 outputs: all eight S models on GS029, with both reset policies
and 3 random starts. Four array tasks run at most two at once. Each requests
1 CPU, 2 GB and 30 minutes. A collect job follows the array.

After collection finishes:

```sh
bash scripts/verify_cv.sh smoke_reward_ladder_v022
```

Expected: `CV verified: 16/16; training converged: 16/16`. The collection log
must also report success; `summary/fit_quality.json` checks the full-data fits.

### Full comparison

```sh
bash scripts/slurm/submit_reward_ladder.sh full
```

This submits 304 outputs in 76 array tasks, at most eight at once. Each task
requests 1 CPU, 2 GB and 2 hours. These are scheduling limits, not runtime
estimates. The fit cache shares parent calculations across dependent models.

```sh
bash scripts/verify_cv.sh reward_ladder_v022
```

Expected: `CV verified: 304/304; training converged: 304/304` plus passing
full-fit collection. An incomplete or failed-quality run is not a passing
experiment. Use a new name and revised fit settings if failed fits need more
starts; completed units of an unchanged plan are cached.

## Download and analyze

Download the whole run, including its plan and per-model JSON records. From
the Mac terminal:

```sh
scp -r kpeng1-swat@firebird.swarthmore.edu:~/rl_ladder_v022/results/runs/reward_ladder_v022 \
  /Users/danceofthepkz/Desktop/00_Inbox/cogsthesis/behavioral-modeling-thesis/rl_ladder/results/firebird/
```

Important summary files:

- `fits.csv`, `bic_wide__*.csv`, `contrasts.csv`: full-data fits and BIC comparisons.
- `fit_quality.json`: full-fit convergence and nested-likelihood checks.
- `cv_scores.csv`, `cv_contrasts.csv`, `cv_wide__*.csv`: saved-parameter test performance.
- `cv_quality.json`, `cv_verification.csv`: CV score/convergence checks.

Create a figure that includes **all R, C and F contrasts**:

```sh
python scripts/plot_reward_ladder.py --run results/firebird/reward_ladder_v022
```

It writes PNG/PDF figures and the numbers behind them to the run's `figures/`
folder. It refuses incomplete/unverified runs and missing matched controls.

## Further validation presets

These are prepared separately and are not submitted by the smoke/full wrapper:

| Preset | Purpose | Units |
| --- | --- | ---: |
| reward_model_recovery | Eight-model confusion matrix; 50 prior datasets per model | 400 |
| reward_param_recovery | Four R-containing models; 100 prior datasets per model | 400 |
| reward_ladder_fixed | Same eight-way comparison with S temperature fixed at 5 | 304 |

The new-mouse dataset can be selected in a separate experiment after the
old-mouse comparison. The S learning-rate/temperature tradeoff, weak/zero
module weights, earlier recovery limitations and F's stimulus-history identity
remain relevant. Execution and likelihood checks do not establish parameter
identifiability or a unique biological explanation.
