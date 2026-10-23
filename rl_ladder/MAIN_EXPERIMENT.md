# Primary 18-model experiment

This is the current experiment entry point. It uses the 18 primary models;
the 15 supplementary variants are not selected by any preset below. Existing
models and historical runs are preserved. Version 0.2.3 adds experiment presets
and submission instructions; model equations and scientific code are unchanged
from 0.2.2.

## Questions and candidates

| Group | Models | Question |
|---|---|---|
| Baselines | M1, M2 | How much can fixed bias or one-trial WSLS explain? |
| S | S0, S+R, S+C, S+F, S+R+C, S+R+F, S+F+C, S+R+F+C | What do independent action value, choice history, and choice-feedback history contribute? |
| A | A0, A+C, A+F, A+F+C | Do C/F help the original simple base? |
| B | B0, B+C, B+F, B+F+C | Do C/F help the original coupled choice-history base? |

R denotes the independent action-value V process. B0's coupling is part of
its base, not an additional R term. The summary includes 29 planned contrasts:
20 matched module comparisons, two base comparisons, six base-versus-baseline
comparisons, and M2 versus M1. All 18 candidates also appear in the wide score
tables, allowing direct inspection without adding models.

## Shared protocol

- Per-mouse maximum likelihood, 20 random starts plus feasible nested-parent
  starts, with the best training objective determining the selected CV fit.
- Full curated learning window; correction trials update latent states but are
  excluded from the likelihood. This does not remove their effects on later trials.
- Full-data BIC/AIC and chronological CV on identical observations across models.
  The existing session split uses `train_frac=0.67`; actual session IDs and
  scored trial counts are saved in each CV record.
- CV uses previous observed choices/outcomes to predict each next held-out
  choice, updating latent states online without changing fitted parameters.
  It measures sequential prediction, not a free-running behavioral simulation.
- Two policies, reported separately: `[wsls, feedback]` and
  `[wsls, choice, feedback]`. Q/V carry across sessions in both. Some models
  give identical results under both policies; those are not independent repeats.
- Archive the resolved plan, raw unit JSON, optimizer records, data/source
  checksums, and verification reports. Keep all baseline outputs.
- Collection requires complete outputs, converged full/CV fits, verified CV
  scores, and no worse training/full likelihood than feasible nested parents
  beyond the numerical tolerance. Passing these checks is not proof of a
  global optimum or parameter identifiability.

## Stages and resources

Every stage requires an explicit submission. Smoke does not launch full fitting
or recovery automatically. Time below is the requested maximum per array task,
not a measured total runtime.

| Mode | Dataset | Units | Units per task | Array tasks | Concurrent tasks | CPU / RAM / time per task |
|---|---|---:|---:|---:|---:|---|
| `smoke` | GS029 | 36 | 4 | 9 | 2 | 1 / 2 GB / 30 min |
| `full` | paper19 | 684 | 4 | 171 | 8 | 1 / 2 GB / 2 h |
| `new` | new5 | 180 | 4 | 45 | 8 | 1 / 2 GB / 2 h |
| `model-recovery` | paper19 templates | 1,800 | 2 | 900 | 8 | 1 / 2 GB / 2 h |
| `param-recovery` | paper19 templates | 3,600 | 4 | 900 | 8 | 1 / 2 GB / 2 h |

The 684 old-mouse units and 180 new-mouse units each contain a full-data fit
and a separate CV training fit. They are not counts of optimizer starts.
Old and new mice have separate run folders and summaries. The new cohort uses
its current curation and has no assigned paper drift/stationarity groups.

Recovery presets retain the same two reset policies and use only primary
models. Model recovery generates 50 datasets per model per policy, fitting all
18 candidates to each (32,400 candidate fits). Parameter recovery generates
100 datasets per model per policy. Neither rejects draws based on accuracy.
These are prepared validation stages, not already completed results. Earlier
12-model recovery cannot establish identifiability of the expanded 18-model set.

## Firebird: smoke first

Upload from the local computer:

```bash
scp /Users/danceofthepkz/Desktop/00_Inbox/cogsthesis/behavioral-modeling-thesis/rl_ladder/dist/rl_ladder_v023_ready.tar.gz kpeng1-swat@firebird.swarthmore.edu:~/
```

On Firebird, unpack into a new folder and reuse the existing environment/data:

```bash
mkdir -p ~/rl_ladder_v023
tar -xzf ~/rl_ladder_v023_ready.tar.gz -C ~/rl_ladder_v023 --strip-components=1
cd ~/rl_ladder_v023
export RL_LADDER_ENV="$HOME/.conda/envs/mice-rl"
source scripts/activate.sh
bash scripts/test.sh
bash scripts/slurm/submit_main_ladder.sh smoke
```

Wait for the printed collect job. Its log should report **36/36** and no
quality failure. Then verify the archived CV scores without refitting:

```bash
bash scripts/verify_cv.sh smoke_main_ladder_v023
cat results/runs/smoke_main_ladder_v023/summary/fit_quality.json
cat results/runs/smoke_main_ladder_v023/summary/cv_quality.json
```

Only after those checks pass, submit the old cohort:

```bash
bash scripts/slurm/submit_main_ladder.sh full
```

After 684/684 and passing quality checks, run the separate new cohort:

```bash
bash scripts/slurm/submit_main_ladder.sh new
```

For either run, call `scripts/verify_cv.sh` with its run name. Review failed
fits before changing settings. Reusing the same experiment skips existing
outputs, including outputs with a failed convergence flag; it does not repair
those fits. Changed optimization settings belong in a fresh run, for example:

```bash
bash scripts/slurm/submit_main_ladder.sh full --fit.n_starts=40 --name-suffix _v023_retry40
```

The optional validation stages are submitted explicitly when ready:

```bash
bash scripts/slurm/submit_main_ladder.sh model-recovery
bash scripts/slurm/submit_main_ladder.sh param-recovery
```

Recovery runtime and convergence should be inspected on a small pilot before
submitting the whole recovery grid: append `--pilot` for one array task. A
single pilot is an execution check, not a recovery result.

## Download and analyze

Run the following on the local computer after collection and verification:

```bash
scp -r kpeng1-swat@firebird.swarthmore.edu:~/rl_ladder_v023/results/runs/main_ladder_v023 /Users/danceofthepkz/Desktop/00_Inbox/cogsthesis/behavioral-modeling-thesis/rl_ladder/results/firebird/
```

Also preserve the run plan, cluster logs, package checksum and environment.
Use `main_new_mice_v023` for the new cohort. Download the entire run folder,
not only `summary/`: raw JSON is needed to verify saved CV parameters/scores.
Primary analysis uses per-mouse held-out NLL per scored trial and matched
module gains. Full-data BIC is a complementary penalized fit comparison;
do not merge different reset policies into independent observations.

Interpretation limits remain: S learning-rate/temperature tradeoffs limit
parameter claims; F is exactly a stimulus-history filter under this dataset's
deterministic feedback. A winning model is a useful predictor, not proof of
a unique biological mechanism. Supplementary variants are not required to
report these predictive comparisons.
