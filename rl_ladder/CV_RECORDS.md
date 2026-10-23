# Complete cross-validation records (v0.2.1)

## What changes

The `cv` task fits early sessions and scores later sessions. It never calls
the full-data fitter or loads a full-data optimum as a starting point.
Existing full-data fits, BIC comparisons and v0.2.0 results remain in their
original folders. The new CV scores belong to a separate run; they do not
retroactively reconstruct missing v0.2.0 training parameters.

The likelihood, model equations, deterministic seeds and chronological split
are unchanged. The default is 20 random starts and a training fraction of 0.67
of sessions, rounded to the nearest integer. Every model for a given mouse
and reset policy uses the same scored test trials. Correction trials still
update latent states but are masked from the main-analysis likelihood.

Parameters are selected using training NLL only. Test choices update latent
states for conditional next-choice prediction, not model parameters. These
records permit exact score reevaluation in the same numerical environment;
they do not establish global optimality or stability across different refits.

## Saved record

Each `results/runs/<run>/<variant>/cv/<model>__<mouse>.json` contains:

| Field inside `result` | Content |
| --- | --- |
| `cv_record.split` | Explicit training/test session IDs and total/scored trial counts |
| `cv_record.training_fit.params`, `.theta` | Named training parameters and full kernel parameter vector |
| `cv_record.training_fit.start_records` | Initial/final parameter vectors, training NLL, convergence, optimizer status, message, iterations and evaluations for every start |
| `cv_record.training_fit.selected_start_index` | The chosen start; selection is based only on training NLL |
| `cv_record.seed`, `.fit_config` | Exact CV seed and optimization settings |
| `cv_record.input_sha256`, `.reset_flags`, `.spec` | Processed-data/weights checksum, reset policy and model definition |
| `cv_record.environment` | Python, numerical package versions, platform, NumPy build configuration and numerical thread settings |
| `data_sha256`, `code_version` | Prepared input-file checksum and scientific-source fingerprint |
| `cv_test_nll`, `cv_n_test`, `cv_nll_per_trial` | Test total NLL, scored trial count and mean NLL |
| `cv_success` | Training optimizer convergence, separate from score verification |
| `cv_verification` | Immediate parameter-based train/test score recheck |

Ordinary `fit` tasks also save the enriched `cv_record` when chronological
CV is enabled. Detailed per-start records are enabled specifically for CV,
so recovery experiments are not enlarged by these diagnostics.

## Firebird: use a separate folder

Upload `dist/rl_ladder_v021_ready.tar.gz` from the Mac, then extract into
`~/rl_ladder_v021`, leaving `~/rl_ladder_v020` unchanged:

```sh
mkdir -p ~/rl_ladder_v021
tar -xzf ~/rl_ladder_v021_ready.tar.gz -C ~/rl_ladder_v021 --strip-components=1
cd ~/rl_ladder_v021
export RL_LADDER_ENV="$HOME/.conda/envs/mice-rl"
source scripts/activate.sh
```

The activation script locates the existing prepared data at
`~/rl_firebird_v2/data`. No new data upload or environment installation is
required. Existing v0.2.0 outputs are not inputs to the CV-only optimizer:
the archived full fits are retained for their full-data analyses, rather
than copied or refitted.

### 1. Small check

Run from the project folder on the login node:

```sh
bash scripts/slurm/submit_cv_audit.sh smoke
```

This schedules 6 CV units: A0, A+C and S+F on GS029 under two reset policies,
3 starts each. The array has 2 tasks, at most 2 running together. Each task
requests 1 CPU, 2 GB memory and 15 minutes; the collect job checks every
saved record and refuses model comparisons if a training fit did not converge
or a saved score does not reproduce.

When the collect job finishes:

```sh
bash scripts/verify_cv.sh smoke_cv_audit_v021
```

The expected report is `CV verified: 6/6; training converged: 6/6`.
This command only reevaluates likelihoods at saved training parameters.

### 2. Full CV record run

After the small check passes:

```sh
bash scripts/slurm/submit_cv_audit.sh full
```

This schedules 456 units: 19 old mice x 12 models x 2 C-reset policies,
20 random starts each. It uses 92 array tasks, at most 8 at once, requesting
1 CPU, 2 GB and 1 hour per task. These are resource requests, not estimated
execution times. The collect job runs after the array finishes.

```sh
bash scripts/verify_cv.sh feedback_cv_audit_v021
```

The expected report is `CV verified: 456/456; training converged: 456/456`.
Check both counts. A verified score from an unconverged fit is not accepted
as a passing run. An unconverged row remains available for diagnosis in
`cv_scores.csv` and `cv_verification.csv`; increase starts or iteration limits
in a new run name if needed. Rerunning a completed plan skips its existing units.

## Outputs and download

The full run is `results/runs/feedback_cv_audit_v021/`. Keep the **whole run
folder**, including `plan.json` and all per-unit JSON files, rather than
downloading only its `summary` directory. Verification needs the prepared
input data and this code release as well.

| Summary file | Meaning |
| --- | --- |
| `cv_scores.csv` | Test scores and training diagnostics; no full-data NLL or BIC |
| `cv_wide__*.csv` | Per-mouse/model mean test NLL, only if every present unit passes |
| `cv_contrasts.csv` | Equal-mouse mean/median gains for added C/F modules |
| `cv_quality.json` | Counts of verified and converged present units; marks partial collection |
| `cv_verification.csv` | Per-unit score reproduction errors and convergence status |
| `cv_verification.json` | Overall verification report, produced by `verify-cv` |
| `collection_status.json` | Completeness against the 456-unit plan |

After running `verify-cv`, download on the Mac:

```sh
scp -r kpeng1-swat@firebird.swarthmore.edu:~/rl_ladder_v021/results/runs/feedback_cv_audit_v021 \
  /Users/danceofthepkz/Desktop/00_Inbox/cogsthesis/behavioral-modeling-thesis/rl_ladder/results/firebird/
```

Store the v0.2.1 code archive alongside the downloaded run. The preserved
`dist/rl_ladder_v020_ready.tar.gz` is the earlier release for the old run.

## Verification versus fitting stability

`verify-cv` checks the input checksum, model, split, reset flags, training
configuration, per-start record consistency, and train/test NLL reevaluated
at the saved training parameters. It never fits a model. Legacy outputs
without training records are reported as unavailable, not silently repaired.

Repeated-seed refits answer a separate question: whether alternative training
optima lead to the same scientific comparisons. This release records what is
needed for that investigation; it does not claim that saving a record resolves
the S-family identifiability problem or makes every exact model rank stable.
