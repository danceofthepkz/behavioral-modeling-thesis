# Model recovery: primary 18-model ladder (v0.2.4)

## Purpose

Generate synthetic choices from a known model, then fit **all 18 primary models**
to those same choices. Select the lowest BIC and count how often the generating
model is recovered. This tests distinguishability under the specified simulation
design. No supplementary models are included.

The old 12-model recovery is a separate, completed experiment. These runs extend
the candidate set to include every primary S-family R combination.

## Design

| Stage | Templates | Synthetic datasets | Candidate fits | Random starts per fit |
| --- | --- | ---: | ---: | ---: |
| Smoke | GS029 and AS06, first 3 sessions | 72 | 1,296 | 5 |
| Old mice | All 19 published learning-period session structures | 2,052 | 36,936 | 20 |
| New mice | All 5 current learning-period session structures | 900 | 16,200 | 20 |

Both C reset policies are tested separately: `[wsls, feedback]` and
`[wsls, choice, feedback]`. Old-mouse runs use 3 independent parameter draws per
template, generating model, and policy; new-mouse runs use 5. Every generating
model therefore has 57 datasets per policy in the old cohort and 25 in the new.
The smoke run has one per model, template and policy.

Generating parameters are sampled from the existing distributions in
`rl_ladder/models/params.py`. These are simulation distributions, not Bayesian
priors used in fitting: estimation remains maximum likelihood. No accuracy
rejection is applied. Some draws switch off a module approximately, making
parent and child models genuinely hard to distinguish. Recovery rates depend on
these distributions and are not estimates of mechanism prevalence in mice.

The `reactive` simulator uses real session lengths, redraws binary stimuli, and
repeats the stimulus after an error, up to 3 consecutive correction trials.
This is the existing task simulator assumption; it does not replay each mouse's
observed correction sequence or claim that every session used an identical
correction protocol. Model state is updated on correction trials, which are
excluded from likelihood and BIC sample size. Old/current file selection and
learning-period curation use the existing data loader.

Each candidate gets independent multistart MLE plus available nested-parent
warm starts. No candidate receives the generating parameters as a privileged
start. BIC is `2*NLL + k*log(n_lik)`. The fitted generating model should achieve
NLL no worse than its known generating parameters, within `1e-4` total NLL.

This experiment tests **BIC recovery**. It does not establish recovery of
chronological CV rankings, repeated-split CV stability, or unique biological
mechanisms. In deterministic feedback, F remains equivalent to filtered stimulus
history. Short smoke windows are execution checks only.

## Upload and activate

Run on the **local Mac**:

```bash
scp /Users/danceofthepkz/Desktop/00_Inbox/cogsthesis/behavioral-modeling-thesis/rl_ladder/dist/rl_ladder_v024_ready.tar.gz kpeng1-swat@firebird.swarthmore.edu:~/
```

Then on **Firebird**, create a new directory (preserve the v0.2.3 archive):

```bash
mkdir ~/rl_ladder_v024
tar -xzf ~/rl_ladder_v024_ready.tar.gz -C ~/rl_ladder_v024 --strip-components=1
cd ~/rl_ladder_v024
export RL_LADDER_ENV="$HOME/.conda/envs/mice-rl"
export RL_LADDER_DATA="$HOME/rl_firebird_v2/data"
export RL_LADDER_RESULTS="$HOME/rl_ladder_v024/results"
source scripts/activate.sh
mkdir -p logs
```

Run tests on a compute node, for example through the existing command wrapper:

```bash
sbatch --partition=unowned --time=00:10:00 scripts/slurm/cmd.sbatch python -m pytest -q
```

## Submit smoke first

```bash
bash scripts/slurm/submit_model_recovery.sh smoke
```

This submits 72 one-CPU tasks, at most 4 running simultaneously, each with 2 GB
and 30 minutes, plus an automatic collection job. After completion:

```bash
bash scripts/verify_recovery.sh smoke_model_recovery_v024
```

Expected: `Recovery verified: 72/72; candidate fits converged: 1296/1296`,
exit code 0, and `passed_present_units: true` in `summary/recovery_quality.json`.
Generating-model feasibility and nested-model checks must also pass.
If they fail, inspect the diagnostic CSV/JSON before submitting full runs.

## Submit full cohorts separately

```bash
bash scripts/slurm/submit_model_recovery.sh old
bash scripts/slurm/submit_model_recovery.sh new
```

Old: 684 array tasks of up to 3 datasets each. New: 450 tasks of up to 2.
Each requests 1 CPU, 2 GB, and a **4-hour limit per array task**, with at most
8 tasks running per submission. Running both at once can use up to 16 CPUs.
These are resource limits, not measured total-runtime forecasts. Use smoke
timings to adjust limits before submission if necessary.

After collection:

```bash
bash scripts/verify_recovery.sh recovery_main_old_v024
bash scripts/verify_recovery.sh recovery_main_new_v024
```

Full expected totals: old `2052/2052` datasets and `36936/36936` converged fits;
new `900/900` datasets and `16200/16200` fits. Download entire run folders,
including `plan.json`, raw unit JSON files and `summary/`.

## Verification and interpretation

The saved record includes simulation RNG state immediately before simulation,
template and synthetic-array signatures, true parameters, environment versions,
candidate parameters, NLL/BIC, selected convergence status, and compact diagnostics
for every optimization start. Selected scores can be replayed; all-start parameter
vectors are deliberately not archived for recovery.

Collection regenerates synthetic trials without fitting and verifies all selected
NLL/BIC values within `1e-6`, candidate completeness, output keys, convergence,
generating-parameter feasibility and nested-model likelihood ordering. It writes:

- `model_recovery.csv`: every candidate fit, including failed fits.
- `recovery_verification.csv` and `recovery_quality.json`: quality results.
- `confusion_counts__*.csv`: rows = generating model, columns = winning model.
- `confusion_frac__*.csv`: row-normalized `P(selected | generating)`.
- `inversion__*.csv`: column-normalized counts under this balanced simulation
  design, not posterior probabilities for real mice.
- `confusion__*.png`: one matrix per cohort/reset policy.

Matrices are withheld if any present dataset fails quality checks. Failed
simulations are not silently excluded. A selected fit may converge even if some
other starts fail; `n_failed` reports those starts separately.

Reusing a run name skips matching saved outputs, including unsuccessful fits.
Changed optimization settings require a fresh suffix, for example
`--fit.n_starts=40 --name-suffix _v024_retry40`. This draws new simulations because
the run name contributes to seeds; it is a fresh recovery experiment, not a
paired refit of the same synthetic datasets. Preserve the original failed run.
For a paired diagnostic retry, replay its archived simulation and compare the
same dataset before deciding how to rerun the study.

No remote job is submitted by packaging or by local tests.
