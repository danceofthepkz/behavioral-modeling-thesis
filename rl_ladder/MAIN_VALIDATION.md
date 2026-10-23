# Primary-experiment preparation checks

Date: 2026-10-22. Release: 0.2.3. Scientific fingerprint: `0e7f4f57de98`.

## Scope

The current presets select 18 primary models only. There are 684 old-cohort
and 180 separately summarized new-cohort empirical fit/CV outputs. Neither
full empirical experiment has been run on Firebird in this release.

## Checks completed locally

- All 282 automated tests passed, including data-dependent original-paper
  likelihood checks, independent update calculations, model nesting, and
  training/test isolation. Submission guards check model identities, equal
  data/settings across candidates, reset policies, and recovery candidates.
- All 19 old-mouse files and all five new-mouse files loaded and preprocessed:
  130,039 old-cohort trials and 24,436 new-cohort trials before likelihood
  masking. Data checksums and correction-trial metadata were inspected.
- The old-mouse GS029 smoke completed all 36 full fits and embedded CV fits
  (18 models, both reset policies, three random starts).
- The new-mouse AS06 smoke completed the same 36-output comparison with
  three random starts. New-mouse data use the current data layout.
- All 72 full fits and 72 CV training fits converged. Neither run had a
  full-fit nested-parent violation. Saved CV scores independently replayed
  successfully for all 72 outputs; training-parent checks also passed.
- Nested warm-start parents of every primary candidate are themselves primary
  models. Supplementary models cannot enter these experiments implicitly.
- Submission dry runs checked all five stage sizes and resources. No remote
  Slurm job was submitted by these local checks.

Local evidence is preserved under `results/runs/`:

- `smoke_main_ladder_v023_localcheck/`
- `smoke_main_new_v023_localcheck/`
- `main_v023_data_preflight.json`

Both smoke folders contain resolved plans, raw unit outputs, full-fit quality
reports, CV quality reports and independent CV verification reports. These are
execution checks on one mouse per cohort, not population findings. Cached
fits with identical scientific/data/configuration keys may be reused.

## Remaining work

Run the Firebird smoke using its installed environment, then full old/new
cohorts. Recovery presets are prepared but have not been executed for the full
18-model set; earlier 12-model recovery does not cover the additional R
combinations. Smoke convergence alone does not establish parameter recovery,
model identifiability, a global optimizer optimum, or stable group conclusions.

The release retains the previous likelihood/update equations and includes no
fixed-temperature, counterfactual or additional stimulus-blind variants in the
primary or recovery candidate lists. See `MAIN_EXPERIMENT.md` for operations.
