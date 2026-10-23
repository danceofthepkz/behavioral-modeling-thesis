# Model-recovery preparation checks

Date: 2026-10-22. Release: 0.2.4. Scientific fingerprint: `d9ac92a5d418`.

## Completed locally

- All **295 tests passed**, including the data-dependent published A0/B0
  likelihood and M1/M2 baseline checks.
- New recovery tests reject altered scores, candidate sets, scientific versions,
  template windows and cached-output keys. They distinguish reproducible scores
  from convergence, generating-parameter feasibility and nested-model quality.
  A failed selected optimizer removes stale confusion matrices and causes the
  strict verifier to return a failure status.
- `smoke_model_recovery_v024`: **72/72 synthetic datasets**, **1,296/1,296
  converged candidate fits**, all simulation signatures and selected NLL/BIC
  scores reproduced. Maximum absolute replay error: **0.0** on this environment.
  All 18 generators/candidates, GS029 and AS06, both reset policies, first three
  sessions, five random starts plus available nested starts.
- `local_full_window_recovery_v024`: **4/4 datasets**, **72/72 converged candidate
  fits**, all quality checks passed, maximum replay error **0.0**. Generating
  model S+R+F+C, full curated learning periods for GS028 and AS06, both reset
  policies, all 18 candidates, 20 random starts plus nested starts. GS028 uses
  its paper-specific file and inferred correction labels; AS06 uses its current
  file and recorded correction flags. These are input/execution checks, not
  evidence of population-level identifiability.
- Submission dry runs confirm 72 smoke tasks; 684 old-cohort tasks (2,052
  datasets); 450 new-cohort tasks (900 datasets). All request one CPU per task;
  no Slurm jobs were submitted.
- Compared 12 source/definition files against the preserved v0.2.3 archive:
  model equations, optimizer, data loader, preprocessing, evaluation and model
  definitions are byte-for-byte unchanged. Only recovery recording/auditing,
  collection/CLI, release labels, tests, submission presets and documentation
  were extended.

The v0.2.3 archive is preserved with SHA256
`4886e1210e8593ffe34339c183777e840acb141e68484e9aa56a1e186f8a2397`.
Old empirical outputs are preserved separately. This release does not refit
those results or claim the full 18-model recovery study is complete.

## Evidence and next gate

Resolved plans, raw simulation/fit records, environment versions, verification
CSVs and quality JSONs are under the two named `results/runs/` folders. Results
and mouse data are excluded from the code archive and Git.

Run the Firebird tests and smoke using the installed cluster environment before
submitting full cohorts. Numerical checks must pass there independently. The
full old/new experiments, prior-distribution sensitivity, empirical-parameter
recovery and chronological-CV model recovery remain unperformed in this stage.
See `MODEL_RECOVERY.md` for the experimental assumptions and commands.
