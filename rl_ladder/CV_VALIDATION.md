# CV record release validation

Validated locally on 2026-10-22. Release: 0.2.1. Scientific fingerprint: `23071acecaa6`.

## Automated checks

- 265 tests passed, including the existing model/reference, nesting and original-paper checks.
- New CV tests cover persisted JSON score replay without an optimizer, altered scores/parameters/splits/start records/seeds, changed masked-trial data, invalid splits, and held-out-choice leakage.
- An exactly reproducible but unconverged fit is rejected by the collection quality gate.
- CV-only execution creates no full-data fit/cache outputs. Completed CV units can be resumed without refitting.

## Actual-data checks

| Check | Cases | Result |
| --- | --- | --- |
| Small CV experiment | A0, A+C, S+F on GS029; two reset policies | 6/6 converged and verified |
| Data-layout coverage | A+C on GS029, JC039, GS028, AS06 | 4/4 converged and verified |
| Previously sensitive examples | S+F on GS028 and JC067; 20 starts | 2/2 converged and verified |

GS029 uses known correction flags. JC039 uses inferred flags. GS028 uses the
published-window override and inferred flags. AS06 uses the new-mouse layout
and known flags. Each input's deterministic rewards and chronological sessions
are validated by the existing loader.

The saved-parameter checks reproduce all 12 train/test scores. GS028 and JC067
also match their previous v0.2.0 **local replay** test NLL exactly in this same
environment. This establishes that the new record keeping did not change those
CV optimizations; it does not establish an exact match to their original
Firebird training optima, whose parameters were not archived.

## Prepared cluster run

The full CV-only plan has 456 units (19 mice x 12 models x 2 policies), 20 starts
each. The shell wrapper requests 1 CPU and 2 GB per array task, chunks 5 units
per task, and caps concurrent array tasks at 8. The time request is 1 hour per
task. Submission is left to the user; no cluster job was submitted locally.

Records from these local checks are in `results/runs/` under these run names:

- `smoke_cv_audit_v021_finalcheck`
- `cv_layout_validation_v021_finalcheck`
- `cv_replay_regression_v021_finalcheck`

The prior v0.2.0 archive and downloaded runs remain intact. See
[CV_RECORDS.md](CV_RECORDS.md) for the new execution and verification steps.
