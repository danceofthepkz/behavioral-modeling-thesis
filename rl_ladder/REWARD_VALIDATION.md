# Reward ladder release validation

Release 0.2.2; validated locally on 2026-10-22. Scientific fingerprint:
`0e7f4f57de98`.

## Automated tests

276 tests passed, including the original-paper 38-likelihood gate and the
M1/M2 baseline gate. New tests check:

- The full S+R+F+C prediction/update rule against independent hand calculations.
- R differs from F under deterministic feedback; zero R weight recovers each matched control exactly.
- Both the free-temperature and fixed-temperature plans include all eight combinations and all 12 matched contrasts.
- Nested CV uses training-only parent fits; changing held-out choices leaves every training solution unchanged.
- CV parent records survive JSON key sorting and are cached independently of full-data fits.
- Full-fit collection rejects a failed convergence flag.

## Actual-data integration checks

| Run | Outputs | Result |
| --- | ---: | --- |
| smoke_reward_ladder_v022_finalcheck | 16 full fits with CV | All eight S models on GS029, both reset policies; full/CV quality checks passed |
| reward_new_layout_v022 | 8 full fits with CV | All eight S models on new mouse AS06; full/CV quality checks passed |
| reward_inferred_ct_v022 | 2 full fits with CV | S+R on GS028/JC039 with 20 random starts; published-window override and inferred correction metadata passed |
| nested_parent_regression_v022 | 1 CV fit | B+F on JC062 no longer violates its feasible training-parent scores |

These are 27 top-level validation outputs, with additional cached parent fits.
All 27 saved CV scores reproduce, and all 26 top-level full-data fits converge
and pass their nested-likelihood checks. The plotting smoke check produced
all 24 module contrasts (12 comparisons x 2 reset policies), including R.
The one-mouse plot is an execution check, not a scientific population result.

## Prepared run and limits

The Firebird smoke plan has 16 outputs. The full plan has 304 outputs:
19 mice x eight S models x two reset policies. Of these, 152 contain R and
152 are matched controls. Each output contains full-data fitting and
chronological CV with 20 random starts plus feasible parent starts.

The full wrapper requests one CPU, 2 GB and two hours per array task,
four outputs per task, at most eight tasks running. No cluster task was
submitted locally. Execution validation does not prove identifiability;
recovery and fixed-temperature sensitivity presets are prepared separately.

Earlier release archives and downloaded experiments remain separate. See
[REWARD_LADDER.md](REWARD_LADDER.md) for submission and verification steps.
