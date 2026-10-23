# Model definitions and interpretation

## Common timing and coding

`s` and `c` are 0 for Low and 1 for High. The prepared learning data satisfy
`r = 1[c == s]`; the loader checks this. Signed choice is `y = 2c - 1`,
signed feedback is `f = 2r - 1`. Every trial is predicted from states available
before its choice and feedback. State updates then use the completed trial.

All models choose High with probability `sigmoid(z)`. Bias `b` is in logit
units except in A/B, where the historical implementation gives `10*b` at
the fixed temperature of 5. C, V, and F contributions are added in logit units.

## Value-learning bases

- **S0:** `z = beta_q * (Q[s,H] - Q[s,L]) + b`.
  Only `Q[s,c] += alpha * (r - Q[s,c])` changes. Initially each stimulus has
  `Q[s,H] = 0.5 + q0/2`, `Q[s,L] = 0.5 - q0/2`.
- **A0:** `z = tau_a * (m + 2*b)`, with `m = Q[H]` on High trials and
  `m = -Q[L]` on Low trials. Initially `Q = [q0, -q0]`; the chosen action
  changes by `alpha * delta`, where `delta = r - Q[c]`. `tau_a = 5` by default.
- **B0:** same initialization, readout, and chosen update as A0, plus
  `Q[1-c] -= gamma * alpha * delta`, using the same pre-update delta.
  This is Collina et al.'s Choice-history model, previously named A+R.
  `gamma = exp(-beta_original)`; gamma is part of the base, not an additive term.
  At gamma zero, B0 exactly reduces to A0. At gamma one the signed initial
  values keep `Q[L] + Q[H] = 0`, making the current readout stimulus independent;
  this does not make the update identical to standard chosen-action RW.

S0cf adds a counterfactual update of the unchosen S action. A0T frees `tau_a`.
These remain optional extensions.

## Independent history modules

### Action value V (the R label in the S family)

`V[c] += alpha_v * (r - V[c])`, starting from `[0.5, 0.5]`.
Its logit contribution is `beta_v * (V[H] - V[L])`. It pools experience across
stimulus categories and updates only the chosen action.

The complete S-family experiment includes all eight switches of R, C and F:
S0, S+R, S+C, S+F, S+R+C, S+R+F, S+F+C and S+R+F+C. Each module adds two
parameters. Matched comparisons toggle just one module while holding the
other two fixed. V/R and F are distinct: V depends on the history of which
action was updated, whereas F reduces to stimulus history in this task.

### Choice history C

`k_next = (1 - alpha_c) * k + alpha_c * y`, starting from zero.
Its logit contribution is `beta_c * k`. Positive beta_c means repetition,
negative beta_c means alternation; neither depends on reward. C has not changed
from version 0.1. It corresponds to the P process in Lebedeva et al. (2026):
`P = beta_c * k`, `alpha_p = alpha_c`, `lambda_p = beta_c`.

### Choice-feedback history F

`h_next = (1 - alpha_f) * h + alpha_f * y * f`, starting from zero.
Its logit contribution is `beta_f * h`. This normalized implementation is
equivalent to the paper's weighted-state equation:

`R_next = R + alpha_r * (lambda_r * C * F - R)`

with `R = beta_f * h`, `alpha_r = alpha_f`, `lambda_r = beta_f`.
The independent Python reference uses the weighted-state equation directly.
The new module does not replace V or C and can coexist with both and with B.

F adds two fitted parameters. The chosen search ranges are `alpha_f` in
`[1e-6, 1]` (optimized in log10 coordinates), and `beta_f` in `[0, 20]`
(linear coordinates). The weight is nonnegative to define a reward-seeking
history process; C retains its signed weight. These are our search bounds,
not a claim about the bounds used by the published authors. Random starts use
log-uniform alpha_f in `[0.001, 0.3]` and uniform beta_f in `[0, 3]`.
At beta_f zero, F is off exactly, and its update rate cannot be identified.

**Task-specific limitation.** Because rewards in our learning data are
deterministic, `(2c-1)*(2r-1) = 2s-1`. F is exactly an exponential stimulus-history
filter, independent of which choice sequence produced that feedback. A good fit
cannot distinguish those two interpretations. No current stimulus enters F's
prediction before the current trial is completed. Correction trials can make
the previous stimulus predictive of the current one and require sensitivity
analysis.

### One-trial WSLS

M2 uses the previous `y*f`, with probability `1 - epsilon/2` of following
win-stay/lose-shift. It starts at chance each session. If alpha_f is fixed at
one, F uses only the previous trial; with `b = 0` and
`beta_f = log((1-epsilon/2)/(epsilon/2))`, it matches M2 when the resets match
and the resulting weight lies within F's search bounds. C alone is not a WSLS
model, even when its update rate equals one.

## Session boundaries and correction trials

Default resets are `[wsls, feedback]`: F resets to zero each session; the
existing Q, V, and C carry policies are preserved. To reproduce the PR model's
P/R initialization, use `[wsls, choice, feedback]` for F+C. The original
Collina replication preset uses no resets and includes all learning trials.
`feedback_ladder` compares both C-reset policies; `sensitivity` also examines
F carryover and correction masking.

Correction trials update states under the default `mask` policy but do not
contribute to likelihood. This does not eliminate their effects on later
predictions. `include` and `drop` are separate sensitivity settings. Data with
missing trial-type metadata retain the existing inferred-correction policy.

## Fitting, comparison, and recovery

The shared multistart MLE fitter, chronological CV, and BIC/AIC calculations
are shared by all combinations. Each F variant has two more free parameters
than its parent; the optional fixed-temperature S variants have one fewer
than their corresponding free-temperature models. B nests A at gamma zero.
Parent fits initialize added modules at zero weight, and plots retain all
fitted models, including optional combinations.

In `reward_ladder`, CV uses additional starts from nested **training-only**
fits. Full-data optima never initialize CV. The same training split, scored
trial mask, reset policy and number of random starts apply to every model.
CV parent fits are cached separately from full-data fits, and the record
stores each parent's training NLL and its initial-vector index. Verification
checks the nested candidate scores and rejects a selected training optimum
more than 1e-4 NLL worse than a feasible parent. This is an optimization gate,
not proof of global optimality or identifiability. The original CV behavior
remains available with `evaluate.cv.warm_start_from_nested=false`.

Recovery simulates deterministic auditory-category learning with either real
stimuli (`replay`) or the reactive correction rule (`reactive`). It is not a
replication of the 80/20 probabilistic reversal environment in the PR paper.
Prior recovery's accuracy filter conditions the sampled parameter population;
use `[0, 1]` to disable selection. Models and parameters can be inseparable at
zero weights or other degenerate settings. A smoke run only checks execution;
it does not establish identifiability or a biological conclusion.

F's history confound, the S learning-rate/temperature tradeoff, convergence,
parameter bounds, held-out prediction, and simulated behavioral summaries
must all inform interpretation. Fixed temperature can improve stability but
does not prove a unique solution.

## Reproducibility and older results

Version 0.2 adds F and renames A+R to B0 and A+R+C to B+C. Historical CLI
names are aliases and do not create duplicate recovery candidates. The raw
archived results keep their original names. Updated source and registry hashes
create new cache keys. Run manifests prevent replacing an older run with a
new code version. An old plan cannot execute against changed model code;
create a new plan with a new run name, for example `--name-suffix _v020`.

## Sources

- Collina et al. (2025), [Individual-specific strategies inform category learning](https://www.nature.com/articles/s41598-024-82219-8).
- Wilson and Collins (2019), [Ten simple rules for the computational modeling of behavioral data](https://elifesciences.org/articles/49547).
- Lebedeva et al. (2026), [Dorsal prefrontal cortex drives perseverative behavior in mice](https://www.nature.com/articles/s41467-026-71664-w).
