# Graph Scripts

The scripts in this directory generate figures and text summaries for Shor
simulator diagnostics. Measurement bit-flip studies live under
`graphs/measurement_error/`; depolarizing-error studies live under
`graphs/depolarizing_error/`. Run them from the repository root with:

```bash
python graphs/measurement_error/<script>.py ...
python graphs/depolarizing_error/<script>.py ...
```

All scripts use Matplotlib's non-interactive `Agg` backend and set
`MPLCONFIGDIR` to `/tmp/matplotlib-shor`, so they work in headless terminals.
When `--output` is omitted, figures are written under `graphs/figures/`.
Every script also writes a text summary under `graphs/data/`; custom `--output`
paths only change the figure path.

Unless noted otherwise, the default phase-bit count is
`2 * ceil(log2(N)) + 1`, `--seed` defaults to random sampling, and
`--measurement-bitflip-error` is disabled by default. Passing
`--measurement-bitflip-error` without a value applies exactly one random
measurement bit flip per phase-estimation run; passing a number `P` applies
independent per-bit measurement flips with probability `P`.

## Measurement Error Scripts

### `plot_phase_estimation_distribution.py`

Plots the empirical Kitaev phase-estimation bit-string distribution for one
fixed `(N, a)`.

- Required: `N`, `--a`
- Defaults: `--bits` = `2 * ceil(log2(N)) + 1`, `--trials 1000`,
  `--seed` = random, measurement bit-flip error disabled
- Other options: `--output`, `--measurement-bitflip-error [P]`

### `plot_one_shot_success_vs_bits.py`

Plots Kitaev one-shot factorization success as the number of phase-estimation
bits varies.

- Required: `N`
- Defaults: `--a` omitted, meaning each trial samples a fresh random coprime
  base; `--min-bits 1`; `--max-bits` = `2 * ceil(log2(N)) + 1`;
  `--trials-per-bits 100`; `--seed` = random
- Other options: `--output`, `--measurement-bitflip-error [P]`

### `plot_one_shot_success_vs_attempts.py`

Plots one-shot factorization success as the allowed number of independent
top-level Shor attempts varies. It can overlay Kitaev and standard Shor curves:

```bash
python graphs/measurement_error/plot_one_shot_success_vs_attempts.py 21 --algorithm kitaev standard
```

- Required: `N`
- Defaults: `--algorithm kitaev`, `--standard-backend statevector`,
  `--analytic-window 100`, `--min-attempts 1`, `--max-attempts 10`,
  `--trials-per-attempt-count 100`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--seed` = random
- Other options: `--output`, `--measurement-bitflip-error [P]`
- `--algorithm` accepts one or more of `kitaev`, `standard`
- `--standard-backend analytic` avoids allocating the full standard-Shor
  state vector by sampling standard-QPE measurement outcomes from the
  classically computed order. This is useful for large control-register counts
  and measurement-error studies.

### `plot_order_recovery_success_vs_order.py`

Plots one-shot order-recovery success versus the true multiplicative order `r`.
In random mode, the script samples odd composite non-perfect-power moduli `N`
and random coprime bases `a`. For controlled experiments, use
`--instances-file` to supply one representative `(N, a)` pair per order. In both
modes, the script computes the true order classically for binning and then runs
one-shot Kitaev and/or standard Shor order finding.

```bash
python graphs/measurement_error/plot_order_recovery_success_vs_order.py --algorithm kitaev standard --standard-backend analytic --bits 29
```

- Required: none
- Defaults: `--algorithm kitaev`, `--bits 11`, `--num-instances 50`,
  `--trials-per-instance 100`, `--min-N 9`, `--max-N` = largest modulus
  satisfying `2 * ceil(log2(N)) + 1 <= --bits`, `--standard-backend statevector`,
  `--analytic-window 100`, `--max-statevector-amplitudes 8000000`,
  `--seed` = random
- Other options: `--output`, `--measurement-bitflip-error [P]`, `--min-order`,
  `--max-order`, `--instances-file`, `--error-probabilities`, `--log-x`
- `--algorithm` accepts one or more of `kitaev`, `standard`
- `--error-probabilities` overlays independent per-bit measurement flip
  probabilities in one figure. Include `0` for the clean curve. It is mutually
  exclusive with `--measurement-bitflip-error`.
- When multiple probabilities are plotted, color identifies measurement error
  probability. Standard Shor uses a lighter shade of the same probability color,
  while marker/line style identifies the algorithm. A combined legend box is
  placed in the right margin of the figure.
- The x-axis bins by exact true order `r`; all sampled `(N, a)` instances with
  the same `r` contribute to that order's success probability
- Success means one-shot continued-fraction post-processing recovers exactly
  the true order `r`, not merely factors of `N`
- `--bits` is fixed globally, so measurement-error comparisons use the same
  number of measured phase bits for every sampled instance and algorithm
- `--instances-file` accepts JSON, either as a bare list or as an object with an
  `"instances"` list. Each entry needs integer `N` and `a` fields. Optional `r`
  or `order` fields are verified against the computed true order.
- In file mode, the script requires at most one representative pair per true
  order. `--num-instances`, `--min-N`, and `--max-N` only control random mode.
- `--standard-backend analytic` is recommended for larger `--bits`; the
  statevector backend has a safety guard to avoid accidental huge allocations
- When `--standard-backend analytic` is used for standard Shor, the data summary
  reports how much exact standard-QPE probability mass is captured by the
  finite `--analytic-window` for each tested order

### `plot_cf_bound_order_recovery_vs_order.py`

Plots continued-fraction sufficient-bound diagnostics versus true order `r`
for configured representative `(N, a)` pairs. The script generates one figure
per selected measurement bit-flip probability and writes one combined text
summary. It is meant to show whether recovery is happening because the measured
phase remains inside the standard CF guarantee, or because continued-fraction
post-processing still succeeds outside that sufficient bound.

```bash
python graphs/measurement_error/plot_cf_bound_order_recovery_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29 --error-probabilities 0 0.05 0.1
```

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--algorithm kitaev standard`, `--bits 29`, single-probability
  fallback `--p 0.1`, `--trials-per-instance 100`,
  `--standard-backend analytic`, `--analytic-window 100`,
  `--max-statevector-amplitudes 8000000`, `--seed` = random
- Other options: `--output`, `--min-order`, `--max-order`,
  `--error-probabilities`, `--overlay-error-probabilities`, `--log-x`
- `--p P` generates one plot. `--error-probabilities P ...` generates one plot
  per probability and is mutually exclusive with `--p`.
- `--overlay-error-probabilities` also generates four extra plots, one for
  each diagnostic quantity, with all selected measurement probabilities
  overlaid in the same figure.
- The plotted events are:
  `A = 2*r^2*min_s distance_mod_1(y/2^t, s/r) <= 1`,
  `B = recovered_order == r`, `B and not A`, and `B | not A`
- `Pr(B)` is the same one-shot true-order recovery probability plotted by
  `plot_order_recovery_success_vs_order.py`
- In the default one-plot-per-probability figures, color identifies the
  diagnostic quantity. In the optional quantity-overlay figures, color
  identifies measurement error probability. Standard Shor uses a lighter shade
  of the same color, while marker/line style identifies the algorithm.
- With multiple probabilities and a custom `--output`, the probability token is
  appended to the output stem so each figure has a separate file.
- With `--standard-backend analytic`, the data summary includes captured-mass
  diagnostics for the finite `--analytic-window`

### `plot_order_recovery_loss_vs_order.py`

Plots how much measurement bit-flip noise reduces one-shot true-order recovery
for configured representative `(N, a)` pairs. For each selected algorithm and
instance, the script estimates a clean recovery probability once, then estimates
noisy recovery probabilities for each requested measurement bit-flip
probability. The plotted value is:

```text
clean recovery probability - noisy recovery probability
```

```bash
python graphs/measurement_error/plot_order_recovery_loss_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29
```

- Required: `--instances-file`
- Defaults: `--algorithm kitaev`, `--bits 11`, `--error-probabilities 0.001 0.01 0.05 0.1`,
  `--trials-per-instance 100`, `--standard-backend statevector`,
  `--analytic-window 100`, `--max-statevector-amplitudes 8000000`,
  `--seed` = random
- Other options: `--output`, `--min-order`, `--max-order`, `--log-x`
- `--instances-file` uses the same JSON format as
  `plot_order_recovery_success_vs_order.py`
- `--error-probabilities` are independent per-bit measurement flip probabilities
- Color identifies measurement error probability. Standard Shor uses a lighter
  shade of the same probability color, while marker/line style identifies the
  algorithm. A combined legend box is placed in the right margin of the figure.
- Negative loss values can occur from finite-trial sampling noise
- For standard Shor with larger `--bits`, use `--standard-backend analytic`
- With `--standard-backend analytic`, the data summary includes captured-mass
  diagnostics for the finite `--analytic-window`

### `plot_phase_hamming_distance_vs_order.py`

Plots average clean/noisy phase-estimation Hamming distance versus true order
`r` for configured representative `(N, a)` pairs. This is intended to compare
measurement-bitflip propagation in Kitaev versus standard Shor before
continued-fraction post-processing.

```bash
python graphs/measurement_error/plot_phase_hamming_distance_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --bits 29 --error-probabilities 0.01 0.05 0.1
```

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--algorithm kitaev standard`, `--bits 29`, single-probability
  fallback `--p 0.1`,
  `--trials-per-instance 100`, `--standard-backend analytic`,
  `--analytic-window 100`, `--max-statevector-amplitudes 8000000`,
  `--seed` = random
- Other options: `--output`, `--min-order`, `--max-order`,
  `--error-probabilities`, `--normalized`, `--log-x`
- `--p P` plots one measurement error probability. `--error-probabilities`
  overlays multiple independent per-bit measurement flip probabilities in one
  figure and is mutually exclusive with `--p`.
- When multiple probabilities are plotted, color identifies measurement error
  probability. Standard Shor uses a lighter shade of the same probability color,
  while marker/line style identifies the algorithm. A combined legend box is
  placed in the right margin of the figure.
- Kitaev noisy trials store flipped measurement bits, so later feed-forward
  phases can change; standard noisy trials flip only the final measured bit
  string
- With `--standard-backend analytic`, the data summary includes captured-mass
  diagnostics for the finite `--analytic-window`

### `plot_phase_decimal_distance_vs_order.py`

Plots average circular decimal distance between clean and noisy phase estimates
versus true order `r` for the same configured instance format. For adaptive
Kitaev feedback this is a paired-trajectory diagnostic rather than a canonical
distance to the unerrored phase.

```bash
python graphs/measurement_error/plot_phase_decimal_distance_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --bits 29 --error-probabilities 0.01 0.05 0.1
```

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--algorithm kitaev standard`, `--bits 29`, single-probability
  fallback `--p 0.1`,
  `--trials-per-instance 100`, `--standard-backend analytic`,
  `--analytic-window 100`, `--max-statevector-amplitudes 8000000`,
  `--seed` = random
- Other options: `--output`, `--min-order`, `--max-order`,
  `--error-probabilities`, `--classical-error-correction`, `--log-x`,
  `--include-half-inverse-order-reference`
- `--p P` plots one measurement error probability. `--error-probabilities`
  overlays multiple independent per-bit measurement flip probabilities in one
  figure and is mutually exclusive with `--p`.
- When multiple probabilities are plotted, color identifies measurement error
  probability. Standard Shor uses a lighter shade of the same probability color,
  while marker/line style identifies the algorithm. A combined legend box is
  placed in the right margin of the figure.
- Decimal distance is `min(|x-y|, 1-|x-y|)` after converting the clean and
  noisy bit strings to binary phase fractions
- With `--standard-backend analytic`, the data summary includes captured-mass
  diagnostics for the finite `--analytic-window`

### `plot_nearest_peak_decimal_distance_vs_order.py`

Plots the average circular distance from each phase-estimation bit string to
its own nearest exact order eigenphase. Without classical correction, each
selected measurement error probability overlays:

```text
Kitaev:       d(b, s/r)
Kitaev:       d(tilde b, s'/r)
standard:     d(b, s/r)
standard:     d(tilde b, s'/r)
```

Here `s/r` is the nearest exact order eigenphase to the clean bit string `b`,
and `s'/r` is the nearest exact order eigenphase to the noisy bit string
`tilde b`.

```bash
python graphs/measurement_error/plot_nearest_peak_decimal_distance_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29 --error-probabilities 0.01 0.05 0.1
```

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--algorithm kitaev standard`, `--bits 29`, single-probability
  fallback `--p 0.1`, `--trials-per-instance 100`,
  `--standard-backend analytic`, `--analytic-window 100`,
  `--max-statevector-amplitudes 8000000`, `--seed` = random
- Other options: `--output`, `--min-order`, `--max-order`,
  `--error-probabilities`, `--classical-error-correction`, `--log-x`
- `--p P` generates one plot. `--error-probabilities P ...` generates one plot
  per probability and is mutually exclusive with `--p`.
- The plotted distance is `min_s distance_mod_1(y/2^t, s/r)`, evaluated
  separately for the clean and noisy bit strings.
- With `--classical-error-correction`, the figure is corrected-only: it plots
  `d(b_ec, s_ec/r)` for Kitaev and standard Shor, where `b_ec` is the noisy bit
  string or one of its one-bit-flip candidates with the smallest distance to
  its own nearest exact order eigenphase. The noisy bit string wins ties.
- Each figure includes the continued-fraction sufficient bound `1/(2r^2)`.
  Add `--include-half-inverse-order-reference` to also plot the solid
  half-inverse-order reference `1/(2r)`. In the vs-order plot, reference curves
  are truncated to the distance-derived y-axis so they do not flatten the
  measured curves.
- This avoids interpreting noisy adaptive Kitaev bits as bitwise errors against
  one arbitrary clean string.
- With `--standard-backend analytic`, the data summary includes captured-mass
  diagnostics for the finite `--analytic-window`

### `plot_nearest_peak_decimal_distance_vs_error_probability.py`

For one fixed input number `N` and coprime base `a`, plots the same nearest-peak
distance diagnostic against measurement bit-flip probability `p`. The
continued-fraction sufficient bound `1/(2r^2)` is plotted as a horizontal
reference line. Add `--include-half-inverse-order-reference` to also plot a
solid horizontal `1/(2r)` reference. Without classical correction, the plot
contains clean/noisy distance curves.

```bash
python graphs/measurement_error/plot_nearest_peak_decimal_distance_vs_error_probability.py 21 --a 11 --algorithm kitaev standard --standard-backend analytic --bits 11 --min-p 0 --max-p 0.2 --num-points 11 --trials 100
```

- Required: `N`, `--a`
- Defaults: `--algorithm kitaev standard`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--trials 100`, `--min-p 0`, `--max-p 1`,
  `--num-points 11`, `--standard-backend analytic`, `--analytic-window 100`,
  `--max-statevector-amplitudes 8000000`, `--seed` = random
- Other options: `--output`, `--classical-error-correction`, `--log-p-grid`,
  `--include-half-inverse-order-reference`
- `--log-p-grid` uses logarithmic spacing and requires `--min-p > 0`
- The plotted clean/noisy distances are measured to their own nearest exact
  order eigenphases, as in
  `plot_nearest_peak_decimal_distance_vs_order.py`
- With `--classical-error-correction`, the plot is corrected-only, using the
  same one-bit minimum-distance diagnostic.

### `plot_noisy_cf_membership_bar.py`

For one fixed input number `N`, coprime base `a`, and measurement bit-flip
probability `p`, plots the empirical proportions of noisy outputs `tilde b`
inside and outside the continued-fraction sufficient set:

```text
S = {tilde b : d(tilde b, s'/r) <= 1/(2r^2), where s'/r is nearest to tilde b}
```

The chart has two membership bars, `S^c` and `S`, with Kitaev and standard Shor
overlaid side by side.

```bash
python graphs/measurement_error/plot_noisy_cf_membership_bar.py 21 --a 11 --p 0.05 --algorithm kitaev standard --standard-backend analytic --bits 11 --trials 1000 --seed 123
```

- Required: `N`, `--a`
- Defaults: `--algorithm kitaev standard`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--p 0.1`, `--trials 1000`,
  `--standard-backend analytic`, `--analytic-window 100`,
  `--max-statevector-amplitudes 8000000`, `--seed` = random
- Other options: `--output`, `--classical-error-correction`
- The plotted quantity is computed only from the noisy bit string `tilde b`; the
  clean paired bit string is not used for membership.
- With `--classical-error-correction`, additional corrected bars are plotted.
  They use the noisy string or one of its one-bit-flip candidates, whichever has
  the smallest distance to its own nearest exact order eigenphase. The noisy bit
  string wins ties.

### `plot_noisy_cf_membership_vs_order.py`

Plots the empirical outside-CF membership probability `Pr(tilde b in S^c)`
versus true order `r` for a configured instance file. This is useful for
checking whether a forced measurement-error window pushes outputs across the CF
threshold as the order scale changes.

```bash
python graphs/measurement_error/plot_noisy_cf_membership_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29 --p 0.01 --reference-curves mp 2mp --include-safe-tail-propagation-reference --trials-per-instance 100 --seed 123
```

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--algorithm kitaev standard`, `--bits 29`, `--p 0.1`,
  `--trials-per-instance 100`, `--standard-backend analytic`,
  `--analytic-window 100`, `--max-statevector-amplitudes 8000000`,
  `--seed` = random
- Other options: `--output`, `--single-measurement-bitflip-range`,
  `--classical-error-correction`, `--reference-curves`,
  `--include-safe-tail-propagation-reference`, `--log-x`
- `--single-measurement-bitflip-range` accepts one or more choices from `all`,
  `leading`, `dangerous`, and `safe-tail`. For `r = 2^m`, these correspond to
  phase-bit ranges `1..t`, `1..m`, `m+1..2m`, and `2m+1..t` respectively. For
  general `r`, the script uses `m = ceil(log2(r))`. Multiple choices, for
  example `leading dangerous`, mean one bit is sampled uniformly from the union
  of those ranges.
- With `--classical-error-correction`, corrected-min curves are overlaid using
  the same one-bit minimum-distance diagnostic as the distance plots.
- `--reference-curves mp 2mp` overlays `1 - exp(-mp)` and/or
  `1 - exp(-2mp)`, where `m = ceil(log2(r))` and `p` is the value passed with
  `--p`. In forced single-bitflip range mode, `--p` is used for these
  references, while the sampled data still uses exactly one forced bitflip.
- `--include-safe-tail-propagation-reference` additionally overlays
  Kitaev-specific references
  `1 - exp(-alpha*m*p - p*(t-2m)*rho_tail)`, with `alpha` selected by
  `--reference-curves`. The main plotted data can remain the independent-`p`
  `Pr(S^c)` curve; the script estimates `rho_tail` internally from auxiliary
  Kitaev safe-tail one-bitflip runs for each order and correction series.

### `plot_noisy_cf_complement_fill.py`

For one fixed input number `N`, coprime base `a`, and measurement bit-flip
probability `p`, measures how broadly noisy outputs fill the complement of the
continued-fraction sufficient set. The plotted quantity is:

```text
|unique noisy bitstrings observed in S^c| / |S^c|
```

This is a coverage/occupancy diagnostic, not a probability. It is useful for
checking whether Kitaev feedback error propagation spreads outside-CF outputs
across many possible bitstrings or concentrates them on a small subset.

```bash
python graphs/measurement_error/plot_noisy_cf_complement_fill.py 21 --a 11 --p 0.1 --algorithm kitaev standard --standard-backend analytic --bits 11 --trials 1000 --seed 123
```

- Required: `N`, `--a`
- Defaults: `--algorithm kitaev standard`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--p 0.1`, `--trials 1000`,
  `--standard-backend analytic`, `--analytic-window 100`,
  `--max-statevector-amplitudes 8000000`, `--seed` = random
- Other options: `--output`, `--cache-dir`, `--rebuild-cache`,
  `--max-cache-bitstrings`, `--classical-error-correction`,
  `--single-measurement-bitflip`, `--single-measurement-bitflip-range`
- With `--single-measurement-bitflip`, each trial uses exactly one uniformly
  random measurement readout bit flip instead of independent per-bit flips from
  `--p`. For standard Shor this means one final control-register bit is flipped;
  for Kitaev, the one readout error occurs during the adaptive sequence and can
  propagate into later reported phase bits through feedback.
- `--single-measurement-bitflip-range` restricts the forced one-flip mode to
  one or more of `all`, `leading`, `dangerous`, or `safe-tail`, with the same
  range convention as `plot_noisy_cf_membership_vs_order.py`.
- The S/S^c membership universe is cached as a packed bitset under
  `graphs/cache/cf_membership/`, keyed by the true order `r`, phase-bit count
  `t`, and cache version.
- Building the cache enumerates all `2^t` bitstrings. The default guard refuses
  to build caches larger than 5,000,000 bitstrings; use
  `--max-cache-bitstrings 0` only when you intentionally want to disable that
  guard.
- The data summary also reports `Pr(S^c)`, the trial probability of landing
  outside S, so the coverage fraction can be interpreted alongside the amount
  of outside-CF probability mass.
- With `--classical-error-correction`, additional corrected bars are plotted
  using the same one-bit minimum-distance diagnostic as the distance plots.

### `plot_noisy_cf_complement_fill_vs_trials.py`

For one fixed input number `N`, coprime base `a`, and measurement bit-flip
probability `p`, plots the fraction of `S^c` uniquely hit as the number of
trials per algorithm changes. Each requested trial count is estimated with an
independent run for each algorithm, so the plotted curves are not forced to be
monotone.

```bash
python graphs/measurement_error/plot_noisy_cf_complement_fill_vs_trials.py 21 --a 11 --p 0.1 --algorithm kitaev standard --standard-backend analytic --bits 11 --trial-counts 10 20 50 100 200 500 1000 --seed 123 --log-x
```

- Required: `N`, `--a`
- Defaults: `--algorithm kitaev standard`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--p 0.1`,
  `--trial-counts 10 20 50 100 200 500 1000`,
  `--standard-backend analytic`, `--analytic-window 100`,
  `--max-statevector-amplitudes 8000000`,
  `--max-cache-bitstrings 5000000`, `--seed` = random
- Other options: `--output`, `--cache-dir`, `--rebuild-cache`,
  `--max-cache-bitstrings`, `--classical-error-correction`,
  `--single-measurement-bitflip`, `--single-measurement-bitflip-range`,
  `--log-x`
- This script uses the same exhaustive packed-bitset cache as
  `plot_noisy_cf_complement_fill.py`: it explicitly determines membership for
  all `2^t` bitstrings when the cache is first built.
- For large bit counts such as `t = 29`, the default guard will refuse to build
  the cache unless `--max-cache-bitstrings` is increased or set to `0`.
- The plotted quantity is coverage,
  `|unique noisy bitstrings observed in S^c| / |S^c|`, not probability mass.
  The text summary also reports `Pr(S^c)` for each trial count.
- With `--single-measurement-bitflip`, each independent run uses exactly one
  uniformly random measurement readout bit flip per trial. This is the mode to
  compare the standard support bound against Kitaev feedback propagation.
- `--single-measurement-bitflip-range` restricts that one-flip mode to one or
  more of `all`, `leading`, `dangerous`, or `safe-tail`.
- With `--classical-error-correction`, additional corrected coverage curves are
  plotted using the same one-bit minimum-distance diagnostic as the distance
  plots.

### `plot_eigenphase_peak_diagnostics_vs_order.py`

Plots clean/noisy nearest-eigenphase diagnostics versus true order `r` for
configured representative `(N, a)` pairs. The script uses paired clean/noisy
phase-estimation trials: `s` is the nearest eigenphase numerator to the clean
bit string and `s'` is the nearest eigenphase numerator to the noisy bit
string. Here `A` means the noisy phase lies inside the continued-fraction
sufficient bound around `s'/r`.

```bash
python graphs/measurement_error/plot_eigenphase_peak_diagnostics_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29 --error-probabilities 0 0.05 0.1
```

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--algorithm kitaev standard`, `--bits 29`, single-probability
  fallback `--p 0.1`, `--trials-per-instance 100`,
  `--standard-backend analytic`, `--analytic-window 100`,
  `--max-statevector-amplitudes 8000000`, `--seed` = random
- Other options: `--output`, `--min-order`, `--max-order`,
  `--error-probabilities`, `--log-x`
- `--p P` plots one measurement error probability. `--error-probabilities`
  overlays multiple independent per-bit measurement flip probabilities in
  every generated figure and is mutually exclusive with `--p`.
- The script generates one figure for each diagnostic:
  `Pr(s' = s)`, `Pr(s' != s and A)`, `Pr(gcd(s', r) = 1)`, and
  `Pr(B | s' != s and A)`, where `B = recovered_order == r`.
- `Pr(gcd(s', r) = 1)` is evaluated for the nearest noisy peak `s'`
  regardless of whether the noisy phase is inside the CF sufficient bound.
- Color identifies measurement error probability. Standard Shor uses a lighter
  shade of the same probability color, while marker/line style identifies the
  algorithm.
- With a custom `--output`, the diagnostic token is appended to the output
  stem so each figure has a separate file.
- With `--standard-backend analytic`, the data summary includes captured-mass
  diagnostics for the finite `--analytic-window`

### `plot_conditional_cf_bound_comparison_vs_order.py`

Directly compares the conditional CF-bound survival probabilities for the same
peak and shifted peak cases. It writes two figures: one for the same-peak
condition and one for the shifted-peak condition.

```text
standard Shor: Pr(A | s' = s)
Kitaev Shor:   Pr(A | s' = s)
standard Shor: Pr(A | s' != s)
Kitaev Shor:   Pr(A | s' != s)
```

Here `s` is the nearest eigenphase numerator to the clean bit string, `s'` is
the nearest eigenphase numerator to the noisy bit string, and `A` means the
noisy phase lies inside the continued-fraction sufficient bound around `s'/r`.

```bash
python graphs/measurement_error/plot_conditional_cf_bound_comparison_vs_order.py --instances-file graphs/config/order_recovery_instances.json --bits 29 --error-probabilities 0.01 0.05 0.1
```

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--bits 29`, single-probability fallback `--p 0.1`,
  `--trials-per-instance 100`, `--standard-backend analytic`,
  `--analytic-window 100`, `--max-statevector-amplitudes 8000000`,
  `--seed` = random
- Other options: `--output`, `--min-order`, `--max-order`,
  `--error-probabilities`, `--log-x`
- `--p P` plots one measurement error probability. `--error-probabilities`
  overlays multiple independent per-bit measurement flip probabilities and is
  mutually exclusive with `--p`.
- A conditional curve is undefined when no trials satisfy its condition; those
  points are omitted from the plotted line and reported as `nan` in the data
  summary.
- Color identifies measurement error probability. The standard curve uses a
  lighter shade of the same probability color, while marker/line style
  identifies the algorithm.
- With a custom `--output`, `_same_peak` and `_shifted_peak` are appended to
  the output stem so each figure has a separate file.
- With `--standard-backend analytic`, the data summary includes captured-mass
  diagnostics for the finite `--analytic-window`

### `plot_one_shot_success_vs_order.py`

Groups Kitaev one-shot success by the true multiplicative order of each
admissible base for a fixed `N`.

- Required: `N`
- Defaults: `--trials-per-a 100`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--seed` = random
- Other options: `--output`, `--measurement-bitflip-error [P]`

### `plot_classical_recovery_vs_error_probability.py`

Plots how often the single-bit classical recovery rule reconstructs the clean
phase-estimation bit string as measurement bit-flip probability varies. It can
overlay Kitaev and standard Shor curves:

```bash
python graphs/measurement_error/plot_classical_recovery_vs_error_probability.py 21 --a 11 --algorithm kitaev standard
```

- Required: one or more `N` values, `--a`
- Defaults: `--algorithm kitaev`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--standard-backend statevector`,
  `--analytic-window 100`, `--trials 100`, `--min-p 0.0`, `--max-p 1.0`,
  `--num-points 11`, `--seed` = random
- Other options: `--output`, `--log-p-grid`, `--show-lower-bound`,
  `--show-first-order`
- `--a` accepts one base for all `N` values, one base per `N`, or multiple
  bases for a single `N`
- `--algorithm` accepts one or more of `kitaev`, `standard`
- `--standard-backend analytic` has the same memory-saving behavior described
  above and is intended for ideal or measurement-error standard Shor sampling

### `measurement_error/plot_hamming_distance_vs_error_probability.py`

Plots average Hamming distance between clean and noisy Kitaev phase-estimation
bit strings as measurement bit-flip probability varies.

- Required: one or more `N` values, `--a`
- Defaults: `--error-model measurement-bitflip`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--trials 100`, `--min-p 0.0`, `--max-p 1.0`,
  `--num-points 11`, `--seed` = random, unnormalized distance
- Other options: `--output`, `--log-p-grid`, `--normalized`
- `--error-model` accepts `measurement-bitflip`
- `--a` accepts one base for all `N` values, one base per `N`, or multiple
  bases for a single `N`

### `measurement_error/plot_decimal_distance_vs_error_probability.py`

Plots average circular decimal distance between clean and noisy Kitaev phase
estimates as measurement bit-flip probability varies.

- Required: one or more `N` values, `--a`
- Defaults: `--error-model measurement-bitflip`, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--trials 100`, `--min-p 0.0`, `--max-p 1.0`,
  `--num-points 11`, `--seed` = random
- Other options: `--output`, `--log-p-grid`
- `--error-model` accepts `measurement-bitflip`
- `--a` accepts one base for all `N` values, one base per `N`, or multiple
  bases for a single `N`

## Depolarizing Error Scripts

### `depolarizing_error/plot_hamming_distance_vs_error_probability.py`

Plots average clean/noisy Kitaev phase-estimation Hamming distance as
controlled-unitary depolarizing probability varies.

- Required: one or more `N` values, `--a`
- Defaults: `--error-model` = both depolarizing models, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--trials 100`, `--min-p 0.0`, `--max-p 1.0`,
  `--num-points 11`, `--seed` = random, unnormalized distance
- `--error-model` accepts `block-depolarizing` and `per-qubit-depolarizing`

### `depolarizing_error/plot_decimal_distance_vs_error_probability.py`

Plots average circular decimal distance between clean and noisy Kitaev phase
estimates as controlled-unitary depolarizing probability varies.

- Required: one or more `N` values, `--a`
- Defaults: `--error-model` = both depolarizing models, `--bits` =
  `2 * ceil(log2(N)) + 1`, `--trials 100`, `--min-p 0.0`, `--max-p 1.0`,
  `--num-points 11`, `--seed` = random
- `--error-model` accepts `block-depolarizing` and `per-qubit-depolarizing`

### `depolarizing_error/plot_order_recovery_success_vs_order.py`

Plots one-shot order-recovery success versus true order under depolarizing
trajectory errors for Kitaev and/or standard Shor.

- Required: none; `--instances-file` defaults to
  `graphs/config/order_recovery_instances.json`
- Defaults: `--algorithm kitaev`, `--error-model block-depolarizing`,
  `--error-probabilities 0 0.01 0.05 0.1`, `--bits 11`,
  `--trials-per-instance 100`, `--standard-backend statevector`
- Standard Shor depolarizing runs require `--standard-backend statevector`.
  The analytic backend samples ideal outcomes from the true order and has no
  evolving control/work state or work-register trajectory to perturb.

## Examples

Plot an empirical Kitaev phase-estimation distribution:

```bash
python graphs/measurement_error/plot_phase_estimation_distribution.py 21 --a 11 --trials 1000 --seed 123
```

Sweep Kitaev one-shot success over bit counts:

```bash
python graphs/measurement_error/plot_one_shot_success_vs_bits.py 21 --a 11 --min-bits 4 --max-bits 12 --trials-per-bits 200 --seed 123
```

Compare Kitaev and standard Shor one-shot success versus allowed attempts:

```bash
python graphs/measurement_error/plot_one_shot_success_vs_attempts.py 21 --algorithm kitaev standard --standard-backend analytic --bits 11 --trials-per-attempt-count 100 --seed 123
```

Compare Kitaev and standard Shor order recovery versus true order:

```bash
python graphs/measurement_error/plot_order_recovery_success_vs_order.py --algorithm kitaev standard --standard-backend analytic --bits 29 --num-instances 100 --trials-per-instance 100 --measurement-bitflip-error 0.01 --seed 123
```

Use a configured representative pair for each true order:

```bash
python graphs/measurement_error/plot_order_recovery_success_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29 --trials-per-instance 100 --seed 123
```

Overlay several measurement error probabilities in the success plot:

```bash
python graphs/measurement_error/plot_order_recovery_success_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29 --error-probabilities 0 0.01 0.05 0.1 --trials-per-instance 100 --seed 123
```

Plot the order-recovery loss caused by several measurement error probabilities:

```bash
python graphs/measurement_error/plot_order_recovery_loss_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --standard-backend analytic --bits 29 --error-probabilities 0.001 0.01 0.05 0.1 --trials-per-instance 100 --seed 123
```

Compare clean/noisy phase-estimation Hamming distance versus true order:

```bash
python graphs/measurement_error/plot_phase_hamming_distance_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --bits 29 --error-probabilities 0.01 0.05 0.1 --trials-per-instance 100 --seed 123
```

Compare clean/noisy phase-estimation decimal distance versus true order:

```bash
python graphs/measurement_error/plot_phase_decimal_distance_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --bits 29 --error-probabilities 0.01 0.05 0.1 --trials-per-instance 100 --seed 123
```

Sweep single-bit recovery for standard Shor measurement errors:

```bash
python graphs/measurement_error/plot_classical_recovery_vs_error_probability.py 15 --algorithm standard --standard-backend analytic --a 2 --bits 8 --trials 100 --num-points 11 --seed 123
```

Compare Kitaev and standard Shor in the same recovery plot:

```bash
python graphs/measurement_error/plot_classical_recovery_vs_error_probability.py 15 --algorithm kitaev standard --standard-backend analytic --a 2 --bits 8 --trials 100 --num-points 11 --seed 123
```

Sweep Hamming distance for measurement bit-flip errors:

```bash
python graphs/measurement_error/plot_hamming_distance_vs_error_probability.py 21 --a 11 --trials 100 --num-points 11 --seed 123
```

Compare multiple `(N, a)` instances in a decimal-distance plot:

```bash
python graphs/measurement_error/plot_decimal_distance_vs_error_probability.py 35 51 65 --a 6 16 2 --bits 29 --trials 100 --seed 123
```

Sweep depolarizing order recovery for Kitaev and standard statevector Shor:

```bash
python graphs/depolarizing_error/plot_order_recovery_success_vs_order.py --instances-file graphs/config/order_recovery_instances.json --algorithm kitaev standard --bits 11 --error-model block-depolarizing --error-probabilities 0 0.01 0.05 0.1 --trials-per-instance 100 --seed 123
```

## Output Files

Default figure names encode the experiment parameters, for example:

```text
graphs/figures/phase_estimation_distribution_N21_a11_t11.png
graphs/data/phase_estimation_distribution_summary_N21_a11_t11.txt
```

For standard Shor or algorithm comparisons, default names include the selected
algorithm/backend, such as:

```text
graphs/figures/one_shot_success_vs_attempts_compare_algorithms_analytic_N21.png
graphs/data/one_shot_success_vs_attempts_summary_compare_algorithms_analytic_N21.txt
graphs/figures/order_recovery_success_vs_order_compare_algorithms_analytic_t29.png
graphs/data/order_recovery_success_vs_order_summary_compare_algorithms_analytic_t29.txt
graphs/figures/order_recovery_loss_vs_order_compare_algorithms_analytic_instances_order_recovery_instances_default_p_t29.png
graphs/data/order_recovery_loss_vs_order_summary_compare_algorithms_analytic_instances_order_recovery_instances_default_p_t29.txt
```

The text summaries are meant to make the plotted values inspectable without
opening the figure. New `graphs/data/*` outputs are ignored by git, while
selected figures may be kept when they are useful reference outputs.
