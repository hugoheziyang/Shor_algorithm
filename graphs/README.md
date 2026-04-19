# Graph Scripts

This directory contains plotting scripts for the hardware-like `one_shot`
workflow, together with generated figures in `graphs/figures/` and text
summaries in `graphs/data/`.

Unless overridden with `--output`, figure files are written into
`graphs/figures/`. Each script also writes a companion `.txt` summary into
`graphs/data/`.

If a graph script omits `--bits`, it uses the repository default

```text
default_num_phase_bits(N) = 2 * ceil(log2(N)) + 1
```

## Success Versus True Order

```bash
python graphs/plot_one_shot_success_vs_order.py 21
```

Default outputs:

```text
graphs/figures/one_shot_success_vs_order_N21.png
graphs/figures/one_shot_success_per_base_N21.png
graphs/data/one_shot_success_vs_order_summary_N21.txt
```

This experiment fixes one odd composite `N`, uses all coprime bases `a` with
`2 <= a <= N - 2`, runs `one_shot` repeatedly for each base, and produces:
- an order-based graph aggregated by the true multiplicative order `r`;
- a per-base graph showing each admissible base `a` and its empirical success rate;
- a text summary containing the figure notes plus the per-base table `a, r, success`.

Useful options:
- `--trials-per-a`: number of one-shot trials per base (default `100`).
- `--seed`: optional master seed for reproducible sampling. If omitted, fresh random sampling is used.
- `--bits`: override the number of phase-estimation bits.
- `--measurement-bitflip-error [P]`: use without a value for exactly one random measurement bit-flip per iterative phase-estimation run, or provide `P` in `[0, 1]` for independent per-detection bit-flips.
- `--output`: custom output path for the order-based figure.

If `--measurement-bitflip-error` is enabled without a probability, the default outputs become:

```text
graphs/figures/one_shot_success_vs_order_N21_measurement_bitflip_error.png
graphs/figures/one_shot_success_per_base_N21_measurement_bitflip_error.png
graphs/data/one_shot_success_vs_order_summary_N21_measurement_bitflip_error.txt
```

If a probability is provided, the default outputs include that value in the
suffix in one-significant-figure scientific notation, for example:

```text
graphs/figures/one_shot_success_vs_order_N21_measurement_bitflip_error_1e-1.png
graphs/figures/one_shot_success_per_base_N21_measurement_bitflip_error_1e-1.png
graphs/data/one_shot_success_vs_order_summary_N21_measurement_bitflip_error_1e-1.txt
```

## Success Versus Phase-Estimation Bits

Fixed-base mode:

```bash
python graphs/plot_one_shot_success_vs_bits.py 21 --a 11
```

Random-base mode:

```bash
python graphs/plot_one_shot_success_vs_bits.py 21
```

Example default outputs for fixed-base mode:

```text
graphs/figures/one_shot_success_vs_bits_N21_a11.png
graphs/data/one_shot_success_vs_bits_summary_N21_a11.txt
```

This experiment shows empirical one-shot factorization success probability
versus the number of phase-estimation bits for either:
- one fixed pair `(N, a)`, if `--a` is provided;
- one fixed `N` with a fresh random coprime base selected each trial, if `--a` is omitted.

Useful options:
- `--min-bits`: minimum bit count in the sweep (default `1`).
- `--max-bits`: maximum bit count in the sweep (default `default_num_phase_bits(N)`).
- `--trials-per-bits`: number of one-shot trials per bit count (default `100`).
- `--a`: optional fixed coprime base. If omitted, a fresh random coprime base is sampled each trial.
- `--measurement-bitflip-error [P]`: use without a value for exactly one random measurement bit-flip per iterative phase-estimation run, or provide `P` in `[0, 1]` for independent per-detection bit-flips.
- `--seed`: optional master seed for reproducible sampling. If omitted, fresh random sampling is used.
- `--output`: custom output path for the figure.

If `--measurement-bitflip-error` is enabled without a probability, the default
outputs gain the suffix `_measurement_bitflip_error`. If a probability is
provided, the default outputs include that value in the suffix in
one-significant-figure scientific notation, for example:

```text
graphs/figures/one_shot_success_vs_bits_N21_a11_measurement_bitflip_error_1e-1.png
graphs/data/one_shot_success_vs_bits_summary_N21_a11_measurement_bitflip_error_1e-1.txt
```

## Success Versus Number of One-Shot Runs

```bash
python graphs/plot_one_shot_success_vs_attempts.py 21
```

Default outputs:

```text
graphs/figures/one_shot_success_vs_attempts_N21.png
graphs/data/one_shot_success_vs_attempts_summary_N21.txt
```

This experiment shows empirical factorization success probability versus the
maximum number of allowed one-shot runs for one fixed odd composite `N`.

Interpretation of the sampling:
- `max-attempts = k` means one top-level factoring run is allowed to make up to `k` one-shot runs before being declared a failure.
- `--trials-per-attempt-count = 100` means that, for each fixed value of `k`, the script repeats that whole top-level run `100` times and estimates the success probability from those `100` outcomes.
- Across those `100` trials, the following stay fixed: `N`, the chosen number of allowed one-shot runs `k`, the number of phase-estimation bits, and `postprocess_mode="one_shot"`.
- Across those `100` trials, the randomness changes: the random base choices and the phase-estimation measurement outcomes.
- Within one single trial at `max-attempts = k`, the code may make up to `k` separate one-shot runs, and each of those runs may use a different random coprime base `a`.

Useful options:
- `--min-attempts`: minimum number of allowed one-shot runs in the sweep (default `1`).
- `--max-attempts`: maximum number of allowed one-shot runs in the sweep (default `10`).
- `--trials-per-attempt-count`: number of independent simulated top-level runs used to estimate success probability for each chosen number of allowed one-shot runs (default `100`).
- `--bits`: optional override for the number of phase-estimation bits.
- `--measurement-bitflip-error [P]`: use without a value for exactly one random measurement bit-flip per iterative phase-estimation run, or provide `P` in `[0, 1]` for independent per-detection bit-flips.
- `--seed`: optional master seed for reproducible sampling. If omitted, fresh random sampling is used.
- `--output`: custom output path for the figure.

If `--measurement-bitflip-error` is enabled without a probability, the default outputs become:

```text
graphs/figures/one_shot_success_vs_attempts_N21_measurement_bitflip_error.png
graphs/data/one_shot_success_vs_attempts_summary_N21_measurement_bitflip_error.txt
```

If a probability is provided, the default outputs include that value in the
suffix in one-significant-figure scientific notation, for example:

```text
graphs/figures/one_shot_success_vs_attempts_N21_measurement_bitflip_error_1e-1.png
graphs/data/one_shot_success_vs_attempts_summary_N21_measurement_bitflip_error_1e-1.txt
```

## Phase-Estimation Outcome Distribution

```bash
python graphs/plot_phase_estimation_distribution.py 21 --a 11
```

Default outputs:

```text
graphs/figures/phase_estimation_distribution_N21_a11.png
graphs/data/phase_estimation_distribution_summary_N21_a11.txt
```

This experiment fixes one odd composite `N` and one coprime base `a`, reruns
the iterative phase-estimation procedure many times, converts each outcome to
the decimal phase estimate `m / 2^t`, and plots the empirical distribution.

Useful options:
- `--a`: fixed coprime base to analyze.
- `--bits`: optional override for the number of phase-estimation bits.
- `--trials`: number of repeated phase-estimation samples to draw (default `1000`).
- `--measurement-bitflip-error [P]`: use without a value for exactly one random measurement bit-flip per iterative phase-estimation run, or provide `P` in `[0, 1]` for independent per-detection bit-flips.
- `--seed`: optional RNG seed for reproducible sampling. If omitted, fresh random sampling is used.
- `--output`: custom output path for the figure.

If `--measurement-bitflip-error` is enabled without a probability, the default outputs become:

```text
graphs/figures/phase_estimation_distribution_N21_a11_measurement_bitflip_error.png
graphs/data/phase_estimation_distribution_summary_N21_a11_measurement_bitflip_error.txt
```

If a probability is provided, the default outputs include that value in the
suffix in one-significant-figure scientific notation, for example:

```text
graphs/figures/phase_estimation_distribution_N21_a11_measurement_bitflip_error_1e-1.png
graphs/data/phase_estimation_distribution_summary_N21_a11_measurement_bitflip_error_1e-1.txt
```

## Classical Recovery Versus Error Probability

```bash
python graphs/plot_classical_recovery_vs_error_probability.py 21 --a 11
```

Default outputs:

```text
graphs/figures/classical_recovery_vs_error_probability_N21_a11.png
graphs/data/classical_recovery_vs_error_probability_summary_N21_a11.txt
```

This experiment fixes one valid modulus `N`, one coprime base `a`, and one
phase-estimation bit count `t`. For each chosen measurement error probability
`p`, it pairs:
- one clean Kitaev phase-estimation run;
- one noisy run using independent per-detection measurement bit-flips with probability `p`.

It then applies a standalone classical single-bit recovery check to the noisy
bitstring by testing the original noisy string together with all `t`
single-bit-flip neighbors.

Useful options:
- `--a`: fixed coprime base to analyze.
- `--bits`: optional override for the number of phase-estimation bits.
- `--trials`: number of paired clean/noisy trials per probability value (default `100`).
- `--min-p`: minimum measurement bit-flip probability in the sweep (default `0.0`).
- `--max-p`: maximum measurement bit-flip probability in the sweep (default `1.0`).
- `--num-points`: number of probability values in the sweep (default `11`).
- `--log-p-grid`: use logarithmically spaced probability values between `min-p` and `max-p`. This requires `min-p > 0`.
- `--seed`: optional master RNG seed for reproducible sampling. If omitted, fresh random sampling is used.
- `--output`: custom output path for the figure.
