# One-Control-Qubit Shor Simulator

Implementation of Shor-style order finding using:
- a reusable single control qubit,
- iterative (Kitaev/semi-classical) phase estimation,
- classical continued-fraction post-processing.

The code is written in plain NumPy and is intended for small, demo-scale integers.

## What This Repository Contains

- `main.py`: command-line entry point.
- `shor_one_qubit/shor.py`: top-level factoring flow and classical shortcuts.
- `shor_one_qubit/kitaev_pe.py`: iterative phase estimation loop.
- `shor_one_qubit/modular_unitaries.py`: modular multiplication unitary action on the work register.
- `shor_one_qubit/state_ops.py`: joint control/work state operations and measurement.
- `shor_one_qubit/gates.py`: small 2x2 gate primitives.
- `shor_one_qubit/classical_post.py`: phase-to-order recovery and factor extraction.
- `shor_one_qubit/tests/`: unit tests for all modules.
- `graphs/plot_one_shot_success_vs_order.py`: graph scripts for success versus true order and per-base success.
- `graphs/plot_one_shot_success_vs_bits.py`: graph script for success versus the number of phase-estimation bits.
- `graphs/figures/`: generated graph image outputs.
- `graphs/data/`: generated text summaries for graph runs.

## Requirements

- Python 3.10+
- NumPy
- pytest (for tests)
- matplotlib (for graph scripts)

Install dependencies in your environment:

```bash
pip install numpy pytest matplotlib
```

## Quick Start

Run one factoring attempt pipeline (random `a` chosen internally):

```bash
python main.py 21 --seed 282
```

Run a single attempt with a fixed base `a`:

```bash
python main.py 21 --a 11 --bits 10 --seed 282
```

CLI options:
- `N` (positional): integer to factor.
- `--a`: optional base for a single attempt.
- `--bits`: number of phase-estimation bits.
- `--attempts`: max attempts (used when `--a` is omitted). Default is `1` for `one_shot` and `10` for `rescue`.
- `--postprocess`: classical post-processing mode, either `one_shot` (default, hardware-like) or `rescue`.
- `--seed`: optional RNG seed for reproducibility. If omitted, fresh randomness is used.

## Running Tests

From the repository root:

```bash
python -m pytest -q
```

Note: `python -m pytest` is the reliable invocation for this layout because running `pytest` directly may not include the package root on `PYTHONPATH` in some environments.

## Generating Graphs

The repository includes a graphing script for the hardware-like `one_shot`
post-processing workflow:

```bash
python graphs/plot_one_shot_success_vs_order.py 21
```

By default this writes:

```text
graphs/figures/one_shot_success_vs_order_N21.png
graphs/figures/one_shot_success_per_base_N21.png
graphs/data/one_shot_success_summary_N21.txt
```

The plot shows empirical factorization success probability versus the true
multiplicative order `r` for one fixed odd composite `N`. The script uses all
coprime bases `a` with `2 <= a <= N - 2`, runs `one_shot` repeatedly for each
base, and then produces:
- an order-based graph aggregated by the true order `r`;
- a per-base graph showing each admissible base `a` and its empirical success rate;
- a text summary containing the figure notes plus the per-base table `a, r, success`.

Useful options:
- `--trials-per-a`: number of one-shot trials per base (default `100`).
- `--seed`: optional master seed for reproducible sampling. If omitted, fresh random sampling is used.
- `--bits`: override the number of phase-estimation bits.
- `--output`: custom output path for the figure.

The repository also includes a fixed-instance bit-sweep script:

```bash
python graphs/plot_one_shot_success_vs_bits.py 21 --a 11
```

If `--a` is omitted, the script instead samples a fresh random coprime base
for each trial:

```bash
python graphs/plot_one_shot_success_vs_bits.py 21
```

By default this writes:

```text
graphs/figures/one_shot_success_vs_bits_N21_a11.png
graphs/data/one_shot_success_vs_bits_summary_N21_a11.txt
```

This plot shows empirical one-shot factorization success probability versus the
number of phase-estimation bits for either:
- one fixed pair `(N, a)`, if `--a` is provided;
- one fixed `N` with fresh random coprime base selection each trial, if `--a` is omitted.

Useful options:
- `--min-bits`: minimum bit count in the sweep (default `1`).
- `--max-bits`: maximum bit count in the sweep (default `default_num_phase_bits(N)`).
- `--trials-per-bits`: number of one-shot trials per bit count (default `100`).
- `--a`: optional fixed coprime base. If omitted, a fresh random coprime base is sampled each trial.
- `--seed`: optional master seed for reproducible sampling. If omitted, fresh random sampling is used.
- `--output`: custom output path for the figure.

## Expected Behavior and Limits

- This project is intended for composite numbers and toy-scale demonstrations.
- Prime inputs are out of scope for this repository. In particular, small primes such as `N=3` can trigger runtime failure when selecting random `a`.
- Some outputs are probabilistic because measurements are sampled; set `--seed` for reproducible runs.
- `final_work_state` is intentionally omitted from CLI output to keep output readable.

## Example Output (abridged)

```text
Result:
N: 21
a: 11
num_phase_bits: 10
postprocess_mode: one_shot
bits: [...]
phase_estimate: ...
order: ...
factors: (3, 7)
success: True
reason: success
final_work_state: <state vector omitted>
```

## Notes for Development

- Tests currently cover gates, state ops, modular unitaries, iterative PE, classical post-processing, and top-level Shor flow.
- There are two separate complexity questions in this repository:
  - the cost of classical post-processing after phase estimation;
  - the cost of classically simulating the quantum state itself.
- For `--postprocess one_shot`, the classical post-processing is polynomial in the input size $L = \log_2 N$: it only uses continued-fraction convergents together with modular-exponentiation checks.
- For `--postprocess rescue`, the classical post-processing can be exponential in $L$, because it may search multiples up to $N$ and may also reduce candidate orders by trial factors up to about $\sqrt{r}$.
- Even with `--postprocess one_shot`, the full simulator is still exponential in $L$, because the work-register state lives in a space of dimension $2^L$ and the NumPy simulation explicitly stores and updates those amplitudes.
- On an actual quantum computer, this simulator bottleneck is avoided: modular multiplication / modular exponentiation would be implemented as a reversible quantum circuit with polynomial complexity in $L$, so the exponential scaling here comes from classical state-vector simulation rather than from Shor's ideal quantum runtime.
- Modular unitaries currently have no gate backend (i.e. we assume the $U^j$ unitaries are made and directly permute the state)
- `classical_post.py` now contains both:
  - hardware-like one-shot continued-fraction helpers, which scan bounded convergents and accept the first one already verified as the order;
  - simulator-side rescue helpers, which additionally search multiples / reduce exponents to recover the true order more aggressively.
- `shor.py` and `main.py` now default to the hardware-like `one_shot` post-processing mode, while still exposing the older `rescue` mode for comparison.
- `config.py` currently unused. Future modifications to include usage in `main.py` and future files.
