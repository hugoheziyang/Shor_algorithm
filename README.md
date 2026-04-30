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
- `shor_one_qubit/errors.py`: configurable simulator error models.
- `error_correction/`: classical recovery and bit-string distance diagnostics.
- `shor_one_qubit/tests/`: unit tests for all modules.
- `graphs/`: graph scripts, generated figures/data, and graph-specific documentation in `graphs/README.md`.

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
- `--bits`: number of phase-estimation bits. If omitted, the default is `default_num_phase_bits(N) = 2 * ceil(log2(N)) + 1`.
- `--attempts`: max attempts (used when `--a` is omitted). Default is `1` for `one_shot` and `10` for `rescue`.
- `--postprocess`: classical post-processing mode, either `one_shot` (default, hardware-like) or `rescue`.
- `--measurement-bitflip-error [P]`: use without a value for exactly one random measurement bit-flip per iterative phase-estimation run, or provide `P` in `[0, 1]` for independent per-detection bit-flips.
- `--controlled-unitary-depolarizing-error P`: after each controlled-$U$ block, apply a block-level trajectory depolarizing error with probability `P` by choosing one random work-register qubit and one random Pauli from `X`, `Y`, `Z`.
- `--controlled-unitary-per-qubit-depolarizing-error P`: after each controlled-$U$ block, independently apply a trajectory depolarizing error to each work-register qubit with probability `P`.
- `--seed`: optional RNG seed for reproducibility. If omitted, fresh randomness is used.

## Running Tests

From the repository root:

```bash
python -m pytest -q
```

Note: `python -m pytest` is the reliable invocation for this layout because running `pytest` directly may not include the package root on `PYTHONPATH` in some environments.

## Generating Graphs

Graphing scripts and their generated outputs live in `graphs/`.
Detailed usage, default output paths, and experiment descriptions are in
`graphs/README.md`.

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
- `errors.py` provides pluggable trajectory error models: measurement bit-flips, block-level controlled-unitary depolarizing Pauli errors, and per-work-qubit controlled-unitary depolarizing Pauli errors.
- `error_correction/code_distance.py` provides paired clean/noisy Kitaev phase-estimation diagnostics using Hamming distance and decimal phase distance between the measured phase-bit strings.
