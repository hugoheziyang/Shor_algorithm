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

## Requirements

- Python 3.10+
- NumPy
- pytest (for tests)

Install dependencies in your environment:

```bash
pip install numpy pytest
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
- `--attempts`: max attempts (used when `--a` is omitted).
- `--seed`: RNG seed for reproducibility.

## Running Tests

From the repository root:

```bash
python -m pytest -q
```

Note: `python -m pytest` is the reliable invocation for this layout because running `pytest` directly may not include the package root on `PYTHONPATH` in some environments.

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
- Code fails at prime inputs only.
- Modular unitaries currently have no gate backend (i.e. we assume the $U^j$ unitaries are made and directly permute the state)
- `classical_post.py` implementation after PE includes a `reduce_to_order` function which checks prime
factors of $r$ up to $\sqrt{r}$ in order to reduce it further to minimal order. If $r$ is large, could take 
$O(\sqrt{N}) = O(\sqrt{2^L})$ tries (inefficient exponential runtime). 
- `classical_post.py` also includes a `recover_order_from_fraction` function which tries to correct the case where
the PE yields $p/q$ instead of $s/r$ by checking if $r \times n$ is the order, for $n \leq N$ (inefficient exponential runtime).
- `config.py` currently unused. Future modifications to include usage in `main.py` and future files.
