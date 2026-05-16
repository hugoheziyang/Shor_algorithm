# Shor Algorithm Simulators

This repository contains small, plain-NumPy simulators for two closely related
Shor order-finding workflows:

- `main.py`: Kitaev / semi-classical Shor using one reusable control qubit.
- `standard_shor.py`: textbook Shor using a full `t`-qubit control register
  and an inverse QFT.

Both implementations are intended for toy-scale integers and for studying the
structure of the algorithms, not for efficient large-number factoring.

## Repository Layout

- `main.py`: command-line entry point for the one-control-qubit simulator.
- `standard_shor.py`: command-line entry point for the standard simulator.
- `shor_common/`: shared modular arithmetic, continued-fraction
  post-processing, simulator error models, and common Shor helper functions.
- `shor_one_qubit/`: Kitaev phase-estimation implementation.
- `shor_standard/`: standard phase-estimation implementation.
- `error_correction/`: classical recovery and bit-string distance diagnostics
  for the Kitaev simulator.
- `graphs/`: plotting scripts grouped by error model, generated figures/data,
  and graph-specific documentation in `graphs/README.md`.
- `tests/`: unit tests for both Shor implementations, diagnostics, and graph
  helpers.

The algorithm-specific packages are intentionally small: `shor_one_qubit/`
contains the iterative phase-estimation flow, while `shor_standard/` contains
the full-control-register phase-estimation flow. Shared pieces live in
`shor_common/`.

## Requirements

- Python 3.10+
- NumPy
- pytest for tests
- matplotlib for graph scripts

Install dependencies in your environment:

```bash
pip install numpy pytest matplotlib
```

## Kitaev / One-Control-Qubit Shor

Run one factoring attempt pipeline with a random coprime base `a`:

```bash
python main.py 21 --seed 282
```

Run a single attempt with a fixed base:

```bash
python main.py 21 --a 11 --bits 10 --seed 282
```

Important options:

- `N`: integer to factor.
- `--a`: optional fixed base for a single attempt.
- `--bits`: number of iterative phase-estimation bits. If omitted, the default
  is `2 * ceil(log2(N)) + 1`.
- `--attempts`: max attempts when `--a` is omitted. Default is `1` for
  `one_shot` and `10` for `rescue`.
- `--postprocess`: `one_shot` or `rescue`.
- `--measurement-bitflip-error [P]`: no value means exactly one random
  measurement bit flip; a value means independent per-detection bit flips with
  probability `P`.
- `--controlled-unitary-depolarizing-error P`: block-level work-register
  depolarizing trajectory error after each controlled-`U` block.
- `--controlled-unitary-per-qubit-depolarizing-error P`: per-work-qubit
  depolarizing trajectory error after each controlled-`U` block.
- `--seed`: optional RNG seed.

## Standard Shor

Run the textbook phase-estimation version:

```bash
python standard_shor.py 15 --a 2 --control-qubits 8 --seed 2
```

The standard simulator uses:

- `t` control qubits, set by `--control-qubits` or its alias `--bits`;
- `L = ceil(log2(N))` work qubits;
- a full control/work state of size `2^t * 2^L`;
- inverse QFT on the control register;
- the same continued-fraction order recovery and factor extraction as the
  Kitaev version.

Important options:

- `N`: integer to factor.
- `--a`: optional fixed base for a single attempt.
- `--control-qubits`, `--bits`: number `t` of control-register qubits. If
  omitted, the default is `2 * ceil(log2(N)) + 1`.
- `--attempts`: max attempts when `--a` is omitted.
- `--standard-backend`: `statevector` (default) stores the full control/work
  state; `analytic` samples ideal standard-QPE measurement outcomes from the
  classically computed order and is useful for large-`t` measurement-error
  diagnostics.
- `--analytic-window`: number of integer outcomes on each side of a QPE peak to
  include when using `--standard-backend analytic`. Analytic runs report the
  captured QPE probability mass for this truncated window, so you can check
  whether the window is wide enough for the tested order.
- `--postprocess`: `one_shot` or `rescue`.
- `--measurement-bitflip-error [P]`: final control-register measurement bit
  flips.
- `--controlled-unitary-depolarizing-error P`: statevector backend only;
  block-level work-register depolarizing trajectory error after each
  controlled-`U` block.
- `--controlled-unitary-per-qubit-depolarizing-error P`: statevector backend
  only; per-work-qubit depolarizing trajectory error after each controlled-`U`
  block.
- `--seed`: optional RNG seed.

Controlled-unitary depolarizing error models are implemented for the
one-control-qubit simulator and for the standard simulator's `statevector`
backend. The analytic standard backend supports only final measurement
bit-flip errors; it samples ideal QPE outcomes from the classically computed
order, so there is no evolving control/work state or work-register trajectory
on which depolarizing Pauli errors could act.

## Running Tests

From the repository root:

```bash
python -m pytest -q
```

Use `python -m pytest` rather than bare `pytest` if your environment does not
automatically put the repository root on `PYTHONPATH`.

## Generating Graphs

Graphing scripts and their generated outputs live in `graphs/`.
See `graphs/README.md` for the current list of scripts and examples.

## Expected Behavior and Limits

- This project is intended for composite numbers and toy-scale demonstrations.
- Prime inputs are out of scope. In particular, small primes such as `N=3` can
  fail when selecting random `a`.
- Outputs are probabilistic because measurements are sampled; set `--seed` for
  reproducibility.
- `final_work_state` is intentionally omitted from CLI output to keep output
  readable.
- The standard simulator stores the full `2^t * 2^L` state, so it grows much
  faster in memory than the one-control-qubit simulator when
  `--standard-backend statevector` is used.

## Development Notes

- The default post-processing mode is `one_shot`, which accepts only orders
  already verified from bounded continued-fraction convergents.
- The `rescue` mode additionally searches multiples and reduces candidate
  exponents; it is useful for simulator-side comparison, but is more aggressive
  than a strict one-shot hardware workflow.
- Modular multiplication is simulated directly as a permutation of the work
  register basis states. There is not yet a reversible gate-level backend.
- Depolarizing-error studies use trajectory Pauli errors after controlled
  unitary blocks. The one-control-qubit simulator steps through those blocks
  naturally; the standard statevector backend uses a sequential controlled-power
  construction when such errors are enabled.
