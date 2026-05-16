import argparse
import time

import numpy as np

from shor_common.errors import (
    ErrorConfig,
    MEASUREMENT_BITFLIP_SINGLE_TOKEN,
    parse_measurement_bitflip_error_arg,
    validate_error_config,
)
from shor_standard.analytic_sampling import qpe_window_mass_diagnostics
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND, STANDARD_BACKEND_CHOICES
from shor_standard.shor import shor_attempt, shor_factor


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Standard Shor simulator with a t-qubit control register"
    )
    parser.add_argument("N", type=int, help="Integer to factor")
    parser.add_argument("--a", type=int, default=None, help="Base a to use")
    parser.add_argument(
        "--control-qubits",
        "--bits",
        dest="num_control_qubits",
        type=int,
        default=None,
        help=(
            "Number t of control-register qubits. If omitted, the default is "
            "2 * ceil(log2(N)) + 1."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=None,
        help="Maximum number of attempts (default: 1 for one_shot, 10 for rescue)",
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default="statevector",
        help=(
            "Backend for standard phase estimation. 'statevector' simulates the "
            "full control/work state; 'analytic' samples from the order-based "
            "measurement distribution and supports measurement bit-flip errors only."
        ),
    )
    parser.add_argument(
        "--analytic-window",
        type=int,
        default=100,
        help=(
            "Number of integer outcomes on each side of a standard-QPE peak to "
            "include when --standard-backend analytic is used."
        ),
    )
    parser.add_argument(
        "--postprocess",
        choices=("one_shot", "rescue"),
        default="one_shot",
        help="Classical post-processing mode after phase estimation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility",
    )
    parser.add_argument(
        "--measurement-bitflip-error",
        nargs="?",
        const=MEASUREMENT_BITFLIP_SINGLE_TOKEN,
        default=None,
        metavar="P",
        help=(
            "Enable final control-register measurement bit-flip errors. Use "
            "without a value for exactly one random bit-flip, or provide P in "
            "[0,1] for independent per-bit flips."
        ),
    )
    parser.add_argument(
        "--controlled-unitary-depolarizing-error",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Statevector backend only: after each controlled-U block, with "
            "probability P in [0,1], apply a random X/Y/Z to one random "
            "work-register qubit."
        ),
    )
    parser.add_argument(
        "--controlled-unitary-per-qubit-depolarizing-error",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Statevector backend only: after each controlled-U block, each "
            "work-register qubit independently gets a random X/Y/Z error with "
            "probability P in [0,1]."
        ),
    )

    args = parser.parse_args()
    if args.analytic_window < 0:
        parser.error("--analytic-window must be >= 0")

    rng = np.random.default_rng(args.seed)
    try:
        measurement_error_config = parse_measurement_bitflip_error_arg(
            args.measurement_bitflip_error
        )
        error_config = validate_error_config(
            ErrorConfig(
                measurement_bitflip=measurement_error_config.measurement_bitflip,
                measurement_bitflip_probability=measurement_error_config.measurement_bitflip_probability,
                controlled_unitary_depolarizing_probability=args.controlled_unitary_depolarizing_error,
                controlled_unitary_per_qubit_depolarizing_probability=args.controlled_unitary_per_qubit_depolarizing_error,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.standard_backend == ANALYTIC_STANDARD_BACKEND and (
        error_config.controlled_unitary_depolarizing_probability is not None
        or error_config.controlled_unitary_per_qubit_depolarizing_probability is not None
    ):
        parser.error(
            "controlled-unitary depolarizing errors require "
            "--standard-backend statevector; the analytic backend samples ideal "
            "outcomes from the true order and has no control/work state to perturb"
        )

    if args.a is None:
        result = shor_factor(
            N=args.N,
            max_attempts=args.attempts,
            num_control_qubits=args.num_control_qubits,
            rng=rng,
            postprocess_mode=args.postprocess,
            error_config=error_config,
            standard_backend=args.standard_backend,
            analytic_window=args.analytic_window,
        )
    else:
        result = shor_attempt(
            N=args.N,
            a=args.a,
            num_control_qubits=args.num_control_qubits,
            rng=rng,
            postprocess_mode=args.postprocess,
            error_config=error_config,
            standard_backend=args.standard_backend,
            analytic_window=args.analytic_window,
        )

    if (
        args.standard_backend == ANALYTIC_STANDARD_BACKEND
        and "sampling_order" in result
    ):
        diagnostics = qpe_window_mass_diagnostics(
            order=result["sampling_order"],
            num_control_qubits=result["num_control_qubits"],
            window=args.analytic_window,
        )
        result.update({
            "analytic_window_min_captured_mass": diagnostics.min_mass,
            "analytic_window_mean_captured_mass": diagnostics.mean_mass,
            "analytic_window_max_captured_mass": diagnostics.max_mass,
            "analytic_window_worst_eigenphase_numerator": (
                diagnostics.worst_eigenphase_numerator
            ),
        })

    print("Result:")
    for key, value in result.items():
        if key == "final_work_state":
            print(f"{key}: <state vector omitted>")
        else:
            print(f"{key}: {value}")
    print(f"time_taken_seconds: {time.perf_counter() - start_time:.3f}")


if __name__ == "__main__":
    main()
