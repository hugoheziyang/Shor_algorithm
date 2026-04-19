import argparse
import time
import numpy as np

from shor_one_qubit.errors import (
    MEASUREMENT_BITFLIP_SINGLE_TOKEN,
    parse_measurement_bitflip_error_arg,
)
from shor_one_qubit.shor import shor_factor, shor_attempt


def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(description="One-control-qubit Shor simulator")
    parser.add_argument("N", type=int, help="Integer to factor")
    parser.add_argument("--a", type=int, default=None, help="Base a to use")
    parser.add_argument("--bits", type=int, default=None, help="Number of phase bits")
    parser.add_argument(
        "--attempts",
        type=int,
        default=None,
        help="Maximum number of attempts (default: 1 for one_shot, 10 for rescue)",
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
            "Enable measurement bit-flip errors. Use without a value for "
            "exactly one random bit-flip per iterative phase-estimation run, "
            "or provide P in [0,1] for independent per-detection bit-flips."
        ),
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    try:
        error_config = parse_measurement_bitflip_error_arg(
            args.measurement_bitflip_error
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.a is None:
        result = shor_factor(
            N=args.N,
            max_attempts=args.attempts,
            num_phase_bits=args.bits,
            rng=rng,
            postprocess_mode=args.postprocess,
            error_config=error_config,
        )
    else:
        result = shor_attempt(
            N=args.N,
            a=args.a,
            num_phase_bits=args.bits,
            rng=rng,
            postprocess_mode=args.postprocess,
            error_config=error_config,
        )

    print("Result:")
    for key, value in result.items():
        if key == "final_work_state":
            print(f"{key}: <state vector omitted>")
        else:
            print(f"{key}: {value}")
    print(f"time_taken_seconds: {time.perf_counter() - start_time:.3f}")


if __name__ == "__main__":
    main()
