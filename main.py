import argparse
import numpy as np

from shor_one_qubit.shor import shor_factor, shor_attempt


def main():
    parser = argparse.ArgumentParser(description="One-control-qubit Shor simulator")
    parser.add_argument("N", type=int, help="Integer to factor")
    parser.add_argument("--a", type=int, default=None, help="Base a to use")
    parser.add_argument("--bits", type=int, default=None, help="Number of phase bits")
    parser.add_argument("--attempts", type=int, default=10, help="Maximum number of attempts")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.a is None:
        result = shor_factor(
            N=args.N,
            max_attempts=args.attempts,
            num_phase_bits=args.bits,
            rng=rng,
        )
    else:
        result = shor_attempt(
            N=args.N,
            a=args.a,
            num_phase_bits=args.bits,
            rng=rng,
        )

    print("Result:")
    for key, value in result.items():
        if key == "final_work_state":
            print(f"{key}: <state vector omitted>")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()