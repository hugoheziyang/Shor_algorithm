from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path("/tmp/matplotlib-shor")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from shor_common.classical_post import shor_factors_from_order
from shor_common.errors import (
    ErrorConfig,
    MEASUREMENT_BITFLIP_SINGLE_TOKEN,
    measurement_bitflip_filename_suffix,
    measurement_bitflip_label,
    measurement_bitflip_title_suffix,
    parse_measurement_bitflip_error_arg,
)
from shor_common.shor_utils import (
    choose_random_a,
    default_num_phase_bits,
    perfect_power_decomposition,
)
from shor_one_qubit.shor import shor_order_finding_attempt


@dataclass
class BitSweepResult:
    num_bits: int
    successful_trials: int
    trials: int

    @property
    def success_rate(self) -> float:
        return self.successful_trials / self.trials


def is_odd_composite(n: int) -> bool:
    """
    Return True for odd composite integers.
    """
    if n < 9 or (n % 2) == 0:
        return False

    for factor in range(3, math.isqrt(n) + 1, 2):
        if (n % factor) == 0:
            return True

    return False


def validate_target_instance(N: int, a: int | None) -> None:
    """
    Validate the fixed Shor instance used in the bit-sweep experiment.
    """
    if not is_odd_composite(N):
        raise ValueError("N must be an odd composite integer >= 9")

    if perfect_power_decomposition(N) is not None:
        raise ValueError(
            "perfect powers are excluded here because this repository treats "
            "them as classical shortcuts rather than quantum order-finding cases"
        )

    if a is None:
        return

    if not (2 <= a <= N - 2):
        raise ValueError("a must satisfy 2 <= a <= N - 2")
    if math.gcd(a, N) != 1:
        raise ValueError("a must be coprime to N")


def multiplicative_order(a: int, N: int) -> int:
    """
    Return the exact multiplicative order of a modulo N.
    """
    if math.gcd(a, N) != 1:
        raise ValueError("a and N must be coprime")

    residue = 1
    for r in range(1, N + 1):
        residue = (residue * a) % N
        if residue == 1:
            return r

    raise RuntimeError("failed to find multiplicative order")


def simulate_for_num_bits(
    N: int,
    a: int | None,
    num_bits: int,
    trials: int,
    rng: np.random.Generator,
    error_config: ErrorConfig | None = None,
) -> BitSweepResult:
    """
    Estimate the one-shot factorization success rate for a fixed bit count.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    successful_trials = 0
    error_config = error_config or ErrorConfig()

    for _ in range(trials):
        trial_a = choose_random_a(N, rng=rng) if a is None else a

        result = shor_order_finding_attempt(
            N=N,
            a=trial_a,
            num_phase_bits=num_bits,
            rng=rng,
            postprocess_mode="one_shot",
            error_config=error_config,
        )

        factors = shor_factors_from_order(trial_a, result["order"], N)
        if factors is not None:
            successful_trials += 1

    return BitSweepResult(
        num_bits=num_bits,
        successful_trials=successful_trials,
        trials=trials,
    )


def write_data_summary(
    output_path: Path,
    N: int,
    a: int | None,
    true_order: int | None,
    results: list[BitSweepResult],
    trials_per_bits: int,
    error_config: ErrorConfig | None,
    seed: int | None,
) -> None:
    """
    Write a text summary containing metadata, graph notes, and per-bit data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)
    error_label = measurement_bitflip_label(error_config)
    if a is None:
        mode_line = "Base selection = fresh random coprime base a for each trial"
        true_order_line = "True order r = not fixed (depends on sampled a)"
        graph_note = "  Fixed odd-composite N and fresh random coprime base a each trial"
    else:
        mode_line = f"a = {a}"
        true_order_line = f"True order r = {true_order}"
        graph_note = "  Fixed odd-composite N and fixed coprime base a"

    lines = [
        "One-Shot Shor Success-vs-Bits Data Summary",
        f"N = {N}",
        mode_line,
        true_order_line,
        f"Trials per bit value = {trials_per_bits}",
        f"Measurement bit-flip error = {error_label}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        graph_note,
        f"  Measurement bit-flip error: {error_label}",
        "  Success means one-shot order recovery followed by successful factor extraction",
        f"  Number of phase bits swept from {results[0].num_bits} to {results[-1].num_bits}",
        "",
        "Per-bit data:",
        "  num bits | success rate | successful trials | total trials",
    ]

    for result in results:
        lines.append(
            f"  {result.num_bits:>8} |"
            f" {result.success_rate:>12.3f} |"
            f" {result.successful_trials:>17} |"
            f" {result.trials:>12}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_success_vs_bits(
    N: int,
    a: int | None,
    true_order: int | None,
    results: list[BitSweepResult],
    error_config: ErrorConfig | None,
    output_path: Path,
) -> None:
    """
    Create and save the success-probability plot versus number of bits.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [result.num_bits for result in results]
    y_values = [result.success_rate for result in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2.0,
        markersize=6,
        color="#1F6F78",
    )

    ax.set_xlabel("Number of phase-estimation bits")
    ax.set_ylabel("Empirical one-shot factorization success probability")
    if a is None:
        ax.set_title(
            "One-Shot Shor Success vs Bits for N = "
            f"{N} (random a per trial){measurement_bitflip_title_suffix(error_config)}"
        )
    else:
        ax.set_title(
            "One-Shot Shor Success vs Bits for N = "
            f"{N}, a = {a}{measurement_bitflip_title_suffix(error_config)}"
        )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)

    if true_order is not None:
        ax.text(
            0.98,
            0.02,
            "true order r = "
            f"{true_order}\nmeasurement bit-flip error = "
            f"{measurement_bitflip_label(error_config)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#D0D0D0"},
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the success-vs-bits graph script.
    """
    parser = argparse.ArgumentParser(
        description="Plot one-shot Shor factorization success versus number of phase-estimation bits."
    )
    parser.add_argument("N", type=int, help="Odd composite number to factor")
    parser.add_argument(
        "--a",
        type=int,
        default=None,
        help="Optional fixed coprime base a to analyze. If omitted, a fresh random coprime base is sampled each trial.",
    )
    parser.add_argument(
        "--min-bits",
        type=int,
        default=1,
        help="Minimum number of phase-estimation bits to include",
    )
    parser.add_argument(
        "--max-bits",
        type=int,
        default=None,
        help="Maximum number of phase-estimation bits to include",
    )
    parser.add_argument(
        "--trials-per-bits",
        type=int,
        default=100,
        help="Number of one-shot trials to simulate for each bit count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional master RNG seed for reproducible sampling",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure",
    )
    return parser


def main() -> None:
    start_time = time.perf_counter()

    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        error_config = parse_measurement_bitflip_error_arg(
            args.measurement_bitflip_error
        )
    except ValueError as exc:
        parser.error(str(exc))

    validate_target_instance(args.N, args.a)

    if args.max_bits is None:
        max_bits = default_num_phase_bits(args.N)
    else:
        max_bits = args.max_bits

    if args.min_bits < 1:
        raise ValueError("min_bits must be >= 1")
    if max_bits < args.min_bits:
        raise ValueError("max_bits must be >= min_bits")
    if args.trials_per_bits < 1:
        raise ValueError("trials_per_bits must be >= 1")

    if args.output is None:
        suffix = f"a{args.a}" if args.a is not None else "random_a"
        filename_suffix = measurement_bitflip_filename_suffix(error_config)
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"one_shot_success_vs_bits_N{args.N}_{suffix}{filename_suffix}.png"
        )
    else:
        figure_output_path = args.output

    suffix = f"a{args.a}" if args.a is not None else "random_a"
    filename_suffix = measurement_bitflip_filename_suffix(error_config)
    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"one_shot_success_vs_bits_summary_N{args.N}_{suffix}{filename_suffix}.txt"
    )

    true_order = None if args.a is None else multiplicative_order(args.a, args.N)
    master_rng = np.random.default_rng(args.seed)
    results: list[BitSweepResult] = []

    for num_bits in range(args.min_bits, max_bits + 1):
        instance_seed = int(master_rng.integers(0, 2 ** 63 - 1))
        instance_rng = np.random.default_rng(instance_seed)
        results.append(
            simulate_for_num_bits(
                N=args.N,
                a=args.a,
                num_bits=num_bits,
                trials=args.trials_per_bits,
                rng=instance_rng,
                error_config=error_config,
            )
        )

    plot_success_vs_bits(
        N=args.N,
        a=args.a,
        true_order=true_order,
        results=results,
        error_config=error_config,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        N=args.N,
        a=args.a,
        true_order=true_order,
        results=results,
        trials_per_bits=args.trials_per_bits,
        error_config=error_config,
        seed=args.seed,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}")
    if args.a is None:
        print("Base selection: fresh random coprime a each trial")
    else:
        print(f"a = {args.a}")
        print(f"True order r = {true_order}")
    print(f"Measurement bit-flip error: {measurement_bitflip_label(error_config)}")
    print(f"Bits swept from {args.min_bits} to {max_bits}")
    print(f"Trials per bit value: {args.trials_per_bits}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
