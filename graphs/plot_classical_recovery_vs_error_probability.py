from __future__ import annotations

import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from error_correction.classical_error_correction import (
    RecoveryProbabilityEstimate,
    estimate_classical_recovery_probability,
    naive_single_bit_recovery_baseline,
    validate_phase_estimation_instance,
)
from shor_one_qubit.shor import default_num_phase_bits


@dataclass
class RecoverySweepResult:
    error_probability: float
    successful_trials: int
    trials: int
    recovery_probability: float
    naive_baseline_probability: float


def validate_probability_sweep(
    min_probability: float,
    max_probability: float,
    num_points: int,
) -> None:
    """
    Validate the requested sweep over measurement-error probabilities.
    """
    if not (0.0 <= min_probability <= 1.0):
        raise ValueError("min-p must satisfy 0 <= min-p <= 1")
    if not (0.0 <= max_probability <= 1.0):
        raise ValueError("max-p must satisfy 0 <= max-p <= 1")
    if min_probability > max_probability:
        raise ValueError("min-p must be <= max-p")
    if num_points < 1:
        raise ValueError("num-points must be >= 1")


def build_probability_values(
    min_probability: float,
    max_probability: float,
    num_points: int,
    log_p_grid: bool = False,
) -> list[float]:
    """
    Build either a linear or logarithmic sweep of measurement-error probabilities.
    """
    validate_probability_sweep(min_probability, max_probability, num_points)

    if num_points == 1:
        return [float(min_probability)]

    if not log_p_grid:
        return [float(value) for value in np.linspace(min_probability, max_probability, num_points)]

    if min_probability <= 0.0:
        raise ValueError("--log-p-grid requires --min-p > 0")

    return [float(value) for value in np.geomspace(min_probability, max_probability, num_points)]


def simulate_for_error_probability(
    N: int,
    a: int,
    num_phase_bits: int,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
) -> RecoverySweepResult:
    """
    Estimate the empirical classical recovery probability for one fixed p.
    """
    estimate: RecoveryProbabilityEstimate = estimate_classical_recovery_probability(
        N=N,
        a=a,
        num_phase_bits=num_phase_bits,
        error_probability=error_probability,
        trials=trials,
        rng=rng,
    )

    return RecoverySweepResult(
        error_probability=estimate.error_probability,
        successful_trials=estimate.successful_trials,
        trials=estimate.trials,
        recovery_probability=estimate.recovery_probability,
        naive_baseline_probability=estimate.naive_baseline_probability,
    )


def write_data_summary(
    output_path: Path,
    N: int,
    a: int,
    num_phase_bits: int,
    trials: int,
    log_p_grid: bool,
    seed: int | None,
    results: list[RecoverySweepResult],
) -> None:
    """
    Write a text summary containing metadata, graph notes, and per-p data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)

    p_header = "error probability p"
    recovery_header = "empirical recovery"
    baseline_header = "non-propagating baseline"
    success_header = "successful trials"
    total_header = "total trials"

    p_width = max(
        len(p_header),
        *(len(f"{result.error_probability:.3f}") for result in results),
    )
    recovery_width = max(
        len(recovery_header),
        *(len(f"{result.recovery_probability:.3f}") for result in results),
    )
    baseline_width = max(
        len(baseline_header),
        *(len(f"{result.naive_baseline_probability:.3f}") for result in results),
    )
    success_width = max(
        len(success_header),
        *(len(str(result.successful_trials)) for result in results),
    )
    total_width = max(
        len(total_header),
        *(len(str(result.trials)) for result in results),
    )

    lines = [
        "Classical Recovery vs Error Probability Summary",
        f"N = {N}",
        f"a = {a}",
        f"Phase-estimation bits = {num_phase_bits}",
        f"Trials per probability value = {trials}",
        "Probability grid = "
        f"{'logarithmic' if log_p_grid else 'linear'}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        "  Fixed odd-composite N and fixed coprime base a",
        "  Each trial pairs a clean Kitaev phase-estimation run with a noisy run",
        "  The clean and noisy runs share the same measurement RNG seed",
        "  The noisy run applies independent per-detection measurement bit-flips with probability p",
        "  Classical recovery checks the noisy bitstring and all single-bit-flip neighbors",
        "  Probability values use a "
        f"{'logarithmic' if log_p_grid else 'linear'} sweep in p",
        "",
        "Per-probability data:",
        "  "
        f"{p_header:>{p_width}} | "
        f"{recovery_header:>{recovery_width}} | "
        f"{baseline_header:>{baseline_width}} | "
        f"{success_header:>{success_width}} | "
        f"{total_header:>{total_width}}",
    ]

    for result in results:
        lines.append(
            "  "
            f"{result.error_probability:>{p_width}.3f} | "
            f"{result.recovery_probability:>{recovery_width}.3f} | "
            f"{result.naive_baseline_probability:>{baseline_width}.3f} | "
            f"{result.successful_trials:>{success_width}} | "
            f"{result.trials:>{total_width}}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_recovery_vs_error_probability(
    N: int,
    a: int,
    num_phase_bits: int,
    trials: int,
    results: list[RecoverySweepResult],
    log_p_grid: bool,
    output_path: Path,
) -> None:
    """
    Create and save the recovery-probability plot versus measurement error p.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [result.error_probability for result in results]
    empirical_values = [result.recovery_probability for result in results]
    if log_p_grid and min(x_values) > 0.0:
        baseline_x_values = np.geomspace(min(x_values), max(x_values), 400)
    else:
        baseline_x_values = np.linspace(min(x_values), max(x_values), 400)
    baseline_y_values = [
        naive_single_bit_recovery_baseline(num_phase_bits, float(value))
        for value in baseline_x_values
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_values,
        empirical_values,
        marker="o",
        linewidth=2.0,
        markersize=6,
        color="#A23B2A",
        label="Empirical recovery probability",
    )
    ax.plot(
        baseline_x_values,
        baseline_y_values,
        linestyle="--",
        linewidth=1.8,
        color="#1F6F78",
        label="Non-propagating baseline",
    )

    ax.set_xlabel("Measurement bit-flip probability p")
    ax.set_ylabel("Recovery success probability")
    ax.set_title(
        "Single-Bit Classical Recovery vs Measurement Error Probability "
        f"for N = {N}, a = {a}"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlim(min(x_values) - 0.01, max(x_values) + 0.01)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax.text(
        0.98,
        0.02,
        f"phase bits = {num_phase_bits}\ntrials per p = {trials}",
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
    Build the command-line interface for the recovery-vs-p graph script.
    """
    parser = argparse.ArgumentParser(
        description="Plot classical single-bit recovery probability versus measurement error probability for a fixed (N, a)."
    )
    parser.add_argument("N", type=int, help="Odd composite number to analyze")
    parser.add_argument(
        "--a",
        type=int,
        required=True,
        help="Fixed coprime base a to analyze",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override the number of phase-estimation bits",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of paired clean/noisy trials per probability value",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Minimum measurement bit-flip probability in the sweep",
    )
    parser.add_argument(
        "--max-p",
        type=float,
        default=1.0,
        help="Maximum measurement bit-flip probability in the sweep",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=11,
        help="Number of probability values to sample between min-p and max-p",
    )
    parser.add_argument(
        "--log-p-grid",
        action="store_true",
        help=(
            "Use logarithmically spaced probability values between min-p and max-p. "
            "This requires min-p > 0."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional master RNG seed for reproducible sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom output path for the figure",
    )
    return parser


def main() -> None:
    start_time = time.perf_counter()

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("trials must be >= 1")

    validate_phase_estimation_instance(args.N, args.a)
    probability_values = build_probability_values(
        args.min_p,
        args.max_p,
        args.num_points,
        log_p_grid=args.log_p_grid,
    )

    if args.bits is None:
        num_phase_bits = default_num_phase_bits(args.N)
    else:
        num_phase_bits = args.bits

    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"classical_recovery_vs_error_probability_N{args.N}_a{args.a}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"classical_recovery_vs_error_probability_summary_N{args.N}_a{args.a}.txt"
    )

    master_rng = np.random.default_rng(args.seed)
    results = []
    for error_probability in probability_values:
        instance_rng = np.random.default_rng(int(master_rng.integers(0, 2 ** 63 - 1)))
        results.append(
            simulate_for_error_probability(
                N=args.N,
                a=args.a,
                num_phase_bits=num_phase_bits,
                error_probability=float(error_probability),
                trials=args.trials,
                rng=instance_rng,
            )
        )

    plot_recovery_vs_error_probability(
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        results=results,
        log_p_grid=args.log_p_grid,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        log_p_grid=args.log_p_grid,
        seed=args.seed,
        results=results,
    )

    seed_label = "random" if args.seed is None else str(args.seed)

    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}")
    print(f"a = {args.a}")
    print(f"Phase-estimation bits = {num_phase_bits}")
    print(f"Trials per probability value: {args.trials}")
    print(
        "Probability grid: "
        f"{'logarithmic' if args.log_p_grid else 'linear'}"
    )
    print(
        f"Measurement bit-flip probability swept from {args.min_p:.3f} "
        f"to {args.max_p:.3f}"
    )
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
