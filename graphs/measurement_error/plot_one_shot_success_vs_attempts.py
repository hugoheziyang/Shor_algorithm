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


from shor_common.errors import (
    ErrorConfig,
    MEASUREMENT_BITFLIP_SINGLE_TOKEN,
    measurement_bitflip_filename_suffix,
    measurement_bitflip_label,
    measurement_bitflip_title_suffix,
    parse_measurement_bitflip_error_arg,
)
from shor_common.shor_utils import (
    default_num_phase_bits,
    perfect_power_decomposition,
)
from shor_one_qubit.shor import shor_factor as kitaev_shor_factor
from shor_standard.backends import STANDARD_BACKEND_CHOICES, STATEVECTOR_STANDARD_BACKEND
from shor_standard.shor import shor_factor as standard_shor_factor


KITAEV_ALGORITHM = "kitaev"
STANDARD_ALGORITHM = "standard"
ALGORITHM_CHOICES = (KITAEV_ALGORITHM, STANDARD_ALGORITHM)


@dataclass
class AttemptSweepResult:
    num_attempts: int
    successful_trials: int
    trials: int
    algorithm: str = KITAEV_ALGORITHM
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND

    @property
    def success_rate(self) -> float:
        return self.successful_trials / self.trials


def normalize_algorithms(algorithms: list[str]) -> list[str]:
    """
    Validate and de-duplicate requested algorithms while preserving order.
    """
    if not algorithms:
        raise ValueError("at least one algorithm is required")

    normalized = []
    for algorithm in algorithms:
        if algorithm not in ALGORITHM_CHOICES:
            raise ValueError("unknown algorithm")
        if algorithm not in normalized:
            normalized.append(algorithm)

    return normalized


def algorithm_display_label(algorithm: str) -> str:
    """
    Return a human-readable label for one algorithm.
    """
    if algorithm == KITAEV_ALGORITHM:
        return "Kitaev"
    if algorithm == STANDARD_ALGORITHM:
        return "standard Shor"
    raise ValueError("unknown algorithm")


def algorithms_display_label(algorithms: list[str]) -> str:
    """
    Return a human-readable label for one or more algorithms.
    """
    algorithms = normalize_algorithms(algorithms)
    if len(algorithms) == 1:
        return algorithm_display_label(algorithms[0])

    return " vs ".join(algorithm_display_label(algorithm) for algorithm in algorithms)


def algorithm_filename_prefix(algorithms: list[str]) -> str:
    """
    Return a filename prefix while preserving default Kitaev filenames.
    """
    algorithms = normalize_algorithms(algorithms)
    if algorithms == [KITAEV_ALGORITHM]:
        return ""
    if algorithms == [STANDARD_ALGORITHM]:
        return "standard_"
    if algorithms == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]:
        return "compare_algorithms_"

    return "compare_" + "_".join(algorithms) + "_"


def standard_backend_filename_segment(
    algorithms: list[str],
    standard_backend: str,
) -> str:
    """
    Return an extra filename segment for non-default standard backends.
    """
    algorithms = normalize_algorithms(algorithms)
    if STANDARD_ALGORITHM not in algorithms:
        return ""
    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        return ""
    return f"{standard_backend}_"


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


def validate_target_number(N: int) -> None:
    """
    Validate the target N for the one-shot run-count sweep experiment.
    """
    if not is_odd_composite(N):
        raise ValueError("N must be an odd composite integer >= 9")

    if perfect_power_decomposition(N) is not None:
        raise ValueError(
            "perfect powers are excluded here because this repository treats "
            "them as classical shortcuts rather than quantum order-finding cases"
        )


def simulate_for_attempt_budget(
    N: int,
    max_attempts: int,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int | None = None,
    error_config: ErrorConfig | None = None,
    algorithm: str = KITAEV_ALGORITHM,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> AttemptSweepResult:
    """
    Estimate the success probability when allowing up to max_attempts one-shot runs.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    successful_trials = 0
    error_config = error_config or ErrorConfig()

    for _ in range(trials):
        if algorithm == KITAEV_ALGORITHM:
            result = kitaev_shor_factor(
                N=N,
                max_attempts=max_attempts,
                num_phase_bits=num_phase_bits,
                rng=rng,
                postprocess_mode="one_shot",
                error_config=error_config,
            )
        elif algorithm == STANDARD_ALGORITHM:
            result = standard_shor_factor(
                N=N,
                max_attempts=max_attempts,
                num_control_qubits=num_phase_bits,
                rng=rng,
                postprocess_mode="one_shot",
                error_config=error_config,
                standard_backend=standard_backend,
                analytic_window=analytic_window,
            )
        else:
            raise ValueError("unknown algorithm")

        if result["success"]:
            successful_trials += 1

    return AttemptSweepResult(
        num_attempts=max_attempts,
        successful_trials=successful_trials,
        trials=trials,
        algorithm=algorithm,
        standard_backend=standard_backend,
    )


def write_data_summary(
    output_path: Path,
    N: int,
    results: list[AttemptSweepResult],
    trials_per_attempt_count: int,
    num_phase_bits: int,
    error_config: ErrorConfig | None,
    seed: int | None,
) -> None:
    """
    Write a text summary containing metadata, graph notes, and per-run-count data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)
    error_label = measurement_bitflip_label(error_config)
    algorithms = normalize_algorithms([result.algorithm for result in results])
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )

    lines = [
        "One-Shot Shor Success-vs-Runs Data Summary",
        f"N = {N}",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"Phase-estimation bits = {num_phase_bits}",
        f"Measurement bit-flip error = {error_label}",
        f"Trials per chosen run count = {trials_per_attempt_count}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        "  Fixed odd-composite N",
        "  Each trial uses the selected top-level shor_factor flow with postprocess_mode = one_shot",
        f"  Measurement bit-flip error: {error_label}",
        "  A fresh random coprime base a may be chosen internally for each one-shot run",
        "  Each plotted point corresponds to a chosen maximum number of allowed one-shot runs",
        f"  Maximum number of allowed one-shot runs swept from {results[0].num_attempts} to {results[-1].num_attempts}",
        "",
        "Per-run-count data:",
        "  algorithm | max one-shot runs | success rate | successful trials | total trials",
    ]

    for result in results:
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>9} |"
            f" {result.num_attempts:>17} |"
            f" {result.success_rate:>12.3f} |"
            f" {result.successful_trials:>17} |"
            f" {result.trials:>12}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_success_vs_attempts(
    N: int,
    results: list[AttemptSweepResult],
    num_phase_bits: int,
    error_config: ErrorConfig | None,
    output_path: Path,
) -> None:
    """
    Create and save the success-probability plot versus number of one-shot runs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    algorithms = normalize_algorithms([result.algorithm for result in results])
    results_by_algorithm = {
        algorithm: [
            result
            for result in results
            if result.algorithm == algorithm
        ]
        for algorithm in algorithms
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        KITAEV_ALGORITHM: "#A23B2A",
        STANDARD_ALGORITHM: "#0072B2",
    }
    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }

    for algorithm in algorithms:
        algorithm_results = results_by_algorithm[algorithm]
        x_values = [result.num_attempts for result in algorithm_results]
        y_values = [result.success_rate for result in algorithm_results]
        ax.plot(
            x_values,
            y_values,
            marker=markers[algorithm],
            linewidth=2.0,
            markersize=6,
            color=colors[algorithm],
            label=algorithm_display_label(algorithm),
        )

    ax.set_xlabel("Maximum number of allowed one-shot runs")
    ax.set_ylabel("Empirical factorization success probability")
    ax.set_title(
        "One-Shot Shor Success vs Number of Runs\n"
        f"{algorithms_display_label(algorithms)} for N = "
        f"{N}{measurement_bitflip_title_suffix(error_config)}"
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    if len(algorithms) > 1:
        ax.legend(fontsize=9)

    ax.text(
        0.98,
        0.02,
        "phase bits = "
        f"{num_phase_bits}\nmeasurement bit-flip error = "
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
    Build the command-line interface for the success-vs-runs graph script.
    """
    parser = argparse.ArgumentParser(
        description="Plot one-shot Shor factorization success versus the allowed number of one-shot runs."
    )
    parser.add_argument("N", type=int, help="Odd composite number to factor")
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM],
        help="One or more phase-estimation algorithms to sample",
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=STATEVECTOR_STANDARD_BACKEND,
        help=(
            "Backend used for standard Shor curves. Use 'analytic' to sample "
            "measurement outcomes from the classically computed order without "
            "allocating the full standard-Shor state vector."
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
        "--min-attempts",
        type=int,
        default=1,
        help="Minimum number of allowed one-shot runs in the sweep",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Maximum number of allowed one-shot runs in the sweep",
    )
    parser.add_argument(
        "--trials-per-attempt-count",
        type=int,
        default=100,
        help="Number of independent simulated top-level runs for each chosen number of allowed one-shot runs",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override the number of phase-estimation bits for all runs",
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
            "exactly one random bit-flip per phase-estimation run, "
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
    algorithms = normalize_algorithms(args.algorithm)
    try:
        error_config = parse_measurement_bitflip_error_arg(
            args.measurement_bitflip_error
        )
    except ValueError as exc:
        parser.error(str(exc))

    validate_target_number(args.N)

    if args.min_attempts < 1:
        raise ValueError("min_attempts must be >= 1")
    if args.max_attempts < args.min_attempts:
        raise ValueError("max_attempts must be >= min_attempts")
    if args.trials_per_attempt_count < 1:
        raise ValueError("trials_per_attempt_count must be >= 1")
    if args.analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    if args.bits is None:
        num_phase_bits = default_num_phase_bits(args.N)
    else:
        num_phase_bits = args.bits

    if args.output is None:
        filename_suffix = measurement_bitflip_filename_suffix(error_config)
        algorithm_prefix = (
            f"{algorithm_filename_prefix(algorithms)}"
            f"{standard_backend_filename_segment(algorithms, args.standard_backend)}"
        )
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"one_shot_success_vs_attempts_{algorithm_prefix}N{args.N}{filename_suffix}.png"
        )
    else:
        figure_output_path = args.output

    filename_suffix = measurement_bitflip_filename_suffix(error_config)
    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, args.standard_backend)}"
    )
    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"one_shot_success_vs_attempts_summary_{algorithm_prefix}N{args.N}{filename_suffix}.txt"
    )

    master_rng = np.random.default_rng(args.seed)
    results: list[AttemptSweepResult] = []

    for algorithm in algorithms:
        for max_attempts in range(args.min_attempts, args.max_attempts + 1):
            instance_seed = int(master_rng.integers(0, 2 ** 63 - 1))
            instance_rng = np.random.default_rng(instance_seed)
            results.append(
                simulate_for_attempt_budget(
                    N=args.N,
                    max_attempts=max_attempts,
                    trials=args.trials_per_attempt_count,
                    rng=instance_rng,
                    num_phase_bits=num_phase_bits,
                    error_config=error_config,
                    algorithm=algorithm,
                    standard_backend=args.standard_backend,
                    analytic_window=args.analytic_window,
                )
            )

    plot_success_vs_attempts(
        N=args.N,
        results=results,
        num_phase_bits=num_phase_bits,
        error_config=error_config,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        N=args.N,
        results=results,
        trials_per_attempt_count=args.trials_per_attempt_count,
        num_phase_bits=num_phase_bits,
        error_config=error_config,
        seed=args.seed,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}")
    print(f"Algorithm: {algorithms_display_label(algorithms)}")
    if STANDARD_ALGORITHM in algorithms:
        print(f"Standard backend: {args.standard_backend}")
        if args.standard_backend != STATEVECTOR_STANDARD_BACKEND:
            print(f"Analytic window: {args.analytic_window}")
    print(f"Phase-estimation bits = {num_phase_bits}")
    print(f"Measurement bit-flip error: {measurement_bitflip_label(error_config)}")
    print(
        "Maximum number of allowed one-shot runs "
        f"swept from {args.min_attempts} to {args.max_attempts}"
    )
    print(f"Trials per chosen run count: {args.trials_per_attempt_count}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
