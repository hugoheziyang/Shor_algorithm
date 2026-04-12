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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from shor_one_qubit.shor import (
    default_num_phase_bits,
    perfect_power_decomposition,
    shor_factor,
)


@dataclass
class AttemptSweepResult:
    num_attempts: int
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
) -> AttemptSweepResult:
    """
    Estimate the success probability when allowing up to max_attempts one-shot runs.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    successful_trials = 0

    for _ in range(trials):
        result = shor_factor(
            N=N,
            max_attempts=max_attempts,
            num_phase_bits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
        )
        if result["success"]:
            successful_trials += 1

    return AttemptSweepResult(
        num_attempts=max_attempts,
        successful_trials=successful_trials,
        trials=trials,
    )


def write_data_summary(
    output_path: Path,
    N: int,
    results: list[AttemptSweepResult],
    trials_per_attempt_count: int,
    num_phase_bits: int,
    seed: int | None,
) -> None:
    """
    Write a text summary containing metadata, graph notes, and per-run-count data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)

    lines = [
        "One-Shot Shor Success-vs-Runs Data Summary",
        f"N = {N}",
        f"Phase-estimation bits = {num_phase_bits}",
        f"Trials per chosen run count = {trials_per_attempt_count}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        "  Fixed odd-composite N",
        "  Each trial uses the top-level shor_factor flow with postprocess_mode = one_shot",
        "  A fresh random coprime base a may be chosen internally for each one-shot run",
        "  Each plotted point corresponds to a chosen maximum number of allowed one-shot runs",
        f"  Maximum number of allowed one-shot runs swept from {results[0].num_attempts} to {results[-1].num_attempts}",
        "",
        "Per-run-count data:",
        "  max one-shot runs | success rate | successful trials | total trials",
    ]

    for result in results:
        lines.append(
            f"  {result.num_attempts:>12} |"
            f" {result.success_rate:>12.3f} |"
            f" {result.successful_trials:>17} |"
            f" {result.trials:>12}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_success_vs_attempts(
    N: int,
    results: list[AttemptSweepResult],
    num_phase_bits: int,
    output_path: Path,
) -> None:
    """
    Create and save the success-probability plot versus number of one-shot runs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [result.num_attempts for result in results]
    y_values = [result.success_rate for result in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2.0,
        markersize=6,
        color="#A23B2A",
    )

    ax.set_xlabel("Maximum number of allowed one-shot runs")
    ax.set_ylabel("Empirical factorization success probability")
    ax.set_title(f"One-Shot Shor Success vs Number of Runs for N = {N}")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)

    ax.text(
        0.98,
        0.02,
        f"phase bits = {num_phase_bits}",
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

    validate_target_number(args.N)

    if args.min_attempts < 1:
        raise ValueError("min_attempts must be >= 1")
    if args.max_attempts < args.min_attempts:
        raise ValueError("max_attempts must be >= min_attempts")
    if args.trials_per_attempt_count < 1:
        raise ValueError("trials_per_attempt_count must be >= 1")

    if args.bits is None:
        num_phase_bits = default_num_phase_bits(args.N)
    else:
        num_phase_bits = args.bits

    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"one_shot_success_vs_attempts_N{args.N}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"one_shot_success_vs_attempts_summary_N{args.N}.txt"
    )

    master_rng = np.random.default_rng(args.seed)
    results: list[AttemptSweepResult] = []

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
            )
        )

    plot_success_vs_attempts(
        N=args.N,
        results=results,
        num_phase_bits=num_phase_bits,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        N=args.N,
        results=results,
        trials_per_attempt_count=args.trials_per_attempt_count,
        num_phase_bits=num_phase_bits,
        seed=args.seed,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}")
    print(f"Phase-estimation bits = {num_phase_bits}")
    print(
        "Maximum number of allowed one-shot runs "
        f"swept from {args.min_attempts} to {args.max_attempts}"
    )
    print(f"Trials per chosen run count: {args.trials_per_attempt_count}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
