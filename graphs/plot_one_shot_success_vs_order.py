from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
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


from shor_one_qubit.classical_post import shor_factors_from_order
from shor_one_qubit.shor import (
    default_num_phase_bits,
    perfect_power_decomposition,
    shor_order_finding_attempt,
)


@dataclass
class InstanceResult:
    N: int
    a: int
    true_order: int
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
    Validate that N is appropriate for the quantum order-finding workflow.
    """
    if not is_odd_composite(N):
        raise ValueError("N must be an odd composite integer >= 9")

    if perfect_power_decomposition(N) is not None:
        raise ValueError(
            "perfect powers are excluded here because the top-level Shor flow "
            "factors them classically before the quantum order-finding step"
        )


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


def admissible_bases_with_orders(N: int) -> list[tuple[int, int]]:
    """
    Return all admissible bases a together with their true orders modulo N.
    """
    validate_target_number(N)

    instances = []
    for a in range(2, N - 1):
        if math.gcd(a, N) != 1:
            continue
        instances.append((a, multiplicative_order(a, N)))

    return instances


def simulate_instance(
    N: int,
    a: int,
    true_order: int,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int | None = None,
) -> InstanceResult:
    """
    Estimate the one-shot factorization success rate for a fixed (N, a).
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    if num_phase_bits is None:
        num_phase_bits = default_num_phase_bits(N)

    successful_trials = 0

    for _ in range(trials):
        result = shor_order_finding_attempt(
            N=N,
            a=a,
            num_phase_bits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
        )

        factors = shor_factors_from_order(a, result["order"], N)
        if factors is not None:
            successful_trials += 1

    return InstanceResult(
        N=N,
        a=a,
        true_order=true_order,
        successful_trials=successful_trials,
        trials=trials,
    )


def summarize_by_order(instance_results: list[InstanceResult]) -> list[dict]:
    """
    Aggregate per-base success rates by the true order r.
    """
    grouped_rates: dict[int, list[float]] = defaultdict(list)

    for result in instance_results:
        grouped_rates[result.true_order].append(result.success_rate)

    summary = []
    for order in sorted(grouped_rates):
        rates = np.asarray(grouped_rates[order], dtype=float)
        summary.append(
            {
                "order": order,
                "mean_success_rate": float(np.mean(rates)),
                "std_success_rate": float(np.std(rates)),
                "num_bases": int(len(rates)),
            }
        )

    return summary


def write_data_summary(
    output_path: Path,
    N: int,
    instance_results: list[InstanceResult],
    summary_rows: list[dict],
    trials_per_a: int,
    seed: int | None,
) -> None:
    """
    Write a text summary containing run metadata, figure notes, and per-base data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)
    ordered_results = sorted(instance_results, key=lambda result: result.a)

    lines = [
        "One-Shot Shor Graph Data Summary",
        f"N = {N}",
        f"Admissible bases used = {len(ordered_results)}",
        f"Trials per base = {trials_per_a}",
        f"Seed = {seed_label}",
        "",
        "Order graph note:",
        "  All coprime bases a with 2 <= a <= N - 2",
        f"  {trials_per_a} one-shot trials per base, seed={seed_label}",
        "  Here, n = number of bases with that true order",
        "",
        "Per-base graph note:",
        "  All coprime bases a with 2 <= a <= N - 2",
        f"  {trials_per_a} one-shot trials per base, seed={seed_label}",
        "  Here, r = true multiplicative order for that base",
        "",
        "Per-base data:",
        "  base a | true order r | success rate",
    ]

    for result in ordered_results:
        lines.append(
            f"  {result.a:>6} | {result.true_order:>12} | {result.success_rate:.3f}"
        )

    lines.extend(
        [
            "",
            "Order summary:",
            "  order r | mean success |   std | number of bases",
        ]
    )

    for row in summary_rows:
        lines.append(
            f"  {row['order']:>7} |"
            f" {row['mean_success_rate']:>12.3f} |"
            f" {row['std_success_rate']:>5.3f} |"
            f" {row['num_bases']:>15}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_success_vs_order(
    N: int,
    instance_results: list[InstanceResult],
    summary_rows: list[dict],
    output_path: Path,
) -> None:
    """
    Create and save the success-probability plot for one fixed number N.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_scatter = [result.true_order for result in instance_results]
    y_scatter = [result.success_rate for result in instance_results]

    x_mean = [row["order"] for row in summary_rows]
    y_mean = [row["mean_success_rate"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        x_scatter,
        y_scatter,
        s=40,
        alpha=0.4,
        color="#5B7B7A",
        edgecolors="none",
        label="Per-base success rate",
    )
    ax.scatter(
        x_mean,
        y_mean,
        marker="x",
        color="#A23B2A",
        s=90,
        linewidths=2.0,
        label="Mean by true order r",
    )

    for row in summary_rows:
        ax.annotate(
            f"n={row['num_bases']}",
            (row["order"], row["mean_success_rate"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#303030",
        )

    ax.set_xlabel("True multiplicative order r", labelpad=18)
    ax.set_ylabel("Empirical one-shot factorization success probability")
    ax.set_title(f"One-Shot Shor Success Probability vs True Order for N = {N}")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_success_vs_base(
    N: int,
    instance_results: list[InstanceResult],
    output_path: Path,
) -> None:
    """
    Create and save the per-base success-probability plot for one fixed N.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_results = sorted(instance_results, key=lambda result: result.a)
    x_values = [result.a for result in ordered_results]
    y_values = [result.success_rate for result in ordered_results]

    fig_width = max(11, 0.28 * len(x_values) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    ax.bar(
        x_values,
        y_values,
        color="#5B7B7A",
        alpha=0.85,
        width=0.75,
    )

    ax.set_xlabel("Base a", labelpad=18)
    ax.set_ylabel("Empirical one-shot factorization success probability")
    ax.set_title(f"One-Shot Shor Success Probability by Base for N = {N}")
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(value) for value in x_values], rotation=90, fontsize=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, axis="y", alpha=0.25)
    ax.margins(x=0.01)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the plotting script.
    """
    parser = argparse.ArgumentParser(
        description="Plot one-shot Shor factorization success versus true order r for a fixed N."
    )
    parser.add_argument("N", type=int, help="Odd composite number to factor")
    parser.add_argument(
        "--trials-per-a",
        type=int,
        default=100,
        help="Number of one-shot trials to simulate for each admissible base a",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional master RNG seed for reproducible sampling",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override the number of phase-estimation bits for all runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output figure file",
    )
    return parser


def main() -> None:
    start_time = time.perf_counter()

    parser = build_argument_parser()
    args = parser.parse_args()

    validate_target_number(args.N)
    bases_with_orders = admissible_bases_with_orders(args.N)
    if not bases_with_orders:
        raise ValueError("no admissible coprime bases found for the requested N")

    if args.output is None:
        output_path = Path(__file__).resolve().parent / "figures" / f"one_shot_success_vs_order_N{args.N}.png"
        per_base_output_path = Path(__file__).resolve().parent / "figures" / f"one_shot_success_per_base_N{args.N}.png"
    else:
        output_path = args.output
        per_base_output_path = args.output.with_name(
            f"{args.output.stem}_per_base{args.output.suffix}"
        )
    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"one_shot_success_vs_order_summary_N{args.N}.txt"
    )

    master_rng = np.random.default_rng(args.seed)
    instance_results: list[InstanceResult] = []

    for a, true_order in bases_with_orders:
        instance_seed = int(master_rng.integers(0, 2 ** 63 - 1))
        instance_rng = np.random.default_rng(instance_seed)
        instance_results.append(
            simulate_instance(
                N=args.N,
                a=a,
                true_order=true_order,
                trials=args.trials_per_a,
                rng=instance_rng,
                num_phase_bits=args.bits,
            )
        )

    summary_rows = summarize_by_order(instance_results)
    plot_success_vs_order(
        N=args.N,
        instance_results=instance_results,
        summary_rows=summary_rows,
        output_path=output_path,
    )
    plot_success_vs_base(
        N=args.N,
        instance_results=instance_results,
        output_path=per_base_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        N=args.N,
        instance_results=instance_results,
        summary_rows=summary_rows,
        trials_per_a=args.trials_per_a,
        seed=args.seed,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved probability-vs-order figure to: {output_path}")
    print(f"Saved probability-vs-base figure to: {per_base_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}")
    print(f"Admissible bases used: {len(bases_with_orders)}")
    print(f"Trials per base: {args.trials_per_a}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")
    print("Order summary:")
    for row in summary_rows:
        print(
            f"  r={row['order']:>2}  "
            f"mean_success={row['mean_success_rate']:.3f}  "
            f"std={row['std_success_rate']:.3f}  "
            f"bases={row['num_bases']}"
        )


if __name__ == "__main__":
    main()
