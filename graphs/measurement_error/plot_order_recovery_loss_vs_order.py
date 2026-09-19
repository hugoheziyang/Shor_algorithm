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
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    ALGORITHM_CHOICES,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
    add_combined_figure_legend,
    algorithm_color_for_base,
    algorithm_display_label,
    algorithm_filename_prefix,
    analytic_window_diagnostics_console_label,
    analytic_window_diagnostics_summary_lines,
    analytic_window_mass_diagnostics_for_instances,
    algorithms_display_label,
    instances_file_filename_segment,
    load_order_recovery_instances_from_file,
    normalize_algorithms,
    simulate_order_recovery_for_instance,
    standard_statevector_allows_subdefault_phase_bits,
    standard_backend_filename_segment,
    validate_statevector_budget,
)
from shor_common.errors import (
    ErrorConfig,
    format_measurement_bitflip_probability,
)
from shor_standard.analytic_sampling import QPEWindowMassDiagnostics
from shor_standard.backends import STANDARD_BACKEND_CHOICES, STATEVECTOR_STANDARD_BACKEND


DEFAULT_ERROR_PROBABILITIES = (0.001, 0.01, 0.05, 0.1)
DEFAULT_MAX_STATEVECTOR_AMPLITUDES = 8_000_000


@dataclass
class OrderRecoveryLossResult:
    instance: OrderRecoveryInstance
    algorithm: str
    error_probability: float
    clean_successful_trials: int
    noisy_successful_trials: int
    trials: int
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND

    @property
    def clean_success_rate(self) -> float:
        return self.clean_successful_trials / self.trials

    @property
    def noisy_success_rate(self) -> float:
        return self.noisy_successful_trials / self.trials

    @property
    def recovery_loss(self) -> float:
        return self.clean_success_rate - self.noisy_success_rate

    @property
    def standard_error(self) -> float:
        if self.error_probability == 0.0:
            return 0.0

        clean_p = self.clean_success_rate
        noisy_p = self.noisy_success_rate
        variance = (
            (clean_p * (1.0 - clean_p))
            + (noisy_p * (1.0 - noisy_p))
        ) / self.trials
        return math.sqrt(variance)


def validate_error_probabilities(probabilities: list[float]) -> list[float]:
    """
    Validate and de-duplicate measurement bit-flip probabilities.
    """
    if not probabilities:
        raise ValueError("at least one error probability is required")

    normalized = []
    for probability in probabilities:
        p = float(probability)
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError("error probabilities must satisfy 0 <= p <= 1")
        if p not in normalized:
            normalized.append(p)

    return normalized


def error_probability_display_label(probability: float) -> str:
    """
    Return a compact plot label for one measurement bit-flip probability.
    """
    return f"p={format_measurement_bitflip_probability(probability)}"


def error_probabilities_filename_segment(probabilities: list[float]) -> str:
    """
    Return a filename segment for the selected probability values.
    """
    probabilities = validate_error_probabilities(probabilities)
    if probabilities == list(DEFAULT_ERROR_PROBABILITIES):
        return "default_p_"

    tokens = [
        format_measurement_bitflip_probability(probability).replace("-", "m")
        for probability in probabilities
    ]
    return "p_" + "_".join(tokens) + "_"


def simulate_loss_for_instance(
    instance: OrderRecoveryInstance,
    algorithm: str,
    clean_successful_trials: int,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> OrderRecoveryLossResult:
    """
    Estimate clean-minus-noisy true-order recovery probability for one instance.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    [error_probability] = validate_error_probabilities([error_probability])

    if error_probability == 0.0:
        noisy_successful_trials = clean_successful_trials
    else:
        noisy_error_config = ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=error_probability,
        )
        noisy_result = simulate_order_recovery_for_instance(
            instance=instance,
            algorithm=algorithm,
            trials=trials,
            rng=rng,
            num_phase_bits=num_phase_bits,
            error_config=noisy_error_config,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
        )
        noisy_successful_trials = noisy_result.successful_trials

    return OrderRecoveryLossResult(
        instance=instance,
        algorithm=algorithm,
        error_probability=error_probability,
        clean_successful_trials=clean_successful_trials,
        noisy_successful_trials=noisy_successful_trials,
        trials=trials,
        standard_backend=standard_backend,
    )


def write_data_summary(
    output_path: Path,
    results: list[OrderRecoveryLossResult],
    instances_file: Path,
    num_phase_bits: int,
    trials_per_instance: int,
    error_probabilities: list[float],
    min_order: int | None,
    max_order: int | None,
    seed: int | None,
    analytic_window_diagnostics: list[QPEWindowMassDiagnostics] | None = None,
) -> None:
    """
    Write metadata plus per-order recovery-loss data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    algorithms = normalize_algorithms([result.algorithm for result in results])
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )
    instances = sorted(
        {
            result.instance
            for result in results
        },
        key=lambda instance: instance.true_order,
    )
    seed_label = "random" if seed is None else str(seed)
    min_order_label = "none" if min_order is None else str(min_order)
    max_order_label = "none" if max_order is None else str(max_order)
    probability_label = ", ".join(
        error_probability_display_label(probability)
        for probability in error_probabilities
    )

    lines = [
        "Order-Recovery Loss-vs-Order Data Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"Instances file = {instances_file}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Instances used = {len(instances)}",
        f"Trials per clean/noisy estimate = {trials_per_instance}",
        f"Measurement bit-flip probabilities = {probability_label}",
        f"Order filter = {min_order_label} to {max_order_label}",
        f"Seed = {seed_label}",
        "",
        *analytic_window_diagnostics_summary_lines(analytic_window_diagnostics),
        *(
            [""]
            if analytic_window_diagnostics
            else []
        ),
        "Graph note:",
        "  The clean success probability is estimated once per instance and algorithm",
        "  Each noisy success probability is estimated with independent per-bit measurement flips",
        "  The plotted value is clean success probability minus noisy success probability",
        "  Negative values can occur from finite-trial sampling noise",
        "",
        "Per-order data:",
        "  algorithm | p | true r | N | a | clean success | noisy success | loss | clean successes | noisy successes | trials",
    ]

    sorted_results = sorted(
        results,
        key=lambda result: (
            result.algorithm,
            result.error_probability,
            result.instance.true_order,
        ),
    )
    for result in sorted_results:
        instance = result.instance
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {format_measurement_bitflip_probability(result.error_probability):>5} |"
            f" {instance.true_order:>6} |"
            f" {instance.N:>5} |"
            f" {instance.a:>5} |"
            f" {result.clean_success_rate:>13.3f} |"
            f" {result.noisy_success_rate:>13.3f} |"
            f" {result.recovery_loss:>6.3f} |"
            f" {result.clean_successful_trials:>15} |"
            f" {result.noisy_successful_trials:>15} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_order_recovery_loss(
    output_path: Path,
    results: list[OrderRecoveryLossResult],
    algorithms: list[str],
    error_probabilities: list[float],
    num_phase_bits: int,
    trials_per_instance: int,
    log_x: bool = False,
) -> None:
    """
    Create and save the order-recovery loss plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    error_probabilities = validate_error_probabilities(error_probabilities)

    fig, ax = plt.subplots(figsize=(11, 6))

    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }
    algorithm_linestyles = {
        KITAEV_ALGORITHM: "-",
        STANDARD_ALGORITHM: "--",
    }
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    probability_colors = {
        probability: color_cycle[index % len(color_cycle)]
        for index, probability in enumerate(error_probabilities)
    }

    for probability in error_probabilities:
        for algorithm in algorithms:
            curve_results = sorted(
                [
                    result
                    for result in results
                    if result.algorithm == algorithm
                    and result.error_probability == probability
                ],
                key=lambda result: result.instance.true_order,
            )
            x_values = [result.instance.true_order for result in curve_results]
            y_values = [result.recovery_loss for result in curve_results]
            ax.plot(
                x_values,
                y_values,
                marker=markers[algorithm],
                linestyle=algorithm_linestyles[algorithm],
                linewidth=1.6,
                markersize=5,
                color=algorithm_color_for_base(
                    probability_colors[probability],
                    algorithm,
                ),
                alpha=0.9,
            )

    ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.45)
    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel("Clean recovery probability minus noisy recovery probability")
    ax.set_title(
        "Measurement-Error Order-Recovery Loss vs True Order\n"
        f"{algorithms_display_label(algorithms)}, t = {num_phase_bits}"
    )
    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    y_values = [result.recovery_loss for result in results]
    y_min = min(y_values)
    y_max = max(y_values)
    ax.set_ylim(min(-0.05, y_min - 0.05), max(0.25, y_max + 0.05))
    ax.grid(True, alpha=0.25)

    algorithm_handles = [
        Line2D(
            [0],
            [0],
            color=algorithm_color_for_base("#333333", algorithm),
            marker=markers[algorithm],
            linestyle=algorithm_linestyles[algorithm],
            linewidth=1.8,
            markersize=6,
            label=algorithm_display_label(algorithm),
        )
        for algorithm in algorithms
    ]
    probability_handles = [
        Line2D(
            [0],
            [0],
            color=probability_colors[probability],
            linestyle="-",
            linewidth=2.0,
            label=error_probability_display_label(probability),
        )
        for probability in error_probabilities
    ]

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=algorithm_handles,
        measurement_handles=probability_handles,
        trials_per_instance=trials_per_instance,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the recovery-loss graph script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot clean-minus-noisy one-shot order-recovery probability versus "
            "true order r for configured Shor order-finding instances."
        )
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        required=True,
        help="JSON file containing configured representative (N, a) pairs",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM],
        help="One or more phase-estimation algorithms to sample",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=11,
        help="Fixed number t of phase-estimation bits/control qubits",
    )
    parser.add_argument(
        "--error-probabilities",
        type=float,
        nargs="+",
        default=list(DEFAULT_ERROR_PROBABILITIES),
        help="Independent per-bit measurement flip probabilities to compare",
    )
    parser.add_argument(
        "--trials-per-instance",
        type=int,
        default=100,
        help="Number of one-shot trials for each clean/noisy estimate",
    )
    parser.add_argument(
        "--min-order",
        type=int,
        default=None,
        help="Optional minimum true order r to keep from the instances file",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=None,
        help="Optional maximum true order r to keep from the instances file",
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
        "--max-statevector-amplitudes",
        type=int,
        default=DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
        help=(
            "Safety limit for standard statevector backend allocations. Use 0 "
            "to disable the guard."
        ),
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a base-2 logarithmic x-axis for the true order r",
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
    algorithms = normalize_algorithms(args.algorithm)

    if args.bits < 1:
        raise ValueError("bits must be >= 1")
    if args.trials_per_instance < 1:
        raise ValueError("trials_per_instance must be >= 1")
    if args.analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    try:
        error_probabilities = validate_error_probabilities(args.error_probabilities)
        instances = load_order_recovery_instances_from_file(
            instances_file=args.instances_file,
            num_phase_bits=args.bits,
            min_order=args.min_order,
            max_order=args.max_order,
            enforce_default_phase_bits=not standard_statevector_allows_subdefault_phase_bits(
                algorithms,
                args.standard_backend,
            ),
        )
        max_N = max(instance.N for instance in instances)
        validate_statevector_budget(
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            num_phase_bits=args.bits,
            max_N=max_N,
            max_statevector_amplitudes=args.max_statevector_amplitudes,
        )
    except ValueError as exc:
        parser.error(str(exc))

    analytic_window_diagnostics = None
    if (
        STANDARD_ALGORITHM in algorithms
        and args.standard_backend != STATEVECTOR_STANDARD_BACKEND
    ):
        analytic_window_diagnostics = analytic_window_mass_diagnostics_for_instances(
            instances=instances,
            num_phase_bits=args.bits,
            analytic_window=args.analytic_window,
        )

    master_rng = np.random.default_rng(args.seed)
    results: list[OrderRecoveryLossResult] = []

    for algorithm in algorithms:
        for instance in instances:
            clean_seed = int(master_rng.integers(0, 2 ** 63 - 1))
            clean_rng = np.random.default_rng(clean_seed)
            clean_result = simulate_order_recovery_for_instance(
                instance=instance,
                algorithm=algorithm,
                trials=args.trials_per_instance,
                rng=clean_rng,
                num_phase_bits=args.bits,
                error_config=ErrorConfig(),
                standard_backend=args.standard_backend,
                analytic_window=args.analytic_window,
            )
            for probability in error_probabilities:
                noisy_seed = int(master_rng.integers(0, 2 ** 63 - 1))
                noisy_rng = np.random.default_rng(noisy_seed)
                results.append(
                    simulate_loss_for_instance(
                        instance=instance,
                        algorithm=algorithm,
                        clean_successful_trials=clean_result.successful_trials,
                        error_probability=probability,
                        trials=args.trials_per_instance,
                        rng=noisy_rng,
                        num_phase_bits=args.bits,
                        standard_backend=args.standard_backend,
                        analytic_window=args.analytic_window,
                    )
                )

    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, args.standard_backend)}"
        f"{instances_file_filename_segment(args.instances_file)}"
    )
    probability_segment = error_probabilities_filename_segment(error_probabilities)
    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / (
                "order_recovery_loss_vs_order_"
                f"{algorithm_prefix}{probability_segment}t{args.bits}.png"
            )
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / (
            "order_recovery_loss_vs_order_summary_"
            f"{algorithm_prefix}{probability_segment}t{args.bits}.txt"
        )
    )

    plot_order_recovery_loss(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
        error_probabilities=error_probabilities,
        num_phase_bits=args.bits,
        trials_per_instance=args.trials_per_instance,
        log_x=args.log_x,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        trials_per_instance=args.trials_per_instance,
        error_probabilities=error_probabilities,
        min_order=args.min_order,
        max_order=args.max_order,
        seed=args.seed,
        analytic_window_diagnostics=analytic_window_diagnostics,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    probability_label = ", ".join(
        error_probability_display_label(probability)
        for probability in error_probabilities
    )
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"Algorithm: {algorithms_display_label(algorithms)}")
    if STANDARD_ALGORITHM in algorithms:
        print(f"Standard backend: {args.standard_backend}")
        if args.standard_backend != STATEVECTOR_STANDARD_BACKEND:
            print(f"Analytic window: {args.analytic_window}")
            print(
                "Analytic window captured mass: "
                f"{analytic_window_diagnostics_console_label(analytic_window_diagnostics)}"
            )
    print(f"Instances file: {args.instances_file}")
    print(f"Instances used: {len(instances)}")
    print(f"Fixed phase-estimation bits: {args.bits}")
    print(f"Measurement bit-flip probabilities: {probability_label}")
    print(f"Trials per clean/noisy estimate: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
