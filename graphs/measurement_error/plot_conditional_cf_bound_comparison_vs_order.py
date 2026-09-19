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
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.measurement_error.phase_distance_vs_order_common import (
    measurement_probability_display_label,
    validate_measurement_error_probabilities,
)
from graphs.measurement_error.plot_cf_bound_order_recovery_vs_order import (
    apply_adaptive_probability_yaxis,
)
from graphs.measurement_error.plot_eigenphase_peak_diagnostics_vs_order import (
    DEFAULT_INSTANCES_FILE,
    DEFAULT_NUM_PHASE_BITS,
    measurement_probabilities_filename_segment,
    run_paired_peak_trial,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
    add_combined_figure_legend,
    algorithm_color_for_base,
    analytic_window_diagnostics_console_label,
    analytic_window_diagnostics_summary_lines,
    analytic_window_mass_diagnostics_for_instances,
    instances_file_filename_segment,
    load_order_recovery_instances_from_file,
    standard_statevector_allows_subdefault_phase_bits,
    standard_backend_filename_segment,
    validate_statevector_budget,
)
from shor_common.errors import format_measurement_bitflip_probability
from shor_standard.analytic_sampling import QPEWindowMassDiagnostics
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)


STANDARD_SAME_PEAK_CONDITION = "standard_same_peak"
KITAEV_SAME_PEAK_CONDITION = "kitaev_same_peak"
STANDARD_SHIFTED_PEAK_CONDITION = "standard_shifted_peak"
KITAEV_SHIFTED_PEAK_CONDITION = "kitaev_shifted_peak"
CONDITIONS = (
    STANDARD_SAME_PEAK_CONDITION,
    KITAEV_SAME_PEAK_CONDITION,
    STANDARD_SHIFTED_PEAK_CONDITION,
    KITAEV_SHIFTED_PEAK_CONDITION,
)
SAME_PEAK_CONDITIONS = (
    STANDARD_SAME_PEAK_CONDITION,
    KITAEV_SAME_PEAK_CONDITION,
)
SHIFTED_PEAK_CONDITIONS = (
    STANDARD_SHIFTED_PEAK_CONDITION,
    KITAEV_SHIFTED_PEAK_CONDITION,
)
CONDITION_GROUPS = {
    "same_peak": SAME_PEAK_CONDITIONS,
    "shifted_peak": SHIFTED_PEAK_CONDITIONS,
}


@dataclass(frozen=True)
class ConditionalCFBoundComparisonResult:
    instance: OrderRecoveryInstance
    condition: str
    measurement_error_probability: float
    trials: int
    num_phase_bits: int
    eligible_trials: int
    within_bound_trials: int
    standard_backend: str = ANALYTIC_STANDARD_BACKEND

    @property
    def conditional_probability(self) -> float:
        if self.eligible_trials == 0:
            return math.nan
        return self.within_bound_trials / self.eligible_trials

    @property
    def eligible_probability(self) -> float:
        return self.eligible_trials / self.trials


def condition_algorithm(condition: str) -> str:
    """
    Return the PE algorithm used by one comparison condition.
    """
    if condition in (
        STANDARD_SAME_PEAK_CONDITION,
        STANDARD_SHIFTED_PEAK_CONDITION,
    ):
        return STANDARD_ALGORITHM
    if condition in (
        KITAEV_SAME_PEAK_CONDITION,
        KITAEV_SHIFTED_PEAK_CONDITION,
    ):
        return KITAEV_ALGORITHM
    raise ValueError("unknown condition")


def condition_display_label(condition: str) -> str:
    """
    Return the legend label for one comparison condition.
    """
    if condition == STANDARD_SAME_PEAK_CONDITION:
        return "standard: Pr(A | s' = s)"
    if condition == KITAEV_SAME_PEAK_CONDITION:
        return "Kitaev: Pr(A | s' = s)"
    if condition == STANDARD_SHIFTED_PEAK_CONDITION:
        return "standard: Pr(A | s' != s)"
    if condition == KITAEV_SHIFTED_PEAK_CONDITION:
        return "Kitaev: Pr(A | s' != s)"
    raise ValueError("unknown condition")


def condition_filename_label(condition: str) -> str:
    """
    Return a filename-safe label for one condition.
    """
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    return condition


def condition_is_eligible(condition: str, clean_peak: int, noisy_peak: int) -> bool:
    """
    Return whether a paired trial contributes to one conditional denominator.
    """
    if condition in (
        STANDARD_SAME_PEAK_CONDITION,
        KITAEV_SAME_PEAK_CONDITION,
    ):
        return noisy_peak == clean_peak
    if condition in (
        STANDARD_SHIFTED_PEAK_CONDITION,
        KITAEV_SHIFTED_PEAK_CONDITION,
    ):
        return noisy_peak != clean_peak
    raise ValueError("unknown condition")


def simulate_conditional_cf_bound_for_instance(
    instance: OrderRecoveryInstance,
    condition: str,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    measurement_error_probability: float,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> ConditionalCFBoundComparisonResult:
    """
    Estimate Pr(A | condition) for one configured instance.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")
    condition_filename_label(condition)

    algorithm = condition_algorithm(condition)
    eligible_trials = 0
    within_bound_trials = 0

    for _ in range(trials):
        clean_peak, noisy_peak, noisy_within_bound, _ = run_paired_peak_trial(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
        )
        if condition_is_eligible(condition, clean_peak, noisy_peak):
            eligible_trials += 1
            if noisy_within_bound:
                within_bound_trials += 1

    return ConditionalCFBoundComparisonResult(
        instance=instance,
        condition=condition,
        measurement_error_probability=measurement_error_probability,
        trials=trials,
        num_phase_bits=num_phase_bits,
        eligible_trials=eligible_trials,
        within_bound_trials=within_bound_trials,
        standard_backend=standard_backend,
    )


def output_path_for_plot(
    output_path: Path | None,
    condition_group: str,
    standard_backend: str,
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
) -> Path:
    """
    Return the figure path for one conditional comparison plot.
    """
    if condition_group not in CONDITION_GROUPS:
        raise ValueError("unknown condition group")

    if output_path is not None:
        return output_path.with_name(
            f"{output_path.stem}_{condition_group}{output_path.suffix}"
        )

    algorithm_prefix = (
        f"{standard_backend_filename_segment([STANDARD_ALGORITHM], standard_backend)}"
        f"{instances_file_filename_segment(instances_file)}"
    )
    probability_segment = measurement_probabilities_filename_segment(
        measurement_error_probabilities
    )
    return (
        Path(__file__).resolve().parent
        / "figures"
        / (
            "conditional_cf_bound_comparison_vs_order_"
            f"{condition_group}_"
            f"{algorithm_prefix}{probability_segment}t{num_phase_bits}.png"
        )
    )


def output_path_for_summary(
    standard_backend: str,
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
) -> Path:
    """
    Return the data-summary path for the conditional comparison plot.
    """
    algorithm_prefix = (
        f"{standard_backend_filename_segment([STANDARD_ALGORITHM], standard_backend)}"
        f"{instances_file_filename_segment(instances_file)}"
    )
    probability_segment = measurement_probabilities_filename_segment(
        measurement_error_probabilities
    )
    return (
        Path(__file__).resolve().parent
        / "data"
        / (
            "conditional_cf_bound_comparison_vs_order_summary_"
            f"{algorithm_prefix}{probability_segment}t{num_phase_bits}.txt"
        )
    )


def plot_conditional_cf_bound_comparison(
    output_path: Path,
    results: list[ConditionalCFBoundComparisonResult],
    condition_group: str,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    trials_per_instance: int,
    log_x: bool = False,
) -> None:
    """
    Plot one peak-relation conditional CF rate for both algorithms.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if condition_group not in CONDITION_GROUPS:
        raise ValueError("unknown condition group")
    conditions = CONDITION_GROUPS[condition_group]
    measurement_error_probabilities = validate_measurement_error_probabilities(
        measurement_error_probabilities
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    markers = {
        STANDARD_SAME_PEAK_CONDITION: "s",
        KITAEV_SAME_PEAK_CONDITION: "o",
        STANDARD_SHIFTED_PEAK_CONDITION: "s",
        KITAEV_SHIFTED_PEAK_CONDITION: "o",
    }
    linestyles = {
        STANDARD_SAME_PEAK_CONDITION: "--",
        KITAEV_SAME_PEAK_CONDITION: "-",
        STANDARD_SHIFTED_PEAK_CONDITION: "--",
        KITAEV_SHIFTED_PEAK_CONDITION: "-",
    }
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    probability_colors = {
        probability: color_cycle[index % len(color_cycle)]
        for index, probability in enumerate(measurement_error_probabilities)
    }
    plotted_values = []

    for probability in measurement_error_probabilities:
        for condition in conditions:
            condition_results = sorted(
                [
                    result
                    for result in results
                    if result.condition == condition
                    and result.measurement_error_probability == probability
                ],
                key=lambda result: result.instance.true_order,
            )
            y_values = [
                result.conditional_probability
                for result in condition_results
            ]
            plotted_values.extend(y_values)
            ax.plot(
                [result.instance.true_order for result in condition_results],
                y_values,
                marker=markers[condition],
                linestyle=linestyles[condition],
                linewidth=1.8,
                markersize=5.5,
                color=algorithm_color_for_base(
                    probability_colors[probability],
                    condition_algorithm(condition),
                ),
                alpha=0.9,
            )

    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel("Conditional probability of satisfying A")
    title_quantity = (
        "Pr(A | s' = s)"
        if condition_group == "same_peak"
        else "Pr(A | s' != s)"
    )
    ax.set_title(
        f"{title_quantity} vs True Order\n"
        f"standard vs Kitaev, t = {num_phase_bits}"
    )
    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    apply_adaptive_probability_yaxis(ax, plotted_values)
    ax.grid(True, alpha=0.25)

    condition_handles = [
        Line2D(
            [0],
            [0],
            color=algorithm_color_for_base("#333333", condition_algorithm(condition)),
            marker=markers[condition],
            linestyle=linestyles[condition],
            linewidth=1.8,
            markersize=6,
            label=condition_display_label(condition),
        )
        for condition in conditions
    ]
    probability_handles = [
        Line2D(
            [0],
            [0],
            color=probability_colors[probability],
            linestyle="-",
            linewidth=2.0,
            label=measurement_probability_display_label(probability),
        )
        for probability in measurement_error_probabilities
    ]

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=condition_handles,
        measurement_handles=probability_handles,
        trials_per_instance=trials_per_instance,
        algorithm_title="Algorithm",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[ConditionalCFBoundComparisonResult],
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    trials_per_instance: int,
    min_order: int | None,
    max_order: int | None,
    seed: int | None,
    analytic_window_diagnostics: list[QPEWindowMassDiagnostics] | None = None,
) -> None:
    """
    Write run metadata and per-order conditional comparison data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_error_probabilities = validate_measurement_error_probabilities(
        measurement_error_probabilities
    )
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if condition_algorithm(result.condition) == STANDARD_ALGORITHM
        }
    )
    seed_label = "random" if seed is None else str(seed)
    min_order_label = "none" if min_order is None else str(min_order)
    max_order_label = "none" if max_order is None else str(max_order)
    probability_label = ", ".join(
        measurement_probability_display_label(probability)
        for probability in measurement_error_probabilities
    )

    lines = [
        "Conditional CF-Bound Comparison Summary",
        "Comparison = standard and Kitaev Pr(A | s'=s) and Pr(A | s'!=s)",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"Instances file = {instances_file}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Instances used = {len({result.instance for result in results})}",
        f"Trials per instance and condition = {trials_per_instance}",
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
        "Definitions:",
        "  s = nearest eigenphase numerator to the clean bit string",
        "  s' = nearest eigenphase numerator to the noisy bit string",
        "  A = noisy phase is inside the CF sufficient bound around s'/r",
        "",
        "Per-order data:",
        "  condition | p | true r | N | a | Pr(A | condition) | Pr(condition) | A and condition count | condition count | trials",
    ]

    for result in sorted(
        results,
        key=lambda item: (
            item.condition,
            item.measurement_error_probability,
            item.instance.true_order,
        ),
    ):
        instance = result.instance
        lines.append(
            f"  {condition_display_label(result.condition):>30} |"
            f" {format_measurement_bitflip_probability(result.measurement_error_probability):>5} |"
            f" {instance.true_order:>6} |"
            f" {instance.N:>5} |"
            f" {instance.a:>5} |"
            f" {result.conditional_probability:>17.3f} |"
            f" {result.eligible_probability:>13.3f} |"
            f" {result.within_bound_trials:>21} |"
            f" {result.eligible_trials:>15} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the conditional CF-bound comparison script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard and Kitaev Pr(A | s'=s) and Pr(A | s'!=s) "
            "versus true order r."
        )
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=DEFAULT_INSTANCES_FILE,
        help="JSON file containing configured representative (N, a) pairs",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=DEFAULT_NUM_PHASE_BITS,
        help="Fixed number t of phase-estimation bits/control qubits",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help=(
            "Single independent per-bit measurement flip probability. If "
            "omitted and --error-probabilities is not used, defaults to 0.1."
        ),
    )
    parser.add_argument(
        "--error-probabilities",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Overlay independent per-bit measurement flip probabilities. "
            "This is mutually exclusive with --p."
        ),
    )
    parser.add_argument(
        "--trials-per-instance",
        type=int,
        default=100,
        help="Number of paired clean/noisy trials per instance and condition",
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
        default=ANALYTIC_STANDARD_BACKEND,
        help=(
            "Backend used for the standard conditions. The analytic "
            "backend is recommended for large t."
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
        help="Use a base-2 logarithmic x-axis for true order r",
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
    """
    Run the conditional CF-bound comparison graph script.
    """
    start_time = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.p is not None and args.error_probabilities is not None:
        parser.error("--p and --error-probabilities are mutually exclusive")
    if args.bits < 1:
        raise ValueError("bits must be >= 1")
    if args.trials_per_instance < 1:
        raise ValueError("trials_per_instance must be >= 1")
    if args.analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    try:
        if args.error_probabilities is None:
            raw_probability = 0.1 if args.p is None else args.p
            measurement_error_probabilities = validate_measurement_error_probabilities([
                raw_probability
            ])
        else:
            measurement_error_probabilities = validate_measurement_error_probabilities(
                args.error_probabilities
            )
        instances = load_order_recovery_instances_from_file(
            instances_file=args.instances_file,
            num_phase_bits=args.bits,
            min_order=args.min_order,
            max_order=args.max_order,
            enforce_default_phase_bits=not standard_statevector_allows_subdefault_phase_bits(
                [KITAEV_ALGORITHM, STANDARD_ALGORITHM],
                args.standard_backend,
            ),
        )
        validate_statevector_budget(
            algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
            standard_backend=args.standard_backend,
            num_phase_bits=args.bits,
            max_N=max(instance.N for instance in instances),
            max_statevector_amplitudes=args.max_statevector_amplitudes,
        )
    except ValueError as exc:
        parser.error(str(exc))

    analytic_window_diagnostics = None
    if args.standard_backend != STATEVECTOR_STANDARD_BACKEND:
        analytic_window_diagnostics = analytic_window_mass_diagnostics_for_instances(
            instances=instances,
            num_phase_bits=args.bits,
            analytic_window=args.analytic_window,
        )

    master_rng = np.random.default_rng(args.seed)
    results: list[ConditionalCFBoundComparisonResult] = []
    for measurement_error_probability in measurement_error_probabilities:
        for condition in CONDITIONS:
            for instance in instances:
                instance_rng = np.random.default_rng(
                    int(master_rng.integers(0, 2 ** 63 - 1))
                )
                results.append(
                    simulate_conditional_cf_bound_for_instance(
                        instance=instance,
                        condition=condition,
                        trials=args.trials_per_instance,
                        rng=instance_rng,
                        num_phase_bits=args.bits,
                        measurement_error_probability=measurement_error_probability,
                        standard_backend=args.standard_backend,
                        analytic_window=args.analytic_window,
                    )
                )

    figure_output_paths = [
        output_path_for_plot(
            output_path=args.output,
            condition_group=condition_group,
            standard_backend=args.standard_backend,
            instances_file=args.instances_file,
            num_phase_bits=args.bits,
            measurement_error_probabilities=measurement_error_probabilities,
        )
        for condition_group in CONDITION_GROUPS
    ]
    data_output_path = output_path_for_summary(
        standard_backend=args.standard_backend,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        measurement_error_probabilities=measurement_error_probabilities,
    )

    for figure_output_path, condition_group in zip(figure_output_paths, CONDITION_GROUPS):
        plot_conditional_cf_bound_comparison(
            output_path=figure_output_path,
            results=results,
            condition_group=condition_group,
            num_phase_bits=args.bits,
            measurement_error_probabilities=measurement_error_probabilities,
            trials_per_instance=args.trials_per_instance,
            log_x=args.log_x,
        )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        measurement_error_probabilities=measurement_error_probabilities,
        trials_per_instance=args.trials_per_instance,
        min_order=args.min_order,
        max_order=args.max_order,
        seed=args.seed,
        analytic_window_diagnostics=analytic_window_diagnostics,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    probability_label = ", ".join(
        measurement_probability_display_label(probability)
        for probability in measurement_error_probabilities
    )
    for figure_output_path in figure_output_paths:
        print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print("Comparison: standard and Kitaev Pr(A | s'=s) and Pr(A | s'!=s)")
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
    print(f"Trials per instance and condition: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
