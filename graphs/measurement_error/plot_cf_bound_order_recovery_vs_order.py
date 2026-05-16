from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
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
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
    add_combined_figure_legend,
    algorithm_color_for_base,
    algorithm_display_label,
    algorithm_filename_prefix,
    algorithms_display_label,
    analytic_window_diagnostics_console_label,
    analytic_window_diagnostics_summary_lines,
    analytic_window_mass_diagnostics_for_instances,
    error_config_for_probability,
    error_probabilities_filename_segment,
    error_probability_display_label,
    instances_file_filename_segment,
    load_order_recovery_instances_from_file,
    normalize_algorithms,
    standard_backend_filename_segment,
    validate_error_probabilities,
    validate_statevector_budget,
)
from shor_common.classical_post import bits_to_fraction
from shor_common.errors import (
    choose_measurement_bitflip_mask,
    format_measurement_bitflip_probability,
)
from shor_common.shor_utils import recover_order_with_mode
from shor_one_qubit.shor import shor_order_finding_attempt as kitaev_order_attempt
from shor_standard.analytic_sampling import (
    QPEWindowMassDiagnostics,
    sample_standard_phase_bits_from_order,
    validate_analytic_error_config,
)
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)
from shor_standard.shor import shor_order_finding_attempt as standard_order_attempt


DEFAULT_INSTANCES_FILE = Path(__file__).resolve().parent / "config" / "order_recovery_instances.json"
DEFAULT_NUM_PHASE_BITS = 29

A_PROBABILITY = "within_bound"
B_PROBABILITY = "recovered_order"
OUTSIDE_B_JOINT_PROBABILITY = "outside_recovered_joint"
OUTSIDE_B_CONDITIONAL_PROBABILITY = "outside_recovered_conditional"
PLOTTED_QUANTITIES = (
    A_PROBABILITY,
    B_PROBABILITY,
    OUTSIDE_B_JOINT_PROBABILITY,
    OUTSIDE_B_CONDITIONAL_PROBABILITY,
)


@dataclass(frozen=True)
class CFBoundOrderRecoveryResult:
    instance: OrderRecoveryInstance
    algorithm: str
    measurement_error_probability: float
    trials: int
    num_phase_bits: int
    within_bound_trials: int
    recovered_trials: int
    recovered_within_bound_trials: int
    recovered_outside_bound_trials: int
    standard_backend: str = ANALYTIC_STANDARD_BACKEND

    @property
    def outside_bound_trials(self) -> int:
        return self.trials - self.within_bound_trials

    @property
    def within_bound_probability(self) -> float:
        return self.within_bound_trials / self.trials

    @property
    def outside_bound_probability(self) -> float:
        return self.outside_bound_trials / self.trials

    @property
    def recovery_probability(self) -> float:
        return self.recovered_trials / self.trials

    @property
    def recovered_outside_bound_probability(self) -> float:
        return self.recovered_outside_bound_trials / self.trials

    @property
    def recovered_given_outside_bound_probability(self) -> float:
        if self.outside_bound_trials == 0:
            return math.nan
        return self.recovered_outside_bound_trials / self.outside_bound_trials

    @property
    def recovered_given_within_bound_probability(self) -> float:
        if self.within_bound_trials == 0:
            return math.nan
        return self.recovered_within_bound_trials / self.within_bound_trials


def circular_fraction_distance(left: Fraction, right: Fraction) -> Fraction:
    """
    Return min(|left-right|, 1-|left-right|) for fractions modulo one.
    """
    if not (0 <= left < 1) or not (0 <= right < 1):
        raise ValueError("fractions must satisfy 0 <= value < 1")

    distance = abs(left - right)
    return min(distance, Fraction(1, 1) - distance)


def nearest_order_phase_distance(
    phase_fraction: Fraction,
    true_order: int,
) -> tuple[Fraction, int]:
    """
    Return the distance from a measured phase to the nearest fraction s/r.
    """
    if not (0 <= phase_fraction < 1):
        raise ValueError("phase_fraction must satisfy 0 <= value < 1")
    if true_order < 1:
        raise ValueError("true_order must be >= 1")

    scaled_phase = phase_fraction * true_order
    floor_candidate = scaled_phase.numerator // scaled_phase.denominator
    candidate_numerators = {
        (floor_candidate - 1) % true_order,
        floor_candidate % true_order,
        (floor_candidate + 1) % true_order,
    }

    best_numerator = min(
        candidate_numerators,
        key=lambda numerator: (
            circular_fraction_distance(
                phase_fraction,
                Fraction(numerator, true_order),
            ),
            numerator,
        ),
    )
    return (
        circular_fraction_distance(
            phase_fraction,
            Fraction(best_numerator, true_order),
        ),
        best_numerator,
    )


def normalized_cf_error(phase_fraction: Fraction, true_order: int) -> Fraction:
    """
    Return 2*r^2 times the distance to the nearest exact eigenphase s/r.

    The sufficient continued-fraction condition |y/2^t - s/r| <= 1/(2r^2)
    is exactly normalized_cf_error <= 1.
    """
    distance, _ = nearest_order_phase_distance(
        phase_fraction=phase_fraction,
        true_order=true_order,
    )
    return Fraction(2 * true_order * true_order, 1) * distance


def within_cf_sufficient_bound(bits: list[int], true_order: int) -> bool:
    """
    Return True if the measured bits lie inside the CF sufficient bound.
    """
    return normalized_cf_error(bits_to_fraction(bits), true_order) <= 1


def run_order_recovery_bit_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    rng: np.random.Generator,
    num_phase_bits: int,
    measurement_error_probability: float,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> tuple[list[int], int | None]:
    """
    Run one order-finding trial and return measured bits plus recovered order.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    error_config = error_config_for_probability(measurement_error_probability)

    if algorithm == KITAEV_ALGORITHM:
        result = kitaev_order_attempt(
            N=instance.N,
            a=instance.a,
            num_phase_bits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
            error_config=error_config,
        )
        return result["bits"], result["order"]

    if algorithm == STANDARD_ALGORITHM:
        if standard_backend == STATEVECTOR_STANDARD_BACKEND:
            result = standard_order_attempt(
                N=instance.N,
                a=instance.a,
                num_control_qubits=num_phase_bits,
                rng=rng,
                postprocess_mode="one_shot",
                error_config=error_config,
                standard_backend=standard_backend,
                analytic_window=analytic_window,
            )
            return result["bits"], result["order"]

        analytic_error_config = validate_analytic_error_config(error_config)
        measurement_bitflip_mask = choose_measurement_bitflip_mask(
            num_phase_bits,
            error_config=analytic_error_config,
            rng=rng,
        )
        bits, _, _, _ = sample_standard_phase_bits_from_order(
            order=instance.true_order,
            num_control_qubits=num_phase_bits,
            rng=rng,
            window=analytic_window,
            measurement_bitflip_mask=measurement_bitflip_mask,
        )
        recovered_order = recover_order_with_mode(
            bits,
            instance.a,
            instance.N,
            postprocess_mode="one_shot",
        )
        return bits, recovered_order

    raise ValueError("unknown algorithm")


def simulate_cf_bound_order_recovery_for_instance(
    instance: OrderRecoveryInstance,
    algorithm: str,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    measurement_error_probability: float,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> CFBoundOrderRecoveryResult:
    """
    Estimate CF-bound and recovery probabilities for one configured instance.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    within_bound_trials = 0
    recovered_trials = 0
    recovered_within_bound_trials = 0
    recovered_outside_bound_trials = 0

    for _ in range(trials):
        bits, recovered_order = run_order_recovery_bit_trial(
            instance=instance,
            algorithm=algorithm,
            rng=rng,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
        )
        is_within_bound = within_cf_sufficient_bound(
            bits,
            true_order=instance.true_order,
        )
        is_recovered = recovered_order == instance.true_order

        if is_within_bound:
            within_bound_trials += 1
        if is_recovered:
            recovered_trials += 1
            if is_within_bound:
                recovered_within_bound_trials += 1
            else:
                recovered_outside_bound_trials += 1

    return CFBoundOrderRecoveryResult(
        instance=instance,
        algorithm=algorithm,
        measurement_error_probability=measurement_error_probability,
        trials=trials,
        num_phase_bits=num_phase_bits,
        within_bound_trials=within_bound_trials,
        recovered_trials=recovered_trials,
        recovered_within_bound_trials=recovered_within_bound_trials,
        recovered_outside_bound_trials=recovered_outside_bound_trials,
        standard_backend=standard_backend,
    )


def quantity_display_label(quantity: str) -> str:
    """
    Return the legend label for a plotted CF diagnostic quantity.
    """
    if quantity == A_PROBABILITY:
        return "Pr(A): within CF bound"
    if quantity == B_PROBABILITY:
        return "Pr(B): recovered r"
    if quantity == OUTSIDE_B_JOINT_PROBABILITY:
        return "Pr(B and not A)"
    if quantity == OUTSIDE_B_CONDITIONAL_PROBABILITY:
        return "Pr(B | not A)"
    raise ValueError("unknown quantity")


def quantity_filename_label(quantity: str) -> str:
    """
    Return a filename-safe label for one diagnostic quantity.
    """
    if quantity not in PLOTTED_QUANTITIES:
        raise ValueError("unknown quantity")
    return quantity


def result_quantity_value(
    result: CFBoundOrderRecoveryResult,
    quantity: str,
) -> float:
    """
    Return the y-value for one plotted quantity.
    """
    if quantity == A_PROBABILITY:
        return result.within_bound_probability
    if quantity == B_PROBABILITY:
        return result.recovery_probability
    if quantity == OUTSIDE_B_JOINT_PROBABILITY:
        return result.recovered_outside_bound_probability
    if quantity == OUTSIDE_B_CONDITIONAL_PROBABILITY:
        return result.recovered_given_outside_bound_probability
    raise ValueError("unknown quantity")


def adaptive_probability_ylim(values: list[float]) -> tuple[float, float]:
    """
    Return a padded y-axis range for probability-valued data.
    """
    finite_values = [
        value
        for value in values
        if math.isfinite(value)
    ]
    if not finite_values:
        return 0.0, 1.0

    min_value = max(0.0, min(finite_values))
    max_value = min(1.0, max(finite_values))
    span = max_value - min_value

    if span == 0.0:
        pad = 0.05
    else:
        pad = max(0.02, 0.10 * span)

    lower = max(0.0, min_value - pad)
    upper = min(1.0, max_value + pad)

    min_span = 0.05
    if upper - lower < min_span:
        midpoint = (upper + lower) / 2.0
        lower = max(0.0, midpoint - (min_span / 2.0))
        upper = min(1.0, midpoint + (min_span / 2.0))
        if upper - lower < min_span:
            if lower == 0.0:
                upper = min(1.0, min_span)
            elif upper == 1.0:
                lower = max(0.0, 1.0 - min_span)

    return lower, upper


def apply_adaptive_probability_yaxis(ax, values: list[float]) -> None:
    """
    Apply an adaptive probability y-axis range and tick formatting.
    """
    lower, upper = adaptive_probability_ylim(values)
    locator = MaxNLocator(nbins=6)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(locator)
    decimals = probability_tick_decimals(lower, upper, locator=locator)
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def probability_tick_decimals(
    lower: float,
    upper: float,
    locator: MaxNLocator | None = None,
) -> int:
    """
    Return enough decimal places for distinct probability tick labels.
    """
    if locator is None:
        locator = MaxNLocator(nbins=6)

    ticks = [
        tick
        for tick in locator.tick_values(lower, upper)
        if lower - 1e-12 <= tick <= upper + 1e-12
    ]
    if len(ticks) < 2:
        return 2

    for decimals in range(1, 7):
        labels = [f"{tick:.{decimals}f}" for tick in ticks]
        roundtrip_errors = [
            abs(float(label) - tick)
            for label, tick in zip(labels, ticks)
        ]
        if len(set(labels)) == len(labels) and max(roundtrip_errors) < 1e-10:
            return decimals

    return 6


def measurement_probability_filename_segment(probability: float) -> str:
    """
    Return a filename segment for one measurement bit-flip probability.
    """
    [probability] = validate_error_probabilities([probability])
    token = format_measurement_bitflip_probability(probability).replace("-", "m")
    return f"p_{token}_"


def figure_output_path_for_probability(
    output_path: Path | None,
    probability: float,
    all_probabilities: list[float],
    algorithms: list[str],
    standard_backend: str,
    instances_file: Path,
    num_phase_bits: int,
) -> Path:
    """
    Return the figure path for one probability-specific plot.
    """
    probability_segment = measurement_probability_filename_segment(probability).rstrip("_")

    if output_path is not None:
        if len(all_probabilities) == 1:
            return output_path
        return output_path.with_name(
            f"{output_path.stem}_{probability_segment}{output_path.suffix}"
        )

    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"{instances_file_filename_segment(instances_file)}"
    )
    return (
        Path(__file__).resolve().parent
        / "figures"
        / (
            "cf_bound_order_recovery_vs_order_"
            f"{algorithm_prefix}{probability_segment}_t{num_phase_bits}.png"
        )
    )


def quantity_overlay_output_path(
    output_path: Path | None,
    quantity: str,
    algorithms: list[str],
    standard_backend: str,
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
) -> Path:
    """
    Return the figure path for one quantity-specific probability overlay.
    """
    quantity_label = quantity_filename_label(quantity)

    if output_path is not None:
        return output_path.with_name(
            f"{output_path.stem}_overlay_{quantity_label}{output_path.suffix}"
        )

    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"{instances_file_filename_segment(instances_file)}"
    )
    probability_segment = error_probabilities_filename_segment(
        measurement_error_probabilities
    )
    return (
        Path(__file__).resolve().parent
        / "figures"
        / (
            "cf_bound_order_recovery_"
            f"{quantity_label}_overlay_vs_order_"
            f"{algorithm_prefix}{probability_segment}t{num_phase_bits}.png"
        )
    )


def write_data_summary(
    output_path: Path,
    results: list[CFBoundOrderRecoveryResult],
    algorithms: list[str],
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
    Write run metadata and per-order CF-bound diagnostic data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    measurement_error_probabilities = validate_error_probabilities(
        measurement_error_probabilities
    )
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )
    seed_label = "random" if seed is None else str(seed)
    min_order_label = "none" if min_order is None else str(min_order)
    max_order_label = "none" if max_order is None else str(max_order)
    probability_label = ", ".join(
        error_probability_display_label(probability)
        for probability in measurement_error_probabilities
    )

    lines = [
        "CF-Bound Order-Recovery Diagnostic Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"Instances file = {instances_file}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Instances used = {len({result.instance for result in results})}",
        f"Trials per instance and algorithm = {trials_per_instance}",
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
        "  A = normalized_error <= 1",
        "  normalized_error = 2*r^2*min_s distance_mod_1(y/2^t, s/r)",
        "  B = recovered_order == true r after one-shot continued fractions",
        "  Pr(B) is the usual one-shot order-recovery success probability",
        "",
        "Per-order data:",
        "  algorithm | p | true r | N | a | Pr(A) | Pr(not A) | Pr(B) | Pr(B and not A) | Pr(B | not A) | Pr(B | A) | A count | not A count | B count | B and not A count | trials",
    ]

    for result in sorted(
        results,
        key=lambda item: (
            item.algorithm,
            item.measurement_error_probability,
            item.instance.true_order,
        ),
    ):
        instance = result.instance
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {format_measurement_bitflip_probability(result.measurement_error_probability):>5} |"
            f" {instance.true_order:>6} |"
            f" {instance.N:>5} |"
            f" {instance.a:>5} |"
            f" {result.within_bound_probability:>6.3f} |"
            f" {result.outside_bound_probability:>9.3f} |"
            f" {result.recovery_probability:>6.3f} |"
            f" {result.recovered_outside_bound_probability:>17.3f} |"
            f" {result.recovered_given_outside_bound_probability:>15.3f} |"
            f" {result.recovered_given_within_bound_probability:>9.3f} |"
            f" {result.within_bound_trials:>7} |"
            f" {result.outside_bound_trials:>11} |"
            f" {result.recovered_trials:>7} |"
            f" {result.recovered_outside_bound_trials:>17} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cf_bound_order_recovery_for_probability(
    output_path: Path,
    results: list[CFBoundOrderRecoveryResult],
    algorithms: list[str],
    num_phase_bits: int,
    measurement_error_probability: float,
    trials_per_instance: int,
    log_x: bool = False,
) -> None:
    """
    Create one CF-bound diagnostic figure for one measurement probability.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    [measurement_error_probability] = validate_error_probabilities([
        measurement_error_probability
    ])

    fig, ax = plt.subplots(figsize=(11, 6))
    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }
    linestyles = {
        KITAEV_ALGORITHM: "-",
        STANDARD_ALGORITHM: "--",
    }
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    quantity_colors = {
        quantity: color_cycle[index % len(color_cycle)]
        for index, quantity in enumerate(PLOTTED_QUANTITIES)
    }
    plotted_values = []

    for quantity in PLOTTED_QUANTITIES:
        for algorithm in algorithms:
            quantity_results = sorted(
                [
                    result
                    for result in results
                    if result.algorithm == algorithm
                    and result.measurement_error_probability
                    == measurement_error_probability
                ],
                key=lambda result: result.instance.true_order,
            )
            y_values = [
                result_quantity_value(result, quantity)
                for result in quantity_results
            ]
            plotted_values.extend(y_values)
            ax.plot(
                [result.instance.true_order for result in quantity_results],
                y_values,
                marker=markers[algorithm],
                linestyle=linestyles[algorithm],
                linewidth=1.8,
                markersize=5.5,
                color=algorithm_color_for_base(
                    quantity_colors[quantity],
                    algorithm,
                ),
                alpha=0.9,
            )

    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel("Empirical probability")
    ax.set_title(
        "Continued-Fraction Bound Diagnostics vs True Order\n"
        f"{algorithms_display_label(algorithms)}, "
        f"{error_probability_display_label(measurement_error_probability)}, "
        f"t = {num_phase_bits}"
    )
    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    apply_adaptive_probability_yaxis(ax, plotted_values)
    ax.grid(True, alpha=0.25)

    algorithm_handles = [
        Line2D(
            [0],
            [0],
            color=algorithm_color_for_base("#333333", algorithm),
            marker=markers[algorithm],
            linestyle=linestyles[algorithm],
            linewidth=1.8,
            markersize=6,
            label=algorithm_display_label(algorithm),
        )
        for algorithm in algorithms
    ]
    quantity_handles = [
        Line2D(
            [0],
            [0],
            color=quantity_colors[quantity],
            linestyle="-",
            linewidth=2.0,
            label=quantity_display_label(quantity),
        )
        for quantity in PLOTTED_QUANTITIES
    ]

    fig.tight_layout(rect=[0.0, 0.0, 0.68, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=algorithm_handles,
        measurement_handles=quantity_handles,
        trials_per_instance=trials_per_instance,
        measurement_title="Quantity",
        extra_text_sections=[
            (
                "Measurement p",
                [error_probability_display_label(measurement_error_probability)],
            )
        ],
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cf_bound_quantity_overlay(
    output_path: Path,
    results: list[CFBoundOrderRecoveryResult],
    algorithms: list[str],
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    quantity: str,
    trials_per_instance: int,
    log_x: bool = False,
) -> None:
    """
    Plot one diagnostic quantity with measurement probabilities overlaid.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    measurement_error_probabilities = validate_error_probabilities(
        measurement_error_probabilities
    )
    quantity_filename_label(quantity)

    fig, ax = plt.subplots(figsize=(11, 6))
    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }
    linestyles = {
        KITAEV_ALGORITHM: "-",
        STANDARD_ALGORITHM: "--",
    }
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    probability_colors = {
        probability: color_cycle[index % len(color_cycle)]
        for index, probability in enumerate(measurement_error_probabilities)
    }
    plotted_values = []

    for probability in measurement_error_probabilities:
        for algorithm in algorithms:
            quantity_results = sorted(
                [
                    result
                    for result in results
                    if result.algorithm == algorithm
                    and result.measurement_error_probability == probability
                ],
                key=lambda result: result.instance.true_order,
            )
            y_values = [
                result_quantity_value(result, quantity)
                for result in quantity_results
            ]
            plotted_values.extend(y_values)
            ax.plot(
                [result.instance.true_order for result in quantity_results],
                y_values,
                marker=markers[algorithm],
                linestyle=linestyles[algorithm],
                linewidth=1.8,
                markersize=5.5,
                color=algorithm_color_for_base(
                    probability_colors[probability],
                    algorithm,
                ),
                alpha=0.9,
            )

    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel("Empirical probability")
    ax.set_title(
        f"{quantity_display_label(quantity)} vs True Order\n"
        f"{algorithms_display_label(algorithms)}, multiple p values, "
        f"t = {num_phase_bits}"
    )
    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    apply_adaptive_probability_yaxis(ax, plotted_values)
    ax.grid(True, alpha=0.25)

    algorithm_handles = [
        Line2D(
            [0],
            [0],
            color=algorithm_color_for_base("#333333", algorithm),
            marker=markers[algorithm],
            linestyle=linestyles[algorithm],
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
        for probability in measurement_error_probabilities
    ]

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=algorithm_handles,
        measurement_handles=probability_handles,
        trials_per_instance=trials_per_instance,
        extra_text_sections=[
            ("Quantity", [quantity_display_label(quantity)])
        ],
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for CF-bound order-recovery diagnostics.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot continued-fraction sufficient-bound survival and true-order "
            "recovery diagnostics versus true order r."
        )
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=DEFAULT_INSTANCES_FILE,
        help="JSON file containing configured representative (N, a) pairs",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        help="One or more phase-estimation algorithms to compare",
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
            "Generate one plot per independent per-bit measurement flip "
            "probability. This is mutually exclusive with --p."
        ),
    )
    parser.add_argument(
        "--overlay-error-probabilities",
        action="store_true",
        help=(
            "Also generate one figure per diagnostic quantity with all "
            "requested measurement error probabilities overlaid."
        ),
    )
    parser.add_argument(
        "--trials-per-instance",
        type=int,
        default=100,
        help="Number of one-shot order-finding trials per instance and algorithm",
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
            "Backend used for standard Shor curves. The analytic backend is "
            "recommended for large t and measurement-error diagnostics."
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
        help=(
            "Optional custom output path. With multiple probabilities, the "
            "probability token is appended to this stem."
        ),
    )
    return parser


def main() -> None:
    """
    Run the CF-bound order-recovery diagnostic script.
    """
    start_time = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()
    algorithms = normalize_algorithms(args.algorithm)

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
            measurement_error_probabilities = validate_error_probabilities([
                raw_probability
            ])
        else:
            measurement_error_probabilities = validate_error_probabilities(
                args.error_probabilities
            )
        instances = load_order_recovery_instances_from_file(
            instances_file=args.instances_file,
            num_phase_bits=args.bits,
            min_order=args.min_order,
            max_order=args.max_order,
        )
        validate_statevector_budget(
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            num_phase_bits=args.bits,
            max_N=max(instance.N for instance in instances),
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
    results: list[CFBoundOrderRecoveryResult] = []
    for measurement_error_probability in measurement_error_probabilities:
        for algorithm in algorithms:
            for instance in instances:
                instance_rng = np.random.default_rng(
                    int(master_rng.integers(0, 2 ** 63 - 1))
                )
                results.append(
                    simulate_cf_bound_order_recovery_for_instance(
                        instance=instance,
                        algorithm=algorithm,
                        trials=args.trials_per_instance,
                        rng=instance_rng,
                        num_phase_bits=args.bits,
                        measurement_error_probability=measurement_error_probability,
                        standard_backend=args.standard_backend,
                        analytic_window=args.analytic_window,
                    )
                )

    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, args.standard_backend)}"
        f"{instances_file_filename_segment(args.instances_file)}"
    )
    probability_segment = error_probabilities_filename_segment(
        measurement_error_probabilities
    )
    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / (
            "cf_bound_order_recovery_vs_order_summary_"
            f"{algorithm_prefix}{probability_segment}t{args.bits}.txt"
        )
    )

    figure_output_paths = []
    for probability in measurement_error_probabilities:
        figure_output_path = figure_output_path_for_probability(
            output_path=args.output,
            probability=probability,
            all_probabilities=measurement_error_probabilities,
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            instances_file=args.instances_file,
            num_phase_bits=args.bits,
        )
        probability_results = [
            result
            for result in results
            if result.measurement_error_probability == probability
        ]
        plot_cf_bound_order_recovery_for_probability(
            output_path=figure_output_path,
            results=probability_results,
            algorithms=algorithms,
            num_phase_bits=args.bits,
            measurement_error_probability=probability,
            trials_per_instance=args.trials_per_instance,
            log_x=args.log_x,
        )
        figure_output_paths.append(figure_output_path)

    if args.overlay_error_probabilities:
        for quantity in PLOTTED_QUANTITIES:
            figure_output_path = quantity_overlay_output_path(
                output_path=args.output,
                quantity=quantity,
                algorithms=algorithms,
                standard_backend=args.standard_backend,
                instances_file=args.instances_file,
                num_phase_bits=args.bits,
                measurement_error_probabilities=measurement_error_probabilities,
            )
            plot_cf_bound_quantity_overlay(
                output_path=figure_output_path,
                results=results,
                algorithms=algorithms,
                num_phase_bits=args.bits,
                measurement_error_probabilities=measurement_error_probabilities,
                quantity=quantity,
                trials_per_instance=args.trials_per_instance,
                log_x=args.log_x,
            )
            figure_output_paths.append(figure_output_path)

    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
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
        error_probability_display_label(probability)
        for probability in measurement_error_probabilities
    )
    for figure_output_path in figure_output_paths:
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
    print(f"Trials per instance and algorithm: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
