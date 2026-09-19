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
import numpy as np
from matplotlib.ticker import MaxNLocator, ScalarFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.measurement_error.phase_distance_vs_order_common import (
    measurement_probability_display_label,
    minimum_single_bit_flip_candidate,
    run_kitaev_phase_distance_trial,
    run_standard_phase_distance_trial,
    validate_measurement_error_probabilities,
)
from graphs.measurement_error.plot_cf_bound_order_recovery_vs_order import (
    nearest_order_phase_distance,
)
from graphs.measurement_error.plot_eigenphase_peak_diagnostics_vs_order import (
    DEFAULT_INSTANCES_FILE,
    DEFAULT_NUM_PHASE_BITS,
    measurement_probabilities_filename_segment,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    ALGORITHM_CHOICES,
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
    add_combined_figure_legend,
    algorithm_display_label,
    algorithm_filename_prefix,
    algorithms_display_label,
    analytic_window_diagnostics_console_label,
    analytic_window_diagnostics_summary_lines,
    analytic_window_mass_diagnostics_for_instances,
    instances_file_filename_segment,
    load_order_recovery_instances_from_file,
    normalize_algorithms,
    standard_statevector_allows_subdefault_phase_bits,
    standard_backend_filename_segment,
    validate_statevector_budget,
)
from shor_common.classical_post import bits_to_fraction
from shor_common.errors import format_measurement_bitflip_probability
from shor_standard.analytic_sampling import QPEWindowMassDiagnostics
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)


CLEAN_PHASE = "clean"
NOISY_PHASE = "noisy"
CORRECTED_PHASE = "corrected"
PHASE_CHOICES = (CLEAN_PHASE, NOISY_PHASE)


@dataclass(frozen=True)
class NearestPeakDistanceTrial:
    clean_distance: Fraction
    noisy_distance: Fraction
    clean_peak_numerator: int
    noisy_peak_numerator: int
    corrected_distance: Fraction | None = None
    corrected_peak_numerator: int | None = None

    @property
    def same_peak(self) -> bool:
        return self.clean_peak_numerator == self.noisy_peak_numerator

    @property
    def corrected_same_peak(self) -> bool:
        return self.clean_peak_numerator == self.corrected_peak_numerator


@dataclass(frozen=True)
class NearestPeakDistanceResult:
    instance: OrderRecoveryInstance
    algorithm: str
    measurement_error_probability: float
    trials: int
    num_phase_bits: int
    total_clean_distance: Fraction
    total_noisy_distance: Fraction
    same_peak_trials: int
    standard_backend: str = ANALYTIC_STANDARD_BACKEND
    total_corrected_distance: Fraction | None = None
    corrected_same_peak_trials: int | None = None

    @property
    def average_clean_distance(self) -> float:
        return float(self.total_clean_distance / self.trials)

    @property
    def average_noisy_distance(self) -> float:
        return float(self.total_noisy_distance / self.trials)

    @property
    def average_corrected_distance(self) -> float:
        if self.total_corrected_distance is None:
            return math.nan
        return float(self.total_corrected_distance / self.trials)

    @property
    def same_peak_probability(self) -> float:
        return self.same_peak_trials / self.trials

    @property
    def corrected_same_peak_probability(self) -> float:
        if self.corrected_same_peak_trials is None:
            return math.nan
        return self.corrected_same_peak_trials / self.trials

    @property
    def has_classical_error_correction(self) -> bool:
        return self.total_corrected_distance is not None


def nearest_peak_distance(bits: list[int], true_order: int) -> tuple[Fraction, int]:
    """
    Return distance from a phase bit string to its nearest order-lattice peak.
    """
    return nearest_order_phase_distance(
        phase_fraction=bits_to_fraction(bits),
        true_order=true_order,
    )


def cf_sufficient_bound_distance(true_order: int) -> Fraction:
    """
    Return the continued-fraction sufficient distance bound 1/(2r^2).
    """
    if true_order < 1:
        raise ValueError("true_order must be >= 1")
    return Fraction(1, 2 * true_order * true_order)


def inverse_order_reference_distance(true_order: int) -> Fraction:
    """
    Return the half-inverse-order reference distance 1/(2r).
    """
    if true_order < 1:
        raise ValueError("true_order must be >= 1")
    return Fraction(1, 2 * true_order)


def phase_choices(include_classical_error_correction: bool = False) -> tuple[str, ...]:
    """
    Return plotted phase-distance series.
    """
    if include_classical_error_correction:
        return (CORRECTED_PHASE,)
    return PHASE_CHOICES


def classical_error_correction_filename_segment(
    include_classical_error_correction: bool,
) -> str:
    """
    Return a filename segment for optional corrected-only plots.
    """
    return "_classical_ec" if include_classical_error_correction else ""


def inverse_order_reference_filename_segment(
    include_inverse_order_reference: bool,
) -> str:
    """
    Return a filename segment for optional half-inverse-order reference plots.
    """
    return "_with_half_inverse_order" if include_inverse_order_reference else ""


def minimum_single_bit_flip_nearest_peak_distance(
    bits: list[int],
    true_order: int,
) -> tuple[Fraction, int, list[int]]:
    """
    Minimize nearest-peak distance over the bit string and all one-bit flips.

    Each candidate is measured against its own nearest order-lattice point.
    The original bit string wins ties, so the diagnostic only updates when a
    one-bit candidate has a strictly smaller nearest-peak distance.
    """
    best_bits, _ = minimum_single_bit_flip_candidate(
        bits,
        score_candidate=lambda candidate: nearest_peak_distance(
            candidate,
            true_order=true_order,
        )[0],
    )
    best_distance, best_peak = nearest_peak_distance(
        best_bits,
        true_order=true_order,
    )
    return best_distance, best_peak, best_bits


def run_nearest_peak_distance_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    include_classical_error_correction: bool = False,
) -> NearestPeakDistanceTrial:
    """
    Run one paired clean/noisy PE trial and measure each phase to its own peak.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    if algorithm == KITAEV_ALGORITHM:
        trial = run_kitaev_phase_distance_trial(
            instance=instance,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
        )
    elif algorithm == STANDARD_ALGORITHM:
        trial = run_standard_phase_distance_trial(
            instance=instance,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
        )
    else:
        raise ValueError("unknown algorithm")

    clean_distance, clean_peak = nearest_peak_distance(
        trial.clean_bits,
        true_order=instance.true_order,
    )
    noisy_distance, noisy_peak = nearest_peak_distance(
        trial.noisy_bits,
        true_order=instance.true_order,
    )
    corrected_distance = None
    corrected_peak = None
    if include_classical_error_correction:
        corrected_distance, corrected_peak, _ = (
            minimum_single_bit_flip_nearest_peak_distance(
                trial.noisy_bits,
                true_order=instance.true_order,
            )
        )

    return NearestPeakDistanceTrial(
        clean_distance=clean_distance,
        noisy_distance=noisy_distance,
        clean_peak_numerator=clean_peak,
        noisy_peak_numerator=noisy_peak,
        corrected_distance=corrected_distance,
        corrected_peak_numerator=corrected_peak,
    )


def simulate_nearest_peak_distance_for_instance(
    instance: OrderRecoveryInstance,
    algorithm: str,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    measurement_error_probability: float,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    include_classical_error_correction: bool = False,
) -> NearestPeakDistanceResult:
    """
    Estimate average clean/noisy nearest-peak decimal distances for one instance.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    total_clean_distance = Fraction(0, 1)
    total_noisy_distance = Fraction(0, 1)
    total_corrected_distance = Fraction(0, 1)
    same_peak_trials = 0
    corrected_same_peak_trials = 0

    for _ in range(trials):
        trial = run_nearest_peak_distance_trial(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
            include_classical_error_correction=include_classical_error_correction,
        )
        total_clean_distance += trial.clean_distance
        total_noisy_distance += trial.noisy_distance
        if include_classical_error_correction:
            total_corrected_distance += trial.corrected_distance
            if trial.corrected_same_peak:
                corrected_same_peak_trials += 1
        if trial.same_peak:
            same_peak_trials += 1

    return NearestPeakDistanceResult(
        instance=instance,
        algorithm=algorithm,
        measurement_error_probability=measurement_error_probability,
        trials=trials,
        num_phase_bits=num_phase_bits,
        total_clean_distance=total_clean_distance,
        total_noisy_distance=total_noisy_distance,
        same_peak_trials=same_peak_trials,
        standard_backend=standard_backend,
        total_corrected_distance=(
            total_corrected_distance
            if include_classical_error_correction
            else None
        ),
        corrected_same_peak_trials=(
            corrected_same_peak_trials
            if include_classical_error_correction
            else None
        ),
    )


def result_phase_distance(result: NearestPeakDistanceResult, phase: str) -> float:
    """
    Return the plotted clean or noisy average distance for one result.
    """
    if phase == CLEAN_PHASE:
        return result.average_clean_distance
    if phase == NOISY_PHASE:
        return result.average_noisy_distance
    if phase == CORRECTED_PHASE:
        return result.average_corrected_distance
    raise ValueError("unknown phase")


def phase_display_label(phase: str) -> str:
    """
    Return a legend label for one phase-distance curve.
    """
    if phase == CLEAN_PHASE:
        return "clean: d(b, s/r)"
    if phase == NOISY_PHASE:
        return "noisy: d(tilde b, s'/r)"
    if phase == CORRECTED_PHASE:
        return "corrected min: d(b_ec, s_ec/r)"
    raise ValueError("unknown phase")


def measurement_probability_filename_segment(probability: float) -> str:
    """
    Return a filename segment for one measurement bit-flip probability.
    """
    [probability] = validate_measurement_error_probabilities([probability])
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
    include_classical_error_correction: bool = False,
    include_inverse_order_reference: bool = False,
) -> Path:
    """
    Return the output figure path for one probability-specific plot.
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
            "nearest_peak_decimal_distance_vs_order_"
            f"{algorithm_prefix}{probability_segment}_t{num_phase_bits}"
            f"{classical_error_correction_filename_segment(include_classical_error_correction)}"
            f"{inverse_order_reference_filename_segment(include_inverse_order_reference)}.png"
        )
    )


def summary_output_path(
    algorithms: list[str],
    standard_backend: str,
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    include_classical_error_correction: bool = False,
    include_inverse_order_reference: bool = False,
) -> Path:
    """
    Return the data-summary path for nearest-peak distance runs.
    """
    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"{instances_file_filename_segment(instances_file)}"
    )
    probability_segment = measurement_probabilities_filename_segment(
        measurement_error_probabilities
    )
    return (
        Path(__file__).resolve().parent
        / "data"
        / (
            "nearest_peak_decimal_distance_vs_order_summary_"
            f"{algorithm_prefix}{probability_segment}t{num_phase_bits}"
            f"{classical_error_correction_filename_segment(include_classical_error_correction)}"
            f"{inverse_order_reference_filename_segment(include_inverse_order_reference)}.txt"
        )
    )


def adaptive_distance_ylim(values: list[float]) -> tuple[float, float]:
    """
    Return a padded y-axis range for nearest-peak decimal distances.
    """
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, 0.05

    max_value = max(0.0, max(finite_values))
    if max_value == 0.0:
        return 0.0, 0.05

    return 0.0, min(0.5, max_value * 1.12)


def apply_scientific_distance_y_axis(ax: plt.Axes) -> None:
    """
    Format distance y-axis ticks with an adaptive scientific scale.
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)


def truncated_cf_bound_curve(
    instances: list[OrderRecoveryInstance],
    y_upper: float,
) -> tuple[list[int], list[float]]:
    """
    Return the visible tail of 1/(2r^2) for a distance-derived y-axis.
    """
    if y_upper <= 0.0:
        return [], []

    instances_by_order = {
        instance.true_order: instance
        for instance in instances
    }
    x_values = []
    y_values = []
    for true_order in sorted(instances_by_order):
        bound = float(cf_sufficient_bound_distance(true_order))
        if bound <= y_upper:
            x_values.append(true_order)
            y_values.append(bound)
    return x_values, y_values


def truncated_inverse_order_curve(
    instances: list[OrderRecoveryInstance],
    y_upper: float,
) -> tuple[list[int], list[float]]:
    """
    Return the visible tail of 1/(2r) for a distance-derived y-axis.
    """
    if y_upper <= 0.0:
        return [], []

    instances_by_order = {
        instance.true_order: instance
        for instance in instances
    }
    x_values = []
    y_values = []
    for true_order in sorted(instances_by_order):
        reference = float(inverse_order_reference_distance(true_order))
        if reference <= y_upper:
            x_values.append(true_order)
            y_values.append(reference)
    return x_values, y_values


def plot_nearest_peak_distance_for_probability(
    output_path: Path,
    results: list[NearestPeakDistanceResult],
    algorithms: list[str],
    num_phase_bits: int,
    measurement_error_probability: float,
    trials_per_instance: int,
    log_x: bool = False,
    include_classical_error_correction: bool = False,
    include_inverse_order_reference: bool = False,
) -> None:
    """
    Plot nearest-peak decimal distance for each algorithm.

    Without classical correction, the plot contains clean and noisy curves.
    With classical correction, it contains only the corrected-min curves.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)

    algorithm_colors = {
        KITAEV_ALGORITHM: "#1F6F78",
        STANDARD_ALGORITHM: "#A23B2A",
    }
    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }
    linestyles = {
        CLEAN_PHASE: "-",
        NOISY_PHASE: "--",
        CORRECTED_PHASE: "-.",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    distance_values = []
    curve_handles = []

    for algorithm in algorithms:
        for phase in phase_choices(include_classical_error_correction):
            phase_results = sorted(
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
                result_phase_distance(result, phase)
                for result in phase_results
            ]
            distance_values.extend(y_values)
            (line,) = ax.plot(
                [result.instance.true_order for result in phase_results],
                y_values,
                marker=markers[algorithm],
                linestyle=linestyles[phase],
                linewidth=1.8,
                markersize=5.5,
                color=algorithm_colors[algorithm],
                alpha=0.92,
                label=(
                    f"{algorithm_display_label(algorithm)} "
                    f"{phase_display_label(phase)}"
                ),
            )
            curve_handles.append(line)

    lower, upper = adaptive_distance_ylim(distance_values)
    bound_instances = [
        result.instance
        for result in results
        if result.measurement_error_probability == measurement_error_probability
    ]
    bound_x_values, bound_y_values = truncated_cf_bound_curve(
        bound_instances,
        y_upper=upper,
    )
    if bound_x_values:
        (bound_line,) = ax.plot(
            bound_x_values,
            bound_y_values,
            linestyle="--",
            linewidth=2.0,
            color="#303030",
            alpha=0.88,
            label="CF bound 1/(2r^2)",
        )
        curve_handles.append(bound_line)

    if include_inverse_order_reference:
        inverse_x_values, inverse_y_values = truncated_inverse_order_curve(
            bound_instances,
            y_upper=upper,
        )
        if inverse_x_values:
            (inverse_line,) = ax.plot(
                inverse_x_values,
                inverse_y_values,
                linestyle="-",
                linewidth=2.0,
                color="#6B5B2A",
                alpha=0.88,
                label="Reference 1/(2r)",
            )
            curve_handles.append(inverse_line)

    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel("Average circular distance to nearest eigenphase")
    title_prefix = (
        "Corrected-Min Nearest-Eigenphase Decimal Distance vs True Order"
        if include_classical_error_correction
        else "Nearest-Eigenphase Decimal Distance vs True Order"
    )
    ax.set_title(
        f"{title_prefix}\n"
        f"{algorithms_display_label(algorithms)}, "
        f"{measurement_probability_display_label(measurement_error_probability)}, "
        f"t = {num_phase_bits}"
    )
    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(lower, upper)
    apply_scientific_distance_y_axis(ax)
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=curve_handles,
        measurement_handles=[],
        trials_per_instance=trials_per_instance,
        algorithm_title="Curve",
        measurement_title="Measurement p",
        measurement_text_labels=[
            measurement_probability_display_label(measurement_error_probability)
        ],
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[NearestPeakDistanceResult],
    algorithms: list[str],
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    trials_per_instance: int,
    min_order: int | None,
    max_order: int | None,
    seed: int | None,
    analytic_window_diagnostics: list[QPEWindowMassDiagnostics] | None = None,
    include_classical_error_correction: bool = False,
    include_inverse_order_reference: bool = False,
) -> None:
    """
    Write run metadata and per-order nearest-peak distance data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    measurement_error_probabilities = validate_measurement_error_probabilities(
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
        measurement_probability_display_label(probability)
        for probability in measurement_error_probabilities
    )

    lines = [
        "Nearest-Eigenphase Decimal Distance Summary",
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
        f"Classical single-bit minimum-distance plot = {'corrected-only' if include_classical_error_correction else 'disabled'}",
        f"Half-inverse-order reference plot = {'enabled' if include_inverse_order_reference else 'disabled'}",
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
        "  b = clean phase-estimation bit string",
        "  tilde b = noisy phase-estimation bit string",
        "  s/r = nearest exact order eigenphase to b",
        "  s'/r = nearest exact order eigenphase to tilde b",
        *(
            [
                "  b_ec = bit string among tilde b and all one-bit flips minimizing distance to its own nearest eigenphase",
                "  s_ec/r = nearest exact order eigenphase to b_ec",
            ]
            if include_classical_error_correction
            else []
        ),
        "  distance = circular distance on [0, 1)",
        "  CF sufficient bound = 1/(2r^2)",
        *(
            ["  half-inverse-order reference = 1/(2r)"]
            if include_inverse_order_reference
            else []
        ),
        "",
        "Per-order data:",
        (
            "  algorithm | p | true r | N | a | CF bound | "
            f"{'1/(2r) | ' if include_inverse_order_reference else ''}"
            "avg d(b,s/r) | "
            "avg d(tilde b,s'/r) | avg d(b_ec,s_ec/r) | Pr(s'=s) | "
            "Pr(s_ec=s) | clean distance total | noisy distance total | "
            "corrected distance total | trials"
            if include_classical_error_correction
            else
            "  algorithm | p | true r | N | a | CF bound | "
            f"{'1/(2r) | ' if include_inverse_order_reference else ''}"
            "avg d(b,s/r) | "
            "avg d(tilde b,s'/r) | Pr(s'=s) | clean distance total | "
            "noisy distance total | trials"
        ),
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
        inverse_order_column = (
            f" {float(inverse_order_reference_distance(instance.true_order)):>8.6f} |"
            if include_inverse_order_reference
            else ""
        )
        prefix = (
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {format_measurement_bitflip_probability(result.measurement_error_probability):>5} |"
            f" {instance.true_order:>6} |"
            f" {instance.N:>5} |"
            f" {instance.a:>5} |"
            f" {float(cf_sufficient_bound_distance(instance.true_order)):>8.6f} |"
            f"{inverse_order_column}"
            f" {result.average_clean_distance:>14.6f} |"
            f" {result.average_noisy_distance:>22.6f} |"
        )
        if include_classical_error_correction:
            lines.append(
                prefix
                + f" {result.average_corrected_distance:>22.6f} |"
                f" {result.same_peak_probability:>8.3f} |"
                f" {result.corrected_same_peak_probability:>10.3f} |"
                f" {float(result.total_clean_distance):>20.6f} |"
                f" {float(result.total_noisy_distance):>20.6f} |"
                f" {float(result.total_corrected_distance):>24.6f} |"
                f" {result.trials:>6}"
            )
        else:
            lines.append(
                prefix
                + f" {result.same_peak_probability:>8.3f} |"
                f" {float(result.total_clean_distance):>20.6f} |"
                f" {float(result.total_noisy_distance):>20.6f} |"
                f" {result.trials:>6}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for nearest-peak decimal-distance diagnostics.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot average circular decimal distance from phase-estimation "
            "outputs to their nearest exact order eigenphase."
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
        "--trials-per-instance",
        type=int,
        default=100,
        help="Number of paired clean/noisy trials per instance and algorithm",
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
        "--include-half-inverse-order-reference",
        "--include-inverse-order-reference",
        dest="include_inverse_order_reference",
        action="store_true",
        help="Plot the optional solid half-inverse-order reference curve 1/(2r)",
    )
    parser.add_argument(
        "--classical-error-correction",
        action="store_true",
        help=(
            "Plot only the single-bit-search minimum nearest-peak distance "
            "diagnostic: choose the noisy bit string or one-bit flip with "
            "the smallest distance to its own nearest eigenphase."
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
        help=(
            "Optional custom output path. With multiple probabilities, the "
            "probability token is appended to this stem."
        ),
    )
    return parser


def main() -> None:
    """
    Run the nearest-peak decimal-distance plotting script.
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
                algorithms,
                args.standard_backend,
            ),
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
    results: list[NearestPeakDistanceResult] = []
    for measurement_error_probability in measurement_error_probabilities:
        for algorithm in algorithms:
            for instance in instances:
                instance_rng = np.random.default_rng(
                    int(master_rng.integers(0, 2 ** 63 - 1))
                )
                results.append(
                    simulate_nearest_peak_distance_for_instance(
                        instance=instance,
                        algorithm=algorithm,
                        trials=args.trials_per_instance,
                        rng=instance_rng,
                        num_phase_bits=args.bits,
                        measurement_error_probability=measurement_error_probability,
                        standard_backend=args.standard_backend,
                        analytic_window=args.analytic_window,
                        include_classical_error_correction=args.classical_error_correction,
                    )
                )

    data_output_path = summary_output_path(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        measurement_error_probabilities=measurement_error_probabilities,
        include_classical_error_correction=args.classical_error_correction,
        include_inverse_order_reference=args.include_inverse_order_reference,
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
            include_classical_error_correction=args.classical_error_correction,
            include_inverse_order_reference=args.include_inverse_order_reference,
        )
        probability_results = [
            result
            for result in results
            if result.measurement_error_probability == probability
        ]
        plot_nearest_peak_distance_for_probability(
            output_path=figure_output_path,
            results=probability_results,
            algorithms=algorithms,
            num_phase_bits=args.bits,
            measurement_error_probability=probability,
            trials_per_instance=args.trials_per_instance,
            log_x=args.log_x,
            include_classical_error_correction=args.classical_error_correction,
            include_inverse_order_reference=args.include_inverse_order_reference,
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
        include_classical_error_correction=args.classical_error_correction,
        include_inverse_order_reference=args.include_inverse_order_reference,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    probability_label = ", ".join(
        measurement_probability_display_label(probability)
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
    print(
        "Classical single-bit minimum-distance plot: "
        f"{'corrected-only' if args.classical_error_correction else 'disabled'}"
    )
    print(
        "Half-inverse-order reference 1/(2r): "
        f"{'enabled' if args.include_inverse_order_reference else 'disabled'}"
    )
    print(f"Trials per instance and algorithm: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
