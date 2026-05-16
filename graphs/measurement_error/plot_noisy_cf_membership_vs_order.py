from __future__ import annotations

import argparse
import math
import os
import sys
import time
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


from graphs.measurement_error.phase_distance_vs_order_common import (
    DEFAULT_INSTANCES_FILE,
    DEFAULT_NUM_PHASE_BITS,
    SINGLE_BITFLIP_SAFE_TAIL_RANGE,
    SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES,
    SingleBitflipRangeSpec,
    normalize_single_measurement_bitflip_ranges,
    order_resolution_bits,
    single_measurement_bitflip_bit_numbers,
    validate_measurement_error_probability,
)
from graphs.measurement_error.plot_noisy_cf_complement_fill import (
    measurement_error_model_display_label,
    measurement_error_model_filename_segment,
)
from graphs.measurement_error.plot_noisy_cf_membership_bar import (
    NoisyCFMembershipResult,
    simulate_noisy_cf_membership,
)
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
    instances_file_filename_segment,
    load_order_recovery_instances_from_file,
    normalize_algorithms,
    standard_backend_filename_segment,
    validate_statevector_budget,
)
from shor_standard.analytic_sampling import QPEWindowMassDiagnostics
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)


MP_REFERENCE_CURVE = "mp"
TWO_MP_REFERENCE_CURVE = "2mp"
REFERENCE_CURVE_CHOICES = (MP_REFERENCE_CURVE, TWO_MP_REFERENCE_CURVE)


def normalize_reference_curves(reference_curves: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """
    Validate and de-duplicate probability reference curves in display order.
    """
    if reference_curves is None:
        return ()
    normalized = []
    for curve in reference_curves:
        if curve not in REFERENCE_CURVE_CHOICES:
            raise ValueError("unknown reference curve")
        if curve not in normalized:
            normalized.append(curve)
    return tuple(normalized)


def single_bitflip_bits_for_instance(
    instance: OrderRecoveryInstance,
    num_phase_bits: int,
    single_measurement_bitflip_range: SingleBitflipRangeSpec,
) -> tuple[int, ...] | None:
    """
    Return forced one-flip bit numbers for an instance, or None for p-mode.
    """
    range_names = normalize_single_measurement_bitflip_ranges(
        single_measurement_bitflip_range
    )
    if range_names is None:
        return None
    return single_measurement_bitflip_bit_numbers(
        num_phase_bits=num_phase_bits,
        true_order=instance.true_order,
        range_name=range_names,
    )


def simulate_noisy_cf_membership_vs_order(
    instances: list[OrderRecoveryInstance],
    algorithms: list[str],
    num_phase_bits: int,
    measurement_error_probability: float,
    trials_per_instance: int,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    include_classical_error_correction: bool = False,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
) -> list[NoisyCFMembershipResult]:
    """
    Estimate Pr(S^c) over configured order instances.
    """
    algorithms = normalize_algorithms(algorithms)
    if trials_per_instance < 1:
        raise ValueError("trials_per_instance must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")
    single_measurement_bitflip_ranges = normalize_single_measurement_bitflip_ranges(
        single_measurement_bitflip_range
    )

    results = []
    for algorithm in algorithms:
        for instance in instances:
            bit_numbers = single_bitflip_bits_for_instance(
                instance=instance,
                num_phase_bits=num_phase_bits,
                single_measurement_bitflip_range=single_measurement_bitflip_ranges,
            )
            instance_seed = int(rng.integers(0, 2 ** 63 - 1))
            results.append(
                simulate_noisy_cf_membership(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    measurement_error_probability=measurement_error_probability,
                    trials=trials_per_instance,
                    rng=np.random.default_rng(instance_seed),
                    standard_backend=standard_backend,
                    analytic_window=analytic_window,
                    single_measurement_bitflip_bits=bit_numbers,
                )
            )
            if include_classical_error_correction:
                results.append(
                    simulate_noisy_cf_membership(
                        instance=instance,
                        algorithm=algorithm,
                        num_phase_bits=num_phase_bits,
                        measurement_error_probability=measurement_error_probability,
                        trials=trials_per_instance,
                        rng=np.random.default_rng(instance_seed),
                        standard_backend=standard_backend,
                        analytic_window=analytic_window,
                        classical_error_correction=True,
                        single_measurement_bitflip_bits=bit_numbers,
                    )
                )
    return results


def bit_numbers_display_label(bit_numbers: tuple[int, ...] | None) -> str:
    """
    Return a compact label for a selected set of 1-indexed phase bits.
    """
    if bit_numbers is None:
        return "p-mode"
    if not bit_numbers:
        return "empty"

    runs = []
    start = previous = bit_numbers[0]
    for bit_number in bit_numbers[1:]:
        if bit_number == previous + 1:
            previous = bit_number
            continue
        runs.append((start, previous))
        start = previous = bit_number
    runs.append((start, previous))

    return ",".join(
        str(start) if start == end else f"{start}..{end}"
        for start, end in runs
    )


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probability: float,
    include_classical_error_correction: bool = False,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
    reference_curves: list[str] | tuple[str, ...] | None = None,
    include_safe_tail_propagation_reference: bool = False,
) -> str:
    """
    Return the filename segment for a Pr(S^c) vs order plot.
    """
    algorithms = normalize_algorithms(algorithms)
    error_token = measurement_error_model_filename_segment(
        measurement_error_probability,
        single_measurement_bitflip_range=single_measurement_bitflip_range,
    )
    reference_curves = normalize_reference_curves(reference_curves)
    reference_token = (
        ""
        if not reference_curves
        else "_refs_" + "_".join(curve.replace("2", "two_") for curve in reference_curves)
    )
    propagation_token = (
        "_safe_tail_prop_ref"
        if include_safe_tail_propagation_reference
        else ""
    )
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"{instances_file_filename_segment(instances_file)}"
        f"{error_token}_t{num_phase_bits}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
        f"{reference_token}"
        f"{propagation_token}"
    )


def adaptive_probability_ylim(values: list[float]) -> tuple[float, float]:
    """
    Return an adaptive y-axis range for outside-S probabilities.
    """
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, 0.05
    max_value = max(finite_values)
    if max_value <= 0.0:
        return 0.0, 0.05
    return 0.0, min(1.0, max_value * 1.12)


def result_series_label(result: NoisyCFMembershipResult) -> str:
    suffix = "corrected" if result.classical_error_correction else "noisy"
    return f"{algorithm_display_label(result.algorithm)} {suffix}"


def reference_curve_label(reference_curve: str) -> str:
    """
    Return a display label for one probability reference curve.
    """
    if reference_curve == MP_REFERENCE_CURVE:
        return "1 - exp(-mp)"
    if reference_curve == TWO_MP_REFERENCE_CURVE:
        return "1 - exp(-2mp)"
    raise ValueError("unknown reference curve")


def reference_curve_values(
    instances: list[OrderRecoveryInstance],
    measurement_error_probability: float,
    reference_curve: str,
) -> list[float]:
    """
    Return the probability-axis values for one exponential reference curve.
    """
    if reference_curve == MP_REFERENCE_CURVE:
        factor = 1.0
    elif reference_curve == TWO_MP_REFERENCE_CURVE:
        factor = 2.0
    else:
        raise ValueError("unknown reference curve")

    return [
        1.0
        - math.exp(
            -factor
            * order_resolution_bits(instance.true_order)
            * measurement_error_probability
        )
        for instance in instances
    ]


def safe_tail_bit_count(num_phase_bits: int, true_order: int) -> int:
    """
    Return the number of phase bits in the safe-tail window 2m+1..t.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    return max(0, num_phase_bits - (2 * order_resolution_bits(true_order)))


def safe_tail_propagation_reference_label(
    result: NoisyCFMembershipResult,
    reference_curve: str,
) -> str:
    """
    Return a display label for one empirical safe-tail propagation reference.
    """
    base_label = "mp" if reference_curve == MP_REFERENCE_CURVE else "2mp"
    return f"{result_series_label(result)} tail-ref {base_label}"


def safe_tail_propagation_reference_values(
    series: list[NoisyCFMembershipResult],
    num_phase_bits: int,
    measurement_error_probability: float,
    reference_curve: str,
) -> list[float]:
    """
    Approximate 1-exp(-alpha*m*p - p*(t-2m)*rho_tail) for one series.

    Here rho_tail is estimated by the plotted safe-tail one-flip probability
    Pr(S^c | one flip uniformly in 2m+1..t).
    """
    if reference_curve == MP_REFERENCE_CURVE:
        factor = 1.0
    elif reference_curve == TWO_MP_REFERENCE_CURVE:
        factor = 2.0
    else:
        raise ValueError("unknown reference curve")

    values = []
    for result in series:
        m = order_resolution_bits(result.instance.true_order)
        tail_count = safe_tail_bit_count(
            num_phase_bits=num_phase_bits,
            true_order=result.instance.true_order,
        )
        exponent = (
            factor * m * measurement_error_probability
            + measurement_error_probability * tail_count * result.outside_probability
        )
        values.append(1.0 - math.exp(-exponent))
    return values


def plot_noisy_cf_membership_vs_order(
    output_path: Path,
    results: list[NoisyCFMembershipResult],
    algorithms: list[str],
    num_phase_bits: int,
    measurement_error_probability: float,
    trials_per_instance: int,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
    reference_curves: list[str] | tuple[str, ...] | None = None,
    include_safe_tail_propagation_reference: bool = False,
    safe_tail_reference_results: list[NoisyCFMembershipResult] | None = None,
    log_x: bool = False,
) -> None:
    """
    Plot Pr(S^c) versus true order r.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")
    reference_curves = normalize_reference_curves(reference_curves)
    single_measurement_bitflip_ranges = normalize_single_measurement_bitflip_ranges(
        single_measurement_bitflip_range
    )
    if include_safe_tail_propagation_reference:
        if not reference_curves:
            raise ValueError("safe-tail propagation references require reference curves")
        if (
            safe_tail_reference_results is None
            and single_measurement_bitflip_ranges != (SINGLE_BITFLIP_SAFE_TAIL_RANGE,)
        ):
            raise ValueError(
                "safe-tail propagation references require "
                "auxiliary safe-tail reference results"
            )

    error_label = measurement_error_model_display_label(
        measurement_error_probability,
        single_measurement_bitflip_range=single_measurement_bitflip_ranges,
    )
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    base_colors = {
        algorithm: color_cycle[index % len(color_cycle)]
        for index, algorithm in enumerate(algorithms)
    }
    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }
    linestyles = {
        False: "-",
        True: "-.",
    }
    result_keys = []
    for result in results:
        key = (result.algorithm, result.classical_error_correction)
        if key not in result_keys:
            result_keys.append(key)

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_values = []
    line_handles = []
    for algorithm, correction_enabled in result_keys:
        series = sorted(
            [
                result
                for result in results
                if result.algorithm == algorithm
                and result.classical_error_correction == correction_enabled
            ],
            key=lambda result: result.instance.true_order,
        )
        x_values = [result.instance.true_order for result in series]
        y_values = [result.outside_probability for result in series]
        plotted_values.extend(y_values)
        color = algorithm_color_for_base(base_colors[algorithm], algorithm)
        (line,) = ax.plot(
            x_values,
            y_values,
            marker=markers[algorithm],
            linestyle=linestyles[correction_enabled],
            linewidth=2.0,
            markersize=5.5,
            color=color,
            alpha=0.92,
            label=result_series_label(series[0]),
        )
        line_handles.append(line)

    instances = sorted(
        {
            result.instance
            for result in results
        },
        key=lambda instance: instance.true_order,
    )
    reference_handles = []
    reference_styles = {
        MP_REFERENCE_CURVE: (":", "#444444"),
        TWO_MP_REFERENCE_CURVE: ("--", "#444444"),
    }
    for reference_curve in reference_curves:
        x_values = [instance.true_order for instance in instances]
        y_values = reference_curve_values(
            instances=instances,
            measurement_error_probability=measurement_error_probability,
            reference_curve=reference_curve,
        )
        plotted_values.extend(y_values)
        linestyle, color = reference_styles[reference_curve]
        (line,) = ax.plot(
            x_values,
            y_values,
            marker="",
            linestyle=linestyle,
            linewidth=1.8,
            color=color,
            alpha=0.85,
            label=reference_curve_label(reference_curve),
        )
        reference_handles.append(line)

    propagation_reference_handles = []
    if include_safe_tail_propagation_reference:
        rho_results = (
            results
            if safe_tail_reference_results is None
            else safe_tail_reference_results
        )
        for algorithm, correction_enabled in result_keys:
            if algorithm != KITAEV_ALGORITHM:
                continue
            rho_series = sorted(
                [
                    result
                    for result in rho_results
                    if result.algorithm == algorithm
                    and result.classical_error_correction == correction_enabled
                ],
                key=lambda result: result.instance.true_order,
            )
            if not rho_series:
                raise ValueError("missing safe-tail reference series")
            x_values = [result.instance.true_order for result in rho_series]
            color = algorithm_color_for_base(base_colors[algorithm], algorithm)
            for reference_curve in reference_curves:
                y_values = safe_tail_propagation_reference_values(
                    series=rho_series,
                    num_phase_bits=num_phase_bits,
                    measurement_error_probability=measurement_error_probability,
                    reference_curve=reference_curve,
                )
                plotted_values.extend(y_values)
                linestyle = ":" if reference_curve == MP_REFERENCE_CURVE else "--"
                (line,) = ax.plot(
                    x_values,
                    y_values,
                    marker="",
                    linestyle=linestyle,
                    linewidth=1.5,
                    color=color,
                    alpha=0.62,
                    label=safe_tail_propagation_reference_label(
                        rho_series[0],
                        reference_curve,
                    ),
                )
                propagation_reference_handles.append(line)

    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lower, upper = adaptive_probability_ylim(plotted_values)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel("Pr(tilde b in S^c)")
    ax.set_title(
        "Noisy CF-Complement Membership vs True Order\n"
        f"{algorithms_display_label(algorithms)}, {error_label}, t = {num_phase_bits}"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=(
            line_handles
            + reference_handles
            + propagation_reference_handles
        ),
        measurement_handles=[],
        trials_per_instance=trials_per_instance,
        algorithm_title="Series",
        measurement_title="Measurement error",
        measurement_text_labels=[error_label],
        trials_title="Trials per instance",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[NoisyCFMembershipResult],
    algorithms: list[str],
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probability: float,
    trials_per_instance: int,
    seed: int | None,
    analytic_window_diagnostics: list[QPEWindowMassDiagnostics] | None = None,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
    reference_curves: list[str] | tuple[str, ...] | None = None,
    include_safe_tail_propagation_reference: bool = False,
    safe_tail_reference_results: list[NoisyCFMembershipResult] | None = None,
) -> None:
    """
    Write metadata and per-order Pr(S^c) data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")
    reference_curves = normalize_reference_curves(reference_curves)
    if include_safe_tail_propagation_reference:
        single_measurement_bitflip_ranges = normalize_single_measurement_bitflip_ranges(
            single_measurement_bitflip_range
        )
        if not reference_curves:
            raise ValueError("safe-tail propagation references require reference curves")
        if (
            safe_tail_reference_results is None
            and single_measurement_bitflip_ranges != (SINGLE_BITFLIP_SAFE_TAIL_RANGE,)
        ):
            raise ValueError(
                "safe-tail propagation references require "
                "auxiliary safe-tail reference results"
            )

    error_label = measurement_error_model_display_label(
        measurement_error_probability,
        single_measurement_bitflip_range=single_measurement_bitflip_range,
    )
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )
    seed_label = "random" if seed is None else str(seed)
    rho_source_label = "not used"
    if include_safe_tail_propagation_reference:
        rho_source_label = (
            "auxiliary single-flip safe-tail runs"
            if safe_tail_reference_results is not None
            else "plotted safe-tail series"
        )
    lines = [
        "Noisy CF-Complement Membership vs Order Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"Instances file = {instances_file}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Measurement error model = {error_label}",
        f"Reference curves = {', '.join(reference_curves) if reference_curves else 'none'}",
        "Safe-tail propagation references = "
        f"{'enabled' if include_safe_tail_propagation_reference else 'disabled'}",
        f"Safe-tail rho source = {rho_source_label}",
        f"Trials per instance and algorithm = {trials_per_instance}",
        f"Seed = {seed_label}",
        "",
        *analytic_window_diagnostics_summary_lines(analytic_window_diagnostics),
        *(
            [""]
            if analytic_window_diagnostics
            else []
        ),
        "Definitions:",
        "  S = {tilde b : d(tilde b, s'/r) <= 1/(2r^2), with s'/r nearest to tilde b}",
        "  Pr(S^c) = empirical probability of landing outside S.",
        "  Corrected uses the bit string among tilde b and all one-bit flips with minimum distance to its own nearest s/r.",
        "  Safe-tail propagation references use 1-exp(-alpha*m*p - p*(t-2m)*rho_tail),",
        "  with alpha in {1, 2} from --reference-curves and rho_tail estimated by auxiliary",
        "  Pr(S^c | exactly one flip uniformly in the safe tail) Kitaev runs unless the plotted series is itself safe-tail.",
        "",
        "Per-order data:",
        "  algorithm | series | true r | N | a | bitflip bits | Pr(S^c) | S^c count | S count | trials",
    ]

    for result in sorted(
        results,
        key=lambda item: (
            item.algorithm,
            item.classical_error_correction,
            item.instance.true_order,
        ),
    ):
        bit_numbers = single_bitflip_bits_for_instance(
            instance=result.instance,
            num_phase_bits=num_phase_bits,
            single_measurement_bitflip_range=single_measurement_bitflip_range,
        )
        bit_label = bit_numbers_display_label(bit_numbers)
        series = "corrected" if result.classical_error_correction else "noisy"
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {series:>9} |"
            f" {result.instance.true_order:>6} |"
            f" {result.instance.N:>5} |"
            f" {result.instance.a:>5} |"
            f" {bit_label:>12} |"
            f" {result.outside_probability:>7.3f} |"
            f" {result.outside_trials:>9} |"
            f" {result.inside_trials:>7} |"
            f" {result.trials:>6}"
        )

    rho_results = safe_tail_reference_results
    if include_safe_tail_propagation_reference and rho_results is not None:
        lines.extend([
            "",
            "Auxiliary safe-tail rho data:",
            "  algorithm | series | true r | rho_tail | S^c count | S count | trials",
        ])
        for result in sorted(
            [
                result
                for result in rho_results
                if result.algorithm == KITAEV_ALGORITHM
            ],
            key=lambda item: (
                item.algorithm,
                item.classical_error_correction,
                item.instance.true_order,
            ),
        ):
            series = "corrected" if result.classical_error_correction else "noisy"
            lines.append(
                f"  {algorithm_display_label(result.algorithm):>13} |"
                f" {series:>9} |"
                f" {result.instance.true_order:>6} |"
                f" {result.outside_probability:>8.4f} |"
                f" {result.outside_trials:>9} |"
                f" {result.inside_trials:>7} |"
                f" {result.trials:>6}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Pr(tilde b in S^c) versus true order r."
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
        default=0.1,
        help="Independent per-bit measurement flip probability for p-mode",
    )
    parser.add_argument(
        "--single-measurement-bitflip-range",
        nargs="+",
        choices=SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES,
        default=None,
        help=(
            "Use exactly one uniformly random measurement bit-flip restricted "
            "to one or more phase-bit ranges. For r=2^m: leading=1..m, "
            "dangerous=m+1..2m, safe-tail=2m+1..t. Multiple ranges are sampled "
            "as one uniform union."
        ),
    )
    parser.add_argument(
        "--trials-per-instance",
        type=int,
        default=100,
        help="Number of noisy trials per instance and algorithm",
    )
    parser.add_argument(
        "--classical-error-correction",
        action="store_true",
        help="Also plot the one-bit minimum-distance corrected series",
    )
    parser.add_argument(
        "--reference-curves",
        "--reference-probability-curves",
        nargs="+",
        choices=REFERENCE_CURVE_CHOICES,
        default=None,
        help=(
            "Overlay exponential reference curves computed from --p and "
            "m=ceil(log2(r)): 1-exp(-mp) and/or 1-exp(-2mp). "
            "Choices: mp, 2mp."
        ),
    )
    parser.add_argument(
        "--include-safe-tail-propagation-reference",
        "--safe-tail-propagation-reference",
        action="store_true",
        help=(
            "Also overlay algorithm-specific references "
            "1-exp(-alpha*m*p - p*(t-2m)*rho_tail), where alpha is chosen "
            "by --reference-curves and rho_tail is estimated from auxiliary "
            "Kitaev safe-tail single-bitflip runs."
        ),
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=ANALYTIC_STANDARD_BACKEND,
        help="Backend used for standard Shor curves",
    )
    parser.add_argument(
        "--analytic-window",
        type=int,
        default=100,
        help="Analytic standard-QPE local window half-width",
    )
    parser.add_argument(
        "--max-statevector-amplitudes",
        type=int,
        default=DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
        help="Safety limit for standard statevector backend allocations",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a base-2 logarithmic x-axis for true order r",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
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
        p = validate_measurement_error_probability(args.p)
        single_bitflip_ranges = normalize_single_measurement_bitflip_ranges(
            args.single_measurement_bitflip_range
        )
        reference_curves = normalize_reference_curves(args.reference_curves)
        if args.include_safe_tail_propagation_reference:
            if not reference_curves:
                raise ValueError(
                    "--include-safe-tail-propagation-reference requires "
                    "--reference-curves mp and/or 2mp"
                )
        instances = load_order_recovery_instances_from_file(
            instances_file=args.instances_file,
            num_phase_bits=args.bits,
        )
        for instance in instances:
            single_bitflip_bits_for_instance(
                instance=instance,
                num_phase_bits=args.bits,
                single_measurement_bitflip_range=single_bitflip_ranges,
            )
            if args.include_safe_tail_propagation_reference:
                single_bitflip_bits_for_instance(
                    instance=instance,
                    num_phase_bits=args.bits,
                    single_measurement_bitflip_range=SINGLE_BITFLIP_SAFE_TAIL_RANGE,
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

    results = simulate_noisy_cf_membership_vs_order(
        instances=instances,
        algorithms=algorithms,
        num_phase_bits=args.bits,
        measurement_error_probability=p,
        trials_per_instance=args.trials_per_instance,
        rng=np.random.default_rng(args.seed),
        standard_backend=args.standard_backend,
        analytic_window=args.analytic_window,
        include_classical_error_correction=args.classical_error_correction,
        single_measurement_bitflip_range=single_bitflip_ranges,
    )
    safe_tail_reference_results = None
    if args.include_safe_tail_propagation_reference:
        safe_tail_seed = (
            None
            if args.seed is None
            else int(np.random.default_rng(args.seed).integers(0, 2 ** 63 - 1))
        )
        safe_tail_reference_results = simulate_noisy_cf_membership_vs_order(
            instances=instances,
            algorithms=algorithms,
            num_phase_bits=args.bits,
            measurement_error_probability=p,
            trials_per_instance=args.trials_per_instance,
            rng=np.random.default_rng(safe_tail_seed),
            standard_backend=args.standard_backend,
            analytic_window=args.analytic_window,
            include_classical_error_correction=args.classical_error_correction,
            single_measurement_bitflip_range=SINGLE_BITFLIP_SAFE_TAIL_RANGE,
        )

    suffix = output_filename_segment(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        measurement_error_probability=p,
        include_classical_error_correction=args.classical_error_correction,
        single_measurement_bitflip_range=single_bitflip_ranges,
        reference_curves=reference_curves,
        include_safe_tail_propagation_reference=(
            args.include_safe_tail_propagation_reference
        ),
    )
    figure_output_path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parent
        / "figures"
        / f"noisy_cf_membership_vs_order_{suffix}.png"
    )
    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"noisy_cf_membership_vs_order_summary_{suffix}.txt"
    )

    plot_noisy_cf_membership_vs_order(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
        num_phase_bits=args.bits,
        measurement_error_probability=p,
        trials_per_instance=args.trials_per_instance,
        single_measurement_bitflip_range=single_bitflip_ranges,
        reference_curves=reference_curves,
        include_safe_tail_propagation_reference=(
            args.include_safe_tail_propagation_reference
        ),
        safe_tail_reference_results=safe_tail_reference_results,
        log_x=args.log_x,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        measurement_error_probability=p,
        trials_per_instance=args.trials_per_instance,
        seed=args.seed,
        analytic_window_diagnostics=analytic_window_diagnostics,
        single_measurement_bitflip_range=single_bitflip_ranges,
        reference_curves=reference_curves,
        include_safe_tail_propagation_reference=(
            args.include_safe_tail_propagation_reference
        ),
        safe_tail_reference_results=safe_tail_reference_results,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    error_label = measurement_error_model_display_label(
        p,
        single_measurement_bitflip_range=single_bitflip_ranges,
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
    print(f"Measurement error model: {error_label}")
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(
        "Reference curves: "
        f"{', '.join(reference_curves) if reference_curves else 'disabled'}"
    )
    print(
        "Safe-tail propagation references: "
        f"{'enabled' if args.include_safe_tail_propagation_reference else 'disabled'}"
    )
    print(f"Trials per instance and algorithm: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
