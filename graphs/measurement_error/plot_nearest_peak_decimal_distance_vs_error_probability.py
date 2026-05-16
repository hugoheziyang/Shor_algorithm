from __future__ import annotations

import argparse
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
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from error_correction.classical_error_correction import validate_phase_estimation_instance
from graphs.measurement_error.plot_classical_recovery_vs_error_probability import build_probability_values
from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_order import (
    CLEAN_PHASE,
    CORRECTED_PHASE,
    NOISY_PHASE,
    NearestPeakDistanceResult,
    adaptive_distance_ylim,
    apply_scientific_distance_y_axis,
    classical_error_correction_filename_segment,
    cf_sufficient_bound_distance,
    inverse_order_reference_filename_segment,
    inverse_order_reference_distance,
    phase_choices,
    phase_display_label,
    result_phase_distance,
    simulate_nearest_peak_distance_for_instance,
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
    normalize_algorithms,
    standard_backend_filename_segment,
    validate_statevector_budget,
)
from shor_common.modular_unitaries import register_size
from shor_common.shor_utils import default_num_phase_bits
from shor_standard.analytic_sampling import multiplicative_order
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)


def build_fixed_instance(N: int, a: int) -> OrderRecoveryInstance:
    """
    Validate a fixed modular instance and compute its true order.
    """
    validate_phase_estimation_instance(N, a)
    return OrderRecoveryInstance(
        N=N,
        a=a,
        true_order=multiplicative_order(a, N),
        num_work_qubits=register_size(N),
        default_phase_bits=default_num_phase_bits(N),
    )


def simulate_nearest_peak_distance_vs_error_probability(
    instance: OrderRecoveryInstance,
    algorithms: list[str],
    probability_values: list[float],
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    include_classical_error_correction: bool = False,
) -> list[NearestPeakDistanceResult]:
    """
    Estimate nearest-peak distances over a measurement-error probability grid.
    """
    algorithms = normalize_algorithms(algorithms)
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    results = []
    for algorithm in algorithms:
        for probability in probability_values:
            probability_rng = np.random.default_rng(
                int(rng.integers(0, 2 ** 63 - 1))
            )
            results.append(
                simulate_nearest_peak_distance_for_instance(
                    instance=instance,
                    algorithm=algorithm,
                    trials=trials,
                    rng=probability_rng,
                    num_phase_bits=num_phase_bits,
                    measurement_error_probability=float(probability),
                    standard_backend=standard_backend,
                    analytic_window=analytic_window,
                    include_classical_error_correction=include_classical_error_correction,
                )
            )
    return results


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    N: int,
    a: int,
    num_phase_bits: int,
    include_classical_error_correction: bool = False,
    include_inverse_order_reference: bool = False,
) -> str:
    """
    Return the filename segment for a fixed-instance probability sweep.
    """
    algorithms = normalize_algorithms(algorithms)
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"N{N}_a{a}_t{num_phase_bits}"
        f"{classical_error_correction_filename_segment(include_classical_error_correction)}"
        f"{inverse_order_reference_filename_segment(include_inverse_order_reference)}"
    )


def plot_nearest_peak_distance_vs_error_probability(
    output_path: Path,
    results: list[NearestPeakDistanceResult],
    algorithms: list[str],
    probability_values: list[float],
    num_phase_bits: int,
    trials_per_probability: int,
    log_p_grid: bool = False,
    include_classical_error_correction: bool = False,
    include_inverse_order_reference: bool = False,
) -> None:
    """
    Plot nearest-peak distance versus measurement error probability.

    Without classical correction, the plot contains clean and noisy curves.
    With classical correction, it contains only the corrected-min curves.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)

    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    bound = float(cf_sufficient_bound_distance(instance.true_order))
    inverse_order_reference = float(
        inverse_order_reference_distance(instance.true_order)
    )
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

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_values = [bound]
    if include_inverse_order_reference:
        plotted_values.append(inverse_order_reference)
    curve_handles = []

    for algorithm in algorithms:
        for phase in phase_choices(include_classical_error_correction):
            phase_results = sorted(
                [
                    result
                    for result in results
                    if result.algorithm == algorithm
                ],
                key=lambda result: result.measurement_error_probability,
            )
            y_values = [
                result_phase_distance(result, phase)
                for result in phase_results
            ]
            plotted_values.extend(y_values)
            (line,) = ax.plot(
                [result.measurement_error_probability for result in phase_results],
                y_values,
                marker=markers[algorithm],
                linestyle=linestyles[phase],
                linewidth=2.0,
                markersize=6,
                color=algorithm_colors[algorithm],
                alpha=0.92,
                label=(
                    f"{algorithm_display_label(algorithm)} "
                    f"{phase_display_label(phase)}"
                ),
            )
            curve_handles.append(line)

    bound_line = ax.axhline(
        bound,
        linestyle="--",
        linewidth=2.0,
        color="#303030",
        alpha=0.88,
        label="CF bound 1/(2r^2)",
    )
    curve_handles.append(bound_line)

    if include_inverse_order_reference:
        inverse_line = ax.axhline(
            inverse_order_reference,
            linestyle="-",
            linewidth=2.0,
            color="#6B5B2A",
            alpha=0.88,
            label="Reference 1/(2r)",
        )
        curve_handles.append(inverse_line)

    ax.set_xlabel("Measurement bit-flip probability p")
    ax.set_ylabel("Average circular distance to nearest eigenphase")
    title_prefix = (
        "Corrected-Min Nearest-Eigenphase Decimal Distance vs Measurement Error"
        if include_classical_error_correction
        else "Nearest-Eigenphase Decimal Distance vs Measurement Error"
    )
    ax.set_title(
        f"{title_prefix}\n"
        f"{algorithms_display_label(algorithms)} for N = {instance.N}, "
        f"a = {instance.a}, r = {instance.true_order}, t = {num_phase_bits}"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    x_min = min(probability_values)
    x_max = max(probability_values)
    if log_p_grid:
        ax.set_xscale("log")
        if x_min == x_max:
            ax.set_xlim(x_min / 1.5, x_max * 1.5)
        else:
            ax.set_xlim(x_min * 0.9, x_max * 1.1)
    else:
        x_padding = max(0.01, (x_max - x_min) * 0.03)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
    lower, upper = adaptive_distance_ylim(plotted_values)
    ax.set_ylim(lower, upper)
    apply_scientific_distance_y_axis(ax)
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=curve_handles,
        measurement_handles=[],
        trials_per_instance=trials_per_probability,
        algorithm_title="Curve",
        measurement_title="p grid",
        measurement_text_labels=[
            "logarithmic" if log_p_grid else "linear",
            f"{x_min:.3g} to {x_max:.3g}",
        ],
        extra_text_sections=[
            (
                "Instance",
                [
                    f"N = {instance.N}, a = {instance.a}",
                    f"r = {instance.true_order}, t = {num_phase_bits}",
                ],
            ),
            (
                "Reference",
                [
                    f"1/(2r^2) = {bound:.6f}",
                    *(
                        [f"1/(2r) = {inverse_order_reference:.6f}"]
                        if include_inverse_order_reference
                        else []
                    ),
                ],
            ),
        ],
        trials_title="Trials per p",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[NearestPeakDistanceResult],
    algorithms: list[str],
    probability_values: list[float],
    num_phase_bits: int,
    trials_per_probability: int,
    log_p_grid: bool,
    seed: int | None,
    include_classical_error_correction: bool = False,
    include_inverse_order_reference: bool = False,
) -> None:
    """
    Write metadata and per-probability nearest-peak distance data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )
    bound = float(cf_sufficient_bound_distance(instance.true_order))
    inverse_order_reference = float(
        inverse_order_reference_distance(instance.true_order)
    )
    seed_label = "random" if seed is None else str(seed)
    probability_label = (
        f"{min(probability_values):.6f} to {max(probability_values):.6f}"
    )

    lines = [
        "Nearest-Eigenphase Decimal Distance vs Error Probability Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"N = {instance.N}",
        f"a = {instance.a}",
        f"True order r = {instance.true_order}",
        f"CF sufficient bound 1/(2r^2) = {bound:.12f}",
        *(
            [f"Half-inverse-order reference 1/(2r) = {inverse_order_reference:.12f}"]
            if include_inverse_order_reference
            else []
        ),
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Trials per probability value and algorithm = {trials_per_probability}",
        f"Probability grid = {'logarithmic' if log_p_grid else 'linear'}",
        f"Measurement bit-flip probability range = {probability_label}",
        f"Probability values = {', '.join(f'{value:.6f}' for value in probability_values)}",
        f"Classical single-bit minimum-distance plot = {'corrected-only' if include_classical_error_correction else 'disabled'}",
        f"Half-inverse-order reference plot = {'enabled' if include_inverse_order_reference else 'disabled'}",
        f"Seed = {seed_label}",
        "",
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
        *(
            ["  half-inverse-order reference = 1/(2r)"]
            if include_inverse_order_reference
            else []
        ),
        "",
        "Per-probability data:",
        (
            "  algorithm | p | avg d(b,s/r) | avg d(tilde b,s'/r) | "
            "avg d(b_ec,s_ec/r) | Pr(s'=s) | Pr(s_ec=s) | "
            "clean distance total | noisy distance total | corrected distance total | trials"
            if include_classical_error_correction
            else
            "  algorithm | p | avg d(b,s/r) | avg d(tilde b,s'/r) | "
            "Pr(s'=s) | clean distance total | noisy distance total | trials"
        ),
    ]

    for result in sorted(
        results,
        key=lambda item: (
            item.algorithm,
            item.measurement_error_probability,
        ),
    ):
        prefix = (
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {result.measurement_error_probability:>8.6f} |"
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
    Build the CLI parser for the fixed-instance nearest-peak p sweep.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot average nearest-eigenphase decimal distance versus measurement "
            "bit-flip probability for one fixed (N, a)."
        )
    )
    parser.add_argument("N", type=int, help="Modulus to analyze")
    parser.add_argument(
        "--a",
        type=int,
        required=True,
        help="Fixed coprime base a to analyze",
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
        default=None,
        help="Override the number of phase-estimation bits/control qubits",
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=ANALYTIC_STANDARD_BACKEND,
        help=(
            "Backend used for standard Shor curves. The analytic backend is "
            "recommended for larger t."
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
        "--classical-error-correction",
        action="store_true",
        help=(
            "Plot only the single-bit-search minimum nearest-peak distance "
            "diagnostic: choose the noisy bit string or one-bit flip with "
            "the smallest distance to its own nearest eigenphase."
        ),
    )
    parser.add_argument(
        "--include-half-inverse-order-reference",
        "--include-inverse-order-reference",
        dest="include_inverse_order_reference",
        action="store_true",
        help="Plot the optional solid half-inverse-order reference line 1/(2r)",
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
    Run the fixed-instance nearest-peak p-sweep plotting script.
    """
    start_time = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()
    algorithms = normalize_algorithms(args.algorithm)

    if args.trials < 1:
        raise ValueError("trials must be >= 1")
    if args.analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")
    if args.bits is not None and args.bits < 1:
        raise ValueError("bits must be >= 1")

    try:
        instance = build_fixed_instance(args.N, args.a)
        num_phase_bits = (
            default_num_phase_bits(args.N)
            if args.bits is None
            else args.bits
        )
        validate_statevector_budget(
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            num_phase_bits=num_phase_bits,
            max_N=args.N,
            max_statevector_amplitudes=args.max_statevector_amplitudes,
        )
        probability_values = build_probability_values(
            args.min_p,
            args.max_p,
            args.num_points,
            log_p_grid=args.log_p_grid,
        )
    except ValueError as exc:
        parser.error(str(exc))

    master_rng = np.random.default_rng(args.seed)
    results = simulate_nearest_peak_distance_vs_error_probability(
        instance=instance,
        algorithms=algorithms,
        probability_values=probability_values,
        trials=args.trials,
        rng=master_rng,
        num_phase_bits=num_phase_bits,
        standard_backend=args.standard_backend,
        analytic_window=args.analytic_window,
        include_classical_error_correction=args.classical_error_correction,
    )

    suffix = output_filename_segment(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        include_classical_error_correction=args.classical_error_correction,
        include_inverse_order_reference=args.include_inverse_order_reference,
    )
    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"nearest_peak_decimal_distance_vs_error_probability_{suffix}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"nearest_peak_decimal_distance_vs_error_probability_summary_{suffix}.txt"
    )

    plot_nearest_peak_distance_vs_error_probability(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
        probability_values=probability_values,
        num_phase_bits=num_phase_bits,
        trials_per_probability=args.trials,
        log_p_grid=args.log_p_grid,
        include_classical_error_correction=args.classical_error_correction,
        include_inverse_order_reference=args.include_inverse_order_reference,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
        probability_values=probability_values,
        num_phase_bits=num_phase_bits,
        trials_per_probability=args.trials,
        log_p_grid=args.log_p_grid,
        seed=args.seed,
        include_classical_error_correction=args.classical_error_correction,
        include_inverse_order_reference=args.include_inverse_order_reference,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"Algorithm: {algorithms_display_label(algorithms)}")
    if STANDARD_ALGORITHM in algorithms:
        print(f"Standard backend: {args.standard_backend}")
        if args.standard_backend != STATEVECTOR_STANDARD_BACKEND:
            print(f"Analytic window: {args.analytic_window}")
    print(f"N = {args.N}")
    print(f"a = {args.a}")
    print(f"True order r = {instance.true_order}")
    print(
        "CF sufficient bound 1/(2r^2) = "
        f"{float(cf_sufficient_bound_distance(instance.true_order)):.6f}"
    )
    print(f"Phase-estimation bits = {num_phase_bits}")
    print(f"Trials per probability value and algorithm: {args.trials}")
    print(
        "Classical single-bit minimum-distance plot: "
        f"{'corrected-only' if args.classical_error_correction else 'disabled'}"
    )
    print(
        "Half-inverse-order reference 1/(2r): "
        f"{'enabled' if args.include_inverse_order_reference else 'disabled'}"
    )
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
