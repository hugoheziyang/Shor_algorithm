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


from graphs.measurement_error.phase_distance_vs_order_common import (
    SingleBitflipRangeSpec,
    normalize_single_measurement_bitflip_ranges,
    single_measurement_bitflip_bit_numbers,
    SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES,
    validate_measurement_error_probability,
)
from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_error_probability import (
    build_fixed_instance,
)
from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_order import (
    cf_sufficient_bound_distance,
)
from graphs.measurement_error.plot_noisy_cf_complement_fill import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_CACHE_BITSTRINGS,
    CFMembershipCache,
    bits_to_index,
    load_or_build_cf_membership_cache,
    measurement_error_model_display_label,
    measurement_error_model_filename_segment,
    run_phase_bits_for_fill_trial,
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
    normalize_algorithms,
    standard_backend_filename_segment,
    validate_statevector_budget,
)
from shor_common.shor_utils import default_num_phase_bits
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)


DEFAULT_TRIAL_COUNTS = [10, 20, 50, 100, 200, 500, 1000]


@dataclass(frozen=True)
class NoisyCFComplementFillTrialSweepResult:
    instance: OrderRecoveryInstance
    algorithm: str
    measurement_error_probability: float
    trial_count: int
    num_phase_bits: int
    outside_unique_hits: int
    outside_trial_hits: int
    total_inside_bitstrings: int
    total_outside_bitstrings: int
    standard_backend: str = ANALYTIC_STANDARD_BACKEND
    classical_error_correction: bool = False
    single_measurement_bitflip: bool = False
    single_measurement_bitflip_range: tuple[str, ...] | None = None

    @property
    def outside_fill_fraction(self) -> float:
        if self.total_outside_bitstrings == 0:
            return math.nan
        return self.outside_unique_hits / self.total_outside_bitstrings

    @property
    def outside_trial_probability(self) -> float:
        return self.outside_trial_hits / self.trial_count


def validate_trial_counts(trial_counts: list[int]) -> list[int]:
    """
    Validate, de-duplicate, and sort trial-count values.
    """
    if not trial_counts:
        raise ValueError("at least one trial count is required")
    normalized = []
    for trial_count in trial_counts:
        count = int(trial_count)
        if count < 1:
            raise ValueError("trial counts must be >= 1")
        if count not in normalized:
            normalized.append(count)
    return sorted(normalized)


def simulate_noisy_cf_complement_fill_vs_trials(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    trial_counts: list[int],
    rng: np.random.Generator,
    membership_cache: CFMembershipCache,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    classical_error_correction: bool = False,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
) -> list[NoisyCFComplementFillTrialSweepResult]:
    """
    Estimate S^c fill fractions with an independent run per trial count.
    """
    trial_counts = validate_trial_counts(trial_counts)
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")
    if membership_cache.true_order != instance.true_order:
        raise ValueError("membership cache order does not match instance")
    if membership_cache.num_phase_bits != num_phase_bits:
        raise ValueError("membership cache bit count does not match simulation")
    single_measurement_bitflip_ranges = normalize_single_measurement_bitflip_ranges(
        single_measurement_bitflip_range
    )

    results = []

    for trial_count in trial_counts:
        trial_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
        outside_hits: set[int] = set()
        outside_trial_hits = 0

        for _ in range(trial_count):
            bits = run_phase_bits_for_fill_trial(
                instance=instance,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                measurement_error_probability=measurement_error_probability,
                rng=trial_rng,
                standard_backend=standard_backend,
                analytic_window=analytic_window,
                classical_error_correction=classical_error_correction,
                single_measurement_bitflip=single_measurement_bitflip,
                single_measurement_bitflip_bits=single_measurement_bitflip_bits,
            )
            bitstring_index = bits_to_index(bits)
            if membership_cache.is_outside_index(bitstring_index):
                outside_trial_hits += 1
                outside_hits.add(bitstring_index)

        results.append(
            NoisyCFComplementFillTrialSweepResult(
                instance=instance,
                algorithm=algorithm,
                measurement_error_probability=measurement_error_probability,
                trial_count=trial_count,
                num_phase_bits=num_phase_bits,
                outside_unique_hits=len(outside_hits),
                outside_trial_hits=outside_trial_hits,
                total_inside_bitstrings=membership_cache.inside_count,
                total_outside_bitstrings=membership_cache.outside_count,
                standard_backend=standard_backend,
                classical_error_correction=classical_error_correction,
                single_measurement_bitflip=single_measurement_bitflip,
                single_measurement_bitflip_range=single_measurement_bitflip_ranges,
            )
        )

    return results


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    N: int,
    a: int,
    num_phase_bits: int,
    measurement_error_probability: float,
    include_classical_error_correction: bool = False,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
) -> str:
    """
    Return the filename segment for one S^c fill-vs-trials chart.
    """
    algorithms = normalize_algorithms(algorithms)
    error_token = measurement_error_model_filename_segment(
        measurement_error_probability,
        single_measurement_bitflip=single_measurement_bitflip,
        single_measurement_bitflip_range=single_measurement_bitflip_range,
    )
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"N{N}_a{a}_t{num_phase_bits}_{error_token}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )


def result_series_label(result: NoisyCFComplementFillTrialSweepResult) -> str:
    """
    Return the plotted label for one noisy/corrected result.
    """
    suffix = "corrected" if result.classical_error_correction else "noisy"
    return f"{algorithm_display_label(result.algorithm)} {suffix}"


def adaptive_fill_sweep_ylim(values: list[float]) -> tuple[float, float]:
    """
    Return a y-axis range that preserves very small occupancy fractions.
    """
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, 1.0
    max_value = max(finite_values)
    if max_value <= 0.0:
        return 0.0, 0.05
    return 0.0, min(1.0, max_value * 1.18)


def fill_tick_format(y_upper: float) -> str:
    if y_upper < 0.001:
        return "%.1e"
    if y_upper < 0.02:
        return "%.4f"
    return "%.3f"


def plot_noisy_cf_complement_fill_vs_trials(
    output_path: Path,
    results: list[NoisyCFComplementFillTrialSweepResult],
    algorithms: list[str],
    trial_counts: list[int],
    membership_cache: CFMembershipCache,
    log_x: bool = False,
) -> None:
    """
    Plot independent-run S^c fill fraction versus trial count.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    trial_counts = validate_trial_counts(trial_counts)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    probability = results[0].measurement_error_probability
    single_measurement_bitflip = results[0].single_measurement_bitflip
    error_label = measurement_error_model_display_label(
        probability,
        single_measurement_bitflip=single_measurement_bitflip,
        single_measurement_bitflip_range=results[0].single_measurement_bitflip_range,
    )
    num_phase_bits = results[0].num_phase_bits
    bound = float(cf_sufficient_bound_distance(instance.true_order))
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
    result_keys = [
        (result.algorithm, result.classical_error_correction)
        for result in results
        if result.trial_count == trial_counts[0]
    ]

    fig, ax = plt.subplots(figsize=(10, 5.8))
    plotted_values = []
    algorithm_handles = []

    for algorithm, correction_enabled in result_keys:
        algorithm_results = sorted(
            [
                result
                for result in results
                if result.algorithm == algorithm
                and result.classical_error_correction == correction_enabled
            ],
            key=lambda result: result.trial_count,
        )
        x_values = [result.trial_count for result in algorithm_results]
        y_values = [result.outside_fill_fraction for result in algorithm_results]
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
            label=result_series_label(algorithm_results[0]),
        )
        algorithm_handles.append(line)

    if log_x:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    lower, upper = adaptive_fill_sweep_ylim(plotted_values)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_formatter(FormatStrFormatter(fill_tick_format(upper)))
    ax.set_xlabel("Trials per algorithm")
    ax.set_ylabel("Fraction of S^c uniquely hit")
    ax.set_title(
        "Noisy S^c Coverage vs Trials\n"
        f"{algorithms_display_label(algorithms)} for N = {instance.N}, "
        f"a = {instance.a}, r = {instance.true_order}, "
        f"{error_label}"
    )
    ax.grid(True, alpha=0.25)

    cache_status = "loaded" if membership_cache.from_cache else "built"
    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=algorithm_handles,
        measurement_handles=[],
        trials_per_instance=trial_counts[-1],
        algorithm_title="Series",
        measurement_title="Measurement error",
        measurement_text_labels=[error_label],
        extra_text_sections=[
            (
                "Instance",
                [
                    f"N = {instance.N}, a = {instance.a}",
                    f"r = {instance.true_order}, t = {num_phase_bits}",
                ],
            ),
            (
                "Universe",
                [
                    f"|S| = {membership_cache.inside_count:,}",
                    f"|S^c| = {membership_cache.outside_count:,}",
                    f"1/(2r^2) = {bound:.6f}",
                ],
            ),
            ("Cache", [cache_status]),
        ],
        trials_title="Max trials",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[NoisyCFComplementFillTrialSweepResult],
    algorithms: list[str],
    trial_counts: list[int],
    membership_cache: CFMembershipCache,
    seed: int | None,
) -> None:
    """
    Write metadata and per-trial-count S^c fill fractions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    trial_counts = validate_trial_counts(trial_counts)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    probability = results[0].measurement_error_probability
    single_measurement_bitflip = results[0].single_measurement_bitflip
    error_label = measurement_error_model_display_label(
        probability,
        single_measurement_bitflip=single_measurement_bitflip,
        single_measurement_bitflip_range=results[0].single_measurement_bitflip_range,
    )
    num_phase_bits = results[0].num_phase_bits
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )
    bound = float(cf_sufficient_bound_distance(instance.true_order))
    seed_label = "random" if seed is None else str(seed)
    cache_label = "loaded" if membership_cache.from_cache else "built"
    cache_path_label = (
        "none" if membership_cache.path is None else str(membership_cache.path)
    )

    lines = [
        "Noisy CF-Complement Fill vs Trials Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"N = {instance.N}",
        f"a = {instance.a}",
        f"True order r = {instance.true_order}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Measurement error model = {error_label}",
        f"CF sufficient bound 1/(2r^2) = {bound:.12f}",
        f"Trial-count values = {', '.join(str(count) for count in trial_counts)}",
        f"Total bitstrings 2^t = {membership_cache.num_bitstrings}",
        f"|S| = {membership_cache.inside_count}",
        f"|S^c| = {membership_cache.outside_count}",
        f"Cache type = packed bitset",
        f"Cache status = {cache_label}",
        f"Cache path = {cache_path_label}",
        f"Seed = {seed_label}",
        "",
        "Definitions:",
        "  S = {tilde b : d(tilde b, s'/r) <= 1/(2r^2), with s'/r nearest to tilde b}",
        *(
            [
                "  Single-measurement-bitflip mode flips exactly one uniformly random readout bit per trial.",
                "  In Kitaev, that one readout error can propagate through later adaptive feedback bits.",
            ]
            if single_measurement_bitflip
            else [
                "  Probability mode flips each measurement readout bit independently with probability p.",
            ]
        ),
        "  Corrected uses the bit string among tilde b and all one-bit flips with minimum distance to its own nearest s/r.",
        "  S^c fill = |unique noisy bitstrings observed in S^c| / |S^c|.",
        "  Each trial-count point is estimated from an independent random run.",
        "",
        "Per-trial-count data:",
        "  algorithm | series | trials | S^c fill | unique S^c hits | |S^c| | Pr(S^c) | S^c trial hits",
    ]

    for result in sorted(
        results,
        key=lambda item: (
            item.algorithm,
            item.classical_error_correction,
            item.trial_count,
        ),
    ):
        series = "corrected" if result.classical_error_correction else "noisy"
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {series:>9} |"
            f" {result.trial_count:>6} |"
            f" {result.outside_fill_fraction:>12.6e} |"
            f" {result.outside_unique_hits:>15} |"
            f" {result.total_outside_bitstrings:>12} |"
            f" {result.outside_trial_probability:>7.3f} |"
            f" {result.outside_trial_hits:>14}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for noisy S^c coverage-vs-trials charts.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot independent-run S^c coverage versus the number of noisy trials "
            "for one fixed (N, a) and measurement bit-flip probability."
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
        "--p",
        type=float,
        default=0.1,
        help="Independent per-bit measurement flip probability",
    )
    parser.add_argument(
        "--single-measurement-bitflip",
        "--one-measurement-bitflip",
        dest="single_measurement_bitflip",
        action="store_true",
        help=(
            "Use exactly one uniformly random measurement bit-flip per trial "
            "instead of independent per-bit flips from --p."
        ),
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
        "--trial-counts",
        type=int,
        nargs="+",
        default=DEFAULT_TRIAL_COUNTS,
        help="Independent trial-count values to plot",
    )
    parser.add_argument(
        "--classical-error-correction",
        action="store_true",
        help=(
            "Also plot the simulator-side single-bit classical correction "
            "diagnostic: use the noisy bit string or one of its one-bit flips "
            "with minimum distance to its own nearest exact order eigenphase."
        ),
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
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached S/S^c membership bitsets keyed by r and t",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the S/S^c membership cache even if it already exists",
    )
    parser.add_argument(
        "--max-cache-bitstrings",
        type=int,
        default=DEFAULT_MAX_CACHE_BITSTRINGS,
        help=(
            "Safety limit for enumerating 2^t bitstrings while building the "
            "cache. Use 0 to disable the guard."
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
        help="Use a logarithmic x-axis for trial counts",
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
    Run the noisy S^c fill-vs-trials chart script.
    """
    start_time = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()
    algorithms = normalize_algorithms(args.algorithm)

    if args.analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")
    if args.bits is not None and args.bits < 1:
        raise ValueError("bits must be >= 1")
    if args.max_cache_bitstrings < 0:
        raise ValueError("max_cache_bitstrings must be >= 0")

    try:
        p = validate_measurement_error_probability(args.p)
        trial_counts = validate_trial_counts(args.trial_counts)
        single_bitflip_ranges = normalize_single_measurement_bitflip_ranges(
            args.single_measurement_bitflip_range
        )
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
        membership_cache = load_or_build_cf_membership_cache(
            cache_dir=args.cache_dir,
            true_order=instance.true_order,
            num_phase_bits=num_phase_bits,
            max_cache_bitstrings=args.max_cache_bitstrings,
            rebuild_cache=args.rebuild_cache,
        )
        single_bitflip_bits = (
            single_measurement_bitflip_bit_numbers(
                num_phase_bits=num_phase_bits,
                true_order=instance.true_order,
                range_name=single_bitflip_ranges or "all",
            )
            if args.single_measurement_bitflip
            or single_bitflip_ranges is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    master_rng = np.random.default_rng(args.seed)
    results = []
    for algorithm in algorithms:
        algorithm_seed = int(master_rng.integers(0, 2 ** 63 - 1))
        results.extend(
            simulate_noisy_cf_complement_fill_vs_trials(
                instance=instance,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                measurement_error_probability=p,
                trial_counts=trial_counts,
                rng=np.random.default_rng(algorithm_seed),
                membership_cache=membership_cache,
                standard_backend=args.standard_backend,
                analytic_window=args.analytic_window,
                single_measurement_bitflip=single_bitflip_bits is not None,
                single_measurement_bitflip_bits=single_bitflip_bits,
                single_measurement_bitflip_range=single_bitflip_ranges,
            )
        )
        if args.classical_error_correction:
            results.extend(
                simulate_noisy_cf_complement_fill_vs_trials(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    measurement_error_probability=p,
                    trial_counts=trial_counts,
                    rng=np.random.default_rng(algorithm_seed),
                    membership_cache=membership_cache,
                    standard_backend=args.standard_backend,
                    analytic_window=args.analytic_window,
                    classical_error_correction=True,
                    single_measurement_bitflip=single_bitflip_bits is not None,
                    single_measurement_bitflip_bits=single_bitflip_bits,
                    single_measurement_bitflip_range=single_bitflip_ranges,
                )
            )

    suffix = output_filename_segment(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        measurement_error_probability=p,
        include_classical_error_correction=args.classical_error_correction,
        single_measurement_bitflip=single_bitflip_bits is not None,
        single_measurement_bitflip_range=single_bitflip_ranges,
    )
    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"noisy_cf_complement_fill_vs_trials_{suffix}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"noisy_cf_complement_fill_vs_trials_summary_{suffix}.txt"
    )

    plot_noisy_cf_complement_fill_vs_trials(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
        trial_counts=trial_counts,
        membership_cache=membership_cache,
        log_x=args.log_x,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
        trial_counts=trial_counts,
        membership_cache=membership_cache,
        seed=args.seed,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    cache_status = "loaded" if membership_cache.from_cache else "built"
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
    print(f"Trial-count values: {', '.join(str(count) for count in trial_counts)}")
    print(f"Total bitstrings 2^t = {membership_cache.num_bitstrings:,}")
    print(f"|S| = {membership_cache.inside_count:,}")
    print(f"|S^c| = {membership_cache.outside_count:,}")
    print(f"Cache status: {cache_status}")
    if membership_cache.path is not None:
        print(f"Cache path: {membership_cache.path}")
    print(
        "Measurement error model: "
        f"{measurement_error_model_display_label(p, single_bitflip_bits is not None, single_bitflip_ranges)}"
    )
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
