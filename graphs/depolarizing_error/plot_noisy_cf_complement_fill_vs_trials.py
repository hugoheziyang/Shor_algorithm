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
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.depolarizing_error.cf_common import (
    ALGORITHM_CHOICES,
    CFMembershipCache,
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_MAX_CACHE_BITSTRINGS,
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    NoisyCFComplementFillTrialSweepResult,
    add_combined_figure_legend,
    algorithm_color_for_base,
    algorithm_display_label,
    algorithm_filename_prefix,
    algorithms_display_label,
    build_fixed_instance,
    cf_sufficient_bound_distance,
    depolarizing_error_display_label,
    depolarizing_error_filename_segment,
    load_or_build_cf_membership_cache,
    normalize_algorithms,
    normalize_error_model,
    simulate_noisy_cf_complement_fill_vs_trials,
    validate_depolarizing_standard_backend,
    validate_error_probabilities,
    validate_statevector_budget,
    validate_trial_counts,
)
from graphs.depolarizing_error.work_register import (
    add_work_register_arguments,
    resolve_work_qubits,
    work_register_display_label,
    work_register_filename_segment,
)
from shor_common.shor_utils import default_num_phase_bits
from shor_standard.backends import (
    STATEVECTOR_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
)
from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    ERROR_MODEL_CHOICES,
    add_depolarizing_region_argument,
    depolarizing_region_display_label,
    depolarizing_region_filename_segment,
)
from shor_common.errors import ControlledUnitaryDepolarizingRangeSpec


DEFAULT_TRIAL_COUNTS = [10, 20, 50, 100, 200, 500, 1000]


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    N: int,
    a: int,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    include_classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> str:
    algorithms = normalize_algorithms(algorithms)
    backend_token = "" if standard_backend == STATEVECTOR_STANDARD_BACKEND else f"{standard_backend}_"
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{backend_token}"
        f"N{N}_a{a}_t{num_phase_bits}_"
        f"{depolarizing_error_filename_segment(error_model, error_probability)}"
        f"{depolarizing_region_filename_segment(depolarizing_region)}"
        f"{work_register_filename_segment(num_work_qubits)}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )


def result_series_label(result: NoisyCFComplementFillTrialSweepResult) -> str:
    suffix = "corrected" if result.classical_error_correction else "noisy"
    return f"{algorithm_display_label(result.algorithm)} {suffix}"


def adaptive_fill_sweep_ylim(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, 1.0
    max_value = max(finite_values)
    if max_value <= 0.0:
        return 0.0, 0.05
    upper = min(1.0, max(0.02, max_value * 1.18))
    return 0.0, upper


def fill_tick_format(y_upper: float) -> str:
    if y_upper <= 0.05:
        return "%.4f"
    if y_upper <= 0.2:
        return "%.3f"
    return "%.2f"


def plot_noisy_cf_complement_fill_vs_trials(
    output_path: Path,
    results: list[NoisyCFComplementFillTrialSweepResult],
    algorithms: list[str],
    trial_counts: list[int],
    membership_cache: CFMembershipCache,
    log_x: bool = False,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    trial_counts = validate_trial_counts(trial_counts)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    error_model = results[0].error_model
    probability = results[0].error_probability
    error_label = depolarizing_error_display_label(error_model, probability)
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
        f"{error_label}, {depolarizing_region_display_label(depolarizing_region)}"
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
        measurement_title="Depolarizing error",
        measurement_text_labels=[error_label],
        extra_text_sections=[
            (
                "Region",
                [depolarizing_region_display_label(depolarizing_region)],
            ),
            (
                "Instance",
                [
                    f"N = {instance.N}, a = {instance.a}",
                    f"r = {instance.true_order}, t = {num_phase_bits}",
                    f"L = {results[0].num_work_qubits}",
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
    fixed_num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    trial_counts = validate_trial_counts(trial_counts)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    error_model = results[0].error_model
    probability = results[0].error_probability
    error_label = depolarizing_error_display_label(error_model, probability)
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
    cache_path_label = "none" if membership_cache.path is None else str(membership_cache.path)

    lines = [
        "Depolarizing CF-Complement Fill vs Trials Summary",
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
        f"Work register = {work_register_display_label(fixed_num_work_qubits)}",
        f"Depolarizing error = {error_label}",
        (
            "Controlled-U depolarizing region = "
            f"{depolarizing_region_display_label(depolarizing_region)}"
        ),
        f"Trial-count values = {', '.join(str(count) for count in trial_counts)}",
        f"CF sufficient bound 1/(2r^2) = {bound:.12f}",
        f"Total bitstrings 2^t = {membership_cache.num_bitstrings}",
        f"|S| = {membership_cache.inside_count}",
        f"|S^c| = {membership_cache.outside_count}",
        f"Cache status = {cache_label}",
        f"Cache path = {cache_path_label}",
        f"Seed = {seed_label}",
        "",
        "Definitions:",
        "  S = {tilde b : d(tilde b, s'/r) <= 1/(2r^2), with s'/r nearest to tilde b}",
        "  Depolarizing trajectory errors are sampled after controlled-U blocks.",
        "  Corrected uses the bit string among tilde b and all one-bit flips with minimum distance to its own nearest s/r.",
        "  S^c fill = |unique noisy bitstrings observed in S^c| / |S^c|.",
        "  Each trial-count point is estimated from an independent run.",
        "",
        "Per-trial-count data:",
        "  algorithm | series | trials | L | S^c fill | unique S^c hits | |S^c| | Pr(S^c) | S^c trial hits",
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
            f" {result.num_work_qubits:>2} |"
            f" {result.outside_fill_fraction:>10.6f} |"
            f" {result.outside_unique_hits:>15} |"
            f" {result.total_outside_bitstrings:>5} |"
            f" {result.outside_trial_probability:>7.3f} |"
            f" {result.outside_trial_hits:>14}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot S^c coverage versus trial count under controlled-unitary "
            "depolarizing errors."
        )
    )
    parser.add_argument("N", type=int, help="Modulus to analyze")
    parser.add_argument("--a", type=int, required=True, help="Fixed coprime base")
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
    )
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument(
        "--error-model",
        choices=ERROR_MODEL_CHOICES,
        default=BLOCK_DEPOLARIZING_MODEL,
    )
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument(
        "--trial-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRIAL_COUNTS),
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=STATEVECTOR_STANDARD_BACKEND,
        help="Backend used for standard Shor. Depolarizing errors require statevector.",
    )
    parser.add_argument("--classical-error-correction", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--max-cache-bitstrings",
        type=int,
        default=DEFAULT_MAX_CACHE_BITSTRINGS,
    )
    parser.add_argument(
        "--max-statevector-amplitudes",
        type=int,
        default=DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    )
    add_work_register_arguments(parser)
    add_depolarizing_region_argument(parser)
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    start_time = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        algorithms = normalize_algorithms(args.algorithm)
        error_model = normalize_error_model(args.error_model)
        p = validate_error_probabilities([args.p])[0]
        trial_counts = validate_trial_counts(args.trial_counts)
        if args.bits is not None and args.bits < 1:
            raise ValueError("bits must be >= 1")
        if args.max_cache_bitstrings < 0:
            raise ValueError("max_cache_bitstrings must be >= 0")
        instance = build_fixed_instance(args.N, args.a)
        num_phase_bits = default_num_phase_bits(args.N) if args.bits is None else args.bits
        num_work_qubits = resolve_work_qubits(
            [instance],
            requested_work_qubits=args.work_qubits,
            fixed_work_register=args.fixed_work_register,
        )
        validate_depolarizing_standard_backend(algorithms, args.standard_backend)
        validate_statevector_budget(
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            num_phase_bits=num_phase_bits,
            max_N=args.N,
            max_statevector_amplitudes=args.max_statevector_amplitudes,
            num_work_qubits=num_work_qubits,
        )
        membership_cache = load_or_build_cf_membership_cache(
            cache_dir=args.cache_dir,
            true_order=instance.true_order,
            num_phase_bits=num_phase_bits,
            max_cache_bitstrings=args.max_cache_bitstrings,
            rebuild_cache=args.rebuild_cache,
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
                error_model=error_model,
                error_probability=p,
                trial_counts=trial_counts,
                rng=np.random.default_rng(algorithm_seed),
                membership_cache=membership_cache,
                standard_backend=args.standard_backend,
                num_work_qubits=num_work_qubits,
                depolarizing_region=args.depolarizing_region,
            )
        )
        if args.classical_error_correction:
            results.extend(
                simulate_noisy_cf_complement_fill_vs_trials(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    error_model=error_model,
                    error_probability=p,
                    trial_counts=trial_counts,
                    rng=np.random.default_rng(algorithm_seed),
                    membership_cache=membership_cache,
                    standard_backend=args.standard_backend,
                    classical_error_correction=True,
                    num_work_qubits=num_work_qubits,
                    depolarizing_region=args.depolarizing_region,
                )
            )

    suffix = output_filename_segment(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        error_model=error_model,
        error_probability=p,
        include_classical_error_correction=args.classical_error_correction,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
    )
    figure_output_path = (
        args.output
        if args.output is not None
        else DEFAULT_FIGURE_DIR
        / f"depolarizing_noisy_cf_complement_fill_vs_trials_{suffix}.png"
    )
    data_output_path = (
        DEFAULT_DATA_DIR
        / f"depolarizing_noisy_cf_complement_fill_vs_trials_summary_{suffix}.txt"
    )

    plot_noisy_cf_complement_fill_vs_trials(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
        trial_counts=trial_counts,
        membership_cache=membership_cache,
        log_x=args.log_x,
        depolarizing_region=args.depolarizing_region,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
        trial_counts=trial_counts,
        membership_cache=membership_cache,
        seed=args.seed,
        fixed_num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    cache_status = "loaded" if membership_cache.from_cache else "built"
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"Algorithm: {algorithms_display_label(algorithms)}")
    if STANDARD_ALGORITHM in algorithms:
        print(f"Standard backend: {args.standard_backend}")
    print(f"N = {args.N}")
    print(f"a = {args.a}")
    print(f"True order r = {instance.true_order}")
    print(f"Phase-estimation bits = {num_phase_bits}")
    print(f"Work register: {work_register_display_label(num_work_qubits)}")
    print(
        "Controlled-U depolarizing region: "
        f"{depolarizing_region_display_label(args.depolarizing_region)}"
    )
    print(f"Trial-count values: {', '.join(str(count) for count in trial_counts)}")
    print(f"Total bitstrings 2^t = {membership_cache.num_bitstrings:,}")
    print(f"|S| = {membership_cache.inside_count:,}")
    print(f"|S^c| = {membership_cache.outside_count:,}")
    print(f"Cache status: {cache_status}")
    if membership_cache.path is not None:
        print(f"Cache path: {membership_cache.path}")
    print(f"Depolarizing error: {depolarizing_error_display_label(error_model, p)}")
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
