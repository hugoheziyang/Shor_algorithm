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
from matplotlib.lines import Line2D
import numpy as np
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.depolarizing_error.cf_common import (
    ALGORITHM_CHOICES,
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    NoisyCFMembershipResult,
    add_combined_figure_legend,
    algorithm_display_label,
    algorithm_filename_prefix,
    algorithms_display_label,
    build_fixed_instance,
    cf_sufficient_bound_distance,
    depolarizing_error_display_label,
    depolarizing_error_filename_segment,
    normalize_algorithms,
    normalize_error_model,
    simulate_noisy_cf_membership,
    validate_depolarizing_standard_backend,
    validate_error_probabilities,
    validate_statevector_budget,
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


S_COMPLEMENT_CATEGORY = "S^c"
S_CATEGORY = "S"
CATEGORIES = (S_COMPLEMENT_CATEGORY, S_CATEGORY)


def result_category_probability(
    result: NoisyCFMembershipResult,
    category: str,
) -> float:
    if category == S_COMPLEMENT_CATEGORY:
        return result.outside_probability
    if category == S_CATEGORY:
        return result.inside_probability
    raise ValueError("unknown category")


def result_series_label(result: NoisyCFMembershipResult) -> str:
    suffix = "corrected" if result.classical_error_correction else "noisy"
    return f"{algorithm_display_label(result.algorithm)} {suffix}"


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
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{'' if standard_backend == STATEVECTOR_STANDARD_BACKEND else standard_backend + '_'}"
        f"N{N}_a{a}_t{num_phase_bits}_"
        f"{depolarizing_error_filename_segment(error_model, error_probability)}"
        f"{depolarizing_region_filename_segment(depolarizing_region)}"
        f"{work_register_filename_segment(num_work_qubits)}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )


def plot_noisy_cf_membership_bar(
    output_path: Path,
    results: list[NoisyCFMembershipResult],
    algorithms: list[str],
) -> None:
    """
    Plot grouped bars for Pr(tilde b in S^c) and Pr(tilde b in S).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    error_model = results[0].error_model
    probability = results[0].error_probability
    error_label = depolarizing_error_display_label(error_model, probability)
    num_phase_bits = results[0].num_phase_bits
    trials = results[0].trials
    bound = float(cf_sufficient_bound_distance(instance.true_order))
    result_keys = [
        (result.algorithm, result.classical_error_correction)
        for result in results
    ]
    result_by_key = {
        (result.algorithm, result.classical_error_correction): result
        for result in results
    }

    colors = {
        KITAEV_ALGORITHM: "#1F6F78",
        STANDARD_ALGORITHM: "#A23B2A",
    }
    hatches = {
        False: "",
        True: "///",
    }
    x_values = np.arange(len(CATEGORIES))
    bar_width = min(0.34, 0.72 / max(1, len(result_keys)))
    offsets = (
        np.linspace(
            -bar_width * (len(result_keys) - 1) / 2,
            bar_width * (len(result_keys) - 1) / 2,
            len(result_keys),
        )
        if len(result_keys) > 1
        else np.array([0.0])
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    legend_handles = []

    for offset, key in zip(offsets, result_keys):
        algorithm, correction_enabled = key
        result = result_by_key[key]
        heights = [
            result_category_probability(result, category)
            for category in CATEGORIES
        ]
        bars = ax.bar(
            x_values + offset,
            heights,
            width=bar_width,
            color=colors[algorithm],
            alpha=0.88,
            hatch=hatches[correction_enabled],
            edgecolor="#222222" if correction_enabled else "none",
            linewidth=0.7 if correction_enabled else 0.0,
            label=result_series_label(result),
        )
        for bar, height in zip(bars, heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.018,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        legend_handle = Line2D(
            [0],
            [0],
            color=colors[algorithm],
            marker="s",
            linestyle="None",
            markersize=8,
            label=result_series_label(result),
        )
        legend_handle.set_markeredgecolor("#222222")
        legend_handle.set_markeredgewidth(0.8 if correction_enabled else 0.0)
        legend_handles.append(legend_handle)

    ax.set_xticks(x_values)
    ax.set_xticklabels(["S^c", "S"])
    ax.set_xlabel("Membership category")
    ax.set_ylabel("Empirical proportion")
    ax.set_ylim(0.0, 1.08)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_title(
        "Noisy CF-Bound Membership\n"
        f"{algorithms_display_label(algorithms)} for N = {instance.N}, "
        f"a = {instance.a}, r = {instance.true_order}, "
        f"{error_label}"
    )
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=legend_handles,
        measurement_handles=[],
        trials_per_instance=trials,
        algorithm_title="Algorithm",
        measurement_title="Depolarizing error",
        measurement_text_labels=[error_label],
        extra_text_sections=[
            (
                "Instance",
                [
                    f"N = {instance.N}, a = {instance.a}",
                    f"r = {instance.true_order}, t = {num_phase_bits}",
                    f"L = {results[0].num_work_qubits}",
                ],
            ),
            (
                "Set S",
                [
                    "d(bits, nearest s/r) <=",
                    f"1/(2r^2) = {bound:.6f}",
                ],
            ),
        ],
        trials_title="Trials per algorithm",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[NoisyCFMembershipResult],
    algorithms: list[str],
    seed: int | None,
    fixed_num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
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

    lines = [
        "Depolarizing CF-Bound Membership Summary",
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
        f"CF sufficient bound 1/(2r^2) = {bound:.12f}",
        f"Seed = {seed_label}",
        "",
        "Definitions:",
        "  S = {tilde b : d(tilde b, s'/r) <= 1/(2r^2), with s'/r nearest to tilde b}",
        "  Depolarizing trajectory errors are sampled after controlled-U blocks.",
        "  Corrected uses the bit string among tilde b and all one-bit flips with minimum distance to its own nearest exact order eigenphase.",
        "  S^c is the complement of S.",
        "",
        "Per-series data:",
        "  algorithm | series | L | Pr(S^c) | Pr(S) | S^c count | S count | trials",
    ]

    for result in sorted(
        results,
        key=lambda item: (item.algorithm, item.classical_error_correction),
    ):
        series = "corrected" if result.classical_error_correction else "noisy"
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {series:>9} |"
            f" {result.num_work_qubits:>2} |"
            f" {result.outside_probability:>7.3f} |"
            f" {result.inside_probability:>5.3f} |"
            f" {result.outside_trials:>9} |"
            f" {result.inside_trials:>7} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Pr(tilde b in S) and Pr(tilde b in S^c) for one fixed "
            "(N, a) under controlled-unitary depolarizing errors."
        )
    )
    parser.add_argument("N", type=int, help="Modulus to analyze")
    parser.add_argument("--a", type=int, required=True, help="Fixed coprime base")
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
        "--error-model",
        choices=ERROR_MODEL_CHOICES,
        default=BLOCK_DEPOLARIZING_MODEL,
        help="Depolarizing model to apply",
    )
    parser.add_argument("--p", type=float, default=0.1, help="Depolarizing probability")
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=STATEVECTOR_STANDARD_BACKEND,
        help="Backend used for standard Shor. Depolarizing errors require statevector.",
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument(
        "--classical-error-correction",
        action="store_true",
        help="Also plot the simulator-side one-bit minimum-distance correction diagnostic.",
    )
    add_depolarizing_region_argument(parser)
    parser.add_argument(
        "--max-statevector-amplitudes",
        type=int,
        default=DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
        help="Safety limit for standard statevector backend allocations. Use 0 to disable.",
    )
    add_work_register_arguments(parser)
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
        if args.trials < 1:
            raise ValueError("trials must be >= 1")
        if args.bits is not None and args.bits < 1:
            raise ValueError("bits must be >= 1")
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
    except ValueError as exc:
        parser.error(str(exc))

    master_rng = np.random.default_rng(args.seed)
    results = []
    for algorithm in algorithms:
        algorithm_seed = int(master_rng.integers(0, 2 ** 63 - 1))
        results.append(
            simulate_noisy_cf_membership(
                instance=instance,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                error_model=error_model,
                error_probability=p,
                trials=args.trials,
                rng=np.random.default_rng(algorithm_seed),
                standard_backend=args.standard_backend,
                num_work_qubits=num_work_qubits,
                depolarizing_region=args.depolarizing_region,
            )
        )
        if args.classical_error_correction:
            results.append(
                simulate_noisy_cf_membership(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    error_model=error_model,
                    error_probability=p,
                    trials=args.trials,
                    rng=np.random.default_rng(algorithm_seed),
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
        else DEFAULT_FIGURE_DIR / f"depolarizing_noisy_cf_membership_bar_{suffix}.png"
    )
    data_output_path = (
        DEFAULT_DATA_DIR / f"depolarizing_noisy_cf_membership_bar_summary_{suffix}.txt"
    )

    plot_noisy_cf_membership_bar(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
        seed=args.seed,
        fixed_num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
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
    print(f"Depolarizing error: {depolarizing_error_display_label(error_model, p)}")
    print(
        "Controlled-U depolarizing region: "
        f"{depolarizing_region_display_label(args.depolarizing_region)}"
    )
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(f"Trials per algorithm: {args.trials}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
