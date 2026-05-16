from __future__ import annotations

import argparse
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
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.measurement_error.phase_distance_vs_order_common import (
    measurement_probability_display_label,
    run_kitaev_phase_distance_trial,
    run_standard_phase_distance_trial,
    validate_measurement_error_probability,
)
from graphs.measurement_error.plot_cf_bound_order_recovery_vs_order import within_cf_sufficient_bound
from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_error_probability import (
    build_fixed_instance,
)
from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_order import (
    cf_sufficient_bound_distance,
    minimum_single_bit_flip_nearest_peak_distance,
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
from shor_common.errors import format_measurement_bitflip_probability
from shor_common.shor_utils import default_num_phase_bits
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)


S_COMPLEMENT_CATEGORY = "S^c"
S_CATEGORY = "S"
CATEGORIES = (S_COMPLEMENT_CATEGORY, S_CATEGORY)


@dataclass(frozen=True)
class NoisyCFMembershipResult:
    instance: OrderRecoveryInstance
    algorithm: str
    measurement_error_probability: float
    trials: int
    num_phase_bits: int
    inside_trials: int
    standard_backend: str = ANALYTIC_STANDARD_BACKEND
    classical_error_correction: bool = False

    @property
    def outside_trials(self) -> int:
        return self.trials - self.inside_trials

    @property
    def inside_probability(self) -> float:
        return self.inside_trials / self.trials

    @property
    def outside_probability(self) -> float:
        return self.outside_trials / self.trials


def run_noisy_membership_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    classical_error_correction: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
) -> bool:
    """
    Return whether one noisy bit string tilde b lies in S.
    """
    if algorithm == KITAEV_ALGORITHM:
        trial = run_kitaev_phase_distance_trial(
            instance=instance,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            single_measurement_bitflip_bits=single_measurement_bitflip_bits,
        )
    elif algorithm == STANDARD_ALGORITHM:
        trial = run_standard_phase_distance_trial(
            instance=instance,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
            single_measurement_bitflip_bits=single_measurement_bitflip_bits,
        )
    else:
        raise ValueError("unknown algorithm")

    if classical_error_correction:
        _, _, bits = minimum_single_bit_flip_nearest_peak_distance(
            trial.noisy_bits,
            true_order=instance.true_order,
        )
    else:
        bits = trial.noisy_bits

    return within_cf_sufficient_bound(
        bits,
        true_order=instance.true_order,
    )


def simulate_noisy_cf_membership(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    trials: int,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    classical_error_correction: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
) -> NoisyCFMembershipResult:
    """
    Estimate Pr(tilde b in S) and Pr(tilde b in S^c) for one algorithm.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    inside_trials = 0
    for _ in range(trials):
        if run_noisy_membership_trial(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
            classical_error_correction=classical_error_correction,
            single_measurement_bitflip_bits=single_measurement_bitflip_bits,
        ):
            inside_trials += 1

    return NoisyCFMembershipResult(
        instance=instance,
        algorithm=algorithm,
        measurement_error_probability=measurement_error_probability,
        trials=trials,
        num_phase_bits=num_phase_bits,
        inside_trials=inside_trials,
        standard_backend=standard_backend,
        classical_error_correction=classical_error_correction,
    )


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    N: int,
    a: int,
    num_phase_bits: int,
    measurement_error_probability: float,
    include_classical_error_correction: bool = False,
) -> str:
    """
    Return the filename segment for one noisy-membership bar chart.
    """
    algorithms = normalize_algorithms(algorithms)
    probability_token = format_measurement_bitflip_probability(
        measurement_error_probability
    ).replace("-", "m")
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"N{N}_a{a}_t{num_phase_bits}_p_{probability_token}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )


def result_category_probability(
    result: NoisyCFMembershipResult,
    category: str,
) -> float:
    """
    Return the plotted probability for one membership category.
    """
    if category == S_COMPLEMENT_CATEGORY:
        return result.outside_probability
    if category == S_CATEGORY:
        return result.inside_probability
    raise ValueError("unknown category")


def result_series_label(result: NoisyCFMembershipResult) -> str:
    """
    Return the plotted label for one noisy/corrected result.
    """
    suffix = "corrected" if result.classical_error_correction else "noisy"
    return f"{algorithm_display_label(result.algorithm)} {suffix}"


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
    probability = results[0].measurement_error_probability
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
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[algorithm],
                marker="s",
                linestyle="None",
                markersize=8,
                label=result_series_label(result),
            )
        )
        legend_handles[-1].set_markeredgecolor("#222222")
        legend_handles[-1].set_markeredgewidth(0.8 if correction_enabled else 0.0)

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
        f"{measurement_probability_display_label(probability)}"
    )
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=legend_handles,
        measurement_handles=[],
        trials_per_instance=trials,
        algorithm_title="Algorithm",
        measurement_title="Measurement p",
        measurement_text_labels=[measurement_probability_display_label(probability)],
        extra_text_sections=[
            (
                "Instance",
                [
                    f"N = {instance.N}, a = {instance.a}",
                    f"r = {instance.true_order}, t = {num_phase_bits}",
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
) -> None:
    """
    Write metadata and per-algorithm noisy-membership proportions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    probability = results[0].measurement_error_probability
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
        "Noisy CF-Bound Membership Summary",
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
        f"Measurement bit-flip probability = {measurement_probability_display_label(probability)}",
        f"CF sufficient bound 1/(2r^2) = {bound:.12f}",
        f"Seed = {seed_label}",
        "",
        "Definitions:",
        "  S = {tilde b : d(tilde b, s'/r) <= 1/(2r^2), with s'/r nearest to tilde b}",
        "  Corrected uses the bit string among tilde b and all one-bit flips with minimum distance to its own nearest s/r.",
        "  S^c is the complement of S.",
        "",
        "Per-series data:",
        "  algorithm | series | Pr(S^c) | Pr(S) | S^c count | S count | trials",
    ]

    for result in sorted(
        results,
        key=lambda item: (item.algorithm, item.classical_error_correction),
    ):
        series = "corrected" if result.classical_error_correction else "noisy"
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {series:>9} |"
            f" {result.outside_probability:>7.3f} |"
            f" {result.inside_probability:>5.3f} |"
            f" {result.outside_trials:>9} |"
            f" {result.inside_trials:>7} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for noisy CF-membership bar charts.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot Pr(tilde b in S) and Pr(tilde b in S^c) for one fixed "
            "(N, a) and measurement bit-flip probability."
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
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=ANALYTIC_STANDARD_BACKEND,
        help=(
            "Backend used for standard Shor bars. The analytic backend is "
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
        default=1000,
        help="Number of noisy trials per algorithm",
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
    Run the noisy CF-membership bar chart script.
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
        p = validate_measurement_error_probability(args.p)
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
                measurement_error_probability=p,
                trials=args.trials,
                rng=np.random.default_rng(algorithm_seed),
                standard_backend=args.standard_backend,
                analytic_window=args.analytic_window,
            )
        )
        if args.classical_error_correction:
            results.append(
                simulate_noisy_cf_membership(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    measurement_error_probability=p,
                    trials=args.trials,
                    rng=np.random.default_rng(algorithm_seed),
                    standard_backend=args.standard_backend,
                    analytic_window=args.analytic_window,
                    classical_error_correction=True,
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
    )
    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"noisy_cf_membership_bar_{suffix}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"noisy_cf_membership_bar_summary_{suffix}.txt"
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
    print(f"Measurement bit-flip probability: {measurement_probability_display_label(p)}")
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(f"Trials per algorithm: {args.trials}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
