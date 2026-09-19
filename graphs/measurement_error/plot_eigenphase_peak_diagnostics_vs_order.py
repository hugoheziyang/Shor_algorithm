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
    run_kitaev_phase_distance_trial,
    run_standard_phase_distance_trial,
    validate_measurement_error_probabilities,
)
from graphs.measurement_error.plot_cf_bound_order_recovery_vs_order import (
    apply_adaptive_probability_yaxis,
    nearest_order_phase_distance,
    normalized_cf_error,
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
    standard_statevector_allows_subdefault_phase_bits,
    standard_backend_filename_segment,
    validate_statevector_budget,
)
from shor_common.classical_post import bits_to_fraction
from shor_common.errors import format_measurement_bitflip_probability
from shor_common.shor_utils import recover_order_with_mode
from shor_standard.analytic_sampling import QPEWindowMassDiagnostics
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)


DEFAULT_INSTANCES_FILE = REPO_ROOT / "graphs" / "config" / "order_recovery_instances.json"
DEFAULT_NUM_PHASE_BITS = 29

SAME_PEAK_PROBABILITY = "same_peak"
SHIFTED_WITHIN_BOUND_PROBABILITY = "shifted_within_bound"
NOISY_PEAK_COPRIME_PROBABILITY = "noisy_peak_coprime"
RECOVERY_GIVEN_SHIFTED_WITHIN_BOUND_PROBABILITY = "recovery_given_shifted_within_bound"
PEAK_DIAGNOSTICS = (
    SAME_PEAK_PROBABILITY,
    SHIFTED_WITHIN_BOUND_PROBABILITY,
    NOISY_PEAK_COPRIME_PROBABILITY,
    RECOVERY_GIVEN_SHIFTED_WITHIN_BOUND_PROBABILITY,
)


@dataclass(frozen=True)
class EigenphasePeakDiagnosticResult:
    instance: OrderRecoveryInstance
    algorithm: str
    measurement_error_probability: float
    trials: int
    num_phase_bits: int
    same_peak_trials: int
    shifted_within_bound_trials: int
    noisy_peak_coprime_trials: int
    recovered_shifted_within_bound_trials: int
    recovered_trials: int
    standard_backend: str = ANALYTIC_STANDARD_BACKEND

    @property
    def same_peak_probability(self) -> float:
        return self.same_peak_trials / self.trials

    @property
    def shifted_within_bound_probability(self) -> float:
        return self.shifted_within_bound_trials / self.trials

    @property
    def noisy_peak_coprime_probability(self) -> float:
        return self.noisy_peak_coprime_trials / self.trials

    @property
    def recovery_probability(self) -> float:
        return self.recovered_trials / self.trials

    @property
    def recovered_given_shifted_within_bound_probability(self) -> float:
        if self.shifted_within_bound_trials == 0:
            return math.nan
        return (
            self.recovered_shifted_within_bound_trials
            / self.shifted_within_bound_trials
        )


def peak_diagnostic_display_label(diagnostic: str) -> str:
    """
    Return a human-readable label for one eigenphase-peak diagnostic.
    """
    if diagnostic == SAME_PEAK_PROBABILITY:
        return "Pr(s' = s)"
    if diagnostic == SHIFTED_WITHIN_BOUND_PROBABILITY:
        return "Pr(s' != s and A)"
    if diagnostic == NOISY_PEAK_COPRIME_PROBABILITY:
        return "Pr(gcd(s', r) = 1)"
    if diagnostic == RECOVERY_GIVEN_SHIFTED_WITHIN_BOUND_PROBABILITY:
        return "Pr(B | s' != s and A)"
    raise ValueError("unknown diagnostic")


def peak_diagnostic_filename_label(diagnostic: str) -> str:
    """
    Return a filename-safe label for one eigenphase-peak diagnostic.
    """
    if diagnostic not in PEAK_DIAGNOSTICS:
        raise ValueError("unknown diagnostic")
    return diagnostic


def result_diagnostic_value(
    result: EigenphasePeakDiagnosticResult,
    diagnostic: str,
) -> float:
    """
    Return the plotted value for one diagnostic result.
    """
    if diagnostic == SAME_PEAK_PROBABILITY:
        return result.same_peak_probability
    if diagnostic == SHIFTED_WITHIN_BOUND_PROBABILITY:
        return result.shifted_within_bound_probability
    if diagnostic == NOISY_PEAK_COPRIME_PROBABILITY:
        return result.noisy_peak_coprime_probability
    if diagnostic == RECOVERY_GIVEN_SHIFTED_WITHIN_BOUND_PROBABILITY:
        return result.recovered_given_shifted_within_bound_probability
    raise ValueError("unknown diagnostic")


def nearest_peak_numerator(bits: list[int], true_order: int) -> int:
    """
    Return the numerator s of the nearest eigenphase s/r.
    """
    _, numerator = nearest_order_phase_distance(
        phase_fraction=bits_to_fraction(bits),
        true_order=true_order,
    )
    return numerator


def noisy_phase_within_cf_bound(noisy_bits: list[int], true_order: int) -> bool:
    """
    Return True if the noisy phase lies inside the nearest-peak CF bound.
    """
    return normalized_cf_error(bits_to_fraction(noisy_bits), true_order) <= 1


def run_paired_peak_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> tuple[int, int, bool, int | None]:
    """
    Run one paired clean/noisy PE trial and return peak/recovery diagnostics.
    """
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

    clean_peak = nearest_peak_numerator(
        trial.clean_bits,
        true_order=instance.true_order,
    )
    noisy_peak = nearest_peak_numerator(
        trial.noisy_bits,
        true_order=instance.true_order,
    )
    noisy_within_bound = noisy_phase_within_cf_bound(
        trial.noisy_bits,
        true_order=instance.true_order,
    )
    recovered_order = recover_order_with_mode(
        trial.noisy_bits,
        instance.a,
        instance.N,
        postprocess_mode="one_shot",
    )

    return clean_peak, noisy_peak, noisy_within_bound, recovered_order


def simulate_peak_diagnostics_for_instance(
    instance: OrderRecoveryInstance,
    algorithm: str,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    measurement_error_probability: float,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> EigenphasePeakDiagnosticResult:
    """
    Estimate eigenphase peak transition diagnostics for one configured instance.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    same_peak_trials = 0
    shifted_within_bound_trials = 0
    noisy_peak_coprime_trials = 0
    recovered_shifted_within_bound_trials = 0
    recovered_trials = 0

    for _ in range(trials):
        clean_peak, noisy_peak, noisy_within_bound, recovered_order = (
            run_paired_peak_trial(
                instance=instance,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                measurement_error_probability=measurement_error_probability,
                rng=rng,
                standard_backend=standard_backend,
                analytic_window=analytic_window,
            )
        )
        same_peak = noisy_peak == clean_peak
        shifted_within_bound = (not same_peak) and noisy_within_bound
        recovered = recovered_order == instance.true_order

        if same_peak:
            same_peak_trials += 1
        if shifted_within_bound:
            shifted_within_bound_trials += 1
        if math.gcd(noisy_peak, instance.true_order) == 1:
            noisy_peak_coprime_trials += 1
        if recovered:
            recovered_trials += 1
        if shifted_within_bound and recovered:
            recovered_shifted_within_bound_trials += 1

    return EigenphasePeakDiagnosticResult(
        instance=instance,
        algorithm=algorithm,
        measurement_error_probability=measurement_error_probability,
        trials=trials,
        num_phase_bits=num_phase_bits,
        same_peak_trials=same_peak_trials,
        shifted_within_bound_trials=shifted_within_bound_trials,
        noisy_peak_coprime_trials=noisy_peak_coprime_trials,
        recovered_shifted_within_bound_trials=recovered_shifted_within_bound_trials,
        recovered_trials=recovered_trials,
        standard_backend=standard_backend,
    )


def measurement_probabilities_filename_segment(probabilities: list[float]) -> str:
    """
    Return a filename segment for one or more measurement bit-flip probabilities.
    """
    probabilities = validate_measurement_error_probabilities(probabilities)
    tokens = [
        format_measurement_bitflip_probability(probability).replace("-", "m")
        for probability in probabilities
    ]
    return "p_" + "_".join(tokens) + "_"


def diagnostic_output_path(
    output_path: Path | None,
    diagnostic: str,
    algorithms: list[str],
    standard_backend: str,
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
) -> Path:
    """
    Return the output figure path for one diagnostic plot.
    """
    diagnostic_label = peak_diagnostic_filename_label(diagnostic)
    if output_path is not None:
        return output_path.with_name(
            f"{output_path.stem}_{diagnostic_label}{output_path.suffix}"
        )

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
        / "figures"
        / (
            "eigenphase_peak_"
            f"{diagnostic_label}_vs_order_"
            f"{algorithm_prefix}{probability_segment}t{num_phase_bits}.png"
        )
    )


def plot_peak_diagnostic_vs_order(
    output_path: Path,
    results: list[EigenphasePeakDiagnosticResult],
    algorithms: list[str],
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    diagnostic: str,
    trials_per_instance: int,
    log_x: bool = False,
) -> None:
    """
    Plot one eigenphase-peak diagnostic with probabilities overlaid.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    measurement_error_probabilities = validate_measurement_error_probabilities(
        measurement_error_probabilities
    )
    peak_diagnostic_filename_label(diagnostic)

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
            diagnostic_results = sorted(
                [
                    result
                    for result in results
                    if result.algorithm == algorithm
                    and result.measurement_error_probability == probability
                ],
                key=lambda result: result.instance.true_order,
            )
            y_values = [
                result_diagnostic_value(result, diagnostic)
                for result in diagnostic_results
            ]
            plotted_values.extend(y_values)
            ax.plot(
                [result.instance.true_order for result in diagnostic_results],
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
        f"{peak_diagnostic_display_label(diagnostic)} vs True Order\n"
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
            label=measurement_probability_display_label(probability),
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
            ("Quantity", [peak_diagnostic_display_label(diagnostic)])
        ],
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[EigenphasePeakDiagnosticResult],
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
    Write run metadata and per-order eigenphase-peak diagnostic data.
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
        "Eigenphase-Peak Diagnostic Summary",
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
        "  s = nearest eigenphase numerator to the clean bit string",
        "  s' = nearest eigenphase numerator to the noisy bit string",
        "  A = noisy phase is inside the CF sufficient bound around s'/r",
        "  B = recovered_order == true r after one-shot continued fractions",
        "",
        "Per-order data:",
        "  algorithm | p | true r | N | a | Pr(s'=s) | Pr(s'!=s and A) | Pr(gcd(s',r)=1) | Pr(B | s'!=s and A) | Pr(B) | same count | shifted A count | coprime s' count | shifted A and B count | B count | trials",
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
            f" {result.same_peak_probability:>9.3f} |"
            f" {result.shifted_within_bound_probability:>16.3f} |"
            f" {result.noisy_peak_coprime_probability:>18.3f} |"
            f" {result.recovered_given_shifted_within_bound_probability:>22.3f} |"
            f" {result.recovery_probability:>6.3f} |"
            f" {result.same_peak_trials:>10} |"
            f" {result.shifted_within_bound_trials:>15} |"
            f" {result.noisy_peak_coprime_trials:>16} |"
            f" {result.recovered_shifted_within_bound_trials:>21} |"
            f" {result.recovered_trials:>7} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for eigenphase peak diagnostics.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot clean/noisy nearest-eigenphase peak diagnostics versus "
            "true order r."
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
            "Overlay independent per-bit measurement flip probabilities in "
            "each diagnostic figure. This is mutually exclusive with --p."
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
            "Optional custom output path. The diagnostic token is appended to "
            "the output stem."
        ),
    )
    return parser


def main() -> None:
    """
    Run the eigenphase peak diagnostic plotting script.
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
    results: list[EigenphasePeakDiagnosticResult] = []
    for measurement_error_probability in measurement_error_probabilities:
        for algorithm in algorithms:
            for instance in instances:
                instance_rng = np.random.default_rng(
                    int(master_rng.integers(0, 2 ** 63 - 1))
                )
                results.append(
                    simulate_peak_diagnostics_for_instance(
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
    probability_segment = measurement_probabilities_filename_segment(
        measurement_error_probabilities
    )
    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / (
            "eigenphase_peak_diagnostics_vs_order_summary_"
            f"{algorithm_prefix}{probability_segment}t{args.bits}.txt"
        )
    )

    figure_output_paths = []
    for diagnostic in PEAK_DIAGNOSTICS:
        figure_output_path = diagnostic_output_path(
            output_path=args.output,
            diagnostic=diagnostic,
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            instances_file=args.instances_file,
            num_phase_bits=args.bits,
            measurement_error_probabilities=measurement_error_probabilities,
        )
        plot_peak_diagnostic_vs_order(
            output_path=figure_output_path,
            results=results,
            algorithms=algorithms,
            num_phase_bits=args.bits,
            measurement_error_probabilities=measurement_error_probabilities,
            diagnostic=diagnostic,
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
    print(f"Trials per instance and algorithm: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
