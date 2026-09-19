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
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_INSTANCES_FILE,
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    DEFAULT_NUM_PHASE_BITS,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    NoisyCFMembershipResult,
    add_combined_figure_legend,
    algorithm_color_for_base,
    algorithm_display_label,
    algorithm_filename_prefix,
    algorithms_display_label,
    depolarizing_error_display_label,
    depolarizing_error_filename_segment,
    instances_file_filename_segment,
    load_instances,
    normalize_algorithms,
    normalize_error_model,
    simulate_noisy_cf_membership_vs_order,
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


def adaptive_probability_ylim(values: list[float]) -> tuple[float, float]:
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


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    instances_file: Path,
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
        f"{instances_file_filename_segment(instances_file)}"
        f"{depolarizing_error_filename_segment(error_model, error_probability)}"
        f"{depolarizing_region_filename_segment(depolarizing_region)}"
        f"_t{num_phase_bits}"
        f"{work_register_filename_segment(num_work_qubits)}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )


def plot_noisy_cf_membership_vs_order(
    output_path: Path,
    results: list[NoisyCFMembershipResult],
    algorithms: list[str],
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials_per_instance: int,
    log_x: bool = False,
    fixed_num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> None:
    """
    Plot Pr(S^c) versus true order r.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")

    error_label = depolarizing_error_display_label(error_model, error_probability)
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
        f"{algorithms_display_label(algorithms)}, {error_label}, "
        f"{depolarizing_region_display_label(depolarizing_region)}, "
        f"t = {num_phase_bits}"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=line_handles,
        measurement_handles=[],
        trials_per_instance=trials_per_instance,
        algorithm_title="Series",
        measurement_title="Depolarizing error",
        measurement_text_labels=[error_label],
        extra_text_sections=[
            (
                "Region",
                [depolarizing_region_display_label(depolarizing_region)],
            ),
            ("Work register", [work_register_display_label(fixed_num_work_qubits)]),
        ],
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
    error_model: str,
    error_probability: float,
    trials_per_instance: int,
    seed: int | None,
    fixed_num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")

    error_label = depolarizing_error_display_label(error_model, error_probability)
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )
    seed_label = "random" if seed is None else str(seed)

    lines = [
        "Depolarizing CF-Complement Membership vs Order Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"Instances file = {instances_file}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Work register = {work_register_display_label(fixed_num_work_qubits)}",
        f"Depolarizing error = {error_label}",
        (
            "Controlled-U depolarizing region = "
            f"{depolarizing_region_display_label(depolarizing_region)}"
        ),
        f"Trials per instance and algorithm = {trials_per_instance}",
        f"Seed = {seed_label}",
        "",
        "Definitions:",
        "  S = {tilde b : d(tilde b, s'/r) <= 1/(2r^2), with s'/r nearest to tilde b}",
        "  Pr(S^c) = empirical probability of landing outside S.",
        "  Depolarizing trajectory errors are sampled after controlled-U blocks.",
        "  Corrected uses the bit string among tilde b and all one-bit flips with minimum distance to its own nearest s/r.",
        "",
        "Per-order data:",
        "  algorithm | series | true r | N | a | L | Pr(S^c) | S^c count | S count | trials",
    ]

    for result in sorted(
        results,
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
            f" {result.instance.N:>5} |"
            f" {result.instance.a:>5} |"
            f" {result.num_work_qubits:>2} |"
            f" {result.outside_probability:>7.3f} |"
            f" {result.outside_trials:>9} |"
            f" {result.inside_trials:>7} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Pr(tilde b in S^c) versus true order r under "
            "controlled-unitary depolarizing errors."
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
    )
    parser.add_argument("--bits", type=int, default=DEFAULT_NUM_PHASE_BITS)
    parser.add_argument(
        "--error-model",
        choices=ERROR_MODEL_CHOICES,
        default=BLOCK_DEPOLARIZING_MODEL,
    )
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--trials-per-instance", type=int, default=100)
    parser.add_argument("--classical-error-correction", action="store_true")
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=STATEVECTOR_STANDARD_BACKEND,
        help="Backend used for standard Shor. Depolarizing errors require statevector.",
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
        if args.bits < 1:
            raise ValueError("bits must be >= 1")
        if args.trials_per_instance < 1:
            raise ValueError("trials_per_instance must be >= 1")
        instances = load_instances(args.instances_file)
        num_work_qubits = resolve_work_qubits(
            instances,
            requested_work_qubits=args.work_qubits,
            fixed_work_register=args.fixed_work_register,
        )
        validate_depolarizing_standard_backend(algorithms, args.standard_backend)
        validate_statevector_budget(
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            num_phase_bits=args.bits,
            max_N=max(instance.N for instance in instances),
            max_statevector_amplitudes=args.max_statevector_amplitudes,
            num_work_qubits=num_work_qubits,
        )
    except ValueError as exc:
        parser.error(str(exc))

    results = simulate_noisy_cf_membership_vs_order(
        instances=instances,
        algorithms=algorithms,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        trials_per_instance=args.trials_per_instance,
        rng=np.random.default_rng(args.seed),
        standard_backend=args.standard_backend,
        include_classical_error_correction=args.classical_error_correction,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
    )

    suffix = output_filename_segment(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        include_classical_error_correction=args.classical_error_correction,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
    )
    figure_output_path = (
        args.output
        if args.output is not None
        else DEFAULT_FIGURE_DIR / f"depolarizing_noisy_cf_membership_vs_order_{suffix}.png"
    )
    data_output_path = (
        DEFAULT_DATA_DIR
        / f"depolarizing_noisy_cf_membership_vs_order_summary_{suffix}.txt"
    )

    plot_noisy_cf_membership_vs_order(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        trials_per_instance=args.trials_per_instance,
        log_x=args.log_x,
        fixed_num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        trials_per_instance=args.trials_per_instance,
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
    print(f"Instances file: {args.instances_file}")
    print(f"Instances used: {len(instances)}")
    print(f"Fixed phase-estimation bits: {args.bits}")
    print(f"Work register: {work_register_display_label(num_work_qubits)}")
    print(
        "Controlled-U depolarizing region: "
        f"{depolarizing_region_display_label(args.depolarizing_region)}"
    )
    print(f"Depolarizing error: {depolarizing_error_display_label(error_model, p)}")
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(f"Trials per instance and algorithm: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
