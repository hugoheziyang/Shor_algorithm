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


from graphs.depolarizing_error.cf_common import (
    ALGORITHM_CHOICES,
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    DEFAULT_NUM_PHASE_BITS,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    NoisyCFMembershipResult,
    add_combined_figure_legend,
    algorithm_display_label,
    algorithm_filename_prefix,
    compact_filename_segment,
    depolarizing_error_display_label,
    depolarizing_error_filename_segment,
    load_instances,
    normalize_error_model,
    simulate_noisy_cf_membership_vs_order,
    simulate_noisy_coprime_cf_membership_vs_order,
    validate_depolarizing_standard_backend,
    validate_error_probabilities,
    validate_statevector_budget,
)
from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    ERROR_MODEL_CHOICES,
    add_depolarizing_region_argument,
    depolarizing_region_display_label,
    depolarizing_region_filename_segment,
)
from graphs.depolarizing_error.plot_noisy_cf_membership_vs_order_by_instance_file import (
    adaptive_probability_ylim,
    instance_files_filename_segment,
    resolve_work_qubits_by_file,
    result_series_label,
    validate_instance_file_labels,
    work_register_overlay_display_label,
    work_register_overlay_filename_segment,
)
from graphs.depolarizing_error.work_register import (
    add_work_register_arguments,
    work_register_display_label,
)
from shor_standard.backends import (
    STATEVECTOR_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
)
from shor_common.errors import ControlledUnitaryDepolarizingRangeSpec


ORDINARY_MEMBERSHIP = "ordinary"
COPRIME_MEMBERSHIP = "coprime"
MEMBERSHIP_CHOICES = (ORDINARY_MEMBERSHIP, COPRIME_MEMBERSHIP)


@dataclass(frozen=True)
class InstanceFileMembershipExcessSweep:
    instances_file: Path
    label: str
    noisy_results: list[NoisyCFMembershipResult]
    baseline_results: list[NoisyCFMembershipResult]
    fixed_num_work_qubits: int | None = None

    @property
    def instance_count(self) -> int:
        return len(
            {
                (result.instance.N, result.instance.a)
                for result in self.noisy_results
            }
        )


def membership_display_label(membership: str) -> str:
    if membership == COPRIME_MEMBERSHIP:
        return "S_coprime"
    if membership == ORDINARY_MEMBERSHIP:
        return "S"
    raise ValueError("unknown membership set")


def membership_filename_segment(membership: str) -> str:
    if membership == COPRIME_MEMBERSHIP:
        return "scoprime"
    if membership == ORDINARY_MEMBERSHIP:
        return "s"
    raise ValueError("unknown membership set")


def result_key(
    result: NoisyCFMembershipResult,
) -> tuple[str, bool, int, int, int]:
    return (
        result.algorithm,
        result.classical_error_correction,
        result.instance.N,
        result.instance.a,
        result.instance.true_order,
    )


def paired_excess_values(
    sweep: InstanceFileMembershipExcessSweep,
    algorithm: str,
    correction_enabled: bool,
) -> list[tuple[NoisyCFMembershipResult, NoisyCFMembershipResult, float]]:
    baseline_by_key = {
        result_key(result): result
        for result in sweep.baseline_results
    }
    pairs = []
    for result in sorted(
        sweep.noisy_results,
        key=lambda item: item.instance.true_order,
    ):
        if result.algorithm != algorithm:
            continue
        if result.classical_error_correction != correction_enabled:
            continue
        baseline = baseline_by_key.get(result_key(result))
        if baseline is None:
            raise ValueError("baseline result missing for noisy result")
        pairs.append(
            (
                result,
                baseline,
                result.outside_probability - baseline.outside_probability,
            )
        )
    return pairs


def adaptive_delta_ylim(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return -0.05, 0.05
    low = min(finite_values)
    high = max(finite_values)
    if low == high:
        margin = 0.05 if low == 0 else abs(low) * 0.15
    else:
        margin = max(0.03, (high - low) * 0.12)
    return max(-1.0, low - margin), min(1.0, high + margin)


def output_filename_segment(
    algorithm: str,
    standard_backend: str,
    instances_files: list[Path],
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    membership: str,
    sweeps: list[InstanceFileMembershipExcessSweep],
    include_classical_error_correction: bool = False,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> str:
    backend_token = (
        ""
        if algorithm != STANDARD_ALGORITHM or standard_backend == STATEVECTOR_STANDARD_BACKEND
        else f"{standard_backend}_"
    )
    segment = (
        f"{algorithm_filename_prefix([algorithm])}"
        f"{backend_token}"
        f"{instance_files_filename_segment(instances_files)}_"
        f"{membership_filename_segment(membership)}_excess_"
        f"{depolarizing_error_filename_segment(error_model, error_probability)}"
        f"{depolarizing_region_filename_segment(depolarizing_region)}"
        f"_minus_p0_t{num_phase_bits}"
        f"{work_register_overlay_filename_segment(sweeps)}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )
    return compact_filename_segment(segment, max_length=220)


def simulate_membership_excess_sweeps(
    instances_files: list[Path],
    labels: list[str],
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials_per_instance: int,
    rng: np.random.Generator,
    membership: str = COPRIME_MEMBERSHIP,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    include_classical_error_correction: bool = False,
    num_work_qubits_by_file: list[int | None] | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> list[InstanceFileMembershipExcessSweep]:
    if len(instances_files) != len(labels):
        raise ValueError("instances files and labels must have the same length")
    if num_work_qubits_by_file is None:
        num_work_qubits_by_file = [None for _ in instances_files]
    if len(num_work_qubits_by_file) != len(instances_files):
        raise ValueError("work-register settings must match instances files")
    if membership not in MEMBERSHIP_CHOICES:
        raise ValueError("unknown membership set")

    simulator = (
        simulate_noisy_coprime_cf_membership_vs_order
        if membership == COPRIME_MEMBERSHIP
        else simulate_noisy_cf_membership_vs_order
    )

    sweeps = []
    for instances_file, label, num_work_qubits in zip(
        instances_files,
        labels,
        num_work_qubits_by_file,
    ):
        instances = load_instances(instances_file)
        noisy_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
        baseline_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
        noisy_results = simulator(
            instances=instances,
            algorithms=[algorithm],
            num_phase_bits=num_phase_bits,
            error_model=error_model,
            error_probability=error_probability,
            trials_per_instance=trials_per_instance,
            rng=noisy_rng,
            standard_backend=standard_backend,
            include_classical_error_correction=include_classical_error_correction,
            num_work_qubits=num_work_qubits,
            depolarizing_region=depolarizing_region,
        )
        baseline_results = simulator(
            instances=instances,
            algorithms=[algorithm],
            num_phase_bits=num_phase_bits,
            error_model=error_model,
            error_probability=0.0,
            trials_per_instance=trials_per_instance,
            rng=baseline_rng,
            standard_backend=standard_backend,
            include_classical_error_correction=include_classical_error_correction,
            num_work_qubits=num_work_qubits,
            depolarizing_region=depolarizing_region,
        )
        sweeps.append(
            InstanceFileMembershipExcessSweep(
                instances_file=instances_file,
                label=label,
                noisy_results=noisy_results,
                baseline_results=baseline_results,
                fixed_num_work_qubits=num_work_qubits,
            )
        )
    return sweeps


def plot_noisy_cf_membership_excess_vs_order_by_instance_file(
    output_path: Path,
    sweeps: list[InstanceFileMembershipExcessSweep],
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials_per_instance: int,
    membership: str = COPRIME_MEMBERSHIP,
    include_classical_error_correction: bool = False,
    log_x: bool = False,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not sweeps:
        raise ValueError("sweeps must be nonempty")

    error_label = depolarizing_error_display_label(error_model, error_probability)
    membership_label = membership_display_label(membership)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "P", "v", "X", "*", "<", ">"]
    linestyles = {
        False: "-",
        True: "-.",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    plotted_values = []
    line_handles: list[Line2D] = []

    for sweep_index, sweep in enumerate(sweeps):
        correction_values = list(
            dict.fromkeys(
                result.classical_error_correction
                for result in sweep.noisy_results
            )
        )
        for correction_enabled in correction_values:
            pairs = paired_excess_values(
                sweep=sweep,
                algorithm=algorithm,
                correction_enabled=correction_enabled,
            )
            if not pairs:
                continue

            x_values = [result.instance.true_order for result, _, _ in pairs]
            y_values = [delta for _, _, delta in pairs]
            plotted_values.extend(y_values)
            label = result_series_label(
                sweep,
                pairs[0][0],
                include_classical_error_correction=include_classical_error_correction,
            )
            (line,) = ax.plot(
                x_values,
                y_values,
                marker=markers[sweep_index % len(markers)],
                linestyle=linestyles[correction_enabled],
                linewidth=2.0,
                markersize=5.5,
                color=colors[sweep_index % len(colors)],
                alpha=0.92,
                label=label,
            )
            line_handles.append(line)

    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lower, upper = adaptive_delta_ylim(plotted_values)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.axhline(0.0, color="0.25", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel(f"Pr_p(tilde b in {membership_label}^c) - Pr_0(tilde b in {membership_label}^c)")
    ax.set_title(
        "Noisy CF-Complement Excess vs True Order\n"
        f"{algorithm_display_label(algorithm)}, {error_label}, "
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
        algorithm_title="Instance file",
        measurement_title="Depolarizing error",
        measurement_text_labels=[error_label],
        extra_text_sections=[
            ("Algorithm", [algorithm_display_label(algorithm)]),
            ("Membership", [membership_label]),
            ("Baseline", ["p = 0"]),
            (
                "Region",
                [depolarizing_region_display_label(depolarizing_region)],
            ),
            ("Work register", [work_register_overlay_display_label(sweeps)]),
        ],
        trials_title="Trials per p",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    sweeps: list[InstanceFileMembershipExcessSweep],
    algorithm: str,
    standard_backend: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials_per_instance: int,
    seed: int | None,
    membership: str = COPRIME_MEMBERSHIP,
    include_classical_error_correction: bool = False,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not sweeps:
        raise ValueError("sweeps must be nonempty")

    seed_label = "random" if seed is None else str(seed)
    error_label = depolarizing_error_display_label(error_model, error_probability)
    membership_label = membership_display_label(membership)
    lines = [
        "Depolarizing CF-Complement Excess vs Order by Instance File Summary",
        f"Algorithm = {algorithm_display_label(algorithm)}",
        *(
            [f"Standard backend = {standard_backend}"]
            if algorithm == STANDARD_ALGORITHM
            else []
        ),
        f"Membership set = {membership_label}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Work register = {work_register_overlay_display_label(sweeps)}",
        f"Depolarizing error = {error_label}",
        (
            "Controlled-U depolarizing region = "
            f"{depolarizing_region_display_label(depolarizing_region)}"
        ),
        "Baseline depolarizing error = p = 0",
        f"Trials per probability = {trials_per_instance}",
        f"Classical single-bit minimum-distance overlay = {include_classical_error_correction}",
        f"Seed = {seed_label}",
        "",
        "Instance files:",
    ]
    for sweep in sweeps:
        lines.append(
            f"  {sweep.label}: {sweep.instances_file} "
            f"({sweep.instance_count} instances, "
            f"{work_register_display_label(sweep.fixed_num_work_qubits)})"
        )

    set_definition = (
        "S_coprime = {tilde b : exists s with gcd(s,r)=1 and "
        "d(tilde b, s/r) <= 1/(2r^2)}"
        if membership == COPRIME_MEMBERSHIP
        else "S = {tilde b : exists s and d(tilde b, s/r) <= 1/(2r^2)}"
    )
    lines.extend(
        [
            "",
            "Definitions:",
            f"  {set_definition}",
            (
                f"  Excess = Pr_p({membership_label}^c) - "
                f"Pr_0({membership_label}^c)."
            ),
            "  Both probabilities are empirical estimates with the same trial count.",
            "",
            "Per-order data:",
            (
                "  file label | series | true r | N | a | L | "
                f"Pr_p({membership_label}^c) | Pr_0({membership_label}^c) | "
                "excess | p-count | p0-count | trials"
            ),
        ]
    )

    for sweep in sweeps:
        correction_values = list(
            dict.fromkeys(
                result.classical_error_correction
                for result in sweep.noisy_results
            )
        )
        for correction_enabled in correction_values:
            pairs = paired_excess_values(
                sweep=sweep,
                algorithm=algorithm,
                correction_enabled=correction_enabled,
            )
            for result, baseline, delta in pairs:
                series = "corrected" if result.classical_error_correction else "noisy"
                lines.append(
                    f"  {sweep.label:>18} |"
                    f" {series:>9} |"
                    f" {result.instance.true_order:>6} |"
                    f" {result.instance.N:>5} |"
                    f" {result.instance.a:>5} |"
                    f" {result.num_work_qubits:>2} |"
                    f" {result.outside_probability:>7.3f} |"
                    f" {baseline.outside_probability:>7.3f} |"
                    f" {delta:>7.3f} |"
                    f" {result.outside_trials:>7} |"
                    f" {baseline.outside_trials:>8} |"
                    f" {result.trials:>6}"
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay baseline-subtracted Pr(tilde b in S^c) versus true order "
            "r for several order-recovery instance files under depolarizing "
            "errors."
        )
    )
    parser.add_argument(
        "--instances-files",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSON files containing configured representative (N, a) pairs",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional legend labels, one per --instances-files entry",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHM_CHOICES,
        default=KITAEV_ALGORITHM,
        help="Single phase-estimation algorithm to use for every instance file",
    )
    parser.add_argument(
        "--membership",
        choices=MEMBERSHIP_CHOICES,
        default=COPRIME_MEMBERSHIP,
        help="Use the ordinary S set or the coprime-numerator S_coprime set",
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
    parser.add_argument(
        "--fixed-work-register-per-file",
        action="store_true",
        help=(
            "Use one fixed work-register size per instance file, sized by the "
            "largest N in that file."
        ),
    )
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
        labels = validate_instance_file_labels(args.instances_files, args.labels)
        algorithm = args.algorithm
        error_model = normalize_error_model(args.error_model)
        p = validate_error_probabilities([args.p])[0]
        if args.bits < 1:
            raise ValueError("bits must be >= 1")
        if args.trials_per_instance < 1:
            raise ValueError("trials_per_instance must be >= 1")
        validate_depolarizing_standard_backend([algorithm], args.standard_backend)

        instances_by_file = [
            load_instances(instances_file)
            for instances_file in args.instances_files
        ]
        num_work_qubits_by_file = resolve_work_qubits_by_file(
            instances_by_file=instances_by_file,
            requested_work_qubits=args.work_qubits,
            fixed_work_register=args.fixed_work_register,
            fixed_work_register_per_file=args.fixed_work_register_per_file,
        )
        for instances, num_work_qubits in zip(
            instances_by_file,
            num_work_qubits_by_file,
        ):
            validate_statevector_budget(
                algorithms=[algorithm],
                standard_backend=args.standard_backend,
                num_phase_bits=args.bits,
                max_N=max(instance.N for instance in instances),
                max_statevector_amplitudes=args.max_statevector_amplitudes,
                num_work_qubits=num_work_qubits,
            )
    except ValueError as exc:
        parser.error(str(exc))

    sweeps = simulate_membership_excess_sweeps(
        instances_files=args.instances_files,
        labels=labels,
        algorithm=algorithm,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        trials_per_instance=args.trials_per_instance,
        rng=np.random.default_rng(args.seed),
        membership=args.membership,
        standard_backend=args.standard_backend,
        include_classical_error_correction=args.classical_error_correction,
        num_work_qubits_by_file=num_work_qubits_by_file,
        depolarizing_region=args.depolarizing_region,
    )

    suffix = output_filename_segment(
        algorithm=algorithm,
        standard_backend=args.standard_backend,
        instances_files=args.instances_files,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        membership=args.membership,
        sweeps=sweeps,
        include_classical_error_correction=args.classical_error_correction,
        depolarizing_region=args.depolarizing_region,
    )
    figure_output_path = (
        args.output
        if args.output is not None
        else DEFAULT_FIGURE_DIR
        / f"dp_noisy_cf_membership_excess_vs_order_by_instance_file_{suffix}.png"
    )
    data_output_path = (
        DEFAULT_DATA_DIR
        / f"dp_noisy_cf_membership_excess_vs_order_by_instance_file_summary_{suffix}.txt"
    )

    plot_noisy_cf_membership_excess_vs_order_by_instance_file(
        output_path=figure_output_path,
        sweeps=sweeps,
        algorithm=algorithm,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        trials_per_instance=args.trials_per_instance,
        membership=args.membership,
        include_classical_error_correction=args.classical_error_correction,
        log_x=args.log_x,
        depolarizing_region=args.depolarizing_region,
    )
    write_data_summary(
        output_path=data_output_path,
        sweeps=sweeps,
        algorithm=algorithm,
        standard_backend=args.standard_backend,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        trials_per_instance=args.trials_per_instance,
        seed=args.seed,
        membership=args.membership,
        include_classical_error_correction=args.classical_error_correction,
        depolarizing_region=args.depolarizing_region,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"Algorithm: {algorithm_display_label(algorithm)}")
    if algorithm == STANDARD_ALGORITHM:
        print(f"Standard backend: {args.standard_backend}")
    print(f"Membership set: {membership_display_label(args.membership)}")
    print("Instance files:")
    for sweep in sweeps:
        print(f"  {sweep.label}: {sweep.instances_file}")
    print(f"Fixed phase-estimation bits: {args.bits}")
    print(f"Work register: {work_register_overlay_display_label(sweeps)}")
    print(
        "Controlled-U depolarizing region: "
        f"{depolarizing_region_display_label(args.depolarizing_region)}"
    )
    print(f"Depolarizing error: {depolarizing_error_display_label(error_model, p)}")
    print("Baseline depolarizing error: p = 0")
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(f"Trials per probability: {args.trials_per_instance}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
