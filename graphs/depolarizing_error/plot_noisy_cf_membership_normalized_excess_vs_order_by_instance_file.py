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


from graphs.depolarizing_error.cf_common import (
    ALGORITHM_CHOICES,
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    DEFAULT_NUM_PHASE_BITS,
    KITAEV_ALGORITHM,
    NoisyCFMembershipResult,
    OrderRecoveryInstance,
    STANDARD_ALGORITHM,
    add_combined_figure_legend,
    algorithm_display_label,
    algorithm_filename_prefix,
    compact_filename_segment,
    depolarizing_error_display_label,
    depolarizing_error_filename_segment,
    load_instances,
    normalize_error_model,
    run_noisy_coprime_membership_trial,
    run_noisy_membership_trial,
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
from graphs.depolarizing_error.plot_noisy_cf_membership_excess_vs_order_by_instance_file import (
    COPRIME_MEMBERSHIP,
    MEMBERSHIP_CHOICES,
    InstanceFileMembershipExcessSweep,
    adaptive_delta_ylim,
    membership_display_label,
    membership_filename_segment,
    paired_excess_values,
)
from graphs.depolarizing_error.plot_noisy_cf_membership_vs_order_by_instance_file import (
    instance_files_filename_segment,
    resolve_work_qubits_by_file,
    result_series_label,
    validate_instance_file_labels,
    work_register_overlay_display_label,
    work_register_overlay_filename_segment,
)
from graphs.depolarizing_error.work_register import (
    add_work_register_arguments,
    effective_work_qubits_for_N,
    work_register_display_label,
)
from shor_standard.backends import (
    STATEVECTOR_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
)
from shor_common.errors import ControlledUnitaryDepolarizingRangeSpec


DEFAULT_CLEAN_GOOD_TRIALS = 1000
DEFAULT_MAX_CLEAN_TRIALS_PER_INSTANCE = 1_000_000
ERROR_SCALING_TL_FILENAME_TOKEN = "scaled_tL"


def validate_depolarizing_error_scale(
    error_probability: float,
    scale_error_by_tl: bool,
) -> float:
    """
    Validate the command-line depolarizing error parameter.

    Without scaling, the value is a per-location probability. With tL scaling,
    it is a nonnegative total error scale and can exceed one, provided every
    derived per-location probability remains at most one.
    """
    error_probability = float(error_probability)
    if not math.isfinite(error_probability) or error_probability < 0.0:
        raise ValueError("depolarizing error probability/scale must be finite and >= 0")
    if not scale_error_by_tl:
        return validate_error_probabilities([error_probability])[0]
    return error_probability


def effective_depolarizing_error_probability(
    error_probability: float,
    num_phase_bits: int,
    instance: OrderRecoveryInstance,
    num_work_qubits: int | None = None,
    scale_error_by_tl: bool = False,
) -> float:
    """
    Return the per-location depolarizing probability used for one instance.
    """
    error_probability = validate_depolarizing_error_scale(
        error_probability=error_probability,
        scale_error_by_tl=scale_error_by_tl,
    )
    if not scale_error_by_tl:
        return error_probability

    effective_num_work_qubits = result_work_qubits(instance, num_work_qubits)
    denominator = num_phase_bits * effective_num_work_qubits
    if denominator < 1:
        raise ValueError("t * L must be positive")
    effective_error_probability = error_probability / denominator
    if effective_error_probability > 1.0:
        raise ValueError(
            "scaled depolarizing probability p_eff = p / (t * L) must be <= 1; "
            f"got p={error_probability:g}, t={num_phase_bits}, "
            f"L={effective_num_work_qubits}, p_eff={effective_error_probability:g}"
        )
    return effective_error_probability


def depolarizing_probability_scaling_filename_segment(
    scale_error_by_tl: bool,
) -> str:
    return f"_{ERROR_SCALING_TL_FILENAME_TOKEN}" if scale_error_by_tl else ""


def depolarizing_probability_scaling_display_label(
    scale_error_by_tl: bool,
) -> str:
    if scale_error_by_tl:
        return "per-location p_eff = p / (t L)"
    return "raw per-location p"


def normalized_excess_failure(
    noisy_outside_probability: float,
    baseline_outside_probability: float,
) -> float:
    baseline_good_probability = 1.0 - baseline_outside_probability
    if baseline_good_probability <= 0.0:
        return math.nan
    return (
        noisy_outside_probability - baseline_outside_probability
    ) / baseline_good_probability


def membership_trial_runner(membership: str):
    if membership == COPRIME_MEMBERSHIP:
        return run_noisy_coprime_membership_trial
    if membership in MEMBERSHIP_CHOICES:
        return run_noisy_membership_trial
    raise ValueError("unknown membership set")


def simulate_membership_for_fixed_trials(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
    membership: str = COPRIME_MEMBERSHIP,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
    scale_error_by_tl: bool = False,
) -> NoisyCFMembershipResult:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    trial_runner = membership_trial_runner(membership)
    effective_error_probability = effective_depolarizing_error_probability(
        error_probability=error_probability,
        num_phase_bits=num_phase_bits,
        instance=instance,
        num_work_qubits=num_work_qubits,
        scale_error_by_tl=scale_error_by_tl,
    )

    inside_trials = 0
    for _ in range(trials):
        if trial_runner(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            error_model=error_model,
            error_probability=effective_error_probability,
            rng=rng,
            standard_backend=standard_backend,
            classical_error_correction=classical_error_correction,
            num_work_qubits=num_work_qubits,
            depolarizing_region=depolarizing_region,
        ):
            inside_trials += 1

    return NoisyCFMembershipResult(
        instance=instance,
        algorithm=algorithm,
        error_model=normalize_error_model(error_model),
        error_probability=effective_error_probability,
        trials=trials,
        num_phase_bits=num_phase_bits,
        inside_trials=inside_trials,
        standard_backend=standard_backend,
        classical_error_correction=classical_error_correction,
        num_work_qubits=result_work_qubits(instance, num_work_qubits),
    )


def simulate_clean_baseline_until_inside(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    clean_good_trials: int,
    rng: np.random.Generator,
    membership: str = COPRIME_MEMBERSHIP,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    max_clean_trials: int = DEFAULT_MAX_CLEAN_TRIALS_PER_INSTANCE,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> NoisyCFMembershipResult:
    if clean_good_trials < 1:
        raise ValueError("clean_good_trials must be >= 1")
    if max_clean_trials < clean_good_trials:
        raise ValueError("max_clean_trials must be >= clean_good_trials")
    trial_runner = membership_trial_runner(membership)

    trials = 0
    inside_trials = 0
    while inside_trials < clean_good_trials:
        if trials >= max_clean_trials:
            raise ValueError(
                "clean baseline did not reach "
                f"{clean_good_trials} inside-{membership_display_label(membership)} "
                f"trials after {max_clean_trials} attempts for "
                f"N={instance.N}, a={instance.a}, r={instance.true_order}. "
                "Increase --max-clean-trials-per-instance, lower "
                "--clean-good-trials, or use more phase-estimation bits."
            )
        trials += 1
        if trial_runner(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            error_model=error_model,
            error_probability=0.0,
            rng=rng,
            standard_backend=standard_backend,
            classical_error_correction=classical_error_correction,
            num_work_qubits=num_work_qubits,
            depolarizing_region=depolarizing_region,
        ):
            inside_trials += 1

    return NoisyCFMembershipResult(
        instance=instance,
        algorithm=algorithm,
        error_model=normalize_error_model(error_model),
        error_probability=0.0,
        trials=trials,
        num_phase_bits=num_phase_bits,
        inside_trials=inside_trials,
        standard_backend=standard_backend,
        classical_error_correction=classical_error_correction,
        num_work_qubits=result_work_qubits(instance, num_work_qubits),
    )


def result_work_qubits(
    instance: OrderRecoveryInstance,
    num_work_qubits: int | None,
) -> int | None:
    return effective_work_qubits_for_N(instance.N, num_work_qubits)


def simulate_membership_normalized_excess_sweeps(
    instances_files: list[Path],
    labels: list[str],
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    clean_good_trials: int,
    max_clean_trials_per_instance: int,
    rng: np.random.Generator,
    membership: str = COPRIME_MEMBERSHIP,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    include_classical_error_correction: bool = False,
    num_work_qubits_by_file: list[int | None] | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
    scale_error_by_tl: bool = False,
) -> list[InstanceFileMembershipExcessSweep]:
    if len(instances_files) != len(labels):
        raise ValueError("instances files and labels must have the same length")
    if num_work_qubits_by_file is None:
        num_work_qubits_by_file = [None for _ in instances_files]
    if len(num_work_qubits_by_file) != len(instances_files):
        raise ValueError("work-register settings must match instances files")
    if clean_good_trials < 1:
        raise ValueError("clean_good_trials must be >= 1")
    if max_clean_trials_per_instance < clean_good_trials:
        raise ValueError(
            "max_clean_trials_per_instance must be >= clean_good_trials"
        )
    if membership not in MEMBERSHIP_CHOICES:
        raise ValueError("unknown membership set")

    correction_values = [False]
    if include_classical_error_correction:
        correction_values.append(True)

    sweeps = []
    for instances_file, label, num_work_qubits in zip(
        instances_files,
        labels,
        num_work_qubits_by_file,
    ):
        noisy_results = []
        baseline_results = []
        instances = load_instances(instances_file)
        for instance in instances:
            for correction_enabled in correction_values:
                baseline_rng = np.random.default_rng(
                    int(rng.integers(0, 2 ** 63 - 1))
                )
                noisy_rng = np.random.default_rng(
                    int(rng.integers(0, 2 ** 63 - 1))
                )
                baseline_result = simulate_clean_baseline_until_inside(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    error_model=error_model,
                    clean_good_trials=clean_good_trials,
                    rng=baseline_rng,
                    membership=membership,
                    standard_backend=standard_backend,
                    classical_error_correction=correction_enabled,
                    num_work_qubits=num_work_qubits,
                    max_clean_trials=max_clean_trials_per_instance,
                    depolarizing_region=depolarizing_region,
                )
                noisy_result = simulate_membership_for_fixed_trials(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    error_model=error_model,
                    error_probability=error_probability,
                    trials=baseline_result.trials,
                    rng=noisy_rng,
                    membership=membership,
                    standard_backend=standard_backend,
                    classical_error_correction=correction_enabled,
                    num_work_qubits=num_work_qubits,
                    depolarizing_region=depolarizing_region,
                    scale_error_by_tl=scale_error_by_tl,
                )
                baseline_results.append(baseline_result)
                noisy_results.append(noisy_result)
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


def output_filename_segment(
    algorithm: str,
    standard_backend: str,
    instances_files: list[Path],
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    membership: str,
    clean_good_trials: int,
    sweeps: list[InstanceFileMembershipExcessSweep],
    include_classical_error_correction: bool = False,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
    scale_error_by_tl: bool = False,
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
        f"{membership_filename_segment(membership)}_norm_excess_"
        f"{depolarizing_error_filename_segment(error_model, error_probability)}"
        f"{depolarizing_probability_scaling_filename_segment(scale_error_by_tl)}"
        f"{depolarizing_region_filename_segment(depolarizing_region)}"
        f"_minus_p0_t{num_phase_bits}"
        f"_cg{clean_good_trials}"
        f"{work_register_overlay_filename_segment(sweeps)}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )
    return compact_filename_segment(segment, max_length=220)


def plot_noisy_cf_membership_normalized_excess_vs_order_by_instance_file(
    output_path: Path,
    sweeps: list[InstanceFileMembershipExcessSweep],
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    clean_good_trials: int,
    membership: str = COPRIME_MEMBERSHIP,
    include_classical_error_correction: bool = False,
    log_x: bool = False,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
    scale_error_by_tl: bool = False,
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
            y_values = [
                normalized_excess_failure(
                    result.outside_probability,
                    baseline.outside_probability,
                )
                for result, baseline, _ in pairs
            ]
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
    ax.set_ylabel(
        (
            f"[Pr_p({membership_label}^c) - Pr_0({membership_label}^c)] "
            f"/ Pr_0({membership_label})"
        )
    )
    ax.set_title(
        "Normalized Noisy CF-Complement Excess vs True Order\n"
        f"{algorithm_display_label(algorithm)}, {error_label}, "
        f"{depolarizing_region_display_label(depolarizing_region)}, "
        f"{depolarizing_probability_scaling_display_label(scale_error_by_tl)}, "
        f"t = {num_phase_bits}"
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=line_handles,
        measurement_handles=[],
        trials_per_instance=clean_good_trials,
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
            (
                "Probability scaling",
                [depolarizing_probability_scaling_display_label(scale_error_by_tl)],
            ),
            ("Work register", [work_register_overlay_display_label(sweeps)]),
        ],
        trials_title="Clean good target",
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
    clean_good_trials: int,
    max_clean_trials_per_instance: int,
    seed: int | None,
    membership: str = COPRIME_MEMBERSHIP,
    include_classical_error_correction: bool = False,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
    scale_error_by_tl: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not sweeps:
        raise ValueError("sweeps must be nonempty")

    seed_label = "random" if seed is None else str(seed)
    error_label = depolarizing_error_display_label(error_model, error_probability)
    membership_label = membership_display_label(membership)
    lines = [
        "Depolarizing Normalized CF-Complement Excess vs Order by Instance File Summary",
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
            "Depolarizing probability scaling = "
            f"{depolarizing_probability_scaling_display_label(scale_error_by_tl)}"
        ),
        (
            "Controlled-U depolarizing region = "
            f"{depolarizing_region_display_label(depolarizing_region)}"
        ),
        "Baseline depolarizing error = p = 0",
        f"Clean inside target = {clean_good_trials}",
        f"Maximum clean trials per instance = {max_clean_trials_per_instance}",
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
                f"  Normalized excess = [Pr_p({membership_label}^c) - "
                f"Pr_0({membership_label}^c)] / Pr_0({membership_label})."
            ),
            f"  Pr_0({membership_label}) = 1 - Pr_0({membership_label}^c).",
            (
                f"  Clean p=0 trials stop when inside count reaches "
                f"{clean_good_trials}; noisy p trials use the same total "
                "trial count as the matched clean run."
            ),
            *(
                [
                    (
                        "  In scaled mode, each noisy instance uses "
                        "p_eff = p / (t * L), where L is that instance's "
                        "effective work-register size."
                    )
                ]
                if scale_error_by_tl
                else []
            ),
            "",
            "Per-order data:",
            (
                "  file label | series | true r | N | a | L | "
                "p_eff | "
                f"Pr_p({membership_label}^c) | Pr_0({membership_label}^c) | "
                f"Pr_0({membership_label}) | excess | normalized excess | "
                "p-out | p0-out | p-in | p0-in | p-trials | p0-trials"
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
                baseline_good_probability = baseline.inside_probability
                normalized_delta = normalized_excess_failure(
                    result.outside_probability,
                    baseline.outside_probability,
                )
                lines.append(
                    f"  {sweep.label:>18} |"
                    f" {series:>9} |"
                    f" {result.instance.true_order:>6} |"
                    f" {result.instance.N:>5} |"
                    f" {result.instance.a:>5} |"
                    f" {result.num_work_qubits:>2} |"
                    f" {result.error_probability:>8.6f} |"
                    f" {result.outside_probability:>7.3f} |"
                    f" {baseline.outside_probability:>7.3f} |"
                    f" {baseline_good_probability:>7.3f} |"
                    f" {delta:>7.3f} |"
                    f" {normalized_delta:>10.3f} |"
                    f" {result.outside_trials:>7} |"
                    f" {baseline.outside_trials:>8} |"
                    f" {result.inside_trials:>4} |"
                    f" {baseline.inside_trials:>5} |"
                    f" {result.trials:>8} |"
                    f" {baseline.trials:>9}"
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay normalized baseline-subtracted Pr(tilde b in S^c) versus "
            "true order r for several order-recovery instance files under "
            "depolarizing errors."
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
    parser.add_argument(
        "--p",
        type=float,
        default=0.1,
        help=(
            "Depolarizing probability. By default this is the raw per-location "
            "probability; with --scale-error-by-tl it is divided by t * L for "
            "each instance."
        ),
    )
    parser.add_argument(
        "--scale-error-by-tl",
        action="store_true",
        help=(
            "Use p_eff = p / (t * L) as the per-location depolarizing "
            "probability for each instance, where t is --bits and L is the "
            "effective work-register size."
        ),
    )
    parser.add_argument(
        "--clean-good-trials",
        type=int,
        default=None,
        help=(
            "Number of clean p=0 inside-membership trials required for each "
            "instance before estimating the matched noisy probability."
        ),
    )
    parser.add_argument(
        "--trials-per-instance",
        type=int,
        default=None,
        help=(
            "Deprecated alias for --clean-good-trials, kept so older commands "
            "request the same clean-good target."
        ),
    )
    parser.add_argument(
        "--max-clean-trials-per-instance",
        type=int,
        default=DEFAULT_MAX_CLEAN_TRIALS_PER_INSTANCE,
        help=(
            "Maximum p=0 trials allowed while collecting the clean-good target "
            "for one instance."
        ),
    )
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
        p = validate_depolarizing_error_scale(
            error_probability=args.p,
            scale_error_by_tl=args.scale_error_by_tl,
        )
        if args.bits < 1:
            raise ValueError("bits must be >= 1")
        if args.clean_good_trials is not None and args.trials_per_instance is not None:
            raise ValueError(
                "choose either --clean-good-trials or --trials-per-instance, not both"
            )
        clean_good_trials = (
            args.clean_good_trials
            if args.clean_good_trials is not None
            else args.trials_per_instance
        )
        if clean_good_trials is None:
            clean_good_trials = DEFAULT_CLEAN_GOOD_TRIALS
        if clean_good_trials < 1:
            raise ValueError("clean_good_trials must be >= 1")
        if args.max_clean_trials_per_instance < clean_good_trials:
            raise ValueError(
                "max_clean_trials_per_instance must be >= clean_good_trials"
            )
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

    try:
        sweeps = simulate_membership_normalized_excess_sweeps(
            instances_files=args.instances_files,
            labels=labels,
            algorithm=algorithm,
            num_phase_bits=args.bits,
            error_model=error_model,
            error_probability=p,
            clean_good_trials=clean_good_trials,
            max_clean_trials_per_instance=args.max_clean_trials_per_instance,
            rng=np.random.default_rng(args.seed),
            membership=args.membership,
            standard_backend=args.standard_backend,
            include_classical_error_correction=args.classical_error_correction,
            num_work_qubits_by_file=num_work_qubits_by_file,
            depolarizing_region=args.depolarizing_region,
            scale_error_by_tl=args.scale_error_by_tl,
        )
    except ValueError as exc:
        parser.error(str(exc))

    suffix = output_filename_segment(
        algorithm=algorithm,
        standard_backend=args.standard_backend,
        instances_files=args.instances_files,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        membership=args.membership,
        clean_good_trials=clean_good_trials,
        sweeps=sweeps,
        include_classical_error_correction=args.classical_error_correction,
        depolarizing_region=args.depolarizing_region,
        scale_error_by_tl=args.scale_error_by_tl,
    )
    figure_output_path = (
        args.output
        if args.output is not None
        else DEFAULT_FIGURE_DIR
        / f"dp_noisy_cf_membership_normalized_excess_vs_order_by_instance_file_{suffix}.png"
    )
    data_output_path = (
        DEFAULT_DATA_DIR
        / f"dp_noisy_cf_membership_normalized_excess_vs_order_by_instance_file_summary_{suffix}.txt"
    )

    plot_noisy_cf_membership_normalized_excess_vs_order_by_instance_file(
        output_path=figure_output_path,
        sweeps=sweeps,
        algorithm=algorithm,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        clean_good_trials=clean_good_trials,
        membership=args.membership,
        include_classical_error_correction=args.classical_error_correction,
        log_x=args.log_x,
        depolarizing_region=args.depolarizing_region,
        scale_error_by_tl=args.scale_error_by_tl,
    )
    write_data_summary(
        output_path=data_output_path,
        sweeps=sweeps,
        algorithm=algorithm,
        standard_backend=args.standard_backend,
        num_phase_bits=args.bits,
        error_model=error_model,
        error_probability=p,
        clean_good_trials=clean_good_trials,
        max_clean_trials_per_instance=args.max_clean_trials_per_instance,
        seed=args.seed,
        membership=args.membership,
        include_classical_error_correction=args.classical_error_correction,
        depolarizing_region=args.depolarizing_region,
        scale_error_by_tl=args.scale_error_by_tl,
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
    print(
        "Depolarizing probability scaling: "
        f"{depolarizing_probability_scaling_display_label(args.scale_error_by_tl)}"
    )
    print(f"Depolarizing error: {depolarizing_error_display_label(error_model, p)}")
    print("Baseline depolarizing error: p = 0")
    print(f"Clean inside target: {clean_good_trials}")
    print(
        "Maximum clean trials per instance: "
        f"{args.max_clean_trials_per_instance}"
    )
    print(
        "Classical single-bit minimum-distance overlay: "
        f"{'enabled' if args.classical_error_correction else 'disabled'}"
    )
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
