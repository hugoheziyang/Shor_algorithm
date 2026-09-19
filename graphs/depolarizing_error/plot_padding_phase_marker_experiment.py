from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
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
    DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    algorithm_display_label,
    algorithm_filename_prefix,
    build_fixed_instance,
    depolarizing_error_display_label,
    depolarizing_error_filename_segment,
    validate_depolarizing_standard_backend,
    validate_statevector_budget,
)
from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    ERROR_MODEL_CHOICES,
    add_depolarizing_region_argument,
    depolarizing_region_display_label,
    depolarizing_region_filename_segment,
    depolarizing_region_phase_bits,
    error_config_for_model,
    normalize_error_model,
)
from graphs.depolarizing_error.plot_order_recovery_success_vs_order import (
    normalize_algorithms,
    validate_error_probabilities,
)
from graphs.depolarizing_error.plot_phase_estimation_distribution import (
    DEFAULT_SMOOTH_SIGMA_BINS,
    DISTRIBUTION_STYLE_BOTH,
    DISTRIBUTION_STYLE_CHOICES,
    DISTRIBUTION_STYLE_SMOOTH,
    DISTRIBUTION_STYLE_STEMS,
    RANDOM_PAULI_TOKEN,
    Y_AXIS_FORMAT_AUTO,
    Y_AXIS_FORMAT_CHOICES,
    PhaseDistributionSample,
    PhaseOutcomeRow,
    apply_probability_y_axis_format,
    bits_to_integer,
    depolarizing_pauli_display_label,
    depolarizing_pauli_filename_segment,
    distribution_style_filename_segment,
    normalize_depolarizing_pauli_arg,
    smoothed_phase_distribution_curve,
    validate_smooth_sigma_bins,
    validate_stem_linewidth,
)
from graphs.depolarizing_error.work_register import (
    add_work_register_arguments,
    effective_work_qubits_for_N,
    resolve_work_qubits,
    work_register_display_label,
    work_register_filename_segment,
)
from shor_common.errors import (
    CONTROLLED_UNITARY_DEPOLARIZING_PAULI_CHOICES,
    ControlledUnitaryDepolarizingPauliSpec,
    ErrorConfig,
)
from shor_common.shor_utils import default_num_phase_bits
from shor_one_qubit.shor import shor_order_finding_attempt as kitaev_order_attempt
from shor_standard.backends import (
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)
from shor_standard.shor import shor_order_finding_attempt as standard_order_attempt


DEFAULT_MARKER_COUNT = 3
DEFAULT_MARKER_WINDOW_BINS = 2
MARKER_COLORS = (
    "#A23B2A",
    "#6A4C93",
    "#C27A1A",
    "#2F855A",
    "#B24C83",
    "#3973AC",
)


@dataclass(frozen=True)
class AlgorithmMarkerExperiment:
    algorithm: str
    clean: PhaseDistributionSample
    noisy_identity: PhaseDistributionSample
    noisy_markers: tuple[PhaseDistributionSample, ...]


def circular_bin_distance(first: int, second: int, num_bins: int) -> int:
    difference = abs(int(first) - int(second))
    return min(difference, num_bins - difference)


def marker_distance_to_order_phases(
    marker_bin: int,
    num_bins: int,
    true_order: int,
) -> float:
    """Circular phase distance from p/Q to the nearest exact phase s/r."""
    residue = (int(marker_bin) * int(true_order)) % int(num_bins)
    return min(residue, num_bins - residue) / (num_bins * true_order)


def random_marker_bins(
    num_phase_bits: int,
    rng: np.random.Generator,
    marker_count: int = DEFAULT_MARKER_COUNT,
) -> list[int]:
    """Choose distinct nonzero marker bins without using the unknown order."""
    if num_phase_bits < 1:
        raise ValueError("bits must be >= 1")
    if marker_count < 2:
        raise ValueError("marker-count must be >= 2 for a marker experiment")

    num_bins = 1 << num_phase_bits
    if marker_count > num_bins - 1:
        raise ValueError(
            "not enough nonzero phase bins for random marker selection; "
            "increase --bits or reduce --marker-count"
        )

    marker_bins: set[int] = set()
    while len(marker_bins) < marker_count:
        marker_bins.add(int(rng.integers(1, num_bins)))
    return sorted(marker_bins)


def validate_marker_bins(marker_bins: list[int], num_phase_bits: int) -> list[int]:
    if num_phase_bits < 1:
        raise ValueError("bits must be >= 1")
    if len(marker_bins) < 2:
        raise ValueError("provide at least two distinct marker bins")

    num_bins = 1 << num_phase_bits
    normalized = [int(marker_bin) for marker_bin in marker_bins]
    if len(set(normalized)) != len(normalized):
        raise ValueError("marker bins must be distinct")
    for marker_bin in normalized:
        if not (0 < marker_bin < num_bins):
            raise ValueError(f"marker bins must satisfy 0 < p < Q = {num_bins}")
    return normalized


def marker_phase_radians(marker_bin: int, num_phase_bits: int) -> float:
    return 2.0 * np.pi * marker_bin / (1 << num_phase_bits)


def run_phase_trial(
    N: int,
    a: int,
    algorithm: str,
    num_phase_bits: int,
    rng: np.random.Generator,
    error_config: ErrorConfig,
    standard_backend: str,
    num_work_qubits: int | None,
    padding_phase_radians: float,
) -> list[int]:
    if algorithm == KITAEV_ALGORITHM:
        result = kitaev_order_attempt(
            N=N,
            a=a,
            num_phase_bits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
            error_config=error_config,
            num_work_qubits=num_work_qubits,
            padding_phase_radians=padding_phase_radians,
        )
    elif algorithm == STANDARD_ALGORITHM:
        result = standard_order_attempt(
            N=N,
            a=a,
            num_control_qubits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
            error_config=error_config,
            standard_backend=standard_backend,
            num_work_qubits=num_work_qubits,
            padding_phase_radians=padding_phase_radians,
        )
    else:
        raise ValueError("unknown algorithm")
    return result["bits"]


def sample_from_counts(
    algorithm: str,
    noise_label: str,
    counts: Counter[int],
    trials: int,
    num_phase_bits: int,
    num_work_qubits: int,
) -> PhaseDistributionSample:
    num_bins = 1 << num_phase_bits
    rows = [
        PhaseOutcomeRow(
            bits=format(outcome, f"0{num_phase_bits}b"),
            outcome_integer=outcome,
            phase_decimal=outcome / num_bins,
            count=count,
            probability=count / trials,
        )
        for outcome, count in sorted(counts.items())
    ]
    return PhaseDistributionSample(
        algorithm=algorithm,
        noise_label=noise_label,
        rows=rows,
        trials=trials,
        num_phase_bits=num_phase_bits,
        num_work_qubits=num_work_qubits,
    )


def simulate_padding_marker_experiment(
    N: int,
    a: int,
    algorithms: list[str],
    num_phase_bits: int,
    trials: int,
    error_model: str,
    error_probability: float,
    marker_bins: list[int],
    rng: np.random.Generator,
    true_order: int,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
) -> list[AlgorithmMarkerExperiment]:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    marker_bins = validate_marker_bins(marker_bins, num_phase_bits)
    algorithms = normalize_algorithms(algorithms)
    allowed_phase_bits = depolarizing_region_phase_bits(
        num_phase_bits=num_phase_bits,
        true_order=true_order,
        depolarizing_region=depolarizing_region,
    )
    noisy_error_config = error_config_for_model(
        error_model=error_model,
        error_probability=error_probability,
        allowed_phase_bits=allowed_phase_bits,
        depolarizing_paulis=depolarizing_paulis,
    )
    effective_work_qubits = effective_work_qubits_for_N(N, num_work_qubits)
    max_seed = np.iinfo(np.uint64).max

    experiments = []
    for algorithm in algorithms:
        clean_counts: Counter[int] = Counter()
        identity_counts: Counter[int] = Counter()
        marker_counts = {marker_bin: Counter() for marker_bin in marker_bins}

        for _ in range(trials):
            clean_seed = int(rng.integers(0, max_seed, dtype=np.uint64))
            noisy_seed = int(rng.integers(0, max_seed, dtype=np.uint64))

            clean_bits = run_phase_trial(
                N=N,
                a=a,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                rng=np.random.default_rng(clean_seed),
                error_config=ErrorConfig(),
                standard_backend=standard_backend,
                num_work_qubits=num_work_qubits,
                padding_phase_radians=0.0,
            )
            clean_counts[bits_to_integer(clean_bits)] += 1

            identity_bits = run_phase_trial(
                N=N,
                a=a,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                rng=np.random.default_rng(noisy_seed),
                error_config=noisy_error_config,
                standard_backend=standard_backend,
                num_work_qubits=num_work_qubits,
                padding_phase_radians=0.0,
            )
            identity_counts[bits_to_integer(identity_bits)] += 1

            for marker_bin in marker_bins:
                marker_bits = run_phase_trial(
                    N=N,
                    a=a,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    rng=np.random.default_rng(noisy_seed),
                    error_config=noisy_error_config,
                    standard_backend=standard_backend,
                    num_work_qubits=num_work_qubits,
                    padding_phase_radians=marker_phase_radians(
                        marker_bin,
                        num_phase_bits,
                    ),
                )
                marker_counts[marker_bin][bits_to_integer(marker_bits)] += 1

        experiments.append(
            AlgorithmMarkerExperiment(
                algorithm=algorithm,
                clean=sample_from_counts(
                    algorithm=algorithm,
                    noise_label="clean p=0",
                    counts=clean_counts,
                    trials=trials,
                    num_phase_bits=num_phase_bits,
                    num_work_qubits=effective_work_qubits,
                ),
                noisy_identity=sample_from_counts(
                    algorithm=algorithm,
                    noise_label="noisy identity padding",
                    counts=identity_counts,
                    trials=trials,
                    num_phase_bits=num_phase_bits,
                    num_work_qubits=effective_work_qubits,
                ),
                noisy_markers=tuple(
                    sample_from_counts(
                        algorithm=algorithm,
                        noise_label=f"noisy marker p_marker={marker_bin}",
                        counts=marker_counts[marker_bin],
                        trials=trials,
                        num_phase_bits=num_phase_bits,
                        num_work_qubits=effective_work_qubits,
                    )
                    for marker_bin in marker_bins
                ),
            )
        )

    return experiments


def probability_map(sample: PhaseDistributionSample) -> dict[int, float]:
    return {row.outcome_integer: row.probability for row in sample.rows}


def difference_rows(
    sample: PhaseDistributionSample,
    baseline: PhaseDistributionSample,
) -> list[PhaseOutcomeRow]:
    sample_probabilities = probability_map(sample)
    baseline_probabilities = probability_map(baseline)
    outcomes = sorted(sample_probabilities.keys() | baseline_probabilities.keys())
    num_bins = 1 << sample.num_phase_bits
    return [
        PhaseOutcomeRow(
            bits=format(outcome, f"0{sample.num_phase_bits}b"),
            outcome_integer=outcome,
            phase_decimal=outcome / num_bins,
            count=0,
            probability=(
                sample_probabilities.get(outcome, 0.0)
                - baseline_probabilities.get(outcome, 0.0)
            ),
        )
        for outcome in outcomes
    ]


def marker_window_probability(
    sample: PhaseDistributionSample,
    marker_bin: int,
    marker_window_bins: int,
) -> float:
    num_bins = 1 << sample.num_phase_bits
    return sum(
        row.probability
        for row in sample.rows
        if circular_bin_distance(row.outcome_integer, marker_bin, num_bins)
        <= marker_window_bins
    )


def marker_window_excess(
    sample: PhaseDistributionSample,
    baseline: PhaseDistributionSample,
    marker_bin: int,
    marker_window_bins: int,
) -> float:
    return marker_window_probability(
        sample,
        marker_bin,
        marker_window_bins,
    ) - marker_window_probability(
        baseline,
        marker_bin,
        marker_window_bins,
    )


def _draw_rows(
    ax,
    rows: list[PhaseOutcomeRow],
    num_phase_bits: int,
    color: str,
    label: str,
    distribution_style: str,
    smooth_sigma_bins: float,
    stem_linewidth: float | None = None,
    linewidth: float = 2.0,
) -> list[float]:
    plotted_values = []
    if distribution_style in (DISTRIBUTION_STYLE_STEMS, DISTRIBUTION_STYLE_BOTH):
        x_values = [row.phase_decimal for row in rows]
        y_values = [row.probability for row in rows]
        stem_label = label if distribution_style == DISTRIBUTION_STYLE_STEMS else None
        ax.vlines(
            x_values,
            0.0,
            y_values,
            color=color,
            linewidth=0.85 if stem_linewidth is None else stem_linewidth,
            alpha=0.34,
            label=stem_label,
        )
        plotted_values.extend(y_values)

    if distribution_style in (DISTRIBUTION_STYLE_SMOOTH, DISTRIBUTION_STYLE_BOTH):
        smooth_x, smooth_y = smoothed_phase_distribution_curve(
            rows=rows,
            num_phase_bits=num_phase_bits,
            sigma_bins=smooth_sigma_bins,
        )
        ax.plot(
            smooth_x,
            smooth_y,
            color=color,
            linestyle=":" if label.startswith("clean") else "-",
            linewidth=linewidth,
            alpha=0.94,
            label=label,
        )
        plotted_values.extend(smooth_y.tolist())

    return plotted_values


def plot_padding_marker_experiment(
    N: int,
    a: int,
    true_order: int,
    num_phase_bits: int,
    trials: int,
    error_model: str,
    error_probability: float,
    marker_bins: list[int],
    experiments: list[AlgorithmMarkerExperiment],
    output_path: Path,
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
    distribution_style: str = DISTRIBUTION_STYLE_SMOOTH,
    smooth_sigma_bins: float = DEFAULT_SMOOTH_SIGMA_BINS,
    y_max: float | None = None,
    difference_y_max: float | None = None,
    y_axis_format: str = Y_AXIS_FORMAT_AUTO,
    stem_linewidth: float | None = None,
) -> None:
    if distribution_style not in DISTRIBUTION_STYLE_CHOICES:
        raise ValueError("unknown distribution style")
    if y_axis_format not in Y_AXIS_FORMAT_CHOICES:
        raise ValueError("unknown y-axis format")
    smooth_sigma_bins = validate_smooth_sigma_bins(smooth_sigma_bins)
    stem_linewidth = validate_stem_linewidth(stem_linewidth)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_bins = 1 << num_phase_bits
    figure_height = 5.0 if len(experiments) == 1 else 4.25 * len(experiments)
    fig, axes = plt.subplots(
        len(experiments),
        2,
        figsize=(14, figure_height),
        sharex=True,
        squeeze=False,
    )

    for row_index, experiment in enumerate(experiments):
        distribution_ax = axes[row_index][0]
        difference_ax = axes[row_index][1]
        distribution_values = []
        difference_values = []

        distribution_values.extend(
            _draw_rows(
                distribution_ax,
                experiment.clean.rows,
                num_phase_bits,
                color="#1F6F78",
                label="clean p=0",
                distribution_style=distribution_style,
                smooth_sigma_bins=smooth_sigma_bins,
                stem_linewidth=stem_linewidth,
                linewidth=2.0,
            )
        )
        distribution_values.extend(
            _draw_rows(
                distribution_ax,
                experiment.noisy_identity.rows,
                num_phase_bits,
                color="#4A4A4A",
                label="noisy theta=0",
                distribution_style=distribution_style,
                smooth_sigma_bins=smooth_sigma_bins,
                stem_linewidth=stem_linewidth,
                linewidth=1.8,
            )
        )

        for marker_index, (marker_bin, marker_sample) in enumerate(
            zip(marker_bins, experiment.noisy_markers)
        ):
            color = MARKER_COLORS[marker_index % len(MARKER_COLORS)]
            marker_phase = marker_bin / num_bins
            marker_label = f"p_marker={marker_bin} ({marker_phase:.4f})"
            distribution_values.extend(
                _draw_rows(
                    distribution_ax,
                    marker_sample.rows,
                    num_phase_bits,
                    color=color,
                    label=marker_label,
                    distribution_style=distribution_style,
                    smooth_sigma_bins=smooth_sigma_bins,
                    stem_linewidth=stem_linewidth,
                )
            )

            rows = difference_rows(marker_sample, experiment.noisy_identity)
            difference_values.extend(
                _draw_rows(
                    difference_ax,
                    rows,
                    num_phase_bits,
                    color=color,
                    label=f"p_marker={marker_bin}",
                    distribution_style=distribution_style,
                    smooth_sigma_bins=smooth_sigma_bins,
                    stem_linewidth=stem_linewidth,
                )
            )
            for ax in (distribution_ax, difference_ax):
                ax.axvline(
                    marker_phase,
                    color=color,
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.55,
                    zorder=0,
                )

        if true_order <= 128:
            for numerator in range(true_order):
                phase = numerator / true_order
                for ax in (distribution_ax, difference_ax):
                    ax.axvline(
                        phase,
                        color="#808080",
                        linewidth=0.5,
                        alpha=0.13,
                        zorder=-1,
                    )

        distribution_upper = y_max
        if distribution_upper is None:
            maximum = max(distribution_values, default=1.0)
            distribution_upper = min(1.0, maximum * 1.16 + 0.01)
        difference_limit = difference_y_max
        if difference_limit is None:
            maximum_absolute = max(
                (abs(value) for value in difference_values),
                default=0.0,
            )
            difference_limit = max(1e-12, maximum_absolute * 1.18)

        distribution_ax.set_ylim(0.0, distribution_upper)
        difference_ax.set_ylim(-difference_limit, difference_limit)
        difference_ax.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.65)
        apply_probability_y_axis_format(
            distribution_ax,
            y_axis_format=y_axis_format,
            upper_y_limit=distribution_upper,
        )
        apply_probability_y_axis_format(
            difference_ax,
            y_axis_format=y_axis_format,
            upper_y_limit=difference_limit,
        )

        display_algorithm = algorithm_display_label(experiment.algorithm)
        distribution_ax.set_title(f"{display_algorithm}: distributions")
        difference_ax.set_title(
            f"{display_algorithm}: marker noisy minus theta=0 noisy"
        )
        distribution_ax.set_ylabel("Empirical probability")
        difference_ax.set_ylabel("Probability difference")
        for ax in (distribution_ax, difference_ax):
            ax.grid(True, alpha=0.2)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.legend(loc="upper right", frameon=True, framealpha=0.88)

    for ax in axes[-1]:
        ax.set_xlabel("Measured phase estimate (decimal)")
    for ax in axes.flat:
        ax.set_xlim(-0.01, 1.01)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.suptitle(
        "Padding-Phase Marker Experiment"
        f" for N = {N}, a = {a}, r = {true_order}",
        y=0.995,
    )
    footer = "\n".join(
        [
            " | ".join(
                [
                    f"t = {num_phase_bits}",
                    f"Q = {num_bins}",
                    f"trials = {trials}",
                    depolarizing_error_display_label(
                        error_model,
                        error_probability,
                    ),
                ]
            ),
            " | ".join(
                [
                    f"Pauli = {depolarizing_pauli_display_label(depolarizing_paulis)}",
                    f"region = {depolarizing_region_display_label(depolarizing_region)}",
                    work_register_display_label(num_work_qubits),
                ]
            ),
            f"marker bins = {', '.join(str(value) for value in marker_bins)} | paired noisy random streams",
        ]
    )
    fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=8.4)
    fig.tight_layout(rect=(0, 0.105, 1, 0.965))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def marker_bins_filename_segment(marker_bins: list[int]) -> str:
    return "_markers_" + "_".join(str(marker_bin) for marker_bin in marker_bins)


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    N: int,
    a: int,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    marker_bins: list[int],
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
    distribution_style: str = DISTRIBUTION_STYLE_SMOOTH,
    smooth_sigma_bins: float = DEFAULT_SMOOTH_SIGMA_BINS,
) -> str:
    backend_segment = ""
    if STANDARD_ALGORITHM in normalize_algorithms(algorithms):
        backend_segment = (
            ""
            if standard_backend == STATEVECTOR_STANDARD_BACKEND
            else f"{standard_backend}_"
        )
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{backend_segment}N{N}_a{a}_t{num_phase_bits}_"
        f"{depolarizing_error_filename_segment(error_model, error_probability)}"
        f"{depolarizing_pauli_filename_segment(depolarizing_paulis)}"
        f"{depolarizing_region_filename_segment(depolarizing_region)}"
        f"{marker_bins_filename_segment(marker_bins)}"
        f"{distribution_style_filename_segment(distribution_style, smooth_sigma_bins)}"
        f"{work_register_filename_segment(num_work_qubits)}"
    )


def write_data_summary(
    output_path: Path,
    N: int,
    a: int,
    true_order: int,
    algorithms: list[str],
    num_phase_bits: int,
    trials: int,
    error_model: str,
    error_probability: float,
    marker_bins: list[int],
    marker_source: str,
    marker_window_bins: int,
    experiments: list[AlgorithmMarkerExperiment],
    seed: int | None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
    distribution_style: str = DISTRIBUTION_STYLE_SMOOTH,
    smooth_sigma_bins: float = DEFAULT_SMOOTH_SIGMA_BINS,
    stem_linewidth: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_bins = 1 << num_phase_bits
    seed_label = "random" if seed is None else str(seed)
    lines = [
        "Padding-Phase Marker Experiment Summary",
        f"N = {N}",
        f"a = {a}",
        f"True order r = {true_order}",
        f"Algorithms = {', '.join(algorithm_display_label(value) for value in algorithms)}",
        f"Standard backend = {standard_backend}",
        f"Phase-estimation bits t = {num_phase_bits}",
        f"Phase bins Q = {num_bins}",
        f"Trials per distribution = {trials}",
        f"Depolarizing error = {depolarizing_error_display_label(error_model, error_probability)}",
        f"Depolarizing Pauli = {depolarizing_pauli_display_label(depolarizing_paulis)}",
        f"Controlled-U depolarizing region = {depolarizing_region_display_label(depolarizing_region)}",
        f"Work register = {work_register_display_label(num_work_qubits)}",
        f"Marker source = {marker_source}",
        f"Marker window radius = {marker_window_bins} bins",
        f"Distribution style = {distribution_style}",
        f"Stem linewidth = {'default' if stem_linewidth is None else stem_linewidth}",
        f"Smooth sigma bins = {smooth_sigma_bins}",
        f"Seed = {seed_label}",
        "Noisy identity and marker runs use matched random seeds per trial.",
        "Probability differences are marker noisy minus theta=0 noisy.",
        "",
        "Markers:",
        "  p_marker | p_marker/Q | theta radians | nearest-order-phase distance",
    ]
    for marker_bin in marker_bins:
        lines.append(
            f"  {marker_bin} | {marker_bin / num_bins:.12g} | "
            f"{marker_phase_radians(marker_bin, num_phase_bits):.12g} | "
            f"{marker_distance_to_order_phases(marker_bin, num_bins, true_order):.12g}"
        )

    for experiment in experiments:
        lines.extend(
            [
                "",
                algorithm_display_label(experiment.algorithm),
                "Marker-window probabilities and excess over noisy theta=0:",
                "  source p_marker | window p_marker | clean | noisy theta=0 | marker noisy | excess",
            ]
        )
        for source_bin, marker_sample in zip(marker_bins, experiment.noisy_markers):
            for window_bin in marker_bins:
                clean_probability = marker_window_probability(
                    experiment.clean,
                    window_bin,
                    marker_window_bins,
                )
                identity_probability = marker_window_probability(
                    experiment.noisy_identity,
                    window_bin,
                    marker_window_bins,
                )
                marker_probability = marker_window_probability(
                    marker_sample,
                    window_bin,
                    marker_window_bins,
                )
                lines.append(
                    f"  {source_bin} | {window_bin} | "
                    f"{clean_probability:.8f} | {identity_probability:.8f} | "
                    f"{marker_probability:.8f} | "
                    f"{marker_probability - identity_probability:+.8f}"
                )

        for sample in (
            experiment.clean,
            experiment.noisy_identity,
            *experiment.noisy_markers,
        ):
            lines.extend(
                [
                    "",
                    f"{sample.noise_label}",
                    "  bits | outcome integer m | decimal phase | count | empirical probability",
                ]
            )
            for row in sample.rows:
                lines.append(
                    "  "
                    f"{row.bits} | {row.outcome_integer:>17} | "
                    f"{row.phase_decimal:>13.8f} | {row.count:>7} | "
                    f"{row.probability:>21.8f}"
                )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed padded-state eigenphase markers against a matched "
            "noisy identity-padding baseline."
        )
    )
    parser.add_argument("N", type=int, help="Composite number to analyze")
    parser.add_argument("--a", type=int, required=True, help="Fixed coprime base")
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Phase-estimation bits t (default: 2 * ceil(log2(N)) + 1)",
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument(
        "--error-model",
        choices=ERROR_MODEL_CHOICES,
        default=BLOCK_DEPOLARIZING_MODEL,
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.1,
        help="Controlled-unitary depolarizing probability",
    )
    parser.add_argument(
        "--depolarizing-pauli",
        choices=(RANDOM_PAULI_TOKEN, *CONTROLLED_UNITARY_DEPOLARIZING_PAULI_CHOICES),
        default=RANDOM_PAULI_TOKEN,
    )
    add_depolarizing_region_argument(parser)
    parser.add_argument(
        "--marker-count",
        type=int,
        default=DEFAULT_MARKER_COUNT,
        help="Number of randomly selected marker bins (default: 3)",
    )
    parser.add_argument(
        "--marker-bins",
        type=int,
        nargs="+",
        default=None,
        help="Explicit marker bins p; overrides random marker selection",
    )
    parser.add_argument(
        "--marker-window-bins",
        type=int,
        default=DEFAULT_MARKER_WINDOW_BINS,
        help="Circular radius around each marker used in summary statistics",
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=STATEVECTOR_STANDARD_BACKEND,
    )
    parser.add_argument(
        "--max-statevector-amplitudes",
        type=int,
        default=DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
    )
    add_work_register_arguments(parser)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--distribution-style",
        choices=DISTRIBUTION_STYLE_CHOICES,
        default=DISTRIBUTION_STYLE_SMOOTH,
    )
    parser.add_argument(
        "--smooth-sigma-bins",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA_BINS,
    )
    parser.add_argument(
        "--stem-linewidth",
        type=float,
        default=None,
        help="Optional linewidth for empirical distribution stems",
    )
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--difference-y-max", type=float, default=None)
    parser.add_argument(
        "--y-format",
        choices=Y_AXIS_FORMAT_CHOICES,
        default=Y_AXIS_FORMAT_AUTO,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    start_time = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        instance = build_fixed_instance(args.N, args.a)
        algorithms = normalize_algorithms(args.algorithm)
        error_model = normalize_error_model(args.error_model)
        error_probability = validate_error_probabilities([args.p])[0]
        num_phase_bits = (
            default_num_phase_bits(args.N)
            if args.bits is None
            else int(args.bits)
        )
        if num_phase_bits < 1:
            raise ValueError("bits must be >= 1")
        if args.trials < 1:
            raise ValueError("trials must be >= 1")
        if args.marker_window_bins < 0:
            raise ValueError("marker-window-bins must be >= 0")
        if args.y_max is not None and args.y_max <= 0.0:
            raise ValueError("y-max must be > 0")
        if args.difference_y_max is not None and args.difference_y_max <= 0.0:
            raise ValueError("difference-y-max must be > 0")
        smooth_sigma_bins = validate_smooth_sigma_bins(args.smooth_sigma_bins)
        stem_linewidth = validate_stem_linewidth(args.stem_linewidth)
        validate_depolarizing_standard_backend(algorithms, args.standard_backend)
        num_work_qubits = resolve_work_qubits(
            [args.N],
            requested_work_qubits=args.work_qubits,
            fixed_work_register=args.fixed_work_register,
        )
        validate_statevector_budget(
            algorithms=algorithms,
            standard_backend=args.standard_backend,
            num_phase_bits=num_phase_bits,
            max_N=args.N,
            max_statevector_amplitudes=args.max_statevector_amplitudes,
            num_work_qubits=num_work_qubits,
        )
        depolarizing_paulis = normalize_depolarizing_pauli_arg(
            args.depolarizing_pauli
        )
        seed_sequence = np.random.SeedSequence(args.seed)
        marker_seed_sequence, simulation_seed_sequence = seed_sequence.spawn(2)
        if args.marker_bins is None:
            marker_bins = random_marker_bins(
                num_phase_bits=num_phase_bits,
                rng=np.random.default_rng(marker_seed_sequence),
                marker_count=args.marker_count,
            )
            marker_source = "random order-independent selection"
        else:
            marker_bins = validate_marker_bins(args.marker_bins, num_phase_bits)
            marker_source = "explicit --marker-bins"
    except ValueError as exc:
        parser.error(str(exc))

    rng = np.random.default_rng(simulation_seed_sequence)
    experiments = simulate_padding_marker_experiment(
        N=args.N,
        a=args.a,
        algorithms=algorithms,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        error_model=error_model,
        error_probability=error_probability,
        marker_bins=marker_bins,
        rng=rng,
        true_order=instance.true_order,
        standard_backend=args.standard_backend,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
    )

    suffix = output_filename_segment(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        error_model=error_model,
        error_probability=error_probability,
        marker_bins=marker_bins,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
        distribution_style=args.distribution_style,
        smooth_sigma_bins=smooth_sigma_bins,
    )
    figure_output_path = (
        args.output
        if args.output is not None
        else DEFAULT_FIGURE_DIR / f"padding_phase_marker_experiment_{suffix}.png"
    )
    data_output_path = (
        DEFAULT_DATA_DIR
        / f"padding_phase_marker_experiment_summary_{suffix}.txt"
    )

    plot_padding_marker_experiment(
        N=args.N,
        a=args.a,
        true_order=instance.true_order,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        error_model=error_model,
        error_probability=error_probability,
        marker_bins=marker_bins,
        experiments=experiments,
        output_path=figure_output_path,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
        distribution_style=args.distribution_style,
        smooth_sigma_bins=smooth_sigma_bins,
        y_max=args.y_max,
        difference_y_max=args.difference_y_max,
        y_axis_format=args.y_format,
        stem_linewidth=stem_linewidth,
    )
    write_data_summary(
        output_path=data_output_path,
        N=args.N,
        a=args.a,
        true_order=instance.true_order,
        algorithms=algorithms,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        error_model=error_model,
        error_probability=error_probability,
        marker_bins=marker_bins,
        marker_source=marker_source,
        marker_window_bins=args.marker_window_bins,
        experiments=experiments,
        seed=args.seed,
        standard_backend=args.standard_backend,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
        distribution_style=args.distribution_style,
        smooth_sigma_bins=smooth_sigma_bins,
        stem_linewidth=stem_linewidth,
    )

    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}, a = {args.a}, true order r = {instance.true_order}")
    print(f"Phase-estimation bits t = {num_phase_bits}, Q = {1 << num_phase_bits}")
    print(f"Marker bins = {marker_bins} ({marker_source})")
    print(f"Trials per distribution = {args.trials}")
    print(
        "Algorithms = "
        + ", ".join(algorithm_display_label(value) for value in algorithms)
    )
    print(
        "Depolarizing error: "
        + depolarizing_error_display_label(error_model, error_probability)
    )
    print(f"Depolarizing Pauli: {depolarizing_pauli_display_label(depolarizing_paulis)}")
    print(
        "Controlled-U depolarizing region: "
        + depolarizing_region_display_label(args.depolarizing_region)
    )
    print(f"Work register: {work_register_display_label(num_work_qubits)}")
    print(f"Stem linewidth: {'default' if stem_linewidth is None else stem_linewidth}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
