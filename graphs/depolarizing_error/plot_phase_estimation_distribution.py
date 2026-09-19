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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter


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
    normalize_controlled_unitary_depolarizing_paulis,
)
from shor_common.modular_unitaries import validate_padding_phase_radians
from shor_common.shor_utils import default_num_phase_bits
from shor_one_qubit.shor import shor_order_finding_attempt as kitaev_order_attempt
from shor_standard.backends import (
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)
from shor_standard.shor import shor_order_finding_attempt as standard_order_attempt


RANDOM_PAULI_TOKEN = "random"
DISTRIBUTION_STYLE_STEMS = "stems"
DISTRIBUTION_STYLE_SMOOTH = "smooth"
DISTRIBUTION_STYLE_BOTH = "both"
DISTRIBUTION_STYLE_CHOICES = (
    DISTRIBUTION_STYLE_STEMS,
    DISTRIBUTION_STYLE_SMOOTH,
    DISTRIBUTION_STYLE_BOTH,
)
DEFAULT_SMOOTH_SIGMA_BINS = 2.0
DEFAULT_SMOOTH_EVAL_POINTS = 2048
Y_AXIS_FORMAT_AUTO = "auto"
Y_AXIS_FORMAT_DECIMAL = "decimal"
Y_AXIS_FORMAT_SCIENTIFIC = "scientific"
Y_AXIS_FORMAT_CHOICES = (
    Y_AXIS_FORMAT_AUTO,
    Y_AXIS_FORMAT_DECIMAL,
    Y_AXIS_FORMAT_SCIENTIFIC,
)
SMALL_Y_AXIS_SCIENTIFIC_THRESHOLD = 0.01


@dataclass(frozen=True)
class PhaseOutcomeRow:
    bits: str
    outcome_integer: int
    phase_decimal: float
    count: int
    probability: float


@dataclass(frozen=True)
class PhaseDistributionSample:
    algorithm: str
    noise_label: str
    rows: list[PhaseOutcomeRow]
    trials: int
    num_phase_bits: int
    num_work_qubits: int


def bits_to_integer(bits: list[int]) -> int:
    """
    Convert fractional phase bits [b1, ..., bt] to the integer m with phase m/2^t.
    """
    outcome = 0
    for bit in bits:
        outcome = (outcome << 1) | int(bit)
    return outcome


def normalize_depolarizing_pauli_arg(
    raw_pauli: str,
) -> tuple[str, ...] | None:
    if raw_pauli == RANDOM_PAULI_TOKEN:
        return None
    return normalize_controlled_unitary_depolarizing_paulis(raw_pauli)


def depolarizing_pauli_filename_segment(
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
) -> str:
    paulis = normalize_controlled_unitary_depolarizing_paulis(depolarizing_paulis)
    if paulis is None:
        return ""
    return "_pauli_" + "_".join(paulis)


def depolarizing_pauli_display_label(
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
) -> str:
    paulis = normalize_controlled_unitary_depolarizing_paulis(depolarizing_paulis)
    if paulis is None:
        return "uniform random X/Y/Z"
    if len(paulis) == 1:
        return f"{paulis[0]} only"
    return "uniform random over " + "/".join(paulis)


def normalize_padding_phase_radians(padding_phase_radians: float) -> float:
    """Normalize the padded-state base-unitary phase to [0, 2 pi)."""
    phase = validate_padding_phase_radians(padding_phase_radians)
    phase = float(np.remainder(phase, 2.0 * np.pi))
    if np.isclose(phase, 0.0, rtol=0.0, atol=1e-14) or np.isclose(
        phase,
        2.0 * np.pi,
        rtol=0.0,
        atol=1e-14,
    ):
        return 0.0
    return phase


def padding_phase_filename_segment(padding_phase_radians: float) -> str:
    phase = normalize_padding_phase_radians(padding_phase_radians)
    if phase == 0.0:
        return ""
    token = format(phase, ".10g").replace("-", "m").replace(".", "p")
    return f"_padtheta_{token}rad"


def padding_phase_display_label(padding_phase_radians: float) -> str:
    phase = normalize_padding_phase_radians(padding_phase_radians)
    if phase == 0.0:
        return "padding theta = 0 (identity)"
    return (
        f"padding theta = {phase:.6g} rad "
        f"(eigenphase = {phase / (2.0 * np.pi):.6g})"
    )


def apply_probability_y_axis_format(
    ax,
    y_axis_format: str,
    upper_y_limit: float,
) -> None:
    if y_axis_format == Y_AXIS_FORMAT_AUTO:
        y_axis_format = (
            Y_AXIS_FORMAT_SCIENTIFIC
            if upper_y_limit < SMALL_Y_AXIS_SCIENTIFIC_THRESHOLD
            else Y_AXIS_FORMAT_DECIMAL
        )
    if y_axis_format == Y_AXIS_FORMAT_DECIMAL:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif y_axis_format == Y_AXIS_FORMAT_SCIENTIFIC:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
    else:
        raise ValueError("unknown y-axis format")


def validate_smooth_sigma_bins(sigma_bins: float) -> float:
    sigma_bins = float(sigma_bins)
    if not np.isfinite(sigma_bins) or sigma_bins <= 0.0:
        raise ValueError("smooth-sigma-bins must be > 0")
    return sigma_bins


def validate_stem_linewidth(stem_linewidth: float | None) -> float | None:
    if stem_linewidth is None:
        return None
    stem_linewidth = float(stem_linewidth)
    if not np.isfinite(stem_linewidth) or stem_linewidth <= 0.0:
        raise ValueError("stem-linewidth must be > 0")
    return stem_linewidth


def smooth_sigma_filename_token(sigma_bins: float) -> str:
    token = format(validate_smooth_sigma_bins(sigma_bins), ".3g")
    return token.replace("-", "m").replace(".", "p").replace("+", "")


def distribution_style_filename_segment(
    distribution_style: str,
    smooth_sigma_bins: float,
) -> str:
    if distribution_style == DISTRIBUTION_STYLE_STEMS:
        return ""
    if distribution_style not in DISTRIBUTION_STYLE_CHOICES:
        raise ValueError("unknown distribution style")
    return (
        f"_{distribution_style}"
        f"_s{smooth_sigma_filename_token(smooth_sigma_bins)}"
    )


def sample_phase_distribution(
    N: int,
    a: int,
    algorithm: str,
    num_phase_bits: int,
    trials: int,
    rng: np.random.Generator,
    error_config: ErrorConfig | None = None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    num_work_qubits: int | None = None,
    noise_label: str = "clean",
    padding_phase_radians: float = 0.0,
) -> PhaseDistributionSample:
    """
    Sample empirical phase-estimation outcomes for one algorithm/noise setting.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    error_config = error_config or ErrorConfig()
    counts: Counter[int] = Counter()
    phase_denominator = 2 ** num_phase_bits

    for _ in range(trials):
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

        counts[bits_to_integer(result["bits"])] += 1

    rows = []
    for outcome in sorted(counts):
        count = counts[outcome]
        rows.append(
            PhaseOutcomeRow(
                bits=format(outcome, f"0{num_phase_bits}b"),
                outcome_integer=outcome,
                phase_decimal=outcome / phase_denominator,
                count=count,
                probability=count / trials,
            )
        )

    return PhaseDistributionSample(
        algorithm=algorithm,
        noise_label=noise_label,
        rows=rows,
        trials=trials,
        num_phase_bits=num_phase_bits,
        num_work_qubits=effective_work_qubits_for_N(N, num_work_qubits),
    )


def simulate_clean_noisy_phase_distributions(
    N: int,
    a: int,
    algorithms: list[str],
    num_phase_bits: int,
    trials: int,
    error_model: str,
    error_probability: float,
    rng: np.random.Generator,
    true_order: int,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
    padding_phase_radians: float = 0.0,
) -> list[PhaseDistributionSample]:
    padding_phase_radians = normalize_padding_phase_radians(
        padding_phase_radians
    )
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

    samples = []
    for algorithm in algorithms:
        samples.append(
            sample_phase_distribution(
                N=N,
                a=a,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                trials=trials,
                rng=rng,
                error_config=ErrorConfig(),
                standard_backend=standard_backend,
                num_work_qubits=num_work_qubits,
                noise_label="clean p=0",
                padding_phase_radians=padding_phase_radians,
            )
        )
        samples.append(
            sample_phase_distribution(
                N=N,
                a=a,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                trials=trials,
                rng=rng,
                error_config=noisy_error_config,
                standard_backend=standard_backend,
                num_work_qubits=num_work_qubits,
                noise_label=f"noisy {depolarizing_error_display_label(error_model, error_probability)}",
                padding_phase_radians=padding_phase_radians,
            )
        )

    return samples


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    N: int,
    a: int,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
    distribution_style: str = DISTRIBUTION_STYLE_STEMS,
    smooth_sigma_bins: float = DEFAULT_SMOOTH_SIGMA_BINS,
    padding_phase_radians: float = 0.0,
) -> str:
    backend_segment = ""
    if STANDARD_ALGORITHM in normalize_algorithms(algorithms):
        backend_segment = "" if standard_backend == STATEVECTOR_STANDARD_BACKEND else f"{standard_backend}_"
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{backend_segment}"
        f"N{N}_a{a}_t{num_phase_bits}_"
        f"{depolarizing_error_filename_segment(error_model, error_probability)}"
        f"{depolarizing_pauli_filename_segment(depolarizing_paulis)}"
        f"{depolarizing_region_filename_segment(depolarizing_region)}"
        f"{padding_phase_filename_segment(padding_phase_radians)}"
        f"{distribution_style_filename_segment(distribution_style, smooth_sigma_bins)}"
        f"{work_register_filename_segment(num_work_qubits)}"
    )


def circular_gaussian_kernel_normalization(
    num_bins: int,
    sigma_bins: float,
) -> float:
    """
    Approximate the normalization for a wrapped Gaussian over phase bins.
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    sigma_bins = validate_smooth_sigma_bins(sigma_bins)
    if num_bins <= 65_536:
        offsets = np.arange(num_bins, dtype=float)
        distances = np.minimum(offsets, num_bins - offsets)
        return float(np.sum(np.exp(-0.5 * (distances / sigma_bins) ** 2)))

    radius = int(np.ceil(8.0 * sigma_bins))
    if radius <= 200_000:
        offsets = np.arange(-radius, radius + 1, dtype=float)
        return float(np.sum(np.exp(-0.5 * (offsets / sigma_bins) ** 2)))

    # For very wide kernels, the infinite-line Gaussian normalization is a
    # close approximation until the circular support saturates.
    return float(min(num_bins, np.sqrt(2.0 * np.pi) * sigma_bins))


def smoothed_phase_distribution_curve(
    rows: list[PhaseOutcomeRow],
    num_phase_bits: int,
    sigma_bins: float,
    max_eval_points: int = DEFAULT_SMOOTH_EVAL_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a wrapped-Gaussian-smoothed curve for observed phase outcomes.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    sigma_bins = validate_smooth_sigma_bins(sigma_bins)
    if max_eval_points < 2:
        raise ValueError("max_eval_points must be >= 2")

    num_bins = 1 << num_phase_bits
    observed_outcomes = np.array(
        [row.outcome_integer for row in rows],
        dtype=float,
    )
    observed_probabilities = np.array(
        [row.probability for row in rows],
        dtype=float,
    )

    if num_bins <= max_eval_points:
        eval_outcomes = np.arange(num_bins, dtype=float)
    else:
        coarse_outcomes = np.linspace(
            0,
            num_bins,
            max_eval_points,
            endpoint=False,
            dtype=float,
        )
        eval_outcomes = np.unique(
            np.concatenate([coarse_outcomes, observed_outcomes])
        )

    if observed_outcomes.size == 0:
        return eval_outcomes / num_bins, np.zeros_like(eval_outcomes)

    normalization = circular_gaussian_kernel_normalization(
        num_bins=num_bins,
        sigma_bins=sigma_bins,
    )
    smoothed = np.zeros(eval_outcomes.shape, dtype=float)
    chunk_size = 1024
    for start in range(0, observed_outcomes.size, chunk_size):
        stop = min(start + chunk_size, observed_outcomes.size)
        distances = np.abs(
            eval_outcomes[:, None] - observed_outcomes[None, start:stop]
        )
        distances = np.minimum(distances, num_bins - distances)
        weights = np.exp(-0.5 * (distances / sigma_bins) ** 2)
        smoothed += weights @ observed_probabilities[start:stop]

    return eval_outcomes / num_bins, smoothed / normalization


def plot_phase_estimation_distribution(
    N: int,
    a: int,
    true_order: int,
    algorithms: list[str],
    num_phase_bits: int,
    trials: int,
    error_model: str,
    error_probability: float,
    samples: list[PhaseDistributionSample],
    output_path: Path,
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
    y_max: float | None = None,
    y_axis_format: str = Y_AXIS_FORMAT_AUTO,
    distribution_style: str = DISTRIBUTION_STYLE_STEMS,
    smooth_sigma_bins: float = DEFAULT_SMOOTH_SIGMA_BINS,
    padding_phase_radians: float = 0.0,
    stem_linewidth: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if distribution_style not in DISTRIBUTION_STYLE_CHOICES:
        raise ValueError("unknown distribution style")
    if y_axis_format not in Y_AXIS_FORMAT_CHOICES:
        raise ValueError("unknown y-axis format")
    smooth_sigma_bins = validate_smooth_sigma_bins(smooth_sigma_bins)
    stem_linewidth = validate_stem_linewidth(stem_linewidth)
    padding_phase_radians = normalize_padding_phase_radians(
        padding_phase_radians
    )
    padding_eigenphase = padding_phase_radians / (2.0 * np.pi)
    effective_num_work_qubits = effective_work_qubits_for_N(
        N,
        num_work_qubits,
    )
    has_padding = N < (1 << effective_num_work_qubits)

    samples_by_key = {
        (sample.algorithm, sample.noise_label): sample
        for sample in samples
    }
    clean_label = "clean p=0"
    noisy_label = f"noisy {depolarizing_error_display_label(error_model, error_probability)}"
    figure_height = 5.0 if len(algorithms) == 1 else 4.3 * len(algorithms)
    fig, axes = plt.subplots(
        len(algorithms),
        1,
        figsize=(10, figure_height),
        sharex=True,
        squeeze=False,
    )

    clean_color = "#1F6F78"
    noisy_color = "#A23B2A"
    draw_stems = distribution_style in (
        DISTRIBUTION_STYLE_STEMS,
        DISTRIBUTION_STYLE_BOTH,
    )
    draw_smooth = distribution_style in (
        DISTRIBUTION_STYLE_SMOOTH,
        DISTRIBUTION_STYLE_BOTH,
    )
    smooth_curves = {}
    plotted_probabilities = []
    for sample in samples:
        if draw_stems:
            plotted_probabilities.extend(row.probability for row in sample.rows)
        if draw_smooth:
            smooth_curves[(sample.algorithm, sample.noise_label)] = (
                smoothed_phase_distribution_curve(
                    rows=sample.rows,
                    num_phase_bits=num_phase_bits,
                    sigma_bins=smooth_sigma_bins,
                )
            )
            plotted_probabilities.extend(
                smooth_curves[(sample.algorithm, sample.noise_label)][1]
            )
    auto_y_max = max(plotted_probabilities) if plotted_probabilities else 1.0

    for row_index, algorithm in enumerate(algorithms):
        ax = axes[row_index][0]
        clean = samples_by_key[(algorithm, clean_label)]
        noisy = samples_by_key[(algorithm, noisy_label)]

        for sample, color, marker, label, linewidth, alpha in (
            (clean, clean_color, "o", "clean p=0", 1.1, 0.82),
            (noisy, noisy_color, "s", "noisy p", 1.0, 0.72),
        ):
            if draw_stems:
                x_values = [row.phase_decimal for row in sample.rows]
                y_values = [row.probability for row in sample.rows]
                stem_label = label if not draw_smooth else f"{label} samples"
                ax.vlines(
                    x_values,
                    0.0,
                    y_values,
                    color=color,
                    linewidth=(
                        linewidth if stem_linewidth is None else stem_linewidth
                    ),
                    alpha=alpha,
                )
                ax.scatter(
                    x_values,
                    y_values,
                    s=24,
                    marker=marker,
                    color=color,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                    label=stem_label,
                )
            if draw_smooth:
                smooth_x, smooth_y = smooth_curves[
                    (sample.algorithm, sample.noise_label)
                ]
                smooth_label = (
                    f"{label} smoothed" if draw_stems else f"{label} smooth"
                )
                ax.plot(
                    smooth_x,
                    smooth_y,
                    color=color,
                    linestyle=":",
                    linewidth=2.2,
                    alpha=0.95,
                    zorder=4,
                    label=smooth_label,
                )

        if true_order <= 128:
            for numerator in range(true_order):
                ax.axvline(
                    numerator / true_order,
                    color="#808080",
                    linewidth=0.55,
                    alpha=0.16,
                    zorder=0,
                )

        if has_padding and padding_phase_radians != 0.0:
            ax.axvline(
                padding_eigenphase,
                color="#6A4C93",
                linestyle="--",
                linewidth=1.25,
                alpha=0.8,
                zorder=1,
                label="padding eigenphase",
            )

        ax.set_ylabel("Empirical probability")
        ax.set_title(algorithm_display_label(algorithm))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        upper_y_limit = (
            y_max
            if y_max is not None
            else min(1.0, auto_y_max * 1.16 + 0.02)
        )
        ax.set_ylim(0.0, upper_y_limit)
        apply_probability_y_axis_format(
            ax,
            y_axis_format=y_axis_format,
            upper_y_limit=upper_y_limit,
        )
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper right", frameon=True, framealpha=0.88)

    axes[-1][0].set_xlabel("Measured phase estimate (decimal)")
    axes[-1][0].set_xlim(-0.01, 1.01)
    axes[-1][0].xaxis.set_major_locator(MaxNLocator(nbins=8))
    axes[-1][0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    title = (
        "Clean/Noisy Phase-Estimation Distribution"
        f" for N = {N}, a = {a}, t = {num_phase_bits}"
    )
    fig.suptitle(title, y=0.995)
    footer_text = "\n".join(
        [
            " | ".join(
                [
                    f"true order r = {true_order}",
                    f"trials = {trials}",
                    depolarizing_error_display_label(error_model, error_probability),
                    f"Pauli = {depolarizing_pauli_display_label(depolarizing_paulis)}",
                ]
            ),
            " | ".join(
                [
                    f"style = {distribution_style}",
                    f"region = {depolarizing_region_display_label(depolarizing_region)}",
                    work_register_display_label(num_work_qubits),
                    padding_phase_display_label(padding_phase_radians),
                ]
            ),
        ]
    )
    fig.text(
        0.5,
        0.018,
        footer_text,
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.965))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


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
    samples: list[PhaseDistributionSample],
    seed: int | None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    num_work_qubits: int | None = None,
    depolarizing_region=None,
    depolarizing_paulis: ControlledUnitaryDepolarizingPauliSpec = None,
    y_max: float | None = None,
    y_axis_format: str = Y_AXIS_FORMAT_AUTO,
    distribution_style: str = DISTRIBUTION_STYLE_STEMS,
    smooth_sigma_bins: float = DEFAULT_SMOOTH_SIGMA_BINS,
    padding_phase_radians: float = 0.0,
    stem_linewidth: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if distribution_style not in DISTRIBUTION_STYLE_CHOICES:
        raise ValueError("unknown distribution style")
    if y_axis_format not in Y_AXIS_FORMAT_CHOICES:
        raise ValueError("unknown y-axis format")
    smooth_sigma_bins = validate_smooth_sigma_bins(smooth_sigma_bins)
    stem_linewidth = validate_stem_linewidth(stem_linewidth)
    padding_phase_radians = normalize_padding_phase_radians(
        padding_phase_radians
    )

    seed_label = "random" if seed is None else str(seed)
    lines = [
        "Depolarizing Phase-Estimation Distribution Summary",
        f"N = {N}",
        f"a = {a}",
        f"True order r = {true_order}",
        f"Algorithms = {', '.join(algorithm_display_label(algorithm) for algorithm in algorithms)}",
        f"Standard backend = {standard_backend}",
        f"Phase-estimation bits = {num_phase_bits}",
        f"Trials per clean/noisy distribution = {trials}",
        f"Depolarizing error = {depolarizing_error_display_label(error_model, error_probability)}",
        f"Depolarizing Pauli = {depolarizing_pauli_display_label(depolarizing_paulis)}",
        f"Controlled-U depolarizing region = {depolarizing_region_display_label(depolarizing_region)}",
        f"Work register = {work_register_display_label(num_work_qubits)}",
        f"Padded-state extension = {padding_phase_display_label(padding_phase_radians)}",
        f"Distribution style = {distribution_style}",
        f"Stem linewidth = {'default' if stem_linewidth is None else stem_linewidth}",
        f"Smooth sigma bins = {smooth_sigma_bins}",
        f"Y-axis upper limit = {'auto' if y_max is None else y_max}",
        f"Y-axis tick format = {y_axis_format}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        "  Clean p=0 and noisy depolarizing empirical histograms are overlaid",
        "  for each selected phase-estimation algorithm.",
        "  Phase values are decimal representations of measured m / 2^t.",
        "  Smoothed curves use a circular Gaussian kernel on the m / 2^t grid.",
        "  Light vertical guide lines mark exact phases s/r when r <= 128.",
        "  For padded y >= N, U applies exp(i * theta) and U^power applies",
        "  exp(i * power * theta). A dashed guide marks theta / (2 pi).",
        "",
    ]

    for sample in samples:
        lines.extend(
            [
                f"{algorithm_display_label(sample.algorithm)} | {sample.noise_label}",
                f"  Effective work-register qubits L = {sample.num_work_qubits}",
                "  bits | outcome integer m | decimal phase | count | empirical probability",
            ]
        )
        for row in sample.rows:
            lines.append(
                "  "
                f"{row.bits} | "
                f"{row.outcome_integer:>17} | "
                f"{row.phase_decimal:>13.6f} | "
                f"{row.count:>5} | "
                f"{row.probability:>21.6f}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay clean and depolarizing-noisy empirical phase-estimation "
            "distributions for a fixed (N, a)."
        )
    )
    parser.add_argument("N", type=int, help="Composite number to analyze")
    parser.add_argument("--a", type=int, required=True, help="Fixed coprime base")
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        help="Algorithm distributions to plot",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Phase-estimation bits t (default: 2 * ceil(log2(N)) + 1)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Trials for each clean/noisy distribution",
    )
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
        "--padding-phase",
        "--padding-phase-radians",
        dest="padding_phase_radians",
        type=float,
        default=0.0,
        help=(
            "Base-unitary phase theta in radians for padded states y >= N; "
            "default 0 keeps the identity extension"
        ),
    )
    parser.add_argument(
        "--depolarizing-pauli",
        choices=(RANDOM_PAULI_TOKEN, *CONTROLLED_UNITARY_DEPOLARIZING_PAULI_CHOICES),
        default=RANDOM_PAULI_TOKEN,
        help="Use random X/Y/Z depolarizing hits, or restrict every hit to X, Y, or Z",
    )
    add_depolarizing_region_argument(parser)
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--distribution-style",
        choices=DISTRIBUTION_STYLE_CHOICES,
        default=DISTRIBUTION_STYLE_STEMS,
        help="Show empirical stems, circular-Gaussian smooth curves, or both",
    )
    parser.add_argument(
        "--smooth-sigma-bins",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA_BINS,
        help=(
            "Circular Gaussian smoothing width in measured phase-grid bins; "
            "used when --distribution-style is smooth or both"
        ),
    )
    parser.add_argument(
        "--stem-linewidth",
        type=float,
        default=None,
        help="Optional linewidth for empirical distribution stems",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=None,
        help="Optional upper limit for the empirical-probability y-axis",
    )
    parser.add_argument(
        "--y-format",
        choices=Y_AXIS_FORMAT_CHOICES,
        default=Y_AXIS_FORMAT_AUTO,
        help=(
            "Y-axis tick format. Auto switches to scientific notation for "
            "small probability ranges."
        ),
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
        p = validate_error_probabilities([args.p])[0]
        if args.bits is None:
            num_phase_bits = default_num_phase_bits(args.N)
        else:
            num_phase_bits = int(args.bits)
        if num_phase_bits < 1:
            raise ValueError("bits must be >= 1")
        if args.trials < 1:
            raise ValueError("trials must be >= 1")
        smooth_sigma_bins = validate_smooth_sigma_bins(args.smooth_sigma_bins)
        stem_linewidth = validate_stem_linewidth(args.stem_linewidth)
        if args.y_max is not None and args.y_max <= 0.0:
            raise ValueError("y-max must be > 0")
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
        padding_phase_radians = normalize_padding_phase_radians(
            args.padding_phase_radians
        )
    except ValueError as exc:
        parser.error(str(exc))

    rng = np.random.default_rng(args.seed)
    samples = simulate_clean_noisy_phase_distributions(
        N=args.N,
        a=args.a,
        algorithms=algorithms,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        error_model=error_model,
        error_probability=p,
        rng=rng,
        true_order=instance.true_order,
        standard_backend=args.standard_backend,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
        padding_phase_radians=padding_phase_radians,
    )

    suffix = output_filename_segment(
        algorithms=algorithms,
        standard_backend=args.standard_backend,
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        error_model=error_model,
        error_probability=p,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
        distribution_style=args.distribution_style,
        smooth_sigma_bins=smooth_sigma_bins,
        padding_phase_radians=padding_phase_radians,
    )
    figure_output_path = (
        args.output
        if args.output is not None
        else DEFAULT_FIGURE_DIR / f"depolarizing_phase_estimation_distribution_{suffix}.png"
    )
    data_output_path = (
        DEFAULT_DATA_DIR
        / f"depolarizing_phase_estimation_distribution_summary_{suffix}.txt"
    )

    plot_phase_estimation_distribution(
        N=args.N,
        a=args.a,
        true_order=instance.true_order,
        algorithms=algorithms,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        error_model=error_model,
        error_probability=p,
        samples=samples,
        output_path=figure_output_path,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
        y_max=args.y_max,
        y_axis_format=args.y_format,
        distribution_style=args.distribution_style,
        smooth_sigma_bins=smooth_sigma_bins,
        padding_phase_radians=padding_phase_radians,
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
        error_probability=p,
        samples=samples,
        seed=args.seed,
        standard_backend=args.standard_backend,
        num_work_qubits=num_work_qubits,
        depolarizing_region=args.depolarizing_region,
        depolarizing_paulis=depolarizing_paulis,
        y_max=args.y_max,
        y_axis_format=args.y_format,
        distribution_style=args.distribution_style,
        smooth_sigma_bins=smooth_sigma_bins,
        padding_phase_radians=padding_phase_radians,
        stem_linewidth=stem_linewidth,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}")
    print(f"a = {args.a}")
    print(f"True order r = {instance.true_order}")
    print(f"Algorithms = {', '.join(algorithm_display_label(algorithm) for algorithm in algorithms)}")
    print(f"Phase-estimation bits = {num_phase_bits}")
    print(f"Trials per clean/noisy distribution = {args.trials}")
    print(f"Depolarizing error: {depolarizing_error_display_label(error_model, p)}")
    print(f"Depolarizing Pauli: {depolarizing_pauli_display_label(depolarizing_paulis)}")
    print(
        "Controlled-U depolarizing region: "
        f"{depolarizing_region_display_label(args.depolarizing_region)}"
    )
    print(f"Work register: {work_register_display_label(num_work_qubits)}")
    print(f"Padded-state extension: {padding_phase_display_label(padding_phase_radians)}")
    print(f"Distribution style: {args.distribution_style}")
    print(f"Stem linewidth: {'default' if stem_linewidth is None else stem_linewidth}")
    print(f"Smooth sigma bins: {smooth_sigma_bins}")
    print(f"Y-axis upper limit: {'auto' if args.y_max is None else args.y_max}")
    print(f"Y-axis tick format: {args.y_format}")
    print(f"Seed = {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
