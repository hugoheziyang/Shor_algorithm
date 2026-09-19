from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

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


from error_correction.code_distance import (
    decimal_distance,
    hamming_distance,
    run_phase_estimation_hamming_distance_trial,
)
from error_correction.classical_error_correction import (
    can_recover_clean_bits_by_single_flip,
    single_bit_flip_candidates,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    ALGORITHM_CHOICES,
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
from shor_common.errors import (
    ErrorConfig,
    apply_measurement_bitflip,
    choose_measurement_bitflip_mask,
    format_measurement_bitflip_probability,
)
from shor_common.modular_unitaries import work_dim
from shor_common.shor_utils import make_basis_state, make_controlled_power_factory
from shor_standard.analytic_sampling import (
    QPEWindowMassDiagnostics,
    sample_standard_phase_bits_from_order,
)
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
    STATEVECTOR_STANDARD_BACKEND,
)
from shor_standard.phase_estimation import (
    bits_to_phase_estimate,
    standard_phase_estimation_with_bitflip_mask,
)


HAMMING_METRIC = "hamming"
DECIMAL_METRIC = "decimal"
METRIC_CHOICES = (HAMMING_METRIC, DECIMAL_METRIC)
DEFAULT_INSTANCES_FILE = REPO_ROOT / "graphs" / "config" / "order_recovery_instances.json"
DEFAULT_NUM_PHASE_BITS = 29
DEFAULT_MAX_STATEVECTOR_AMPLITUDES = 8_000_000
SINGLE_BITFLIP_ALL_RANGE = "all"
SINGLE_BITFLIP_LEADING_RANGE = "leading"
SINGLE_BITFLIP_DANGEROUS_RANGE = "dangerous"
SINGLE_BITFLIP_SAFE_TAIL_RANGE = "safe-tail"
SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES = (
    SINGLE_BITFLIP_ALL_RANGE,
    SINGLE_BITFLIP_LEADING_RANGE,
    SINGLE_BITFLIP_DANGEROUS_RANGE,
    SINGLE_BITFLIP_SAFE_TAIL_RANGE,
)
SingleBitflipRangeSpec = str | tuple[str, ...] | list[str] | None
ScoreT = TypeVar("ScoreT")


@dataclass(frozen=True)
class PhaseDistanceTrial:
    clean_bits: list[int]
    noisy_bits: list[int]
    clean_phase_estimate: float
    noisy_phase_estimate: float
    measurement_bitflip_mask: list[bool]

    @property
    def hamming_distance(self) -> int:
        return hamming_distance(self.clean_bits, self.noisy_bits)

    @property
    def decimal_distance(self) -> float:
        return decimal_distance(self.clean_bits, self.noisy_bits)


@dataclass(frozen=True)
class PhaseDistanceOrderResult:
    instance: OrderRecoveryInstance
    algorithm: str
    trials: int
    num_phase_bits: int
    measurement_error_probability: float
    total_hamming_distance: int
    total_decimal_distance: float
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND

    @property
    def average_hamming_distance(self) -> float:
        return self.total_hamming_distance / self.trials

    @property
    def average_normalized_hamming_distance(self) -> float:
        return self.total_hamming_distance / (self.trials * self.num_phase_bits)

    @property
    def average_decimal_distance(self) -> float:
        return self.total_decimal_distance / self.trials


def validate_measurement_error_probability(probability: float) -> float:
    """
    Validate an independent per-bit measurement flip probability.
    """
    p = float(probability)
    if not math.isfinite(p) or not (0.0 <= p <= 1.0):
        raise ValueError("measurement error probability must satisfy 0 <= p <= 1")
    return p


def measurement_probability_filename_segment(probability: float) -> str:
    """
    Return a filename segment for one measurement bit-flip probability.
    """
    probability = validate_measurement_error_probability(probability)
    token = format_measurement_bitflip_probability(probability).replace("-", "m")
    return f"p_{token}_"


def validate_measurement_error_probabilities(probabilities: list[float]) -> list[float]:
    """
    Validate and de-duplicate independent per-bit measurement flip probabilities.
    """
    if not probabilities:
        raise ValueError("at least one error probability is required")

    normalized = []
    for probability in probabilities:
        p = validate_measurement_error_probability(probability)
        if p not in normalized:
            normalized.append(p)

    return normalized


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


def measurement_probability_display_label(probability: float) -> str:
    """
    Return a compact display label for one measurement bit-flip probability.
    """
    probability = validate_measurement_error_probability(probability)
    return f"p={format_measurement_bitflip_probability(probability)}"


def measurement_error_config_for_probability(
    probability: float,
    single_measurement_bitflip: bool = False,
) -> ErrorConfig:
    """
    Return the measurement bit-flip ErrorConfig for one error mode.
    """
    if single_measurement_bitflip:
        return ErrorConfig(measurement_bitflip=True)
    probability = validate_measurement_error_probability(probability)
    if probability == 0.0:
        return ErrorConfig()
    return ErrorConfig(
        measurement_bitflip=True,
        measurement_bitflip_probability=probability,
    )


def order_resolution_bits(true_order: int) -> int:
    """
    Return m = ceil(log2(r)), the bit scale for order spacing 1/r.
    """
    if true_order < 1:
        raise ValueError("true_order must be >= 1")
    if true_order == 1:
        return 0
    return (true_order - 1).bit_length()


def normalize_single_measurement_bitflip_ranges(
    range_names: SingleBitflipRangeSpec,
) -> tuple[str, ...] | None:
    """
    Normalize one or more forced bit-flip range names.
    """
    if range_names is None:
        return None
    raw_names = [range_names] if isinstance(range_names, str) else list(range_names)
    if not raw_names:
        raise ValueError("at least one single-measurement-bitflip range is required")

    normalized = []
    for range_name in raw_names:
        if range_name not in SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES:
            raise ValueError("unknown single-measurement-bitflip range")
        if range_name not in normalized:
            normalized.append(range_name)
    return tuple(normalized)


def _single_measurement_bitflip_bit_numbers_for_one_range(
    num_phase_bits: int,
    true_order: int,
    range_name: str,
) -> tuple[int, ...]:
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if range_name not in SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES:
        raise ValueError("unknown single-measurement-bitflip range")

    if range_name == SINGLE_BITFLIP_ALL_RANGE:
        start = 1
        end = num_phase_bits
    else:
        m = order_resolution_bits(true_order)
        if range_name == SINGLE_BITFLIP_LEADING_RANGE:
            start = 1
            end = min(m, num_phase_bits)
        elif range_name == SINGLE_BITFLIP_DANGEROUS_RANGE:
            start = m + 1
            end = min(2 * m, num_phase_bits)
        else:
            start = (2 * m) + 1
            end = num_phase_bits

    if start > end:
        raise ValueError(
            f"single-measurement-bitflip range {range_name!r} is empty for "
            f"r={true_order}, t={num_phase_bits}"
        )
    return tuple(range(start, end + 1))


def single_measurement_bitflip_bit_numbers(
    num_phase_bits: int,
    true_order: int,
    range_name: SingleBitflipRangeSpec,
) -> tuple[int, ...]:
    """
    Return 1-indexed phase-bit numbers eligible for one forced readout flip.

    Bit number j refers to the coefficient of 2^-j in 0.b1...bt. For
    r=2^m, the ranges are exact: leading bits 1..m, dangerous bits
    m+1..2m, and safe-tail bits 2m+1..t. Multiple ranges mean the union.
    """
    range_names = normalize_single_measurement_bitflip_ranges(range_name)
    if range_names is None:
        range_names = (SINGLE_BITFLIP_ALL_RANGE,)

    bit_numbers: set[int] = set()
    for name in range_names:
        bit_numbers.update(
            _single_measurement_bitflip_bit_numbers_for_one_range(
                num_phase_bits=num_phase_bits,
                true_order=true_order,
                range_name=name,
            )
        )
    return tuple(sorted(bit_numbers))


def choose_single_measurement_bitflip_mask(
    num_phase_bits: int,
    rng: np.random.Generator,
    bit_numbers: tuple[int, ...] | None = None,
) -> list[bool]:
    """
    Choose a one-hot measurement bit-flip mask over 1-indexed phase bits.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if bit_numbers is None:
        bit_numbers = tuple(range(1, num_phase_bits + 1))
    if not bit_numbers:
        raise ValueError("single-measurement-bitflip range must be nonempty")
    for bit_number in bit_numbers:
        if not (1 <= int(bit_number) <= num_phase_bits):
            raise ValueError("single-measurement-bitflip bit numbers are out of range")

    chosen_bit = int(rng.choice(list(bit_numbers)))
    mask = [False] * num_phase_bits
    mask[chosen_bit - 1] = True
    return mask


def single_bit_flip_corrected_bits(
    clean_bits: list[int],
    noisy_bits: list[int],
) -> list[int]:
    """
    Return the single-bit-search corrected bit string for diagnostics.

    This matches the existing classical recovery diagnostic: if the clean
    paired bit string is among the noisy string and its one-bit-flip neighbors,
    use the clean string; otherwise keep the noisy string.
    """
    if can_recover_clean_bits_by_single_flip(
        noisy_bits=noisy_bits,
        clean_bits=clean_bits,
    ):
        return list(clean_bits)
    return list(noisy_bits)


def minimum_single_bit_flip_candidate(
    bits: list[int],
    score_candidate: Callable[[list[int]], ScoreT],
) -> tuple[list[int], ScoreT]:
    """
    Return the original bit string or a one-bit neighbor with lower score.

    The original bit string is considered first and wins ties. This is useful
    for diagnostics where a classical single-bit search should only update a
    noisy bit string when a one-bit correction strictly improves the measured
    criterion.
    """
    candidates = single_bit_flip_candidates(bits)
    best_candidate = candidates[0]
    best_score = score_candidate(best_candidate)

    for candidate in candidates[1:]:
        candidate_score = score_candidate(candidate)
        if candidate_score < best_score:
            best_candidate = candidate
            best_score = candidate_score

    return list(best_candidate), best_score


def run_kitaev_phase_distance_trial(
    instance: OrderRecoveryInstance,
    num_phase_bits: int,
    measurement_error_probability: float,
    rng: np.random.Generator,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
) -> PhaseDistanceTrial:
    """
    Run one paired clean/noisy Kitaev PE trial.
    """
    measurement_bitflip_mask = None
    if single_measurement_bitflip or single_measurement_bitflip_bits is not None:
        measurement_bitflip_mask = choose_single_measurement_bitflip_mask(
            num_phase_bits,
            rng=rng,
            bit_numbers=single_measurement_bitflip_bits,
        )
    trial = run_phase_estimation_hamming_distance_trial(
        N=instance.N,
        a=instance.a,
        num_phase_bits=num_phase_bits,
        error_config=measurement_error_config_for_probability(
            measurement_error_probability,
            single_measurement_bitflip=(
                single_measurement_bitflip
                or measurement_bitflip_mask is not None
            ),
        ),
        measurement_bitflip_mask=measurement_bitflip_mask,
        rng=rng,
    )
    return PhaseDistanceTrial(
        clean_bits=trial.clean_bits,
        noisy_bits=trial.noisy_bits,
        clean_phase_estimate=trial.clean_phase_estimate,
        noisy_phase_estimate=trial.noisy_phase_estimate,
        measurement_bitflip_mask=trial.measurement_bitflip_mask,
    )


def run_standard_phase_distance_trial(
    instance: OrderRecoveryInstance,
    num_phase_bits: int,
    measurement_error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
) -> PhaseDistanceTrial:
    """
    Run one paired clean/noisy standard PE trial.

    The paired standard trial shares the same raw control-register outcome;
    noise is applied only to the final measured bits.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")
    measurement_error_probability = validate_measurement_error_probability(
        measurement_error_probability
    )
    if standard_backend not in STANDARD_BACKEND_CHOICES:
        raise ValueError("unknown standard backend")

    measurement_seed = int(rng.integers(0, 2 ** 63 - 1))
    measurement_error_seed = int(rng.integers(0, 2 ** 63 - 1))
    if single_measurement_bitflip or single_measurement_bitflip_bits is not None:
        measurement_bitflip_mask = choose_single_measurement_bitflip_mask(
            num_phase_bits,
            rng=np.random.default_rng(measurement_error_seed),
            bit_numbers=single_measurement_bitflip_bits,
        )
    else:
        measurement_bitflip_mask = choose_measurement_bitflip_mask(
            num_phase_bits,
            error_config=measurement_error_config_for_probability(
                measurement_error_probability,
            ),
            rng=np.random.default_rng(measurement_error_seed),
        )

    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        initial_work_state = make_basis_state(1, work_dim(instance.N))
        controlled_power_factory = make_controlled_power_factory(
            instance.a,
            instance.N,
        )
        clean_bits, clean_phase_estimate, _ = standard_phase_estimation_with_bitflip_mask(
            initial_work_state=initial_work_state,
            controlled_power_factory=controlled_power_factory,
            num_control_qubits=num_phase_bits,
            measurement_bitflip_mask=[False] * num_phase_bits,
            measurement_rng=np.random.default_rng(measurement_seed),
        )
    else:
        clean_bits, clean_phase_estimate, _, _ = sample_standard_phase_bits_from_order(
            order=instance.true_order,
            num_control_qubits=num_phase_bits,
            rng=np.random.default_rng(measurement_seed),
            window=analytic_window,
            measurement_bitflip_mask=[False] * num_phase_bits,
        )

    noisy_bits = [
        apply_measurement_bitflip(bit, should_flip=should_flip)
        for bit, should_flip in zip(clean_bits, measurement_bitflip_mask)
    ]
    noisy_phase_estimate = bits_to_phase_estimate(noisy_bits)

    return PhaseDistanceTrial(
        clean_bits=clean_bits,
        noisy_bits=noisy_bits,
        clean_phase_estimate=clean_phase_estimate,
        noisy_phase_estimate=noisy_phase_estimate,
        measurement_bitflip_mask=measurement_bitflip_mask,
    )


def simulate_phase_distance_for_instance(
    instance: OrderRecoveryInstance,
    algorithm: str,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    measurement_error_probability: float,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> PhaseDistanceOrderResult:
    """
    Estimate clean/noisy PE distances for one configured order instance.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    measurement_error_probability = validate_measurement_error_probability(
        measurement_error_probability
    )

    total_hamming_distance = 0
    total_decimal_distance = 0.0
    for _ in range(trials):
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

        total_hamming_distance += trial.hamming_distance
        total_decimal_distance += trial.decimal_distance

    return PhaseDistanceOrderResult(
        instance=instance,
        algorithm=algorithm,
        trials=trials,
        num_phase_bits=num_phase_bits,
        measurement_error_probability=measurement_error_probability,
        total_hamming_distance=total_hamming_distance,
        total_decimal_distance=total_decimal_distance,
        standard_backend=standard_backend,
    )


def result_y_value(
    result: PhaseDistanceOrderResult,
    metric: str,
    normalized: bool = False,
) -> float:
    """
    Return the plotted metric for one result.
    """
    if metric == HAMMING_METRIC:
        if normalized:
            return result.average_normalized_hamming_distance
        return result.average_hamming_distance
    if metric == DECIMAL_METRIC:
        return result.average_decimal_distance
    raise ValueError("unknown metric")


def metric_display_label(metric: str, normalized: bool = False) -> str:
    """
    Return a human-readable metric label.
    """
    if metric == HAMMING_METRIC:
        if normalized:
            return "Average normalized Hamming distance"
        return "Average Hamming distance"
    if metric == DECIMAL_METRIC:
        return "Average circular decimal distance"
    raise ValueError("unknown metric")


def metric_title_label(metric: str, normalized: bool = False) -> str:
    """
    Return the shorter metric label used in plot titles.
    """
    if metric == DECIMAL_METRIC:
        return "Average decimal distance"
    return metric_display_label(metric, normalized)


def metric_filename_label(metric: str, normalized: bool = False) -> str:
    """
    Return a filename-safe metric label.
    """
    if metric == HAMMING_METRIC:
        return "normalized_hamming" if normalized else "hamming"
    if metric == DECIMAL_METRIC:
        return "decimal"
    raise ValueError("unknown metric")


def write_data_summary(
    output_path: Path,
    metric: str,
    results: list[PhaseDistanceOrderResult],
    algorithms: list[str],
    instances_file: Path,
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    trials_per_instance: int,
    min_order: int | None,
    max_order: int | None,
    seed: int | None,
    normalized: bool = False,
    analytic_window_diagnostics: list[QPEWindowMassDiagnostics] | None = None,
) -> None:
    """
    Write run metadata plus per-order distance data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    seed_label = "random" if seed is None else str(seed)
    min_order_label = "none" if min_order is None else str(min_order)
    max_order_label = "none" if max_order is None else str(max_order)
    measurement_error_probabilities = validate_measurement_error_probabilities(
        measurement_error_probabilities
    )
    probability_label = ", ".join(
        measurement_probability_display_label(probability)
        for probability in measurement_error_probabilities
    )

    lines = [
        f"Phase-Estimation {metric_display_label(metric, normalized)} vs Order Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
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
        "Graph note:",
        "  Each configured (N, a) pair contributes one point at its true order r",
        "  Each trial compares a clean phase-estimation bit string to a noisy bit string",
        "  Kitaev noisy trials store flipped bits, so later feedback can change",
        "  Standard noisy trials apply flips only to the final measured bit string",
        "  Distance is measured before continued-fraction order recovery",
        "",
        "Per-order data:",
        "  algorithm | p | true r | N | a | avg hamming | avg normalized hamming | avg decimal | total hamming | total decimal | trials",
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
            f" {result.average_hamming_distance:>11.3f} |"
            f" {result.average_normalized_hamming_distance:>22.6f} |"
            f" {result.average_decimal_distance:>11.6f} |"
            f" {result.total_hamming_distance:>13} |"
            f" {result.total_decimal_distance:>13.6f} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_phase_distance_vs_order(
    output_path: Path,
    metric: str,
    results: list[PhaseDistanceOrderResult],
    algorithms: list[str],
    num_phase_bits: int,
    measurement_error_probabilities: list[float],
    trials_per_instance: int,
    normalized: bool = False,
    log_x: bool = False,
) -> None:
    """
    Create and save a phase-distance-vs-order plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    measurement_error_probabilities = validate_measurement_error_probabilities(
        measurement_error_probabilities
    )

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

    for probability in measurement_error_probabilities:
        for algorithm in algorithms:
            algorithm_results = sorted(
                [
                    result
                    for result in results
                    if result.algorithm == algorithm
                    and result.measurement_error_probability == probability
                ],
                key=lambda result: result.instance.true_order,
            )
            ax.plot(
                [result.instance.true_order for result in algorithm_results],
                [
                    result_y_value(result, metric=metric, normalized=normalized)
                    for result in algorithm_results
                ],
                marker=markers[algorithm],
                linestyle=linestyles[algorithm],
                linewidth=2.0,
                markersize=5.5,
                color=algorithm_color_for_base(
                    probability_colors[probability],
                    algorithm,
                ),
                alpha=0.9,
            )

    probability_title = (
        measurement_probability_display_label(measurement_error_probabilities[0])
        if len(measurement_error_probabilities) == 1
        else "multiple p values"
    )
    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel(metric_display_label(metric, normalized))
    ax.set_title(
        f"{metric_title_label(metric, normalized)} vs True Order\n"
        f"{algorithms_display_label(algorithms)}, {probability_title}, "
        f"t = {num_phase_bits}"
    )
    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if metric == DECIMAL_METRIC:
        ax.set_ylim(-0.015, 0.515)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    elif normalized:
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        max_y = max(
            result.average_hamming_distance
            for result in results
        )
        ax.set_ylim(-0.05, max(num_phase_bits, max_y) + 0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

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
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_argument_parser(metric: str) -> argparse.ArgumentParser:
    """
    Build the command-line parser for a phase-distance-vs-order script.
    """
    if metric not in METRIC_CHOICES:
        raise ValueError("unknown metric")

    parser = argparse.ArgumentParser(
        description=(
            f"Plot {metric_display_label(metric).lower()} between clean and "
            "measurement-noisy phase-estimation bit strings versus true order r."
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
            "one figure. This is mutually exclusive with --p."
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
    if metric == HAMMING_METRIC:
        parser.add_argument(
            "--normalized",
            action="store_true",
            help="Plot average Hamming distance divided by t",
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
        help="Optional custom output path for the figure",
    )
    return parser


def main(metric: str) -> None:
    """
    Run one phase-distance-vs-order graph script.
    """
    start_time = time.perf_counter()
    parser = build_argument_parser(metric)
    args = parser.parse_args()
    algorithms = normalize_algorithms(args.algorithm)
    normalized = bool(getattr(args, "normalized", False))

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
    results = []
    for measurement_error_probability in measurement_error_probabilities:
        for algorithm in algorithms:
            for instance in instances:
                instance_rng = np.random.default_rng(
                    int(master_rng.integers(0, 2 ** 63 - 1))
                )
                results.append(
                    simulate_phase_distance_for_instance(
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

    probability_segment = measurement_probabilities_filename_segment(
        measurement_error_probabilities
    )
    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, args.standard_backend)}"
        f"{instances_file_filename_segment(args.instances_file)}"
    )
    metric_label = metric_filename_label(metric, normalized=normalized)
    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / (
                f"phase_{metric_label}_distance_vs_order_"
                f"{algorithm_prefix}{probability_segment}t{args.bits}.png"
            )
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / (
            f"phase_{metric_label}_distance_vs_order_summary_"
            f"{algorithm_prefix}{probability_segment}t{args.bits}.txt"
        )
    )

    plot_phase_distance_vs_order(
        output_path=figure_output_path,
        metric=metric,
        results=results,
        algorithms=algorithms,
        num_phase_bits=args.bits,
        measurement_error_probabilities=measurement_error_probabilities,
        trials_per_instance=args.trials_per_instance,
        normalized=normalized,
        log_x=args.log_x,
    )
    write_data_summary(
        output_path=data_output_path,
        metric=metric,
        results=results,
        algorithms=algorithms,
        instances_file=args.instances_file,
        num_phase_bits=args.bits,
        measurement_error_probabilities=measurement_error_probabilities,
        trials_per_instance=args.trials_per_instance,
        min_order=args.min_order,
        max_order=args.max_order,
        seed=args.seed,
        normalized=normalized,
        analytic_window_diagnostics=analytic_window_diagnostics,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    probability_label = ", ".join(
        measurement_probability_display_label(probability)
        for probability in measurement_error_probabilities
    )
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"Algorithm: {algorithms_display_label(algorithms)}")
    print(f"Metric: {metric_display_label(metric, normalized)}")
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
