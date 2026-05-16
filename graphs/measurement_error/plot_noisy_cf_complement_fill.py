from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
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
    SingleBitflipRangeSpec,
    measurement_probability_display_label,
    normalize_single_measurement_bitflip_ranges,
    run_kitaev_phase_distance_trial,
    run_standard_phase_distance_trial,
    single_measurement_bitflip_bit_numbers,
    SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES,
    validate_measurement_error_probability,
)
from graphs.measurement_error.plot_cf_bound_order_recovery_vs_order import (
    nearest_order_phase_distance,
)
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


CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "cf_membership"
DEFAULT_MAX_CACHE_BITSTRINGS = 5_000_000


@dataclass(frozen=True)
class CFMembershipCache:
    true_order: int
    num_phase_bits: int
    inside_count: int
    outside_count: int
    inside_bits_packed: np.ndarray
    path: Path | None = None
    from_cache: bool = False

    @property
    def num_bitstrings(self) -> int:
        return 1 << self.num_phase_bits

    def is_inside_index(self, index: int) -> bool:
        if not (0 <= index < self.num_bitstrings):
            raise ValueError("bitstring index is outside the phase-bit universe")
        byte = int(self.inside_bits_packed[index >> 3])
        return bool(byte & (1 << (index & 7)))

    def is_outside_index(self, index: int) -> bool:
        return not self.is_inside_index(index)


@dataclass(frozen=True)
class NoisyCFComplementFillResult:
    instance: OrderRecoveryInstance
    algorithm: str
    measurement_error_probability: float
    trials: int
    num_phase_bits: int
    outside_unique_hits: int
    outside_trial_hits: int
    total_inside_bitstrings: int
    total_outside_bitstrings: int
    standard_backend: str = ANALYTIC_STANDARD_BACKEND
    classical_error_correction: bool = False
    single_measurement_bitflip: bool = False
    single_measurement_bitflip_range: tuple[str, ...] | None = None

    @property
    def outside_fill_fraction(self) -> float:
        if self.total_outside_bitstrings == 0:
            return math.nan
        return self.outside_unique_hits / self.total_outside_bitstrings

    @property
    def outside_trial_probability(self) -> float:
        return self.outside_trial_hits / self.trials


def bits_to_index(bits: list[int]) -> int:
    """
    Convert phase bits [b1, ..., bt] to the integer index of b1...bt.
    """
    index = 0
    for bit in bits:
        bit_value = int(bit)
        if bit_value not in (0, 1):
            raise ValueError("bits must be 0 or 1")
        index = (index << 1) | bit_value
    return index


def validate_cache_parameters(true_order: int, num_phase_bits: int) -> None:
    if true_order < 1:
        raise ValueError("true_order must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")


def cache_output_path(
    cache_dir: Path,
    true_order: int,
    num_phase_bits: int,
) -> Path:
    """
    Return the cache path for one (r, t) membership universe.
    """
    validate_cache_parameters(true_order, num_phase_bits)
    return (
        cache_dir
        / f"cf_membership_r{true_order}_t{num_phase_bits}_v{CACHE_VERSION}.npz"
    )


def save_cf_membership_cache(cache: CFMembershipCache, output_path: Path) -> None:
    """
    Save a packed S-membership bitset for one (r, t).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.array(CACHE_VERSION, dtype=np.int64),
        true_order=np.array(cache.true_order, dtype=np.int64),
        num_phase_bits=np.array(cache.num_phase_bits, dtype=np.int64),
        inside_count=np.array(cache.inside_count, dtype=np.int64),
        outside_count=np.array(cache.outside_count, dtype=np.int64),
        bitorder=np.array("little"),
        inside_bits_packed=cache.inside_bits_packed.astype(np.uint8, copy=False),
    )


def load_cf_membership_cache(
    cache_path: Path,
    true_order: int,
    num_phase_bits: int,
) -> CFMembershipCache:
    """
    Load and validate a cached S-membership bitset.
    """
    validate_cache_parameters(true_order, num_phase_bits)
    with np.load(cache_path, allow_pickle=False) as data:
        version = int(data["version"])
        cached_order = int(data["true_order"])
        cached_bits = int(data["num_phase_bits"])
        inside_count = int(data["inside_count"])
        outside_count = int(data["outside_count"])
        bitorder = str(data["bitorder"])
        inside_bits_packed = data["inside_bits_packed"].astype(
            np.uint8,
            copy=True,
        )

    if version != CACHE_VERSION:
        raise ValueError("cached CF membership version is incompatible")
    if cached_order != true_order or cached_bits != num_phase_bits:
        raise ValueError("cached CF membership key does not match requested r,t")
    if bitorder != "little":
        raise ValueError("cached CF membership bit order is incompatible")

    num_bitstrings = 1 << num_phase_bits
    expected_packed_length = (num_bitstrings + 7) // 8
    if len(inside_bits_packed) != expected_packed_length:
        raise ValueError("cached CF membership bitset has the wrong length")
    if inside_count + outside_count != num_bitstrings:
        raise ValueError("cached CF membership counts do not sum to 2^t")

    return CFMembershipCache(
        true_order=true_order,
        num_phase_bits=num_phase_bits,
        inside_count=inside_count,
        outside_count=outside_count,
        inside_bits_packed=inside_bits_packed,
        path=cache_path,
        from_cache=True,
    )


def build_cf_membership_cache(
    true_order: int,
    num_phase_bits: int,
    max_cache_bitstrings: int = DEFAULT_MAX_CACHE_BITSTRINGS,
    output_path: Path | None = None,
) -> CFMembershipCache:
    """
    Enumerate the 2^t bitstrings and cache which lie in S.
    """
    validate_cache_parameters(true_order, num_phase_bits)
    if max_cache_bitstrings < 0:
        raise ValueError("max_cache_bitstrings must be >= 0")

    num_bitstrings = 1 << num_phase_bits
    if max_cache_bitstrings and num_bitstrings > max_cache_bitstrings:
        raise ValueError(
            "building the CF membership cache would enumerate "
            f"{num_bitstrings:,} bitstrings; increase --max-cache-bitstrings "
            "or reduce --bits"
        )

    denominator = 1 << num_phase_bits
    bound = cf_sufficient_bound_distance(true_order)
    inside_bits_packed = np.zeros((num_bitstrings + 7) // 8, dtype=np.uint8)
    inside_count = 0

    for index in range(num_bitstrings):
        distance, _ = nearest_order_phase_distance(
            phase_fraction=Fraction(index, denominator),
            true_order=true_order,
        )
        if distance <= bound:
            inside_bits_packed[index >> 3] |= np.uint8(1 << (index & 7))
            inside_count += 1

    cache = CFMembershipCache(
        true_order=true_order,
        num_phase_bits=num_phase_bits,
        inside_count=inside_count,
        outside_count=num_bitstrings - inside_count,
        inside_bits_packed=inside_bits_packed,
        path=output_path,
        from_cache=False,
    )
    if output_path is not None:
        save_cf_membership_cache(cache, output_path)
    return cache


def load_or_build_cf_membership_cache(
    cache_dir: Path,
    true_order: int,
    num_phase_bits: int,
    max_cache_bitstrings: int = DEFAULT_MAX_CACHE_BITSTRINGS,
    rebuild_cache: bool = False,
) -> CFMembershipCache:
    """
    Load an existing (r, t) cache, or build and save it if absent.
    """
    cache_path = cache_output_path(cache_dir, true_order, num_phase_bits)
    if cache_path.exists() and not rebuild_cache:
        return load_cf_membership_cache(
            cache_path=cache_path,
            true_order=true_order,
            num_phase_bits=num_phase_bits,
        )
    return build_cf_membership_cache(
        true_order=true_order,
        num_phase_bits=num_phase_bits,
        max_cache_bitstrings=max_cache_bitstrings,
        output_path=cache_path,
    )


def run_noisy_phase_bits_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
) -> list[int]:
    """
    Run one phase-estimation trial and return only the noisy measured bits.
    """
    if algorithm == KITAEV_ALGORITHM:
        trial = run_kitaev_phase_distance_trial(
            instance=instance,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            single_measurement_bitflip=single_measurement_bitflip,
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
            single_measurement_bitflip=single_measurement_bitflip,
            single_measurement_bitflip_bits=single_measurement_bitflip_bits,
        )
    else:
        raise ValueError("unknown algorithm")
    return trial.noisy_bits


def run_phase_bits_for_fill_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    classical_error_correction: bool = False,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
) -> list[int]:
    """
    Run one phase-estimation trial and return noisy or corrected bits.
    """
    if algorithm == KITAEV_ALGORITHM:
        trial = run_kitaev_phase_distance_trial(
            instance=instance,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            single_measurement_bitflip=single_measurement_bitflip,
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
            single_measurement_bitflip=single_measurement_bitflip,
            single_measurement_bitflip_bits=single_measurement_bitflip_bits,
        )
    else:
        raise ValueError("unknown algorithm")

    if classical_error_correction:
        _, _, bits = minimum_single_bit_flip_nearest_peak_distance(
            trial.noisy_bits,
            true_order=instance.true_order,
        )
        return bits
    return trial.noisy_bits


def simulate_noisy_cf_complement_fill(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    measurement_error_probability: float,
    trials: int,
    rng: np.random.Generator,
    membership_cache: CFMembershipCache,
    standard_backend: str = ANALYTIC_STANDARD_BACKEND,
    analytic_window: int = 100,
    classical_error_correction: bool = False,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_bits: tuple[int, ...] | None = None,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
) -> NoisyCFComplementFillResult:
    """
    Estimate what fraction of S^c is uniquely hit by noisy outputs.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")
    if membership_cache.true_order != instance.true_order:
        raise ValueError("membership cache order does not match instance")
    if membership_cache.num_phase_bits != num_phase_bits:
        raise ValueError("membership cache bit count does not match simulation")

    outside_hits: set[int] = set()
    outside_trial_hits = 0
    for _ in range(trials):
        bits = run_phase_bits_for_fill_trial(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            measurement_error_probability=measurement_error_probability,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
            classical_error_correction=classical_error_correction,
            single_measurement_bitflip=single_measurement_bitflip,
            single_measurement_bitflip_bits=single_measurement_bitflip_bits,
        )
        bitstring_index = bits_to_index(bits)
        if membership_cache.is_outside_index(bitstring_index):
            outside_trial_hits += 1
            outside_hits.add(bitstring_index)

    return NoisyCFComplementFillResult(
        instance=instance,
        algorithm=algorithm,
        measurement_error_probability=measurement_error_probability,
        trials=trials,
        num_phase_bits=num_phase_bits,
        outside_unique_hits=len(outside_hits),
        outside_trial_hits=outside_trial_hits,
        total_inside_bitstrings=membership_cache.inside_count,
        total_outside_bitstrings=membership_cache.outside_count,
        standard_backend=standard_backend,
        classical_error_correction=classical_error_correction,
        single_measurement_bitflip=single_measurement_bitflip,
        single_measurement_bitflip_range=normalize_single_measurement_bitflip_ranges(
            single_measurement_bitflip_range
        ),
    )


def output_filename_segment(
    algorithms: list[str],
    standard_backend: str,
    N: int,
    a: int,
    num_phase_bits: int,
    measurement_error_probability: float,
    include_classical_error_correction: bool = False,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
) -> str:
    """
    Return the filename segment for one S^c fill chart.
    """
    algorithms = normalize_algorithms(algorithms)
    error_token = measurement_error_model_filename_segment(
        measurement_error_probability,
        single_measurement_bitflip=single_measurement_bitflip,
        single_measurement_bitflip_range=single_measurement_bitflip_range,
    )
    return (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, standard_backend)}"
        f"N{N}_a{a}_t{num_phase_bits}_{error_token}"
        f"{'_classical_ec' if include_classical_error_correction else ''}"
    )


def measurement_error_model_filename_segment(
    probability: float,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
) -> str:
    """
    Return the filename token for the chosen measurement-error model.
    """
    range_names = normalize_single_measurement_bitflip_ranges(
        single_measurement_bitflip_range
    )
    if range_names is not None:
        range_token = "_".join(name.replace("-", "_") for name in range_names)
        return f"single_measurement_bitflip_{range_token}"
    if single_measurement_bitflip:
        return "single_measurement_bitflip"
    probability_token = format_measurement_bitflip_probability(probability).replace(
        "-",
        "m",
    )
    return f"p_{probability_token}"


def measurement_error_model_display_label(
    probability: float,
    single_measurement_bitflip: bool = False,
    single_measurement_bitflip_range: SingleBitflipRangeSpec = None,
) -> str:
    """
    Return a compact display label for the chosen measurement-error model.
    """
    range_names = normalize_single_measurement_bitflip_ranges(
        single_measurement_bitflip_range
    )
    if range_names is not None:
        range_label = "+".join(range_names)
        return f"single random measurement bit-flip ({range_label})"
    if single_measurement_bitflip:
        return "single random measurement bit-flip"
    return measurement_probability_display_label(probability)


def result_series_label(result: NoisyCFComplementFillResult) -> str:
    """
    Return the plotted label for one noisy/corrected result.
    """
    suffix = "corrected" if result.classical_error_correction else "noisy"
    return f"{algorithm_display_label(result.algorithm)} {suffix}"


def fill_label(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if value == 0.0:
        return "0"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def adaptive_fill_ylim(values: list[float]) -> tuple[float, float]:
    """
    Return a readable y-axis range for small S^c fill fractions.
    """
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return 0.0, 1.0
    max_value = max(finite_values)
    if max_value <= 0.0:
        return 0.0, 0.05
    upper = min(1.0, max(0.02, max_value * 1.25))
    return 0.0, upper


def plot_noisy_cf_complement_fill(
    output_path: Path,
    results: list[NoisyCFComplementFillResult],
    algorithms: list[str],
    membership_cache: CFMembershipCache,
) -> None:
    """
    Plot the fraction of S^c uniquely filled by noisy sampled bitstrings.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    probability = results[0].measurement_error_probability
    single_measurement_bitflip = results[0].single_measurement_bitflip
    error_label = measurement_error_model_display_label(
        probability,
        single_measurement_bitflip=single_measurement_bitflip,
        single_measurement_bitflip_range=results[0].single_measurement_bitflip_range,
    )
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
    x_values = np.arange(len(result_keys))
    heights = [
        result_by_key[key].outside_fill_fraction
        for key in result_keys
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    bars = ax.bar(
        x_values,
        heights,
        width=0.52,
        color=[colors[algorithm] for algorithm, _ in result_keys],
        alpha=0.88,
    )
    for bar, (_, correction_enabled) in zip(bars, result_keys):
        bar.set_hatch(hatches[correction_enabled])
        bar.set_edgecolor("#222222" if correction_enabled else "none")
        bar.set_linewidth(0.7 if correction_enabled else 0.0)
    lower, upper = adaptive_fill_ylim(heights)
    ax.set_ylim(lower, upper)

    for bar, height in zip(bars, heights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(upper * 0.025, 1e-12),
            fill_label(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [
            result_series_label(result_by_key[key])
            for key in result_keys
        ],
        rotation=15 if len(result_keys) > 2 else 0,
        ha="right" if len(result_keys) > 2 else "center",
    )
    ax.set_xlabel("Series")
    ax.set_ylabel("Fraction of S^c uniquely hit")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.set_title(
        "Noisy S^c Coverage\n"
        f"{algorithms_display_label(algorithms)} for N = {instance.N}, "
        f"a = {instance.a}, r = {instance.true_order}, "
        f"{error_label}"
    )
    ax.grid(True, axis="y", alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[algorithm],
            marker="s",
            linestyle="None",
            markersize=8,
            label=result_series_label(result_by_key[(algorithm, correction_enabled)]),
        )
        for algorithm, correction_enabled in result_keys
    ]
    cache_status = "loaded" if membership_cache.from_cache else "built"
    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=legend_handles,
        measurement_handles=[],
        trials_per_instance=trials,
        algorithm_title="Series",
        measurement_title="Measurement error",
        measurement_text_labels=[error_label],
        extra_text_sections=[
            (
                "Instance",
                [
                    f"N = {instance.N}, a = {instance.a}",
                    f"r = {instance.true_order}, t = {num_phase_bits}",
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
        trials_title="Trials per algorithm",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_data_summary(
    output_path: Path,
    results: list[NoisyCFComplementFillResult],
    algorithms: list[str],
    membership_cache: CFMembershipCache,
    seed: int | None,
) -> None:
    """
    Write metadata and per-algorithm S^c fill fractions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)
    if not results:
        raise ValueError("results must be nonempty")

    instance = results[0].instance
    probability = results[0].measurement_error_probability
    single_measurement_bitflip = results[0].single_measurement_bitflip
    error_label = measurement_error_model_display_label(
        probability,
        single_measurement_bitflip=single_measurement_bitflip,
        single_measurement_bitflip_range=results[0].single_measurement_bitflip_range,
    )
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
    cache_path_label = (
        "none" if membership_cache.path is None else str(membership_cache.path)
    )

    lines = [
        "Noisy CF-Complement Fill Summary",
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
        f"Measurement error model = {error_label}",
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
        *(
            [
                "  Single-measurement-bitflip mode flips exactly one uniformly random readout bit per trial.",
                "  In Kitaev, that one readout error can propagate through later adaptive feedback bits.",
            ]
            if single_measurement_bitflip
            else [
                "  Probability mode flips each measurement readout bit independently with probability p.",
            ]
        ),
        "  Corrected uses the bit string among tilde b and all one-bit flips with minimum distance to its own nearest s/r.",
        "  S^c fill = |unique noisy bitstrings observed in S^c| / |S^c|.",
        "  Pr(S^c) is the trial probability of landing outside S, not the fill fraction.",
        "",
        "Per-series data:",
        "  algorithm | series | S^c fill | unique S^c hits | |S^c| | Pr(S^c) | S^c trial hits | trials",
    ]

    for result in sorted(
        results,
        key=lambda item: (item.algorithm, item.classical_error_correction),
    ):
        series = "corrected" if result.classical_error_correction else "noisy"
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {series:>9} |"
            f" {result.outside_fill_fraction:>10.6f} |"
            f" {result.outside_unique_hits:>15} |"
            f" {result.total_outside_bitstrings:>5} |"
            f" {result.outside_trial_probability:>7.3f} |"
            f" {result.outside_trial_hits:>14} |"
            f" {result.trials:>6}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for noisy S^c coverage charts.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot how much of the continued-fraction complement S^c is "
            "uniquely hit by noisy phase-estimation outputs."
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
        "--single-measurement-bitflip",
        "--one-measurement-bitflip",
        dest="single_measurement_bitflip",
        action="store_true",
        help=(
            "Use exactly one uniformly random measurement bit-flip per trial "
            "instead of independent per-bit flips from --p."
        ),
    )
    parser.add_argument(
        "--single-measurement-bitflip-range",
        nargs="+",
        choices=SINGLE_MEASUREMENT_BITFLIP_RANGE_CHOICES,
        default=None,
        help=(
            "Use exactly one uniformly random measurement bit-flip restricted "
            "to one or more phase-bit ranges. For r=2^m: leading=1..m, "
            "dangerous=m+1..2m, safe-tail=2m+1..t. Multiple ranges are sampled "
            "as one uniform union."
        ),
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
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached S/S^c membership bitsets keyed by r and t",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the S/S^c membership cache even if it already exists",
    )
    parser.add_argument(
        "--max-cache-bitstrings",
        type=int,
        default=DEFAULT_MAX_CACHE_BITSTRINGS,
        help=(
            "Safety limit for enumerating 2^t bitstrings while building the "
            "cache. Use 0 to disable the guard."
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
    Run the noisy S^c fill chart script.
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
    if args.max_cache_bitstrings < 0:
        raise ValueError("max_cache_bitstrings must be >= 0")

    try:
        p = validate_measurement_error_probability(args.p)
        single_bitflip_ranges = normalize_single_measurement_bitflip_ranges(
            args.single_measurement_bitflip_range
        )
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
        membership_cache = load_or_build_cf_membership_cache(
            cache_dir=args.cache_dir,
            true_order=instance.true_order,
            num_phase_bits=num_phase_bits,
            max_cache_bitstrings=args.max_cache_bitstrings,
            rebuild_cache=args.rebuild_cache,
        )
        single_bitflip_bits = (
            single_measurement_bitflip_bit_numbers(
                num_phase_bits=num_phase_bits,
                true_order=instance.true_order,
                range_name=single_bitflip_ranges or "all",
            )
            if args.single_measurement_bitflip
            or single_bitflip_ranges is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))

    master_rng = np.random.default_rng(args.seed)
    results = []
    for algorithm in algorithms:
        algorithm_seed = int(master_rng.integers(0, 2 ** 63 - 1))
        results.append(
            simulate_noisy_cf_complement_fill(
                instance=instance,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                measurement_error_probability=p,
                trials=args.trials,
                rng=np.random.default_rng(algorithm_seed),
                membership_cache=membership_cache,
                standard_backend=args.standard_backend,
                analytic_window=args.analytic_window,
                single_measurement_bitflip=single_bitflip_bits is not None,
                single_measurement_bitflip_bits=single_bitflip_bits,
                single_measurement_bitflip_range=single_bitflip_ranges,
            )
        )
        if args.classical_error_correction:
            results.append(
                simulate_noisy_cf_complement_fill(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    measurement_error_probability=p,
                    trials=args.trials,
                    rng=np.random.default_rng(algorithm_seed),
                    membership_cache=membership_cache,
                    standard_backend=args.standard_backend,
                    analytic_window=args.analytic_window,
                    classical_error_correction=True,
                    single_measurement_bitflip=single_bitflip_bits is not None,
                    single_measurement_bitflip_bits=single_bitflip_bits,
                    single_measurement_bitflip_range=single_bitflip_ranges,
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
        single_measurement_bitflip=single_bitflip_bits is not None,
        single_measurement_bitflip_range=single_bitflip_ranges,
    )
    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"noisy_cf_complement_fill_{suffix}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"noisy_cf_complement_fill_summary_{suffix}.txt"
    )

    plot_noisy_cf_complement_fill(
        output_path=figure_output_path,
        results=results,
        algorithms=algorithms,
        membership_cache=membership_cache,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        algorithms=algorithms,
        membership_cache=membership_cache,
        seed=args.seed,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    cache_status = "loaded" if membership_cache.from_cache else "built"
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
    print(f"Total bitstrings 2^t = {membership_cache.num_bitstrings:,}")
    print(f"|S| = {membership_cache.inside_count:,}")
    print(f"|S^c| = {membership_cache.outside_count:,}")
    print(f"Cache status: {cache_status}")
    if membership_cache.path is not None:
        print(f"Cache path: {membership_cache.path}")
    print(
        "Measurement error model: "
        f"{measurement_error_model_display_label(p, single_bitflip_bits is not None, single_bitflip_ranges)}"
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
