from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    DrawingArea,
    HPacker,
    TextArea,
    VPacker,
)

from error_correction.classical_error_correction import (
    single_bit_flip_candidates,
    validate_phase_estimation_instance,
)
from shor_common.classical_post import bits_to_fraction
from shor_common.errors import (
    ControlledUnitaryDepolarizingRangeSpec,
    controlled_unitary_depolarizing_phase_bits,
)
from shor_one_qubit.shor import shor_order_finding_attempt as kitaev_order_attempt
from shor_standard.analytic_sampling import multiplicative_order
from shor_standard.backends import (
    STATEVECTOR_STANDARD_BACKEND,
)
from shor_standard.shor import shor_order_finding_attempt as standard_order_attempt

from graphs.statevector_precision import (
    warn_if_statevector_phase_bits_below_default,
)
from graphs.depolarizing_error.work_register import effective_work_qubits_for_N
from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    error_config_for_model,
    error_model_display_label,
    error_model_filename_label,
    normalize_error_model,
)
from graphs.depolarizing_error.plot_order_recovery_success_vs_order import (
    ALGORITHM_CHOICES,
    DEFAULT_INSTANCES_FILE,
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
    load_instances,
    normalize_algorithms,
    probability_token,
    validate_error_probabilities,
)


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "cf_membership"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_MAX_CACHE_BITSTRINGS = 5_000_000
DEFAULT_MAX_STATEVECTOR_AMPLITUDES = 8_000_000
DEFAULT_NUM_PHASE_BITS = 11
STANDARD_COLOR_LIGHTENING = 0.58
CACHE_VERSION = 1


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
class NoisyCFMembershipResult:
    instance: OrderRecoveryInstance
    algorithm: str
    error_model: str
    error_probability: float
    trials: int
    num_phase_bits: int
    inside_trials: int
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND
    classical_error_correction: bool = False
    num_work_qubits: int | None = None

    @property
    def outside_trials(self) -> int:
        return self.trials - self.inside_trials

    @property
    def inside_probability(self) -> float:
        return self.inside_trials / self.trials

    @property
    def outside_probability(self) -> float:
        return self.outside_trials / self.trials


@dataclass(frozen=True)
class NoisyCFComplementFillResult:
    instance: OrderRecoveryInstance
    algorithm: str
    error_model: str
    error_probability: float
    trials: int
    num_phase_bits: int
    outside_unique_hits: int
    outside_trial_hits: int
    total_inside_bitstrings: int
    total_outside_bitstrings: int
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND
    classical_error_correction: bool = False
    num_work_qubits: int | None = None

    @property
    def outside_fill_fraction(self) -> float:
        if self.total_outside_bitstrings == 0:
            return math.nan
        return self.outside_unique_hits / self.total_outside_bitstrings

    @property
    def outside_trial_probability(self) -> float:
        return self.outside_trial_hits / self.trials


@dataclass(frozen=True)
class NoisyCFComplementFillTrialSweepResult:
    instance: OrderRecoveryInstance
    algorithm: str
    error_model: str
    error_probability: float
    trial_count: int
    num_phase_bits: int
    outside_unique_hits: int
    outside_trial_hits: int
    total_inside_bitstrings: int
    total_outside_bitstrings: int
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND
    classical_error_correction: bool = False
    num_work_qubits: int | None = None

    @property
    def outside_fill_fraction(self) -> float:
        if self.total_outside_bitstrings == 0:
            return math.nan
        return self.outside_unique_hits / self.total_outside_bitstrings

    @property
    def outside_trial_probability(self) -> float:
        return self.outside_trial_hits / self.trial_count


def build_fixed_instance(N: int, a: int) -> OrderRecoveryInstance:
    """
    Validate a fixed modular instance and compute its true order.
    """
    validate_phase_estimation_instance(N, a)
    return OrderRecoveryInstance(N=N, a=a, true_order=multiplicative_order(a, N))


def algorithm_display_label(algorithm: str) -> str:
    if algorithm == KITAEV_ALGORITHM:
        return "Kitaev"
    if algorithm == STANDARD_ALGORITHM:
        return "standard Shor"
    raise ValueError("unknown algorithm")


def algorithms_display_label(algorithms: list[str]) -> str:
    algorithms = normalize_algorithms(algorithms)
    if len(algorithms) == 1:
        return algorithm_display_label(algorithms[0])
    return " vs ".join(algorithm_display_label(algorithm) for algorithm in algorithms)


def algorithm_filename_prefix(algorithms: list[str]) -> str:
    algorithms = normalize_algorithms(algorithms)
    if algorithms == [KITAEV_ALGORITHM]:
        return "kitaev_"
    if algorithms == [STANDARD_ALGORITHM]:
        return "standard_"
    if algorithms == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]:
        return "compare_algorithms_"
    return "compare_" + "_".join(algorithms) + "_"


def standard_backend_filename_segment(
    algorithms: list[str],
    standard_backend: str,
) -> str:
    algorithms = normalize_algorithms(algorithms)
    if STANDARD_ALGORITHM not in algorithms:
        return ""
    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        return ""
    return f"{standard_backend}_"


def safe_filename_token(raw_text: str) -> str:
    token = "".join(
        character if character.isalnum() else "_"
        for character in raw_text.lower()
    ).strip("_")
    return token or "instances"


def compact_filename_segment(raw_text: str, max_length: int = 150) -> str:
    """
    Keep generated filename components under filesystem limits.

    The returned value is unchanged when it already fits. Long values first get
    semantic abbreviations rather than opaque hash suffixes.
    """
    if max_length < 16:
        raise ValueError("max_length must be >= 16")
    if len(raw_text) <= max_length:
        return raw_text

    compacted = raw_text
    replacements = (
        ("per_qubit_depolarizing", "pqdp"),
        ("block_depolarizing", "dp"),
        ("order_recover_instances", "ori"),
        ("order_recovery_instances", "ori"),
        ("depolarizing", "dp"),
        ("membership", "mbr"),
        ("instance_file", "files"),
    )
    for old, new in replacements:
        compacted = compacted.replace(old, new)
    if len(compacted) <= max_length:
        return compacted

    return compacted[:max_length].rstrip("_")


def instances_file_filename_segment(instances_file: Path | None) -> str:
    if instances_file is None:
        return ""
    return f"instances_{safe_filename_token(instances_file.stem)}_"


def lighten_color(color: str, amount: float) -> tuple[float, float, float]:
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must satisfy 0 <= amount <= 1")
    red, green, blue = to_rgb(color)
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
    )


def algorithm_color_for_base(
    base_color: str,
    algorithm: str,
) -> str | tuple[float, float, float]:
    if algorithm == KITAEV_ALGORITHM:
        return base_color
    if algorithm == STANDARD_ALGORITHM:
        return lighten_color(base_color, STANDARD_COLOR_LIGHTENING)
    raise ValueError("unknown algorithm")


def add_combined_figure_legend(
    fig,
    algorithm_handles: list[Line2D],
    measurement_handles: list[Line2D],
    trials_per_instance: int,
    algorithm_title: str = "Algorithm",
    measurement_title: str = "Depolarizing error",
    extra_text_sections: list[tuple[str, list[str]]] | None = None,
    measurement_text_labels: list[str] | None = None,
    trials_title: str = "Trials per instance",
) -> None:
    """
    Add the same grouped right-margin legend used by the measurement plots.
    """

    def section_title(label: str) -> TextArea:
        return TextArea(
            label,
            textprops={"fontsize": 8, "fontweight": "bold", "ha": "left"},
        )

    def text_row(label: str) -> TextArea:
        return TextArea(label, textprops={"fontsize": 8, "ha": "left"})

    def handle_row(handle: Line2D) -> HPacker:
        drawing = DrawingArea(28, 12, 0, 0)
        sample = Line2D(
            [2, 26],
            [6, 6],
            color=handle.get_color(),
            marker=handle.get_marker(),
            linestyle=handle.get_linestyle(),
            linewidth=handle.get_linewidth(),
            markersize=handle.get_markersize(),
        )
        drawing.add_artist(sample)
        return HPacker(
            children=[drawing, text_row(handle.get_label())],
            align="center",
            pad=0,
            sep=4,
        )

    def section(title: str, rows: list) -> VPacker:
        return VPacker(
            children=[section_title(title), *rows],
            align="left",
            pad=0,
            sep=3,
        )

    measurement_rows = (
        [text_row(label) for label in measurement_text_labels]
        if measurement_text_labels is not None
        else [handle_row(handle) for handle in measurement_handles]
    )
    children = [
        section(algorithm_title, [handle_row(handle) for handle in algorithm_handles]),
        section(measurement_title, measurement_rows),
    ]
    if extra_text_sections is not None:
        children.extend(
            section(title, [text_row(label) for label in labels])
            for title, labels in extra_text_sections
        )
    children.append(section(trials_title, [text_row(str(trials_per_instance))]))

    legend_box = VPacker(
        children=children,
        align="left",
        pad=0,
        sep=10,
    )
    anchored_legend = AnchoredOffsetbox(
        loc="upper left",
        child=legend_box,
        bbox_to_anchor=(0.75, 0.88),
        bbox_transform=fig.transFigure,
        frameon=True,
        pad=0.45,
        borderpad=0.0,
    )
    anchored_legend.patch.set_facecolor("white")
    anchored_legend.patch.set_alpha(0.9)
    anchored_legend.patch.set_edgecolor("#D0D0D0")
    fig.add_artist(anchored_legend)


def validate_depolarizing_standard_backend(
    algorithms: list[str],
    standard_backend: str,
) -> None:
    """
    Depolarizing trajectory errors need an explicit evolving standard state.
    """
    algorithms = normalize_algorithms(algorithms)
    if STANDARD_ALGORITHM not in algorithms:
        return
    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        return
    raise ValueError(
        "depolarizing errors require the standard statevector backend; the "
        "analytic backend samples ideal outcomes from the true order and has "
        "no control/work state or work-register trajectory to perturb"
    )


def validate_statevector_budget(
    algorithms: list[str],
    standard_backend: str,
    num_phase_bits: int,
    max_N: int,
    max_statevector_amplitudes: int,
    num_work_qubits: int | None = None,
) -> None:
    warn_if_statevector_phase_bits_below_default(
        algorithms=algorithms,
        standard_backend=standard_backend,
        num_phase_bits=num_phase_bits,
        instances_or_N_values=[max_N],
        num_work_qubits=num_work_qubits,
        stacklevel=3,
    )
    if STANDARD_ALGORITHM not in algorithms:
        return
    if standard_backend != STATEVECTOR_STANDARD_BACKEND:
        return
    if max_statevector_amplitudes <= 0:
        return

    max_work_qubits = effective_work_qubits_for_N(max_N, num_work_qubits)
    max_amplitudes = (2 ** num_phase_bits) * (2 ** max_work_qubits)
    if max_amplitudes > max_statevector_amplitudes:
        raise ValueError(
            "standard statevector backend could allocate up to "
            f"{max_amplitudes:,} complex amplitudes for this sweep. Lower "
            "--bits/--max-N, choose smaller instances, or raise "
            "--max-statevector-amplitudes."
        )


def depolarizing_error_filename_segment(
    error_model: str,
    error_probability: float,
) -> str:
    return (
        f"{error_model_filename_label(error_model)}"
        f"_p_{probability_token(float(error_probability))}"
    )


def depolarizing_error_display_label(
    error_model: str,
    error_probability: float,
) -> str:
    return f"{error_model_display_label(error_model)}, p = {error_probability:g}"


def circular_fraction_distance(left: Fraction, right: Fraction) -> Fraction:
    if not (0 <= left < 1) or not (0 <= right < 1):
        raise ValueError("fractions must satisfy 0 <= value < 1")
    distance = abs(left - right)
    return min(distance, Fraction(1, 1) - distance)


def nearest_order_phase_distance(
    phase_fraction: Fraction,
    true_order: int,
) -> tuple[Fraction, int]:
    if not (0 <= phase_fraction < 1):
        raise ValueError("phase_fraction must satisfy 0 <= value < 1")
    if true_order < 1:
        raise ValueError("true_order must be >= 1")

    scaled_phase = phase_fraction * true_order
    floor_candidate = scaled_phase.numerator // scaled_phase.denominator
    candidate_numerators = {
        (floor_candidate - 1) % true_order,
        floor_candidate % true_order,
        (floor_candidate + 1) % true_order,
    }
    best_numerator = min(
        candidate_numerators,
        key=lambda numerator: (
            circular_fraction_distance(
                phase_fraction,
                Fraction(numerator, true_order),
            ),
            numerator,
        ),
    )
    return (
        circular_fraction_distance(
            phase_fraction,
            Fraction(best_numerator, true_order),
        ),
        best_numerator,
    )


@lru_cache(maxsize=512)
def coprime_order_phase_numerators(true_order: int) -> tuple[int, ...]:
    if true_order < 1:
        raise ValueError("true_order must be >= 1")
    return tuple(
        numerator
        for numerator in range(true_order)
        if math.gcd(numerator, true_order) == 1
    )


def nearest_coprime_order_phase_distance(
    phase_fraction: Fraction,
    true_order: int,
) -> tuple[Fraction, int]:
    if not (0 <= phase_fraction < 1):
        raise ValueError("phase_fraction must satisfy 0 <= value < 1")

    candidate_numerators = coprime_order_phase_numerators(true_order)
    best_numerator = min(
        candidate_numerators,
        key=lambda numerator: (
            circular_fraction_distance(
                phase_fraction,
                Fraction(numerator, true_order),
            ),
            numerator,
        ),
    )
    return (
        circular_fraction_distance(
            phase_fraction,
            Fraction(best_numerator, true_order),
        ),
        best_numerator,
    )


def normalized_cf_error(phase_fraction: Fraction, true_order: int) -> Fraction:
    distance, _ = nearest_order_phase_distance(
        phase_fraction=phase_fraction,
        true_order=true_order,
    )
    return Fraction(2 * true_order * true_order, 1) * distance


def normalized_coprime_cf_error(
    phase_fraction: Fraction,
    true_order: int,
) -> Fraction:
    distance, _ = nearest_coprime_order_phase_distance(
        phase_fraction=phase_fraction,
        true_order=true_order,
    )
    return Fraction(2 * true_order * true_order, 1) * distance


def within_cf_sufficient_bound(bits: list[int], true_order: int) -> bool:
    return normalized_cf_error(bits_to_fraction(bits), true_order) <= 1


def within_coprime_cf_sufficient_bound(
    bits: list[int],
    true_order: int,
) -> bool:
    return normalized_coprime_cf_error(bits_to_fraction(bits), true_order) <= 1


def cf_sufficient_bound_distance(true_order: int) -> Fraction:
    if true_order < 1:
        raise ValueError("true_order must be >= 1")
    return Fraction(1, 2 * true_order * true_order)


def nearest_peak_distance(
    bits: list[int],
    true_order: int,
) -> tuple[Fraction, int]:
    return nearest_order_phase_distance(
        phase_fraction=bits_to_fraction(bits),
        true_order=true_order,
    )


def minimum_single_bit_flip_nearest_peak_distance(
    bits: list[int],
    true_order: int,
) -> tuple[Fraction, int, list[int]]:
    best_bits = min(
        single_bit_flip_candidates(bits),
        key=lambda candidate: (
            nearest_peak_distance(candidate, true_order=true_order)[0],
            candidate != list(bits),
        ),
    )
    best_distance, best_peak = nearest_peak_distance(
        best_bits,
        true_order=true_order,
    )
    return best_distance, best_peak, best_bits


def minimum_single_bit_flip_nearest_coprime_peak_distance(
    bits: list[int],
    true_order: int,
) -> tuple[Fraction, int, list[int]]:
    best_bits = min(
        single_bit_flip_candidates(bits),
        key=lambda candidate: (
            nearest_coprime_order_phase_distance(
                bits_to_fraction(candidate),
                true_order=true_order,
            )[0],
            candidate != list(bits),
        ),
    )
    best_distance, best_peak = nearest_coprime_order_phase_distance(
        bits_to_fraction(best_bits),
        true_order=true_order,
    )
    return best_distance, best_peak, best_bits


def bits_to_index(bits: list[int]) -> int:
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
    validate_cache_parameters(true_order, num_phase_bits)
    return (
        cache_dir
        / f"cf_membership_r{true_order}_t{num_phase_bits}_v{CACHE_VERSION}.npz"
    )


def save_cf_membership_cache(cache: CFMembershipCache, output_path: Path) -> None:
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


def run_depolarizing_phase_bits_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> list[int]:
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    error_model = normalize_error_model(error_model)
    error_probability = validate_error_probabilities([error_probability])[0]
    allowed_phase_bits = controlled_unitary_depolarizing_phase_bits(
        num_phase_bits=num_phase_bits,
        true_order=instance.true_order,
        range_names=depolarizing_region,
    )
    error_config = error_config_for_model(
        error_model,
        error_probability,
        allowed_phase_bits=allowed_phase_bits,
    )

    if algorithm == KITAEV_ALGORITHM:
        result = kitaev_order_attempt(
            N=instance.N,
            a=instance.a,
            num_phase_bits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
            error_config=error_config,
            num_work_qubits=num_work_qubits,
        )
    elif algorithm == STANDARD_ALGORITHM:
        validate_depolarizing_standard_backend([algorithm], standard_backend)
        result = standard_order_attempt(
            N=instance.N,
            a=instance.a,
            num_control_qubits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
            error_config=error_config,
            standard_backend=standard_backend,
            num_work_qubits=num_work_qubits,
        )
    else:
        raise ValueError("unknown algorithm")

    return list(result["bits"])


def run_noisy_membership_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> bool:
    bits = run_depolarizing_phase_bits_trial(
        instance=instance,
        algorithm=algorithm,
        num_phase_bits=num_phase_bits,
        error_model=error_model,
        error_probability=error_probability,
        rng=rng,
        standard_backend=standard_backend,
        num_work_qubits=num_work_qubits,
        depolarizing_region=depolarizing_region,
    )

    if classical_error_correction:
        _, _, bits = minimum_single_bit_flip_nearest_peak_distance(
            bits,
            true_order=instance.true_order,
        )

    return within_cf_sufficient_bound(bits, true_order=instance.true_order)


def run_noisy_coprime_membership_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> bool:
    bits = run_depolarizing_phase_bits_trial(
        instance=instance,
        algorithm=algorithm,
        num_phase_bits=num_phase_bits,
        error_model=error_model,
        error_probability=error_probability,
        rng=rng,
        standard_backend=standard_backend,
        num_work_qubits=num_work_qubits,
        depolarizing_region=depolarizing_region,
    )

    if classical_error_correction:
        _, _, bits = minimum_single_bit_flip_nearest_coprime_peak_distance(
            bits,
            true_order=instance.true_order,
        )

    return within_coprime_cf_sufficient_bound(
        bits,
        true_order=instance.true_order,
    )


def simulate_noisy_cf_membership(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> NoisyCFMembershipResult:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    inside_trials = 0
    for _ in range(trials):
        if run_noisy_membership_trial(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            error_model=error_model,
            error_probability=error_probability,
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
        error_probability=float(error_probability),
        trials=trials,
        num_phase_bits=num_phase_bits,
        inside_trials=inside_trials,
        standard_backend=standard_backend,
        classical_error_correction=classical_error_correction,
        num_work_qubits=effective_work_qubits_for_N(instance.N, num_work_qubits),
    )


def simulate_noisy_coprime_cf_membership(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> NoisyCFMembershipResult:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    inside_trials = 0
    for _ in range(trials):
        if run_noisy_coprime_membership_trial(
            instance=instance,
            algorithm=algorithm,
            num_phase_bits=num_phase_bits,
            error_model=error_model,
            error_probability=error_probability,
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
        error_probability=float(error_probability),
        trials=trials,
        num_phase_bits=num_phase_bits,
        inside_trials=inside_trials,
        standard_backend=standard_backend,
        classical_error_correction=classical_error_correction,
        num_work_qubits=effective_work_qubits_for_N(instance.N, num_work_qubits),
    )


def run_phase_bits_for_fill_trial(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> list[int]:
    bits = run_depolarizing_phase_bits_trial(
        instance=instance,
        algorithm=algorithm,
        num_phase_bits=num_phase_bits,
        error_model=error_model,
        error_probability=error_probability,
        rng=rng,
        standard_backend=standard_backend,
        num_work_qubits=num_work_qubits,
        depolarizing_region=depolarizing_region,
    )
    if classical_error_correction:
        _, _, bits = minimum_single_bit_flip_nearest_peak_distance(
            bits,
            true_order=instance.true_order,
        )
    return bits


def simulate_noisy_cf_complement_fill(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
    membership_cache: CFMembershipCache,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> NoisyCFComplementFillResult:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
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
            error_model=error_model,
            error_probability=error_probability,
            rng=rng,
            standard_backend=standard_backend,
            classical_error_correction=classical_error_correction,
            num_work_qubits=num_work_qubits,
            depolarizing_region=depolarizing_region,
        )
        bitstring_index = bits_to_index(bits)
        if membership_cache.is_outside_index(bitstring_index):
            outside_trial_hits += 1
            outside_hits.add(bitstring_index)

    return NoisyCFComplementFillResult(
        instance=instance,
        algorithm=algorithm,
        error_model=normalize_error_model(error_model),
        error_probability=float(error_probability),
        trials=trials,
        num_phase_bits=num_phase_bits,
        outside_unique_hits=len(outside_hits),
        outside_trial_hits=outside_trial_hits,
        total_inside_bitstrings=membership_cache.inside_count,
        total_outside_bitstrings=membership_cache.outside_count,
        standard_backend=standard_backend,
        classical_error_correction=classical_error_correction,
        num_work_qubits=effective_work_qubits_for_N(instance.N, num_work_qubits),
    )


def validate_trial_counts(trial_counts: list[int]) -> list[int]:
    if not trial_counts:
        raise ValueError("at least one trial count is required")
    normalized = []
    for trial_count in trial_counts:
        count = int(trial_count)
        if count < 1:
            raise ValueError("trial counts must be >= 1")
        if count not in normalized:
            normalized.append(count)
    return sorted(normalized)


def simulate_noisy_cf_complement_fill_vs_trials(
    instance: OrderRecoveryInstance,
    algorithm: str,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trial_counts: list[int],
    rng: np.random.Generator,
    membership_cache: CFMembershipCache,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> list[NoisyCFComplementFillTrialSweepResult]:
    trial_counts = validate_trial_counts(trial_counts)
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if membership_cache.true_order != instance.true_order:
        raise ValueError("membership cache order does not match instance")
    if membership_cache.num_phase_bits != num_phase_bits:
        raise ValueError("membership cache bit count does not match simulation")

    results = []
    for trial_count in trial_counts:
        trial_rng = np.random.default_rng(int(rng.integers(0, 2 ** 63 - 1)))
        outside_hits: set[int] = set()
        outside_trial_hits = 0

        for _ in range(trial_count):
            bits = run_phase_bits_for_fill_trial(
                instance=instance,
                algorithm=algorithm,
                num_phase_bits=num_phase_bits,
                error_model=error_model,
                error_probability=error_probability,
                rng=trial_rng,
                standard_backend=standard_backend,
                classical_error_correction=classical_error_correction,
                num_work_qubits=num_work_qubits,
                depolarizing_region=depolarizing_region,
            )
            bitstring_index = bits_to_index(bits)
            if membership_cache.is_outside_index(bitstring_index):
                outside_trial_hits += 1
                outside_hits.add(bitstring_index)

        results.append(
            NoisyCFComplementFillTrialSweepResult(
                instance=instance,
                algorithm=algorithm,
                error_model=normalize_error_model(error_model),
                error_probability=float(error_probability),
                trial_count=trial_count,
                num_phase_bits=num_phase_bits,
                outside_unique_hits=len(outside_hits),
                outside_trial_hits=outside_trial_hits,
                total_inside_bitstrings=membership_cache.inside_count,
                total_outside_bitstrings=membership_cache.outside_count,
                standard_backend=standard_backend,
                classical_error_correction=classical_error_correction,
                num_work_qubits=effective_work_qubits_for_N(
                    instance.N,
                    num_work_qubits,
                ),
            )
        )

    return results


def simulate_noisy_cf_membership_vs_order(
    instances: list[OrderRecoveryInstance],
    algorithms: list[str],
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials_per_instance: int,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    include_classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> list[NoisyCFMembershipResult]:
    algorithms = normalize_algorithms(algorithms)
    if trials_per_instance < 1:
        raise ValueError("trials_per_instance must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    results = []
    for algorithm in algorithms:
        for instance in instances:
            instance_seed = int(rng.integers(0, 2 ** 63 - 1))
            results.append(
                simulate_noisy_cf_membership(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    error_model=error_model,
                    error_probability=error_probability,
                    trials=trials_per_instance,
                    rng=np.random.default_rng(instance_seed),
                    standard_backend=standard_backend,
                    num_work_qubits=num_work_qubits,
                    depolarizing_region=depolarizing_region,
                )
            )
            if include_classical_error_correction:
                results.append(
                    simulate_noisy_cf_membership(
                        instance=instance,
                        algorithm=algorithm,
                        num_phase_bits=num_phase_bits,
                        error_model=error_model,
                        error_probability=error_probability,
                        trials=trials_per_instance,
                        rng=np.random.default_rng(instance_seed),
                        standard_backend=standard_backend,
                        classical_error_correction=True,
                        num_work_qubits=num_work_qubits,
                        depolarizing_region=depolarizing_region,
                    )
                )
    return results


def simulate_noisy_coprime_cf_membership_vs_order(
    instances: list[OrderRecoveryInstance],
    algorithms: list[str],
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials_per_instance: int,
    rng: np.random.Generator,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    include_classical_error_correction: bool = False,
    num_work_qubits: int | None = None,
    depolarizing_region: ControlledUnitaryDepolarizingRangeSpec = None,
) -> list[NoisyCFMembershipResult]:
    algorithms = normalize_algorithms(algorithms)
    if trials_per_instance < 1:
        raise ValueError("trials_per_instance must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    results = []
    for algorithm in algorithms:
        for instance in instances:
            instance_seed = int(rng.integers(0, 2 ** 63 - 1))
            results.append(
                simulate_noisy_coprime_cf_membership(
                    instance=instance,
                    algorithm=algorithm,
                    num_phase_bits=num_phase_bits,
                    error_model=error_model,
                    error_probability=error_probability,
                    trials=trials_per_instance,
                    rng=np.random.default_rng(instance_seed),
                    standard_backend=standard_backend,
                    num_work_qubits=num_work_qubits,
                    depolarizing_region=depolarizing_region,
                )
            )
            if include_classical_error_correction:
                results.append(
                    simulate_noisy_coprime_cf_membership(
                        instance=instance,
                        algorithm=algorithm,
                        num_phase_bits=num_phase_bits,
                        error_model=error_model,
                        error_probability=error_probability,
                        trials=trials_per_instance,
                        rng=np.random.default_rng(instance_seed),
                        standard_backend=standard_backend,
                        classical_error_correction=True,
                        num_work_qubits=num_work_qubits,
                        depolarizing_region=depolarizing_region,
                    )
                )
    return results
