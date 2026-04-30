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
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from error_correction.code_distance import (
    AverageHammingDistanceEstimate,
    estimate_average_hamming_distance,
)
from error_correction.classical_error_correction import validate_phase_estimation_instance
from shor_one_qubit.errors import ErrorConfig
from shor_one_qubit.shor import default_num_phase_bits


MEASUREMENT_BITFLIP_MODEL = "measurement-bitflip"
BLOCK_DEPOLARIZING_MODEL = "block-depolarizing"
PER_QUBIT_DEPOLARIZING_MODEL = "per-qubit-depolarizing"

ERROR_MODEL_CHOICES = (
    MEASUREMENT_BITFLIP_MODEL,
    BLOCK_DEPOLARIZING_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
)

ERROR_MODEL_ALIASES = {
    "measurement": MEASUREMENT_BITFLIP_MODEL,
    "measurement-bitflip": MEASUREMENT_BITFLIP_MODEL,
    "measurement_bitflip": MEASUREMENT_BITFLIP_MODEL,
    "block": BLOCK_DEPOLARIZING_MODEL,
    "block-depolarizing": BLOCK_DEPOLARIZING_MODEL,
    "block_depolarizing": BLOCK_DEPOLARIZING_MODEL,
    "controlled-unitary-depolarizing": BLOCK_DEPOLARIZING_MODEL,
    "controlled_unitary_depolarizing": BLOCK_DEPOLARIZING_MODEL,
    "per-qubit": PER_QUBIT_DEPOLARIZING_MODEL,
    "per-qubit-depolarizing": PER_QUBIT_DEPOLARIZING_MODEL,
    "per_qubit_depolarizing": PER_QUBIT_DEPOLARIZING_MODEL,
    "controlled-unitary-per-qubit-depolarizing": PER_QUBIT_DEPOLARIZING_MODEL,
    "controlled_unitary_per_qubit_depolarizing": PER_QUBIT_DEPOLARIZING_MODEL,
}


@dataclass
class HammingDistanceSweepResult:
    error_probability: float
    trials: int
    total_hamming_distance: int
    average_hamming_distance: float
    average_normalized_hamming_distance: float


@dataclass
class HammingDistanceInstance:
    N: int
    a: int


@dataclass
class ErrorModelHammingDistanceSweep:
    N: int
    a: int
    error_model: str
    num_phase_bits: int
    results: list[HammingDistanceSweepResult]


def validate_probability_sweep(
    min_probability: float,
    max_probability: float,
    num_points: int,
) -> None:
    """
    Validate the requested sweep over error probabilities.
    """
    if not (0.0 <= min_probability <= 1.0):
        raise ValueError("min-p must satisfy 0 <= min-p <= 1")
    if not (0.0 <= max_probability <= 1.0):
        raise ValueError("max-p must satisfy 0 <= max-p <= 1")
    if min_probability > max_probability:
        raise ValueError("min-p must be <= max-p")
    if num_points < 1:
        raise ValueError("num-points must be >= 1")


def build_probability_values(
    min_probability: float,
    max_probability: float,
    num_points: int,
    log_p_grid: bool = False,
) -> list[float]:
    """
    Build either a linear or logarithmic sweep of error probabilities.
    """
    validate_probability_sweep(min_probability, max_probability, num_points)

    if log_p_grid and min_probability <= 0.0:
        raise ValueError("--log-p-grid requires --min-p > 0")

    if num_points == 1:
        return [float(min_probability)]

    if not log_p_grid:
        return [
            float(value)
            for value in np.linspace(min_probability, max_probability, num_points)
        ]

    return [
        float(value)
        for value in np.geomspace(min_probability, max_probability, num_points)
    ]


def normalize_error_model(error_model: str) -> str:
    """
    Return the canonical name for one error-model label or alias.
    """
    normalized_label = error_model.strip().lower()
    try:
        return ERROR_MODEL_ALIASES[normalized_label]
    except KeyError as exc:
        choices = ", ".join(ERROR_MODEL_CHOICES)
        raise ValueError(f"error model must be one of: {choices}") from exc


def normalize_error_models(error_models: list[str] | None) -> list[str]:
    """
    Normalize and validate the error models selected for plotting.
    """
    if error_models is None:
        return list(ERROR_MODEL_CHOICES)
    if not error_models:
        raise ValueError("at least one error model is required")

    normalized = [normalize_error_model(error_model) for error_model in error_models]
    if len(set(normalized)) != len(normalized):
        raise ValueError("error models must be distinct")

    return normalized


def error_config_for_model(error_model: str, error_probability: float) -> ErrorConfig:
    """
    Build the simulator error configuration for one error model and p value.
    """
    error_model = normalize_error_model(error_model)
    p = float(error_probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("error_probability must satisfy 0 <= p <= 1")

    if error_model == MEASUREMENT_BITFLIP_MODEL:
        return ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=p,
        )

    if error_model == BLOCK_DEPOLARIZING_MODEL:
        return ErrorConfig(controlled_unitary_depolarizing_probability=p)

    return ErrorConfig(controlled_unitary_per_qubit_depolarizing_probability=p)


def error_model_display_label(error_model: str) -> str:
    """
    Return a concise legend label for one error model.
    """
    error_model = normalize_error_model(error_model)
    if error_model == MEASUREMENT_BITFLIP_MODEL:
        return "Measurement bit-flip"
    if error_model == BLOCK_DEPOLARIZING_MODEL:
        return "Block depolarizing"
    return "Per-work-qubit depolarizing"


def error_model_filename_label(error_model: str) -> str:
    """
    Return a compact filename label for one error model.
    """
    error_model = normalize_error_model(error_model)
    if error_model == MEASUREMENT_BITFLIP_MODEL:
        return "measurement_bitflip"
    if error_model == BLOCK_DEPOLARIZING_MODEL:
        return "block_depolarizing"
    return "per_qubit_depolarizing"


def error_models_filename_suffix(error_models: list[str]) -> str:
    """
    Return a compact filename suffix for one or more error models.
    """
    normalized = normalize_error_models(error_models)
    if normalized == list(ERROR_MODEL_CHOICES):
        return "all_error_models"

    return "_".join(
        error_model_filename_label(error_model)
        for error_model in normalized
    )


def validate_bases(N: int, bases: list[int]) -> list[int]:
    """
    Validate the fixed coprime bases used for overlaid instance curves.
    """
    if not bases:
        raise ValueError("at least one base a is required")
    if len(set(bases)) != len(bases):
        raise ValueError("bases must be distinct")

    for a in bases:
        validate_phase_estimation_instance(N, a)

    return list(bases)


def base_filename_suffix(bases: list[int]) -> str:
    """
    Return the filename suffix describing the selected base or bases.
    """
    if len(bases) == 1:
        return f"a{bases[0]}"

    return "bases" + "_".join(str(a) for a in bases)


def base_filename_suffix_with_bits(
    bases: list[int],
    num_phase_bits_by_base: dict[int, int],
) -> str:
    """
    Return the filename suffix for selected bases including phase-bit counts.
    """
    if len(bases) == 1:
        a = bases[0]
        return f"a{a}_t{num_phase_bits_by_base[a]}"

    return "bases" + "_".join(
        f"{a}_t{num_phase_bits_by_base[a]}"
        for a in bases
    )


def build_instances(
    N_values: list[int],
    bases: list[int],
) -> list[HammingDistanceInstance]:
    """
    Build and validate the requested fixed (N, a) instances.

    Supported forms:
      - one N with one or more bases: N --a a1 a2 ...
      - multiple N values with one shared base: N1 N2 ... --a a
      - multiple N values paired with equal-length bases: N1 N2 ... --a a1 a2 ...
    """
    if not N_values:
        raise ValueError("at least one N is required")
    if not bases:
        raise ValueError("at least one base a is required")

    if len(N_values) == 1:
        return [
            HammingDistanceInstance(N=N_values[0], a=a)
            for a in validate_bases(N_values[0], bases)
        ]

    if len(bases) == 1:
        instances = [
            HammingDistanceInstance(N=N, a=bases[0])
            for N in N_values
        ]
    elif len(bases) == len(N_values):
        instances = [
            HammingDistanceInstance(N=N, a=a)
            for N, a in zip(N_values, bases)
        ]
    else:
        raise ValueError(
            "when multiple N values are provided, --a must contain either "
            "one shared base or the same number of bases as N values"
        )

    instance_keys = [(instance.N, instance.a) for instance in instances]
    if len(set(instance_keys)) != len(instance_keys):
        raise ValueError("instances must be distinct")

    for instance in instances:
        validate_phase_estimation_instance(instance.N, instance.a)

    return instances


def instance_filename_suffix(instances: list[HammingDistanceInstance]) -> str:
    """
    Return the output filename suffix for one or more instances.
    """
    N_values = {instance.N for instance in instances}
    if len(N_values) == 1:
        N = instances[0].N
        bases = [instance.a for instance in instances]
        return f"N{N}_{base_filename_suffix(bases)}"

    return "instances_" + "_".join(
        f"N{instance.N}_a{instance.a}"
        for instance in instances
    )


def instance_filename_suffix_with_bits(
    sweeps: list[ErrorModelHammingDistanceSweep],
) -> str:
    """
    Return the output filename suffix for one or more instances including t.
    """
    instance_keys = list(dict.fromkeys((sweep.N, sweep.a) for sweep in sweeps))
    bit_counts_by_instance = {
        (sweep.N, sweep.a): sweep.num_phase_bits
        for sweep in sweeps
    }
    N_values = {N for N, _ in instance_keys}
    if len(N_values) == 1:
        N = instance_keys[0][0]
        bases = [a for _, a in instance_keys]
        bit_counts_by_base = {
            a: bit_counts_by_instance[(N, a)]
            for a in bases
        }
        return f"N{N}_{base_filename_suffix_with_bits(bases, bit_counts_by_base)}"

    return "instances_" + "_".join(
        f"N{N}_a{a}_t{bit_counts_by_instance[(N, a)]}"
        for N, a in instance_keys
    )


def instance_display_label(instances: list[HammingDistanceInstance]) -> str:
    """
    Return a compact display label for one or more instances.
    """
    if len(instances) == 1:
        instance = instances[0]
        return f"N = {instance.N}, a = {instance.a}"

    return "instances = " + "; ".join(
        f"(N = {instance.N}, a = {instance.a})"
        for instance in instances
    )


def instance_display_label_with_bits(
    sweeps: list[ErrorModelHammingDistanceSweep],
) -> str:
    """
    Return a compact display label including phase-bit counts.
    """
    instance_keys = list(dict.fromkeys((sweep.N, sweep.a) for sweep in sweeps))
    bit_counts_by_instance = {
        (sweep.N, sweep.a): sweep.num_phase_bits
        for sweep in sweeps
    }
    if len(instance_keys) == 1:
        N, a = instance_keys[0]
        return f"N = {N}, a = {a}, t = {bit_counts_by_instance[(N, a)]}"

    N_values = {N for N, _ in instance_keys}
    if len(N_values) == 1:
        N = instance_keys[0][0]
        base_parts = [
            f"a = {a}, t = {bit_counts_by_instance[(N, a)]}"
            for _, a in instance_keys
        ]
        return f"N = {N}, " + "; ".join(base_parts)

    return "instances = " + "; ".join(
        f"(N = {N}, a = {a}, t = {bit_counts_by_instance[(N, a)]})"
        for N, a in instance_keys
    )


def validate_overlay_mode(
    instances: list[HammingDistanceInstance],
    error_models: list[str],
) -> None:
    """
    Require the plot to overlay either instances or error models, not both.
    """
    if len(instances) > 1 and len(error_models) > 1:
        raise ValueError(
            "multiple (N, a) instances require exactly one --error-model; "
            "use multiple error models only for a single instance"
        )


def curve_label(
    sweep: ErrorModelHammingDistanceSweep,
    overlay_instances: bool,
    include_bits: bool,
) -> str:
    """
    Return the legend label for one Hamming-distance curve.
    """
    if overlay_instances:
        parts = [f"N = {sweep.N}", f"a = {sweep.a}"]
        parts.append(f"t = {sweep.num_phase_bits}")
        return ", ".join(parts)

    return error_model_display_label(sweep.error_model)


def simulate_for_error_probability(
    N: int,
    a: int,
    num_phase_bits: int,
    error_model: str,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
) -> HammingDistanceSweepResult:
    """
    Estimate the average clean/noisy Hamming distance for one fixed p.
    """
    estimate: AverageHammingDistanceEstimate = estimate_average_hamming_distance(
        N=N,
        a=a,
        num_phase_bits=num_phase_bits,
        trials=trials,
        error_config=error_config_for_model(error_model, error_probability),
        rng=rng,
    )

    return HammingDistanceSweepResult(
        error_probability=float(error_probability),
        trials=estimate.trials,
        total_hamming_distance=estimate.total_hamming_distance,
        average_hamming_distance=estimate.average_hamming_distance,
        average_normalized_hamming_distance=estimate.average_normalized_hamming_distance,
    )


def simulate_for_error_model(
    N: int,
    a: int,
    num_phase_bits: int,
    error_model: str,
    probability_values: list[float],
    trials: int,
    master_rng: np.random.Generator,
) -> ErrorModelHammingDistanceSweep:
    """
    Estimate average Hamming distance across all requested p values.
    """
    results = []
    for error_probability in probability_values:
        model_rng = np.random.default_rng(int(master_rng.integers(0, 2 ** 63 - 1)))
        results.append(
            simulate_for_error_probability(
                N=N,
                a=a,
                num_phase_bits=num_phase_bits,
                error_model=error_model,
                error_probability=float(error_probability),
                trials=trials,
                rng=model_rng,
            )
        )

    return ErrorModelHammingDistanceSweep(
        N=N,
        a=a,
        error_model=normalize_error_model(error_model),
        num_phase_bits=num_phase_bits,
        results=results,
    )


def write_data_summary(
    output_path: Path,
    trials: int,
    log_p_grid: bool,
    normalized: bool,
    seed: int | None,
    sweeps: list[ErrorModelHammingDistanceSweep],
) -> None:
    """
    Write a text summary containing metadata, graph notes, and per-p data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)
    selected_models = list(dict.fromkeys(sweep.error_model for sweep in sweeps))
    model_label = ", ".join(
        error_model_display_label(error_model)
        for error_model in selected_models
    )
    distinct_instances = list(dict.fromkeys((sweep.N, sweep.a) for sweep in sweeps))
    overlay_instances = len(distinct_instances) > 1
    bit_counts = sorted({sweep.num_phase_bits for sweep in sweeps})
    bit_count_label = (
        str(bit_counts[0])
        if len(bit_counts) == 1
        else "per instance: " + ", ".join(str(bit_count) for bit_count in bit_counts)
    )
    all_results = [
        result
        for sweep in sweeps
        for result in sweep.results
    ]

    N_header = "N"
    base_header = "base a"
    model_header = "error model"
    bits_header = "phase bits"
    p_header = "error probability p"
    average_header = "avg Hamming distance"
    normalized_header = "avg normalized distance"
    total_header = "total Hamming distance"
    trials_header = "trials"

    N_width = max(
        len(N_header),
        *(len(str(sweep.N)) for sweep in sweeps),
    )
    base_width = max(
        len(base_header),
        *(len(str(sweep.a)) for sweep in sweeps),
    )
    model_width = max(
        len(model_header),
        *(len(error_model_display_label(sweep.error_model)) for sweep in sweeps),
    )
    bits_width = max(
        len(bits_header),
        *(len(str(sweep.num_phase_bits)) for sweep in sweeps),
    )
    p_width = max(
        len(p_header),
        *(len(f"{result.error_probability:.3f}") for result in all_results),
    )
    average_width = max(
        len(average_header),
        *(len(f"{result.average_hamming_distance:.3f}") for result in all_results),
    )
    normalized_width = max(
        len(normalized_header),
        *(
            len(f"{result.average_normalized_hamming_distance:.3f}")
            for result in all_results
        ),
    )
    total_width = max(
        len(total_header),
        *(len(str(result.total_hamming_distance)) for result in all_results),
    )
    trials_width = max(
        len(trials_header),
        *(len(str(result.trials)) for result in all_results),
    )

    lines = [
        "Kitaev Phase-Estimation Hamming Distance vs Error Probability Summary",
        instance_display_label([
            HammingDistanceInstance(N=N, a=a)
            for N, a in distinct_instances
        ]),
        f"Phase-estimation bits = {bit_count_label}",
        f"Trials per probability value = {trials}",
        f"Error models = {model_label}",
        "Probability grid = "
        f"{'logarithmic' if log_p_grid else 'linear'}",
        "Plotted quantity = "
        f"{'average normalized Hamming distance' if normalized else 'average Hamming distance'}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        "  "
        + (
            "Fixed valid (N, a) instances for one error model"
            if overlay_instances
            else "One fixed valid (N, a) instance with selected error models"
        ),
        "  Each trial pairs a clean Kitaev phase-estimation run with a noisy run",
        "  The clean and noisy runs share the same measurement RNG seed",
        "  Measurement bit-flip uses independent per-detection flips with probability p",
        "  Depolarizing models apply trajectory Pauli errors after each controlled-U block",
        "  The distance is computed between clean and noisy measured phase-bit strings",
        "  Probability values use a "
        f"{'logarithmic' if log_p_grid else 'linear'} sweep in p",
        "",
        "Per-probability data:",
        "  "
        f"{N_header:>{N_width}} | "
        f"{base_header:>{base_width}} | "
        f"{model_header:>{model_width}} | "
        f"{bits_header:>{bits_width}} | "
        f"{p_header:>{p_width}} | "
        f"{average_header:>{average_width}} | "
        f"{normalized_header:>{normalized_width}} | "
        f"{total_header:>{total_width}} | "
        f"{trials_header:>{trials_width}}",
    ]

    for sweep in sweeps:
        for result in sweep.results:
            lines.append(
                "  "
                f"{sweep.N:>{N_width}} | "
                f"{sweep.a:>{base_width}} | "
                f"{error_model_display_label(sweep.error_model):>{model_width}} | "
                f"{sweep.num_phase_bits:>{bits_width}} | "
                f"{result.error_probability:>{p_width}.3f} | "
                f"{result.average_hamming_distance:>{average_width}.3f} | "
                f"{result.average_normalized_hamming_distance:>{normalized_width}.3f} | "
                f"{result.total_hamming_distance:>{total_width}} | "
                f"{result.trials:>{trials_width}}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_hamming_distance_vs_error_probability(
    trials: int,
    sweeps: list[ErrorModelHammingDistanceSweep],
    log_p_grid: bool,
    normalized: bool,
    output_path: Path,
) -> None:
    """
    Create and save the Hamming-distance plot versus error probability p.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [result.error_probability for result in sweeps[0].results]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.get_cmap("tab20").colors
    markers = ["o", "s", "^", "D", "P", "v", "X", "*", "<", ">"]
    overlay_instances = len({(sweep.N, sweep.a) for sweep in sweeps}) > 1
    include_bits = len({sweep.num_phase_bits for sweep in sweeps}) > 1

    for index, sweep in enumerate(sweeps):
        if normalized:
            y_values = [
                result.average_normalized_hamming_distance
                for result in sweep.results
            ]
        else:
            y_values = [
                result.average_hamming_distance
                for result in sweep.results
            ]

        ax.plot(
            x_values,
            y_values,
            marker=markers[index % len(markers)],
            linewidth=2.4,
            markersize=7,
            color=colors[index % len(colors)],
            label=curve_label(
                sweep,
                overlay_instances=overlay_instances,
                include_bits=include_bits,
            ),
        )

    ax.set_xlabel("Error probability p")
    if normalized:
        ax.set_ylabel("Average normalized Hamming distance")
        ax.set_ylim(-0.03, 1.03)
    else:
        ax.set_ylabel("Average Hamming distance")
        ax.set_ylim(-0.03, max(sweep.num_phase_bits for sweep in sweeps) + 0.3)

    selected_models = sorted({sweep.error_model for sweep in sweeps})
    if overlay_instances:
        title_target = "selected (N, a) instances"
        title_error = error_model_display_label(selected_models[0])
    else:
        title_target = instance_display_label_with_bits(sweeps)
        title_error = "selected error models"
    ax.set_title(
        "Kitaev Phase-Estimation Bit Distance vs Error Probability "
        f"for {title_target} ({title_error})"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    x_min = min(x_values)
    x_max = max(x_values)
    if log_p_grid:
        ax.set_xscale("log")
        ax.set_xlim(x_min * 0.9, x_max * 1.1)
    else:
        x_padding = max(0.01, (x_max - x_min) * 0.03)
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    ax.text(
        0.98,
        0.02,
        (
            f"phase bits = {sweeps[0].num_phase_bits}"
            if len({sweep.num_phase_bits for sweep in sweeps}) == 1
            else "phase bits vary"
        )
        + f"\ntrials per p = {trials}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "#D0D0D0",
        },
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the Hamming-distance graph script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot average clean/noisy Kitaev PE Hamming distance versus error "
            "probability for fixed (N, a) instances."
        )
    )
    parser.add_argument("N", type=int, nargs="+", help="One or more moduli to analyze")
    parser.add_argument(
        "--a",
        type=int,
        nargs="+",
        required=True,
        help=(
            "One or more fixed coprime bases a to analyze. With multiple N "
            "values, provide one shared base or one base per N."
        ),
    )
    parser.add_argument(
        "--error-model",
        nargs="+",
        default=None,
        help=(
            "One or more error models to overlay. Choices: measurement-bitflip, "
            "block-depolarizing, per-qubit-depolarizing. Defaults to all three."
        ),
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override the number of phase-estimation bits",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of paired clean/noisy trials per probability value",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Minimum error probability in the sweep",
    )
    parser.add_argument(
        "--max-p",
        type=float,
        default=1.0,
        help="Maximum error probability in the sweep",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=11,
        help="Number of probability values to sample between min-p and max-p",
    )
    parser.add_argument(
        "--log-p-grid",
        action="store_true",
        help=(
            "Use logarithmically spaced probability values between min-p and max-p. "
            "This requires min-p > 0."
        ),
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="Plot average Hamming distance divided by the number of phase bits",
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
    start_time = time.perf_counter()

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("trials must be >= 1")

    if args.bits is not None and args.bits < 1:
        raise ValueError("bits must be >= 1")

    instances = build_instances(args.N, args.a)
    error_models = normalize_error_models(args.error_model)
    validate_overlay_mode(instances, error_models)
    probability_values = build_probability_values(
        args.min_p,
        args.max_p,
        args.num_points,
        log_p_grid=args.log_p_grid,
    )

    master_rng = np.random.default_rng(args.seed)
    sweeps = []
    for instance in instances:
        num_phase_bits = (
            default_num_phase_bits(instance.N)
            if args.bits is None
            else args.bits
        )
        for error_model in error_models:
            sweeps.append(
                simulate_for_error_model(
                    N=instance.N,
                    a=instance.a,
                    num_phase_bits=num_phase_bits,
                    error_model=error_model,
                    probability_values=probability_values,
                    trials=args.trials,
                    master_rng=master_rng,
                )
            )

    suffix = (
        f"{instance_filename_suffix_with_bits(sweeps)}_"
        f"{error_models_filename_suffix(error_models)}"
    )
    if args.normalized:
        suffix = f"{suffix}_normalized"

    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"hamming_distance_vs_error_probability_{suffix}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"hamming_distance_vs_error_probability_summary_{suffix}.txt"
    )

    plot_hamming_distance_vs_error_probability(
        trials=args.trials,
        sweeps=sweeps,
        log_p_grid=args.log_p_grid,
        normalized=args.normalized,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        trials=args.trials,
        log_p_grid=args.log_p_grid,
        normalized=args.normalized,
        seed=args.seed,
        sweeps=sweeps,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    model_label = ", ".join(
        error_model_display_label(error_model)
        for error_model in error_models
    )

    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(instance_display_label(instances))
    bit_counts = sorted({sweep.num_phase_bits for sweep in sweeps})
    if len(bit_counts) == 1:
        print(f"Phase-estimation bits = {bit_counts[0]}")
    else:
        print(
            "Phase-estimation bits per instance: "
            + ", ".join(
                f"N={sweep.N}, a={sweep.a}: {sweep.num_phase_bits}"
                for sweep in sweeps
            )
        )
    print(f"Error models: {model_label}")
    print(f"Trials per probability value: {args.trials}")
    print(
        "Probability grid: "
        f"{'logarithmic' if args.log_p_grid else 'linear'}"
    )
    print(
        f"Error probability swept from {args.min_p:.3f} "
        f"to {args.max_p:.3f}"
    )
    print(
        "Plotted quantity: "
        f"{'average normalized Hamming distance' if args.normalized else 'average Hamming distance'}"
    )
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
