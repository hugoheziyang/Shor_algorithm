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
    AverageDecimalDistanceEstimate,
    estimate_average_decimal_distance,
)
from graphs.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    MEASUREMENT_BITFLIP_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
    ErrorModelHammingDistanceSweep,
    HammingDistanceInstance,
    build_instances as build_hamming_instances,
    build_probability_values,
    error_config_for_model,
    error_model_display_label,
    error_models_filename_suffix,
    instance_display_label as hamming_instance_display_label,
    instance_display_label_with_bits as hamming_instance_display_label_with_bits,
    instance_filename_suffix as hamming_instance_filename_suffix,
    instance_filename_suffix_with_bits as hamming_instance_filename_suffix_with_bits,
    normalize_error_model,
    normalize_error_models,
    validate_overlay_mode as validate_hamming_overlay_mode,
)
from shor_one_qubit.shor import default_num_phase_bits


@dataclass(frozen=True)
class DecimalDistanceSweepResult:
    error_probability: float
    trials: int
    total_decimal_distance: float
    average_decimal_distance: float


@dataclass(frozen=True)
class DecimalDistanceInstance:
    N: int
    a: int


@dataclass(frozen=True)
class ErrorModelDecimalDistanceSweep:
    N: int
    a: int
    error_model: str
    num_phase_bits: int
    results: list[DecimalDistanceSweepResult]


def _as_hamming_instances(
    instances: list[DecimalDistanceInstance],
) -> list[HammingDistanceInstance]:
    return [
        HammingDistanceInstance(N=instance.N, a=instance.a)
        for instance in instances
    ]


def build_instances(
    N_values: list[int],
    bases: list[int],
) -> list[DecimalDistanceInstance]:
    """
    Build and validate the requested fixed (N, a) instances.
    """
    return [
        DecimalDistanceInstance(N=instance.N, a=instance.a)
        for instance in build_hamming_instances(N_values, bases)
    ]


def instance_filename_suffix(instances: list[DecimalDistanceInstance]) -> str:
    """
    Return the output filename suffix for one or more instances.
    """
    return hamming_instance_filename_suffix(_as_hamming_instances(instances))


def instance_filename_suffix_with_bits(
    sweeps: list[ErrorModelDecimalDistanceSweep],
) -> str:
    """
    Return the output filename suffix for one or more instances including t.
    """
    hamming_sweeps = [
        ErrorModelHammingDistanceSweep(
            N=sweep.N,
            a=sweep.a,
            error_model=sweep.error_model,
            num_phase_bits=sweep.num_phase_bits,
            results=[],
        )
        for sweep in sweeps
    ]
    return hamming_instance_filename_suffix_with_bits(hamming_sweeps)


def instance_display_label(instances: list[DecimalDistanceInstance]) -> str:
    """
    Return a compact display label for one or more instances.
    """
    return hamming_instance_display_label(_as_hamming_instances(instances))


def instance_display_label_with_bits(
    sweeps: list[ErrorModelDecimalDistanceSweep],
) -> str:
    """
    Return a compact display label including phase-bit counts.
    """
    hamming_sweeps = [
        ErrorModelHammingDistanceSweep(
            N=sweep.N,
            a=sweep.a,
            error_model=sweep.error_model,
            num_phase_bits=sweep.num_phase_bits,
            results=[],
        )
        for sweep in sweeps
    ]
    return hamming_instance_display_label_with_bits(hamming_sweeps)


def validate_overlay_mode(
    instances: list[DecimalDistanceInstance],
    error_models: list[str],
) -> None:
    """
    Require the plot to overlay either instances or error models, not both.
    """
    validate_hamming_overlay_mode(_as_hamming_instances(instances), error_models)


def curve_label(
    sweep: ErrorModelDecimalDistanceSweep,
    overlay_instances: bool,
    include_bits: bool,
) -> str:
    """
    Return the legend label for one decimal-distance curve.
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
) -> DecimalDistanceSweepResult:
    """
    Estimate the average clean/noisy decimal distance for one fixed p.
    """
    estimate: AverageDecimalDistanceEstimate = estimate_average_decimal_distance(
        N=N,
        a=a,
        num_phase_bits=num_phase_bits,
        trials=trials,
        error_config=error_config_for_model(error_model, error_probability),
        rng=rng,
    )

    return DecimalDistanceSweepResult(
        error_probability=float(error_probability),
        trials=estimate.trials,
        total_decimal_distance=estimate.total_decimal_distance,
        average_decimal_distance=estimate.average_decimal_distance,
    )


def simulate_for_error_model(
    N: int,
    a: int,
    num_phase_bits: int,
    error_model: str,
    probability_values: list[float],
    trials: int,
    master_rng: np.random.Generator,
) -> ErrorModelDecimalDistanceSweep:
    """
    Estimate average decimal distance across all requested p values.
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

    return ErrorModelDecimalDistanceSweep(
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
    seed: int | None,
    sweeps: list[ErrorModelDecimalDistanceSweep],
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
    average_header = "avg decimal distance"
    total_header = "total decimal distance"
    trials_header = "trials"

    N_width = max(len(N_header), *(len(str(sweep.N)) for sweep in sweeps))
    base_width = max(len(base_header), *(len(str(sweep.a)) for sweep in sweeps))
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
        *(len(f"{result.average_decimal_distance:.6f}") for result in all_results),
    )
    total_width = max(
        len(total_header),
        *(len(f"{result.total_decimal_distance:.6f}") for result in all_results),
    )
    trials_width = max(
        len(trials_header),
        *(len(str(result.trials)) for result in all_results),
    )

    lines = [
        "Decimal Distance vs Error Probability Summary",
        instance_display_label([
            DecimalDistanceInstance(N=N, a=a)
            for N, a in distinct_instances
        ]),
        f"Phase-estimation bits = {bit_count_label}",
        f"Trials per probability value = {trials}",
        f"Error models = {model_label}",
        "Probability grid = "
        f"{'logarithmic' if log_p_grid else 'linear'}",
        "Plotted quantity = average decimal distance",
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
        "  The distance is min(|x-y|, 1-|x-y|) after converting bit strings to phases",
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
                f"{result.average_decimal_distance:>{average_width}.6f} | "
                f"{result.total_decimal_distance:>{total_width}.6f} | "
                f"{result.trials:>{trials_width}}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_decimal_distance_vs_error_probability(
    trials: int,
    sweeps: list[ErrorModelDecimalDistanceSweep],
    log_p_grid: bool,
    output_path: Path,
) -> None:
    """
    Create and save the decimal-distance plot versus error probability p.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [result.error_probability for result in sweeps[0].results]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.get_cmap("tab20").colors
    markers = ["o", "s", "^", "D", "P", "v", "X", "*", "<", ">"]
    overlay_instances = len({(sweep.N, sweep.a) for sweep in sweeps}) > 1
    include_bits = len({sweep.num_phase_bits for sweep in sweeps}) > 1

    for index, sweep in enumerate(sweeps):
        y_values = [
            result.average_decimal_distance
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
    ax.set_ylabel("Average decimal distance")
    ax.set_ylim(-0.015, 0.515)

    selected_models = sorted({sweep.error_model for sweep in sweeps})
    if overlay_instances:
        title_target = "selected (N, a) instances"
        title_error = error_model_display_label(selected_models[0])
    else:
        title_target = instance_display_label_with_bits(sweeps)
        title_error = "selected error models"
    ax.set_title(
        "Decimal Distance vs Error Probability "
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
    Build the command-line interface for the decimal-distance graph script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot average clean/noisy Kitaev PE decimal distance versus "
            "error probability for fixed (N, a) instances."
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

    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"decimal_distance_vs_error_probability_{suffix}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"decimal_distance_vs_error_probability_summary_{suffix}.txt"
    )

    plot_decimal_distance_vs_error_probability(
        trials=args.trials,
        sweeps=sweeps,
        log_p_grid=args.log_p_grid,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        trials=args.trials,
        log_p_grid=args.log_p_grid,
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
    print("Plotted quantity: average decimal distance")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
