from __future__ import annotations

import argparse
import math
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


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from shor_one_qubit.errors import (
    ErrorConfig,
    MEASUREMENT_BITFLIP_SINGLE_TOKEN,
    measurement_bitflip_filename_suffix,
    measurement_bitflip_label,
    measurement_bitflip_title_suffix,
    parse_measurement_bitflip_error_arg,
)
from shor_one_qubit.shor import (
    default_num_phase_bits,
    perfect_power_decomposition,
    shor_order_finding_attempt,
)


@dataclass
class PhaseOutcomeRow:
    bits: str
    outcome_integer: int
    phase_decimal: float
    count: int
    probability: float


def is_odd_composite(n: int) -> bool:
    """
    Return True for odd composite integers.
    """
    if n < 9 or (n % 2) == 0:
        return False

    for factor in range(3, math.isqrt(n) + 1, 2):
        if (n % factor) == 0:
            return True

    return False


def validate_target_instance(N: int, a: int) -> None:
    """
    Validate the fixed Shor instance used in the phase-distribution experiment.
    """
    if not is_odd_composite(N):
        raise ValueError("N must be an odd composite integer >= 9")

    if perfect_power_decomposition(N) is not None:
        raise ValueError(
            "perfect powers are excluded here because this repository treats "
            "them as classical shortcuts rather than quantum order-finding cases"
        )

    if not (2 <= a <= N - 2):
        raise ValueError("a must satisfy 2 <= a <= N - 2")
    if math.gcd(a, N) != 1:
        raise ValueError("a must be coprime to N")


def multiplicative_order(a: int, N: int) -> int:
    """
    Return the exact multiplicative order of a modulo N.
    """
    if math.gcd(a, N) != 1:
        raise ValueError("a and N must be coprime")

    residue = 1
    for r in range(1, N + 1):
        residue = (residue * a) % N
        if residue == 1:
            return r

    raise RuntimeError("failed to find multiplicative order")


def bits_to_integer(bits: list[int]) -> int:
    """
    Convert a binary fractional bit list [b1, b2, ..., bt] into the integer
    m represented by 0.b1b2...bt = m / 2^t.
    """
    outcome = 0
    for bit in bits:
        outcome = (outcome << 1) | bit
    return outcome


def sample_phase_distribution(
    N: int,
    a: int,
    num_phase_bits: int,
    trials: int,
    rng: np.random.Generator,
    error_config: ErrorConfig | None = None,
) -> list[PhaseOutcomeRow]:
    """
    Sample the iterative phase-estimation outcomes for a fixed (N, a).
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")

    counts: Counter[int] = Counter()
    phase_denominator = 2 ** num_phase_bits
    error_config = error_config or ErrorConfig()

    for _ in range(trials):
        result = shor_order_finding_attempt(
            N=N,
            a=a,
            num_phase_bits=num_phase_bits,
            rng=rng,
            postprocess_mode="one_shot",
            error_config=error_config,
        )
        outcome = bits_to_integer(result["bits"])
        counts[outcome] += 1

    rows = []
    for outcome in sorted(counts):
        count = counts[outcome]
        bits = format(outcome, f"0{num_phase_bits}b")
        rows.append(
            PhaseOutcomeRow(
                bits=bits,
                outcome_integer=outcome,
                phase_decimal=outcome / phase_denominator,
                count=count,
                probability=count / trials,
            )
        )

    return rows


def write_data_summary(
    output_path: Path,
    N: int,
    a: int,
    true_order: int,
    num_phase_bits: int,
    trials: int,
    error_config: ErrorConfig | None,
    seed: int | None,
    rows: list[PhaseOutcomeRow],
) -> None:
    """
    Write a text summary containing metadata, graph notes, and observed outcomes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)
    error_label = measurement_bitflip_label(error_config)
    bits_header = "bits"
    outcome_header = "outcome integer m"
    phase_header = "decimal phase"
    count_header = "count"
    probability_header = "empirical probability"

    bits_width = max(len(bits_header), *(len(row.bits) for row in rows))
    outcome_width = max(
        len(outcome_header),
        *(len(str(row.outcome_integer)) for row in rows),
    )
    phase_width = max(
        len(phase_header),
        *(len(f"{row.phase_decimal:.6f}") for row in rows),
    )
    count_width = max(len(count_header), *(len(str(row.count)) for row in rows))
    probability_width = max(
        len(probability_header),
        *(len(f"{row.probability:.6f}") for row in rows),
    )

    lines = [
        "Phase-Estimation Distribution Summary",
        f"N = {N}",
        f"a = {a}",
        f"True order r = {true_order}",
        f"Phase-estimation bits = {num_phase_bits}",
        f"Trials = {trials}",
        f"Measurement bit-flip error = {error_label}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        "  Fixed odd-composite N and fixed coprime base a",
        "  Each trial reruns the iterative phase-estimation procedure for the same (N, a)",
        f"  Measurement bit-flip error: {error_label}",
        "  Horizontal axis shows the measured phase estimate in decimal form",
        "  Only observed phase outcomes are listed below",
        "",
        "Observed phase-outcome data:",
        "  "
        f"{bits_header:<{bits_width}} | "
        f"{outcome_header:>{outcome_width}} | "
        f"{phase_header:>{phase_width}} | "
        f"{count_header:>{count_width}} | "
        f"{probability_header:>{probability_width}}",
    ]

    for row in rows:
        lines.append(
            "  "
            f"{row.bits:<{bits_width}} | "
            f"{row.outcome_integer:>{outcome_width}} | "
            f"{row.phase_decimal:>{phase_width}.6f} | "
            f"{row.count:>{count_width}} | "
            f"{row.probability:>{probability_width}.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_phase_distribution(
    N: int,
    a: int,
    true_order: int,
    num_phase_bits: int,
    trials: int,
    error_config: ErrorConfig | None,
    rows: list[PhaseOutcomeRow],
    output_path: Path,
) -> None:
    """
    Create and save the empirical phase-estimation outcome distribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [row.phase_decimal for row in rows]
    y_values = [row.probability for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.vlines(
        x_values,
        0.0,
        y_values,
        color="#1F6F78",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.scatter(
        x_values,
        y_values,
        s=28,
        color="#A23B2A",
        zorder=3,
    )

    ax.set_xlim(-0.01, 1.01)
    ax.set_xlabel("Measured phase estimate (decimal)")
    ax.set_ylabel("Empirical probability")
    ax.set_title(
        "Phase-Estimation Outcome Distribution for N = "
        f"{N}, a = {a}, t = {num_phase_bits}"
        f"{measurement_bitflip_title_suffix(error_config)}"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)

    ax.text(
        0.98,
        0.98,
        "true order r = "
        f"{true_order}\nphase bits = {num_phase_bits}\ntrials = {trials}\n"
        f"measurement bit-flip error = {measurement_bitflip_label(error_config)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#D0D0D0"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the phase-distribution graph script.
    """
    parser = argparse.ArgumentParser(
        description="Plot the empirical iterative phase-estimation outcome distribution for a fixed (N, a)."
    )
    parser.add_argument("N", type=int, help="Odd composite number to factor")
    parser.add_argument(
        "--a",
        type=int,
        required=True,
        help="Fixed coprime base a to analyze",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override the number of phase-estimation bits (default: 2 * ceil(log2(N)) + 1)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of repeated phase-estimation samples to draw",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling",
    )
    parser.add_argument(
        "--measurement-bitflip-error",
        nargs="?",
        const=MEASUREMENT_BITFLIP_SINGLE_TOKEN,
        default=None,
        metavar="P",
        help=(
            "Enable measurement bit-flip errors. Use without a value for "
            "exactly one random bit-flip per iterative phase-estimation run, "
            "or provide P in [0,1] for independent per-detection bit-flips."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure",
    )
    return parser


def main() -> None:
    start_time = time.perf_counter()

    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        error_config = parse_measurement_bitflip_error_arg(
            args.measurement_bitflip_error
        )
    except ValueError as exc:
        parser.error(str(exc))

    validate_target_instance(args.N, args.a)

    if args.trials < 1:
        raise ValueError("trials must be >= 1")

    if args.bits is None:
        num_phase_bits = default_num_phase_bits(args.N)
    else:
        num_phase_bits = args.bits

    true_order = multiplicative_order(args.a, args.N)

    if args.output is None:
        filename_suffix = measurement_bitflip_filename_suffix(error_config)
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / (
                f"phase_estimation_distribution_N{args.N}_a{args.a}"
                f"_t{num_phase_bits}{filename_suffix}.png"
            )
        )
    else:
        figure_output_path = args.output

    filename_suffix = measurement_bitflip_filename_suffix(error_config)
    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / (
            f"phase_estimation_distribution_summary_N{args.N}_a{args.a}"
            f"_t{num_phase_bits}{filename_suffix}.txt"
        )
    )

    rng = np.random.default_rng(args.seed)
    rows = sample_phase_distribution(
        N=args.N,
        a=args.a,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        rng=rng,
        error_config=error_config,
    )

    plot_phase_distribution(
        N=args.N,
        a=args.a,
        true_order=true_order,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        error_config=error_config,
        rows=rows,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        N=args.N,
        a=args.a,
        true_order=true_order,
        num_phase_bits=num_phase_bits,
        trials=args.trials,
        error_config=error_config,
        seed=args.seed,
        rows=rows,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"N = {args.N}")
    print(f"a = {args.a}")
    print(f"True order r = {true_order}")
    print(f"Phase-estimation bits = {num_phase_bits}")
    print(f"Trials = {args.trials}")
    print(f"Measurement bit-flip error: {measurement_bitflip_label(error_config)}")
    print(f"Seed = {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
