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


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from error_correction.classical_error_correction import (
    RecoveryProbabilityEstimate,
    classical_recovery_first_order_upper_bound,
    classical_recovery_lower_bound as propagating_error_lower_bound,
    estimate_classical_recovery_probability,
    estimate_standard_classical_recovery_probability,
    kitaev_first_order_recovery_estimate,
    kitaev_single_error_recoverability_coefficient,
    validate_phase_estimation_instance,
)
from shor_common.shor_utils import default_num_phase_bits
from shor_standard.backends import STANDARD_BACKEND_CHOICES, STATEVECTOR_STANDARD_BACKEND


MAX_CURVES = 5
KITAEV_ALGORITHM = "kitaev"
STANDARD_ALGORITHM = "standard"
ALGORITHM_CHOICES = (KITAEV_ALGORITHM, STANDARD_ALGORITHM)


@dataclass
class RecoverySweepResult:
    error_probability: float
    successful_trials: int
    trials: int
    recovery_probability: float
    naive_baseline_probability: float


@dataclass
class RecoveryInstance:
    N: int
    a: int


@dataclass
class InstanceRecoverySweep:
    N: int
    a: int
    num_phase_bits: int
    results: list[RecoverySweepResult]
    algorithm: str = KITAEV_ALGORITHM
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND
    analytic_window: int = 100


def algorithm_display_label(algorithm: str) -> str:
    """
    Return a human-readable algorithm label.
    """
    if algorithm == KITAEV_ALGORITHM:
        return "Kitaev"
    if algorithm == STANDARD_ALGORITHM:
        return "standard Shor"
    raise ValueError("unknown algorithm")


def algorithm_filename_prefix(algorithm: str) -> str:
    """
    Return a filename prefix segment while preserving Kitaev file compatibility.
    """
    if algorithm == KITAEV_ALGORITHM:
        return ""
    if algorithm == STANDARD_ALGORITHM:
        return "standard_"
    raise ValueError("unknown algorithm")


def normalize_algorithms(algorithms: list[str]) -> list[str]:
    """
    Validate and de-duplicate requested algorithms while preserving order.
    """
    if not algorithms:
        raise ValueError("at least one algorithm is required")

    normalized = []
    for algorithm in algorithms:
        if algorithm not in ALGORITHM_CHOICES:
            raise ValueError("unknown algorithm")
        if algorithm not in normalized:
            normalized.append(algorithm)

    return normalized


def algorithms_display_label(algorithms: list[str]) -> str:
    """
    Return a human-readable label for one or more algorithms.
    """
    algorithms = normalize_algorithms(algorithms)
    if len(algorithms) == 1:
        return algorithm_display_label(algorithms[0])

    return " vs ".join(algorithm_display_label(algorithm) for algorithm in algorithms)


def algorithms_filename_prefix(algorithms: list[str]) -> str:
    """
    Return a filename prefix for one or more selected algorithms.
    """
    algorithms = normalize_algorithms(algorithms)
    if len(algorithms) == 1:
        return algorithm_filename_prefix(algorithms[0])

    if algorithms == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]:
        return "compare_algorithms_"

    return "compare_" + "_".join(algorithms) + "_"


def standard_backend_filename_segment(
    algorithms: list[str],
    standard_backend: str,
) -> str:
    """
    Return an extra filename segment for non-default standard backends.
    """
    algorithms = normalize_algorithms(algorithms)
    if STANDARD_ALGORITHM not in algorithms:
        return ""
    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        return ""
    return f"{standard_backend}_"


def bit_count_label(algorithm: str) -> str:
    """
    Return the display name for the swept register/measurement size.
    """
    if algorithm == KITAEV_ALGORITHM:
        return "phase bits"
    if algorithm == STANDARD_ALGORITHM:
        return "control qubits"
    raise ValueError("unknown algorithm")


def bit_count_label_for_algorithms(algorithms: list[str]) -> str:
    """
    Return the display name for t when one or more algorithms are selected.
    """
    algorithms = normalize_algorithms(algorithms)
    if len(algorithms) == 1:
        return bit_count_label(algorithms[0])

    return "phase bits/control qubits"


def baseline_curve_label(algorithm: str) -> str:
    """
    Return the analytic baseline label for the selected algorithm.
    """
    if algorithm == KITAEV_ALGORITHM:
        return "First-order upper bound"
    if algorithm == STANDARD_ALGORITHM:
        return "Exact single-bit recovery"
    raise ValueError("unknown algorithm")


def baseline_curve_label_for_algorithms(algorithms: list[str]) -> str:
    """
    Return the analytic baseline label for one or more algorithms.
    """
    algorithms = normalize_algorithms(algorithms)
    if len(algorithms) == 1:
        return baseline_curve_label(algorithms[0])

    return "Single-bit recovery baseline"


def validate_probability_sweep(
    min_probability: float,
    max_probability: float,
    num_points: int,
) -> None:
    """
    Validate the requested sweep over measurement-error probabilities.
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
    Build either a linear or logarithmic sweep of measurement-error probabilities.
    """
    validate_probability_sweep(min_probability, max_probability, num_points)

    if num_points == 1:
        return [float(min_probability)]

    if not log_p_grid:
        return [float(value) for value in np.linspace(min_probability, max_probability, num_points)]

    if min_probability <= 0.0:
        raise ValueError("--log-p-grid requires --min-p > 0")

    return [float(value) for value in np.geomspace(min_probability, max_probability, num_points)]


def validate_bases(N: int, bases: list[int]) -> list[int]:
    """
    Validate the fixed coprime bases used for the overlaid recovery curves.
    """
    if not bases:
        raise ValueError("at least one base a is required")
    if len(bases) > MAX_CURVES:
        raise ValueError(f"at most {MAX_CURVES} bases can be overlaid")
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


def base_display_label(bases: list[int]) -> str:
    """
    Return a compact display label for one or more bases.
    """
    if len(bases) == 1:
        return f"a = {bases[0]}"

    return "bases = " + ", ".join(str(a) for a in bases)


def build_instances(N_values: list[int], bases: list[int]) -> list[RecoveryInstance]:
    """
    Build and validate the list of (N, a) curves requested by the CLI.

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
            RecoveryInstance(N=N_values[0], a=a)
            for a in validate_bases(N_values[0], bases)
        ]

    if len(bases) == 1:
        instances = [
            RecoveryInstance(N=N, a=bases[0])
            for N in N_values
        ]
    elif len(bases) == len(N_values):
        instances = [
            RecoveryInstance(N=N, a=a)
            for N, a in zip(N_values, bases)
        ]
    else:
        raise ValueError(
            "when multiple N values are provided, --a must contain either "
            "one shared base or the same number of bases as N values"
        )

    if len(instances) > MAX_CURVES:
        raise ValueError(f"at most {MAX_CURVES} curves can be overlaid")

    instance_keys = [(instance.N, instance.a) for instance in instances]
    if len(set(instance_keys)) != len(instance_keys):
        raise ValueError("instances must be distinct")

    for instance in instances:
        validate_phase_estimation_instance(instance.N, instance.a)

    return instances


def instance_filename_suffix(instances: list[RecoveryInstance]) -> str:
    """
    Return the output filename suffix for one or more recovery instances.
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


def instance_filename_suffix_with_bits(sweeps: list[InstanceRecoverySweep]) -> str:
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


def instance_display_label(instances: list[RecoveryInstance]) -> str:
    """
    Return a compact display label for one or more recovery instances.
    """
    N_values = {instance.N for instance in instances}
    if len(N_values) == 1:
        N = instances[0].N
        bases = [instance.a for instance in instances]
        return f"N = {N}, {base_display_label(bases)}"

    if len(instances) == 1:
        instance = instances[0]
        return f"N = {instance.N}, a = {instance.a}"

    return "instances = " + "; ".join(
        f"(N = {instance.N}, a = {instance.a})"
        for instance in instances
    )


def instance_display_label_with_bits(sweeps: list[InstanceRecoverySweep]) -> str:
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


def curve_label(
    sweep: InstanceRecoverySweep,
    include_N: bool,
    include_bits: bool,
    include_algorithm: bool = False,
) -> str:
    """
    Return the legend label for one empirical recovery curve.
    """
    parts = []
    if include_algorithm:
        parts.append(algorithm_display_label(sweep.algorithm))
    if include_N:
        parts.append(f"N = {sweep.N}")
    parts.append(f"a = {sweep.a}")
    if include_N or include_bits:
        parts.append(f"t = {sweep.num_phase_bits}")
    return ", ".join(parts)


def simulate_for_error_probability(
    N: int,
    a: int,
    num_phase_bits: int,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
    algorithm: str = KITAEV_ALGORITHM,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> RecoverySweepResult:
    """
    Estimate the empirical classical recovery probability for one fixed p.
    """
    if algorithm == KITAEV_ALGORITHM:
        estimate: RecoveryProbabilityEstimate = estimate_classical_recovery_probability(
            N=N,
            a=a,
            num_phase_bits=num_phase_bits,
            error_probability=error_probability,
            trials=trials,
            rng=rng,
        )
    elif algorithm == STANDARD_ALGORITHM:
        estimate = estimate_standard_classical_recovery_probability(
            N=N,
            a=a,
            num_control_qubits=num_phase_bits,
            error_probability=error_probability,
            trials=trials,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
        )
    else:
        raise ValueError("unknown algorithm")

    return RecoverySweepResult(
        error_probability=estimate.error_probability,
        successful_trials=estimate.successful_trials,
        trials=estimate.trials,
        recovery_probability=estimate.recovery_probability,
        naive_baseline_probability=estimate.naive_baseline_probability,
    )


def simulate_for_instance(
    N: int,
    a: int,
    num_phase_bits: int,
    probability_values: list[float],
    trials: int,
    master_rng: np.random.Generator,
    algorithm: str = KITAEV_ALGORITHM,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> InstanceRecoverySweep:
    """
    Estimate recovery probability across all requested p values for one instance.
    """
    results = []
    for error_probability in probability_values:
        instance_rng = np.random.default_rng(int(master_rng.integers(0, 2 ** 63 - 1)))
        results.append(
            simulate_for_error_probability(
                N=N,
                a=a,
                num_phase_bits=num_phase_bits,
                error_probability=float(error_probability),
                trials=trials,
                rng=instance_rng,
                algorithm=algorithm,
                standard_backend=standard_backend,
                analytic_window=analytic_window,
            )
        )

    return InstanceRecoverySweep(
        N=N,
        a=a,
        num_phase_bits=num_phase_bits,
        results=results,
        algorithm=algorithm,
        standard_backend=standard_backend,
        analytic_window=analytic_window,
    )


def write_data_summary(
    output_path: Path,
    trials: int,
    log_p_grid: bool,
    show_lower_bound: bool,
    show_first_order: bool,
    seed: int | None,
    sweeps: list[InstanceRecoverySweep],
) -> None:
    """
    Write a text summary containing metadata, graph notes, and per-p data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seed_label = "random" if seed is None else str(seed)
    all_results = [
        result
        for sweep in sweeps
        for result in sweep.results
    ]
    instances = [
        RecoveryInstance(N=N, a=a)
        for N, a in dict.fromkeys((sweep.N, sweep.a) for sweep in sweeps)
    ]
    bit_counts = sorted({sweep.num_phase_bits for sweep in sweeps})
    algorithms = normalize_algorithms([sweep.algorithm for sweep in sweeps])
    standard_backends = sorted(
        {
            sweep.standard_backend
            for sweep in sweeps
            if sweep.algorithm == STANDARD_ALGORITHM
        }
    )
    analytic_windows = sorted(
        {
            sweep.analytic_window
            for sweep in sweeps
            if (
                sweep.algorithm == STANDARD_ALGORITHM
                and sweep.standard_backend != STATEVECTOR_STANDARD_BACKEND
            )
        }
    )
    bit_label = bit_count_label_for_algorithms(algorithms)
    baseline_label = baseline_curve_label_for_algorithms(algorithms)
    bit_count_value_label = (
        str(bit_counts[0])
        if len(bit_counts) == 1
        else "per instance: " + ", ".join(str(bit_count) for bit_count in bit_counts)
    )

    algorithm_header = "algorithm"
    N_header = "N"
    base_header = "base a"
    bits_header = bit_label
    p_header = "error probability p"
    recovery_header = "empirical recovery"
    baseline_header = baseline_label
    lower_bound_header = "propagating lower bound"
    first_order_header = "First-order estimate"
    coefficient_header = "C_t"
    success_header = "successful trials"
    total_header = "total trials"

    algorithm_width = max(
        len(algorithm_header),
        *(len(algorithm_display_label(sweep.algorithm)) for sweep in sweeps),
    )
    N_width = max(
        len(N_header),
        *(len(str(sweep.N)) for sweep in sweeps),
    )
    base_width = max(
        len(base_header),
        *(len(str(sweep.a)) for sweep in sweeps),
    )
    bits_width = max(
        len(bits_header),
        *(len(str(sweep.num_phase_bits)) for sweep in sweeps),
    )
    p_width = max(
        len(p_header),
        *(len(f"{result.error_probability:.3f}") for result in all_results),
    )
    recovery_width = max(
        len(recovery_header),
        *(len(f"{result.recovery_probability:.3f}") for result in all_results),
    )
    baseline_width = max(
        len(baseline_header),
        *(len(f"{result.naive_baseline_probability:.3f}") for result in all_results),
    )
    lower_bound_width = max(
        len(lower_bound_header),
        *(
            len(f"{propagating_error_lower_bound(sweep.num_phase_bits, result.error_probability):.3f}")
            for sweep in sweeps
            for result in sweep.results
        ),
    )
    first_order_width = max(
        len(first_order_header),
        *(
            len(f"{kitaev_first_order_recovery_estimate(sweep.num_phase_bits, result.error_probability):.3f}")
            for sweep in sweeps
            for result in sweep.results
        ),
    )
    coefficient_width = max(
        len(coefficient_header),
        *(
            len(f"{kitaev_single_error_recoverability_coefficient(sweep.num_phase_bits):.3f}")
            for sweep in sweeps
        ),
    )
    success_width = max(
        len(success_header),
        *(len(str(result.successful_trials)) for result in all_results),
    )
    total_width = max(
        len(total_header),
        *(len(str(result.trials)) for result in all_results),
    )

    lines = [
        "Classical Recovery vs Error Probability Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        *(
            [f"Analytic window = {analytic_windows[0]}"]
            if analytic_windows
            else []
        ),
        instance_display_label(instances),
        f"{bit_label.capitalize()} = {bit_count_value_label}",
        f"Trials per probability value = {trials}",
        "Probability grid = "
        f"{'logarithmic' if log_p_grid else 'linear'}",
        "Propagating lower bound plotted = "
        f"{'yes' if show_lower_bound else 'no'}",
        "First-order estimate plotted = "
        f"{'yes' if show_first_order else 'no'}",
        f"Seed = {seed_label}",
        "",
        "Graph note:",
        "  Up to five fixed (N, a) instances",
        *(
            [
                "  Kitaev trials pair a clean iterative phase-estimation run with a noisy run",
                "  Kitaev clean/noisy runs share the same measurement RNG seed",
                "  Kitaev noisy runs apply independent per-detection measurement bit-flips with probability p",
            ]
            if KITAEV_ALGORITHM in algorithms
            else []
        ),
        *(
            [
                "  Standard trials pair a clean standard phase-estimation run with a noisy run",
                "  Standard clean/noisy runs share the same control-register measurement RNG seed",
                "  Standard noisy runs apply independent final control-bit flips with probability p",
                "  The standard analytic baseline is exact for this non-propagating bit-flip model",
            ]
            if STANDARD_ALGORITHM in algorithms
            else []
        ),
        "  Classical recovery checks the noisy bitstring and all single-bit-flip neighbors",
        *(
            [
                "  Optional lower bound counts no raw errors plus one error on the final measured bit",
                "  Optional first-order estimate uses analytic C_t for single raw Kitaev errors",
            ]
            if KITAEV_ALGORITHM in algorithms and STANDARD_ALGORITHM not in algorithms
            else []
        ),
        "  Probability values use a "
        f"{'logarithmic' if log_p_grid else 'linear'} sweep in p",
        "",
        "Per-probability data:",
        "  "
        f"{algorithm_header:>{algorithm_width}} | "
        f"{N_header:>{N_width}} | "
        f"{base_header:>{base_width}} | "
        f"{bits_header:>{bits_width}} | "
        f"{p_header:>{p_width}} | "
        f"{recovery_header:>{recovery_width}} | "
        f"{baseline_header:>{baseline_width}} | "
        + (
            f"{lower_bound_header:>{lower_bound_width}} | "
            if show_lower_bound
            else ""
        )
        + (
            f"{first_order_header:>{first_order_width}} | "
            f"{coefficient_header:>{coefficient_width}} | "
            if show_first_order
            else ""
        )
        + f"{success_header:>{success_width}} | "
        f"{total_header:>{total_width}}",
    ]

    for sweep in sweeps:
        for result in sweep.results:
            lower_bound = propagating_error_lower_bound(
                sweep.num_phase_bits,
                result.error_probability,
            )
            first_order = kitaev_first_order_recovery_estimate(
                sweep.num_phase_bits,
                result.error_probability,
            )
            coefficient = kitaev_single_error_recoverability_coefficient(
                sweep.num_phase_bits
            )
            lines.append(
                "  "
                f"{algorithm_display_label(sweep.algorithm):>{algorithm_width}} | "
                f"{sweep.N:>{N_width}} | "
                f"{sweep.a:>{base_width}} | "
                f"{sweep.num_phase_bits:>{bits_width}} | "
                f"{result.error_probability:>{p_width}.3f} | "
                f"{result.recovery_probability:>{recovery_width}.3f} | "
                f"{result.naive_baseline_probability:>{baseline_width}.3f} | "
                + (
                    f"{lower_bound:>{lower_bound_width}.3f} | "
                    if show_lower_bound
                    else ""
                )
                + (
                    f"{first_order:>{first_order_width}.3f} | "
                    f"{coefficient:>{coefficient_width}.3f} | "
                    if show_first_order
                    else ""
                )
                + f"{result.successful_trials:>{success_width}} | "
                f"{result.trials:>{total_width}}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_recovery_vs_error_probability(
    trials: int,
    sweeps: list[InstanceRecoverySweep],
    log_p_grid: bool,
    show_lower_bound: bool,
    show_first_order: bool,
    output_path: Path,
) -> None:
    """
    Create and save the recovery-probability plot versus measurement error p.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [result.error_probability for result in sweeps[0].results]
    algorithms = normalize_algorithms([sweep.algorithm for sweep in sweeps])
    bit_label = bit_count_label_for_algorithms(algorithms)
    baseline_label = baseline_curve_label_for_algorithms(algorithms)
    if log_p_grid and min(x_values) > 0.0:
        baseline_x_values = np.geomspace(min(x_values), max(x_values), 400)
    else:
        baseline_x_values = np.linspace(min(x_values), max(x_values), 400)
    baseline_y_values = [
        {
            "x": baseline_x_values,
            "y": [
                classical_recovery_first_order_upper_bound(bit_count, float(value))
                for value in baseline_x_values
            ],
            "bit_count": bit_count,
        }
        for bit_count in sorted({sweep.num_phase_bits for sweep in sweeps})
    ]
    lower_bound_y_values = [
        {
            "x": baseline_x_values,
            "y": [
                propagating_error_lower_bound(bit_count, float(value))
                for value in baseline_x_values
            ],
            "bit_count": bit_count,
        }
        for bit_count in sorted({sweep.num_phase_bits for sweep in sweeps})
    ]
    first_order_y_values = [
        {
            "x": baseline_x_values,
            "y": [
                kitaev_first_order_recovery_estimate(bit_count, float(value))
                for value in baseline_x_values
            ],
            "bit_count": bit_count,
        }
        for bit_count in sorted({sweep.num_phase_bits for sweep in sweeps})
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    markers = ["o", "s", "^", "D", "P"]
    include_N = len({sweep.N for sweep in sweeps}) > 1
    include_bits = len({sweep.num_phase_bits for sweep in sweeps}) > 1
    include_algorithm = len(algorithms) > 1
    show_analytic_overlay = show_lower_bound or show_first_order
    for index, sweep in enumerate(sweeps):
        empirical_values = [result.recovery_probability for result in sweep.results]
        ax.plot(
            x_values,
            empirical_values,
            marker=markers[index % len(markers)],
            linestyle="None" if show_analytic_overlay else "-",
            linewidth=0.0 if show_analytic_overlay else 2.4,
            markersize=7,
            color=colors[index % len(colors)],
            label=curve_label(
                sweep,
                include_N=include_N,
                include_bits=include_bits,
                include_algorithm=include_algorithm,
            ),
        )

    baseline_styles = ["--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
    for index, baseline in enumerate(baseline_y_values):
        ax.plot(
            baseline["x"],
            baseline["y"],
            linestyle=baseline_styles[index],
            linewidth=1.8,
            color="#303030",
            alpha=0.85,
            label=(
                baseline_label
                if len(baseline_y_values) == 1
                else f"{baseline_label}, t = {baseline['bit_count']}"
            ),
        )

    if show_lower_bound:
        lower_bound_styles = [":", (0, (1, 1)), (0, (1, 2)), (0, (3, 1, 1, 1)), "-."]
        for index, lower_bound in enumerate(lower_bound_y_values):
            ax.plot(
                lower_bound["x"],
                lower_bound["y"],
                linestyle=lower_bound_styles[index],
                linewidth=1.8,
                color="#707070",
                alpha=0.9,
                label=(
                    "Propagating lower bound"
                    if len(lower_bound_y_values) == 1
                    else f"Lower bound, t = {lower_bound['bit_count']}"
                ),
            )

    if show_first_order:
        first_order_styles = ["-", "--", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
        for index, first_order in enumerate(first_order_y_values):
            ax.plot(
                first_order["x"],
                first_order["y"],
                linestyle=first_order_styles[index],
                linewidth=1.9,
                color="#505050",
                alpha=0.9,
                label=(
                    "First-order estimate"
                    if len(first_order_y_values) == 1
                    else f"First-order estimate, t = {first_order['bit_count']}"
                ),
            )

    ax.set_xlabel("Measurement bit-flip probability p")
    ax.set_ylabel("Recovery success probability")
    title_instance_label = (
        instance_display_label_with_bits(sweeps)
        if len({sweep.N for sweep in sweeps}) == 1
        else "selected (N, a) instances"
    )
    ax.set_title(
        "Single-Bit Classical Recovery vs Measurement Error Probability\n"
        f"{algorithms_display_label(algorithms)} for {title_instance_label}"
    )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlim(min(x_values) - 0.01, max(x_values) + 0.01)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    bit_counts = sorted({sweep.num_phase_bits for sweep in sweeps})
    bit_count_text = (
        f"{bit_label} = {bit_counts[0]}"
        if len(bit_counts) == 1
        else f"{bit_label} vary"
    )
    ax.text(
        0.98,
        0.02,
        f"{bit_count_text}\ntrials per p = {trials}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#D0D0D0"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the recovery-vs-p graph script.
    """
    parser = argparse.ArgumentParser(
        description="Plot classical single-bit recovery probability versus measurement error probability for fixed (N, a) instances."
    )
    parser.add_argument("N", type=int, nargs="+", help="One or more valid moduli to analyze")
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM],
        help="One or more phase-estimation algorithms to sample",
    )
    parser.add_argument(
        "--a",
        type=int,
        nargs="+",
        required=True,
        help=(
            f"One to {MAX_CURVES} fixed coprime bases a to analyze. "
            "With multiple N values, provide one shared base or one base per N."
        ),
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Override the number of phase-estimation bits/control qubits",
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=STATEVECTOR_STANDARD_BACKEND,
        help=(
            "Backend used for standard Shor curves. Use 'analytic' to sample "
            "measurement outcomes from the classically computed order without "
            "allocating the full standard-Shor state vector."
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
        default=100,
        help="Number of paired clean/noisy trials per probability value",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Minimum measurement bit-flip probability in the sweep",
    )
    parser.add_argument(
        "--max-p",
        type=float,
        default=1.0,
        help="Maximum measurement bit-flip probability in the sweep",
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
        "--show-lower-bound",
        action="store_true",
        help=(
            "Plot the guaranteed propagating-error lower bound "
            "(1-p)^t + p(1-p)^(t-1)."
        ),
    )
    parser.add_argument(
        "--show-first-order",
        action="store_true",
        help=(
            "Plot the first-order estimate "
            "(1-p)^t + C_t p(1-p)^(t-1)."
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
    algorithms = normalize_algorithms(args.algorithm)

    if algorithms != [KITAEV_ALGORITHM] and (args.show_lower_bound or args.show_first_order):
        parser.error(
            "--show-lower-bound and --show-first-order currently apply only "
            "to --algorithm kitaev"
        )

    if args.trials < 1:
        raise ValueError("trials must be >= 1")
    if args.analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    instances = build_instances(args.N, args.a)
    probability_values = build_probability_values(
        args.min_p,
        args.max_p,
        args.num_points,
        log_p_grid=args.log_p_grid,
    )

    if args.bits is not None and args.bits < 1:
        raise ValueError("bits must be >= 1")

    master_rng = np.random.default_rng(args.seed)
    sweeps = []
    for instance in instances:
        for algorithm in algorithms:
            num_phase_bits = (
                default_num_phase_bits(instance.N)
                if args.bits is None
                else args.bits
            )
            sweeps.append(
                simulate_for_instance(
                    N=instance.N,
                    a=instance.a,
                    num_phase_bits=num_phase_bits,
                    probability_values=probability_values,
                    trials=args.trials,
                    master_rng=master_rng,
                    algorithm=algorithm,
                    standard_backend=args.standard_backend,
                    analytic_window=args.analytic_window,
                )
            )

    suffix = instance_filename_suffix_with_bits(sweeps)
    suffix = (
        f"{algorithms_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, args.standard_backend)}"
        f"{suffix}"
    )
    if args.show_lower_bound:
        suffix = f"{suffix}_with_lower_bound"
    if args.show_first_order:
        suffix = f"{suffix}_with_first_order"

    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / f"classical_recovery_vs_error_probability_{suffix}.png"
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / f"classical_recovery_vs_error_probability_summary_{suffix}.txt"
    )

    plot_recovery_vs_error_probability(
        trials=args.trials,
        sweeps=sweeps,
        log_p_grid=args.log_p_grid,
        show_lower_bound=args.show_lower_bound,
        show_first_order=args.show_first_order,
        output_path=figure_output_path,
    )
    write_data_summary(
        output_path=data_output_path,
        trials=args.trials,
        log_p_grid=args.log_p_grid,
        show_lower_bound=args.show_lower_bound,
        show_first_order=args.show_first_order,
        seed=args.seed,
        sweeps=sweeps,
    )

    seed_label = "random" if args.seed is None else str(args.seed)

    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"Algorithm: {algorithms_display_label(algorithms)}")
    if STANDARD_ALGORITHM in algorithms:
        print(f"Standard backend: {args.standard_backend}")
        if args.standard_backend != STATEVECTOR_STANDARD_BACKEND:
            print(f"Analytic window: {args.analytic_window}")
    print(instance_display_label(instances))
    bit_counts = sorted({sweep.num_phase_bits for sweep in sweeps})
    bit_label = bit_count_label_for_algorithms(algorithms)
    if len(bit_counts) == 1:
        print(f"{bit_label.capitalize()} = {bit_counts[0]}")
    else:
        print(
            f"{bit_label.capitalize()} per instance: "
            + ", ".join(
                f"N={sweep.N}, a={sweep.a}: {sweep.num_phase_bits}"
                for sweep in sweeps
            )
        )
    print(f"Trials per probability value: {args.trials}")
    print(
        "Propagating lower bound plotted: "
        f"{'yes' if args.show_lower_bound else 'no'}"
    )
    print(
        "First-order estimate plotted: "
        f"{'yes' if args.show_first_order else 'no'}"
    )
    print(
        "Probability grid: "
        f"{'logarithmic' if args.log_p_grid else 'linear'}"
    )
    print(
        f"Measurement bit-flip probability swept from {args.min_p:.3f} "
        f"to {args.max_p:.3f}"
    )
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
