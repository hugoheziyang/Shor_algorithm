from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path("/tmp/matplotlib-shor")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from shor_common.errors import (
    ErrorConfig,
    MEASUREMENT_BITFLIP_SINGLE_TOKEN,
    choose_measurement_bitflip_mask,
    format_measurement_bitflip_probability,
    measurement_bitflip_filename_suffix,
    measurement_bitflip_label,
    measurement_bitflip_title_suffix,
    parse_measurement_bitflip_error_arg,
)
from shor_common.modular_unitaries import register_size
from shor_common.shor_utils import (
    choose_random_a,
    default_num_phase_bits,
    perfect_power_decomposition,
    recover_order_with_mode,
)
from shor_one_qubit.shor import shor_order_finding_attempt as kitaev_order_attempt
from shor_standard.analytic_sampling import (
    QPEWindowMassDiagnostics,
    multiplicative_order,
    qpe_window_mass_diagnostics,
    sample_standard_phase_bits_from_order,
    validate_analytic_error_config,
)
from shor_standard.backends import STANDARD_BACKEND_CHOICES, STATEVECTOR_STANDARD_BACKEND
from shor_standard.shor import shor_order_finding_attempt as standard_order_attempt


KITAEV_ALGORITHM = "kitaev"
STANDARD_ALGORITHM = "standard"
ALGORITHM_CHOICES = (KITAEV_ALGORITHM, STANDARD_ALGORITHM)
DEFAULT_NUM_PHASE_BITS = 11
DEFAULT_MAX_STATEVECTOR_AMPLITUDES = 8_000_000
STANDARD_COLOR_LIGHTENING = 0.58


@dataclass(frozen=True)
class OrderRecoveryInstance:
    N: int
    a: int
    true_order: int
    num_work_qubits: int
    default_phase_bits: int


@dataclass
class OrderRecoveryResult:
    instance: OrderRecoveryInstance
    algorithm: str
    successful_trials: int
    trials: int
    num_phase_bits: int
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND
    measurement_error_label: str = "disabled"
    error_probability: float | None = None

    @property
    def success_rate(self) -> float:
        return self.successful_trials / self.trials


@dataclass
class OrderBinSummary:
    true_order: int
    algorithm: str
    successful_trials: int
    trials: int
    num_instances: int
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND
    measurement_error_label: str = "disabled"
    error_probability: float | None = None

    @property
    def success_rate(self) -> float:
        return self.successful_trials / self.trials

    @property
    def standard_error(self) -> float:
        p = self.success_rate
        return math.sqrt((p * (1.0 - p)) / self.trials)


def lighten_color(color: str, amount: float) -> tuple[float, float, float]:
    """
    Return a lighter RGB shade by mixing a color with white.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must satisfy 0 <= amount <= 1")

    red, green, blue = to_rgb(color)
    return (
        red + (1.0 - red) * amount,
        green + (1.0 - green) * amount,
        blue + (1.0 - blue) * amount,
    )


def algorithm_color_for_base(base_color: str, algorithm: str) -> str | tuple[float, float, float]:
    """
    Return the plotted curve color for an algorithm at one probability color.
    """
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
    measurement_title: str = "Measurement p",
    extra_text_sections: list[tuple[str, list[str]]] | None = None,
    measurement_text_labels: list[str] | None = None,
    trials_title: str = "Trials per instance",
) -> None:
    """
    Add one grouped right-margin legend for algorithm, a colored category, and
    trial count.
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


def algorithm_display_label(algorithm: str) -> str:
    """
    Return a human-readable label for one algorithm.
    """
    if algorithm == KITAEV_ALGORITHM:
        return "Kitaev"
    if algorithm == STANDARD_ALGORITHM:
        return "standard Shor"
    raise ValueError("unknown algorithm")


def algorithms_display_label(algorithms: list[str]) -> str:
    """
    Return a human-readable label for one or more algorithms.
    """
    algorithms = normalize_algorithms(algorithms)
    if len(algorithms) == 1:
        return algorithm_display_label(algorithms[0])

    return " vs ".join(algorithm_display_label(algorithm) for algorithm in algorithms)


def algorithm_filename_prefix(algorithms: list[str]) -> str:
    """
    Return a filename segment for the selected algorithms.
    """
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
    """
    Return an extra filename segment for non-default standard backends.
    """
    algorithms = normalize_algorithms(algorithms)
    if STANDARD_ALGORITHM not in algorithms:
        return ""
    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        return ""
    return f"{standard_backend}_"


def validate_error_probabilities(probabilities: list[float]) -> list[float]:
    """
    Validate and de-duplicate measurement bit-flip probabilities.
    """
    if not probabilities:
        raise ValueError("at least one error probability is required")

    normalized = []
    for probability in probabilities:
        p = float(probability)
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError("error probabilities must satisfy 0 <= p <= 1")
        if p not in normalized:
            normalized.append(p)

    return normalized


def error_probability_display_label(probability: float) -> str:
    """
    Return a compact label for one measurement bit-flip probability.
    """
    return f"p={format_measurement_bitflip_probability(probability)}"


def error_probabilities_filename_segment(probabilities: list[float]) -> str:
    """
    Return a filename segment for selected measurement bit-flip probabilities.
    """
    probabilities = validate_error_probabilities(probabilities)
    tokens = [
        format_measurement_bitflip_probability(probability).replace("-", "m")
        for probability in probabilities
    ]
    return "p_" + "_".join(tokens) + "_"


def error_config_for_probability(probability: float) -> ErrorConfig:
    """
    Return an ErrorConfig for independent per-bit measurement flips.
    """
    [probability] = validate_error_probabilities([probability])
    if probability == 0.0:
        return ErrorConfig()

    return ErrorConfig(
        measurement_bitflip=True,
        measurement_bitflip_probability=probability,
    )


def measurement_error_sort_key(label: str, probability: float | None) -> tuple[float, str]:
    """
    Sort measurement-error labels by probability when available.
    """
    if probability is None:
        return (-1.0, label)
    return (probability, label)


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


def max_modulus_for_phase_bits(num_phase_bits: int) -> int:
    """
    Return the largest N whose default 2L+1 phase-bit count fits in t bits.
    """
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    max_work_qubits = (num_phase_bits - 1) // 2
    return 2 ** max_work_qubits


def validate_modulus_for_fixed_bits(N: int, num_phase_bits: int) -> None:
    """
    Require N to be an odd non-perfect-power composite fitting the fixed t.
    """
    if not is_odd_composite(N):
        raise ValueError("N must be an odd composite integer >= 9")
    if perfect_power_decomposition(N) is not None:
        raise ValueError("perfect-power moduli are excluded")
    if default_num_phase_bits(N) > num_phase_bits:
        raise ValueError("N requires more default phase bits than the fixed t")


def validate_instance_order_filter(
    min_order: int | None,
    max_order: int | None,
) -> None:
    """
    Validate optional true-order filters.
    """
    if min_order is not None and min_order < 1:
        raise ValueError("min_order must be >= 1")
    if max_order is not None and max_order < 1:
        raise ValueError("max_order must be >= 1")
    if (
        min_order is not None
        and max_order is not None
        and max_order < min_order
    ):
        raise ValueError("max_order must be >= min_order")


def parse_positive_json_int(raw_value, field_name: str, index: int) -> int:
    """
    Parse a JSON integer field while rejecting booleans.
    """
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(
            f"instances[{index}].{field_name} must be an integer"
        )
    return int(raw_value)


def safe_filename_token(raw_text: str) -> str:
    """
    Return a conservative filename token.
    """
    token = "".join(
        character if character.isalnum() else "_"
        for character in raw_text.lower()
    ).strip("_")
    return token or "instances"


def instances_file_filename_segment(instances_file: Path | None) -> str:
    """
    Return a filename segment for configured instance files.
    """
    if instances_file is None:
        return ""
    return f"instances_{safe_filename_token(instances_file.stem)}_"


def instance_from_pair(N: int, a: int, num_phase_bits: int) -> OrderRecoveryInstance:
    """
    Validate one configured (N, a) pair and compute its true order.
    """
    validate_modulus_for_fixed_bits(N, num_phase_bits)
    if not (2 <= a <= N - 2):
        raise ValueError(f"a must satisfy 2 <= a <= N - 2 for N={N}")
    if math.gcd(a, N) != 1:
        raise ValueError(f"a={a} must be coprime to N={N}")

    return OrderRecoveryInstance(
        N=N,
        a=a,
        true_order=multiplicative_order(a, N),
        num_work_qubits=register_size(N),
        default_phase_bits=default_num_phase_bits(N),
    )


def load_order_recovery_instances_from_file(
    instances_file: Path,
    num_phase_bits: int,
    min_order: int | None = None,
    max_order: int | None = None,
) -> list[OrderRecoveryInstance]:
    """
    Load configured representative (N, a) pairs from a JSON file.

    The file may be either a JSON list of objects or an object with an
    "instances" list. Each instance object must contain integer "N" and "a"
    fields. The true order is computed here and must be unique in file mode.
    """
    validate_instance_order_filter(min_order, max_order)

    try:
        payload = json.loads(instances_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read instances file: {instances_file}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in instances file: {instances_file}") from exc

    if isinstance(payload, dict):
        raw_instances = payload.get("instances")
    else:
        raw_instances = payload

    if not isinstance(raw_instances, list):
        raise ValueError("instances file must contain a JSON list or an 'instances' list")

    instances = []
    seen_orders: dict[int, OrderRecoveryInstance] = {}
    for index, raw_instance in enumerate(raw_instances):
        if not isinstance(raw_instance, dict):
            raise ValueError(f"instances[{index}] must be an object")
        if "N" not in raw_instance or "a" not in raw_instance:
            raise ValueError(f"instances[{index}] must contain N and a")

        N = parse_positive_json_int(raw_instance["N"], "N", index)
        a = parse_positive_json_int(raw_instance["a"], "a", index)
        expected_order = raw_instance.get("r", raw_instance.get("order"))
        if expected_order is not None:
            expected_order = parse_positive_json_int(expected_order, "r", index)
            if min_order is not None and expected_order < min_order:
                continue
            if max_order is not None and expected_order > max_order:
                continue

        instance = instance_from_pair(N=N, a=a, num_phase_bits=num_phase_bits)
        if expected_order is not None:
            if expected_order != instance.true_order:
                raise ValueError(
                    f"instances[{index}] expected r={expected_order}, "
                    f"but computed r={instance.true_order} for N={N}, a={a}"
                )

        if min_order is not None and instance.true_order < min_order:
            continue
        if max_order is not None and instance.true_order > max_order:
            continue

        previous_instance = seen_orders.get(instance.true_order)
        if previous_instance is not None:
            raise ValueError(
                "instances file must contain at most one representative per "
                f"true order r; r={instance.true_order} appears for "
                f"(N={previous_instance.N}, a={previous_instance.a}) and "
                f"(N={instance.N}, a={instance.a})"
            )

        seen_orders[instance.true_order] = instance
        instances.append(instance)

    if not instances:
        raise ValueError("instances file did not provide any usable instances")

    return sorted(instances, key=lambda instance: instance.true_order)


def valid_moduli_for_fixed_bits(
    num_phase_bits: int,
    min_N: int,
    max_N: int | None = None,
) -> list[int]:
    """
    Return eligible odd composite moduli with 2L+1 <= num_phase_bits.
    """
    if min_N < 2:
        raise ValueError("min_N must be >= 2")

    allowed_max_N = max_modulus_for_phase_bits(num_phase_bits)
    if max_N is None:
        max_N = allowed_max_N
    elif max_N > allowed_max_N:
        raise ValueError(
            f"max_N={max_N} exceeds the largest modulus allowed by "
            f"--bits {num_phase_bits}: {allowed_max_N}"
        )

    if max_N < min_N:
        raise ValueError("max_N must be >= min_N")

    return [
        N
        for N in range(min_N, max_N + 1, 2)
        if is_odd_composite(N)
        and perfect_power_decomposition(N) is None
        and default_num_phase_bits(N) <= num_phase_bits
    ]


def sample_order_recovery_instances(
    num_instances: int,
    num_phase_bits: int,
    rng: np.random.Generator,
    min_N: int = 9,
    max_N: int | None = None,
    min_order: int | None = None,
    max_order: int | None = None,
) -> list[OrderRecoveryInstance]:
    """
    Sample random coprime (N, a) instances and compute their true orders.
    """
    if num_instances < 1:
        raise ValueError("num_instances must be >= 1")
    if min_order is not None and min_order < 1:
        raise ValueError("min_order must be >= 1")
    if max_order is not None and max_order < 1:
        raise ValueError("max_order must be >= 1")
    validate_instance_order_filter(min_order, max_order)

    candidate_moduli = valid_moduli_for_fixed_bits(
        num_phase_bits=num_phase_bits,
        min_N=min_N,
        max_N=max_N,
    )
    if not candidate_moduli:
        raise ValueError(
            "no valid odd composite non-perfect-power moduli fit the requested "
            "N range and fixed phase-bit count"
        )

    instances: list[OrderRecoveryInstance] = []
    max_sampling_attempts = max(1000, 100 * num_instances)

    for _ in range(max_sampling_attempts):
        if len(instances) >= num_instances:
            break

        N = int(rng.choice(candidate_moduli))
        a = choose_random_a(N, rng=rng)
        true_order = multiplicative_order(a, N)

        if min_order is not None and true_order < min_order:
            continue
        if max_order is not None and true_order > max_order:
            continue

        instances.append(
            OrderRecoveryInstance(
                N=N,
                a=a,
                true_order=true_order,
                num_work_qubits=register_size(N),
                default_phase_bits=default_num_phase_bits(N),
            )
        )

    if len(instances) < num_instances:
        raise ValueError(
            "could not sample enough instances with the requested order filters; "
            "try increasing --max-order, lowering --min-order, or increasing "
            "the N range"
        )

    return instances


def simulate_order_recovery_for_instance(
    instance: OrderRecoveryInstance,
    algorithm: str,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    error_config: ErrorConfig | None = None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
    measurement_error_label: str | None = None,
    error_probability: float | None = None,
) -> OrderRecoveryResult:
    """
    Estimate one-shot probability of recovering the true order for one instance.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    successful_trials = 0

    for _ in range(trials):
        if algorithm == KITAEV_ALGORITHM:
            result = kitaev_order_attempt(
                N=instance.N,
                a=instance.a,
                num_phase_bits=num_phase_bits,
                rng=rng,
                postprocess_mode="one_shot",
                error_config=error_config,
            )
            recovered_order = result["order"]
        elif algorithm == STANDARD_ALGORITHM:
            if standard_backend == STATEVECTOR_STANDARD_BACKEND:
                result = standard_order_attempt(
                    N=instance.N,
                    a=instance.a,
                    num_control_qubits=num_phase_bits,
                    rng=rng,
                    postprocess_mode="one_shot",
                    error_config=error_config,
                    standard_backend=standard_backend,
                    analytic_window=analytic_window,
                )
                recovered_order = result["order"]
            else:
                analytic_error_config = validate_analytic_error_config(error_config)
                measurement_bitflip_mask = choose_measurement_bitflip_mask(
                    num_phase_bits,
                    error_config=analytic_error_config,
                    rng=rng,
                )
                bits, _, _, _ = sample_standard_phase_bits_from_order(
                    order=instance.true_order,
                    num_control_qubits=num_phase_bits,
                    rng=rng,
                    window=analytic_window,
                    measurement_bitflip_mask=measurement_bitflip_mask,
                )
                recovered_order = recover_order_with_mode(
                    bits,
                    instance.a,
                    instance.N,
                    postprocess_mode="one_shot",
                )
        else:
            raise ValueError("unknown algorithm")

        if recovered_order == instance.true_order:
            successful_trials += 1

    return OrderRecoveryResult(
        instance=instance,
        algorithm=algorithm,
        successful_trials=successful_trials,
        trials=trials,
        num_phase_bits=num_phase_bits,
        standard_backend=standard_backend,
        measurement_error_label=(
            measurement_error_label
            if measurement_error_label is not None
            else measurement_bitflip_label(error_config)
        ),
        error_probability=error_probability,
    )


def aggregate_results_by_order(
    results: list[OrderRecoveryResult],
) -> list[OrderBinSummary]:
    """
    Group instance-level order-recovery results by true order and algorithm.
    """
    grouped: dict[tuple[str, str, int], list[OrderRecoveryResult]] = defaultdict(list)
    for result in results:
        grouped[
            (
                result.algorithm,
                result.measurement_error_label,
                result.instance.true_order,
            )
        ].append(result)

    summaries = []
    for (algorithm, measurement_error_label, true_order), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            measurement_error_sort_key(item[0][1], item[1][0].error_probability),
            item[0][2],
        ),
    ):
        summaries.append(
            OrderBinSummary(
                true_order=true_order,
                algorithm=algorithm,
                successful_trials=sum(result.successful_trials for result in group),
                trials=sum(result.trials for result in group),
                num_instances=len(group),
                standard_backend=group[0].standard_backend,
                measurement_error_label=measurement_error_label,
                error_probability=group[0].error_probability,
            )
        )

    return summaries


def validate_statevector_budget(
    algorithms: list[str],
    standard_backend: str,
    num_phase_bits: int,
    max_N: int,
    max_statevector_amplitudes: int,
) -> None:
    """
    Guard against accidental huge standard statevector allocations.
    """
    if STANDARD_ALGORITHM not in algorithms:
        return
    if standard_backend != STATEVECTOR_STANDARD_BACKEND:
        return
    if max_statevector_amplitudes <= 0:
        return

    max_amplitudes = (2 ** num_phase_bits) * (2 ** register_size(max_N))
    if max_amplitudes > max_statevector_amplitudes:
        raise ValueError(
            "standard statevector backend could allocate up to "
            f"{max_amplitudes:,} complex amplitudes for this sweep. Use "
            "--standard-backend analytic, lower --bits/--max-N, or raise "
            "--max-statevector-amplitudes."
        )


def analytic_window_mass_diagnostics_for_instances(
    instances: list[OrderRecoveryInstance],
    num_phase_bits: int,
    analytic_window: int,
) -> list[QPEWindowMassDiagnostics]:
    """
    Compute captured-mass diagnostics for each unique true order in a sweep.
    """
    true_orders = sorted({instance.true_order for instance in instances})
    return [
        qpe_window_mass_diagnostics(
            order=true_order,
            num_control_qubits=num_phase_bits,
            window=analytic_window,
        )
        for true_order in true_orders
    ]


def analytic_window_diagnostics_summary_lines(
    diagnostics: list[QPEWindowMassDiagnostics] | None,
) -> list[str]:
    """
    Format analytic-window captured-mass diagnostics for data summaries.
    """
    if not diagnostics:
        return []

    worst = min(diagnostics, key=lambda diagnostic: diagnostic.min_mass)
    mean_of_order_means = sum(
        diagnostic.mean_mass for diagnostic in diagnostics
    ) / len(diagnostics)

    lines = [
        "Analytic window captured-mass diagnostics:",
        f"  Window = {worst.window}",
        f"  Orders checked = {len(diagnostics)}",
        (
            "  Worst captured mass = "
            f"{worst.min_mass:.12f} "
            f"(r={worst.order}, s={worst.worst_eigenphase_numerator})"
        ),
        f"  Mean captured mass across order means = {mean_of_order_means:.12f}",
        "  Per-order captured mass:",
        "    r | min mass | mean mass | max mass | worst s | best s",
    ]
    for diagnostic in diagnostics:
        lines.append(
            f"    {diagnostic.order:>3} |"
            f" {diagnostic.min_mass:>8.6f} |"
            f" {diagnostic.mean_mass:>9.6f} |"
            f" {diagnostic.max_mass:>8.6f} |"
            f" {diagnostic.worst_eigenphase_numerator:>7} |"
            f" {diagnostic.best_eigenphase_numerator:>6}"
        )

    return lines


def analytic_window_diagnostics_console_label(
    diagnostics: list[QPEWindowMassDiagnostics] | None,
) -> str:
    """
    Return a compact one-line captured-mass diagnostic.
    """
    if not diagnostics:
        return "not computed"

    worst = min(diagnostics, key=lambda diagnostic: diagnostic.min_mass)
    return (
        f"min={worst.min_mass:.6f} "
        f"(r={worst.order}, s={worst.worst_eigenphase_numerator}), "
        f"window={worst.window}"
    )


def write_data_summary(
    output_path: Path,
    results: list[OrderRecoveryResult],
    order_summaries: list[OrderBinSummary],
    instance_source: str,
    num_phase_bits: int,
    num_instances: int,
    trials_per_instance: int,
    min_N: int,
    max_N: int,
    min_order: int | None,
    max_order: int | None,
    error_config: ErrorConfig | None,
    seed: int | None,
    measurement_error_summary: str | None = None,
    analytic_window_diagnostics: list[QPEWindowMassDiagnostics] | None = None,
) -> None:
    """
    Write run metadata plus per-order and per-instance recovery data.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    algorithms = normalize_algorithms([result.algorithm for result in results])
    standard_backends = sorted(
        {
            result.standard_backend
            for result in results
            if result.algorithm == STANDARD_ALGORITHM
        }
    )
    seed_label = "random" if seed is None else str(seed)
    error_label = measurement_error_summary or measurement_bitflip_label(error_config)
    min_order_label = "none" if min_order is None else str(min_order)
    max_order_label = "none" if max_order is None else str(max_order)

    lines = [
        "Order-Recovery Success-vs-Order Data Summary",
        f"Algorithm = {algorithms_display_label(algorithms)}",
        *(
            [f"Standard backend = {standard_backends[0]}"]
            if standard_backends
            else []
        ),
        f"Instance source = {instance_source}",
        f"Fixed phase-estimation bits = {num_phase_bits}",
        f"Instances used = {num_instances}",
        f"Trials per instance and algorithm = {trials_per_instance}",
        f"N range = {min_N} to {max_N}",
        f"Order filter = {min_order_label} to {max_order_label}",
        f"Measurement bit-flip error setting = {error_label}",
        f"Seed = {seed_label}",
        "",
        *analytic_window_diagnostics_summary_lines(analytic_window_diagnostics),
        *(
            [""]
            if analytic_window_diagnostics
            else []
        ),
        "Graph note:",
        "  Each sampled N is an odd composite non-perfect power with 2L+1 <= t",
        "  Each sampled base a is coprime to N",
        "  True orders are computed classically for diagnostic binning",
        "  Quantum sampling uses one_shot continued-fraction post-processing",
        "  Success means the recovered order equals the true order r exactly",
        "",
        "Per-order data:",
        "  algorithm | measurement error | true r | success rate | successful trials | total trials | sampled instances",
    ]

    for summary in order_summaries:
        lines.append(
            f"  {algorithm_display_label(summary.algorithm):>13} |"
            f" {summary.measurement_error_label:>17} |"
            f" {summary.true_order:>6} |"
            f" {summary.success_rate:>12.3f} |"
            f" {summary.successful_trials:>17} |"
            f" {summary.trials:>12} |"
            f" {summary.num_instances:>17}"
        )

    lines.extend([
        "",
        "Per-instance data:",
        "  algorithm | measurement error | N | a | L | default t | true r | success rate | successful trials | total trials",
    ])

    sorted_results = sorted(
        results,
        key=lambda result: (
            result.algorithm,
            measurement_error_sort_key(
                result.measurement_error_label,
                result.error_probability,
            ),
            result.instance.true_order,
            result.instance.N,
            result.instance.a,
        ),
    )
    for result in sorted_results:
        instance = result.instance
        lines.append(
            f"  {algorithm_display_label(result.algorithm):>13} |"
            f" {result.measurement_error_label:>17} |"
            f" {instance.N:>5} |"
            f" {instance.a:>5} |"
            f" {instance.num_work_qubits:>2} |"
            f" {instance.default_phase_bits:>9} |"
            f" {instance.true_order:>6} |"
            f" {result.success_rate:>12.3f} |"
            f" {result.successful_trials:>17} |"
            f" {result.trials:>12}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_order_recovery_success(
    output_path: Path,
    order_summaries: list[OrderBinSummary],
    algorithms: list[str],
    num_phase_bits: int,
    trials_per_instance: int,
    error_config: ErrorConfig | None,
    measurement_error_summary: str | None = None,
    log_x: bool = False,
) -> None:
    """
    Create and save the success-probability plot versus true order r.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = normalize_algorithms(algorithms)

    measurement_error_labels = []
    for summary in order_summaries:
        if summary.measurement_error_label not in measurement_error_labels:
            measurement_error_labels.append(summary.measurement_error_label)
    measurement_error_labels.sort(
        key=lambda label: measurement_error_sort_key(
            label,
            next(
                summary.error_probability
                for summary in order_summaries
                if summary.measurement_error_label == label
            ),
        )
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }
    algorithm_linestyles = {
        KITAEV_ALGORITHM: "-",
        STANDARD_ALGORITHM: "--",
    }
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    measurement_error_colors = {
        label: color_cycle[index % len(color_cycle)]
        for index, label in enumerate(measurement_error_labels)
    }

    for measurement_error_label in measurement_error_labels:
        for algorithm in algorithms:
            summaries = sorted(
                [
                    summary
                    for summary in order_summaries
                    if summary.algorithm == algorithm
                    and summary.measurement_error_label == measurement_error_label
                ],
                key=lambda summary: summary.true_order,
            )
            x_values = [summary.true_order for summary in summaries]
            y_values = [summary.success_rate for summary in summaries]
            ax.plot(
                x_values,
                y_values,
                marker=markers[algorithm],
                linestyle=algorithm_linestyles[algorithm],
                linewidth=1.8,
                markersize=6,
                color=algorithm_color_for_base(
                    measurement_error_colors[measurement_error_label],
                    algorithm,
                ),
                alpha=0.9,
            )

    ax.set_xlabel("True multiplicative order r")
    ax.set_ylabel("Empirical probability of recovering the true order")
    title_suffix = (
        measurement_bitflip_title_suffix(error_config)
        if len(measurement_error_labels) == 1
        else ""
    )
    ax.set_title(
        "One-Shot Order Recovery Success vs True Order\n"
        f"{algorithms_display_label(algorithms)}, t = "
        f"{num_phase_bits}{title_suffix}"
    )
    if log_x:
        ax.set_xscale("log", base=2)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)

    algorithm_handles = [
        Line2D(
            [0],
            [0],
            color=algorithm_color_for_base("#333333", algorithm),
            marker=markers[algorithm],
            linestyle=algorithm_linestyles[algorithm],
            linewidth=1.8,
            markersize=6,
            label=algorithm_display_label(algorithm),
        )
        for algorithm in algorithms
    ]
    measurement_handles = [
        Line2D(
            [0],
            [0],
            color=measurement_error_colors[label],
            linestyle="-",
            linewidth=2.0,
            label=label,
        )
        for label in measurement_error_labels
    ]

    fig.tight_layout(rect=[0.0, 0.0, 0.72, 1.0])
    add_combined_figure_legend(
        fig=fig,
        algorithm_handles=algorithm_handles,
        measurement_handles=measurement_handles,
        trials_per_instance=trials_per_instance,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for the order-recovery graph script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot one-shot probability of recovering the true order r versus "
            "r for randomly sampled Shor order-finding instances."
        )
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=[KITAEV_ALGORITHM],
        help="One or more phase-estimation algorithms to sample",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=DEFAULT_NUM_PHASE_BITS,
        help=(
            "Fixed number t of phase-estimation bits/control qubits. Generated "
            "N values must satisfy 2 * ceil(log2(N)) + 1 <= t."
        ),
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=50,
        help="Number of random coprime (N, a) instances to sample when --instances-file is omitted",
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing configured representative (N, a) "
            "pairs. The script computes r for each pair and requires one pair "
            "per true order."
        ),
    )
    parser.add_argument(
        "--trials-per-instance",
        type=int,
        default=100,
        help="Number of one-shot order-finding trials per instance and algorithm",
    )
    parser.add_argument(
        "--min-N",
        type=int,
        default=9,
        help="Smallest candidate modulus N",
    )
    parser.add_argument(
        "--max-N",
        type=int,
        default=None,
        help="Largest candidate modulus N. Defaults to the largest N fitting --bits.",
    )
    parser.add_argument(
        "--min-order",
        type=int,
        default=None,
        help="Optional minimum true order r to keep",
    )
    parser.add_argument(
        "--max-order",
        type=int,
        default=None,
        help="Optional maximum true order r to keep",
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
        "--max-statevector-amplitudes",
        type=int,
        default=DEFAULT_MAX_STATEVECTOR_AMPLITUDES,
        help=(
            "Safety limit for standard statevector backend allocations. Use 0 "
            "to disable the guard."
        ),
    )
    parser.add_argument(
        "--measurement-bitflip-error",
        nargs="?",
        const=MEASUREMENT_BITFLIP_SINGLE_TOKEN,
        default=None,
        metavar="P",
        help=(
            "Enable measurement bit-flip errors. Use without a value for "
            "exactly one random bit-flip per phase-estimation run, or provide "
            "P in [0,1] for independent per-detection bit-flips."
        ),
    )
    parser.add_argument(
        "--error-probabilities",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Overlay independent per-bit measurement flip probabilities. "
            "Include 0 for the clean curve. This is mutually exclusive with "
            "--measurement-bitflip-error."
        ),
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a base-2 logarithmic x-axis for the true order r",
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

    if (
        args.error_probabilities is not None
        and args.measurement_bitflip_error is not None
    ):
        parser.error(
            "--error-probabilities and --measurement-bitflip-error are mutually exclusive"
        )

    try:
        if args.error_probabilities is None:
            error_config = parse_measurement_bitflip_error_arg(
                args.measurement_bitflip_error
            )
            error_settings = [
                (
                    measurement_bitflip_label(error_config),
                    (
                        error_config.measurement_bitflip_probability
                        if error_config.measurement_bitflip_probability is not None
                        else None
                    ),
                    error_config,
                )
            ]
            measurement_error_summary = measurement_bitflip_label(error_config)
        else:
            error_probabilities = validate_error_probabilities(
                args.error_probabilities
            )
            error_settings = [
                (
                    error_probability_display_label(probability),
                    probability,
                    error_config_for_probability(probability),
                )
                for probability in error_probabilities
            ]
            measurement_error_summary = ", ".join(
                error_probability_display_label(probability)
                for probability in error_probabilities
            )
            error_config = ErrorConfig()
    except ValueError as exc:
        parser.error(str(exc))

    if args.bits < 1:
        raise ValueError("bits must be >= 1")
    if args.num_instances < 1:
        raise ValueError("num_instances must be >= 1")
    if args.trials_per_instance < 1:
        raise ValueError("trials_per_instance must be >= 1")
    if args.analytic_window < 0:
        raise ValueError("analytic_window must be >= 0")

    master_rng = np.random.default_rng(args.seed)
    try:
        if args.instances_file is None:
            allowed_max_N = max_modulus_for_phase_bits(args.bits)
            max_N = allowed_max_N if args.max_N is None else args.max_N
            validate_statevector_budget(
                algorithms=algorithms,
                standard_backend=args.standard_backend,
                num_phase_bits=args.bits,
                max_N=max_N,
                max_statevector_amplitudes=args.max_statevector_amplitudes,
            )
            instances = sample_order_recovery_instances(
                num_instances=args.num_instances,
                num_phase_bits=args.bits,
                rng=master_rng,
                min_N=args.min_N,
                max_N=args.max_N,
                min_order=args.min_order,
                max_order=args.max_order,
            )
            min_N = args.min_N
            instance_source = "random sampling"
        else:
            instances = load_order_recovery_instances_from_file(
                instances_file=args.instances_file,
                num_phase_bits=args.bits,
                min_order=args.min_order,
                max_order=args.max_order,
            )
            min_N = min(instance.N for instance in instances)
            max_N = max(instance.N for instance in instances)
            validate_statevector_budget(
                algorithms=algorithms,
                standard_backend=args.standard_backend,
                num_phase_bits=args.bits,
                max_N=max_N,
                max_statevector_amplitudes=args.max_statevector_amplitudes,
            )
            instance_source = f"configured file: {args.instances_file}"
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

    results: list[OrderRecoveryResult] = []
    for measurement_error_label, error_probability, current_error_config in error_settings:
        for algorithm in algorithms:
            for instance in instances:
                instance_seed = int(master_rng.integers(0, 2 ** 63 - 1))
                instance_rng = np.random.default_rng(instance_seed)
                results.append(
                    simulate_order_recovery_for_instance(
                        instance=instance,
                        algorithm=algorithm,
                        trials=args.trials_per_instance,
                        rng=instance_rng,
                        num_phase_bits=args.bits,
                        error_config=current_error_config,
                        standard_backend=args.standard_backend,
                        analytic_window=args.analytic_window,
                        measurement_error_label=measurement_error_label,
                        error_probability=error_probability,
                    )
                )

    order_summaries = aggregate_results_by_order(results)

    if args.error_probabilities is None:
        filename_suffix = measurement_bitflip_filename_suffix(error_config)
    else:
        filename_suffix = "_" + error_probabilities_filename_segment(
            error_probabilities
        ).rstrip("_")
    algorithm_prefix = (
        f"{algorithm_filename_prefix(algorithms)}"
        f"{standard_backend_filename_segment(algorithms, args.standard_backend)}"
        f"{instances_file_filename_segment(args.instances_file)}"
    )
    if args.output is None:
        figure_output_path = (
            Path(__file__).resolve().parent
            / "figures"
            / (
                "order_recovery_success_vs_order_"
                f"{algorithm_prefix}t{args.bits}{filename_suffix}.png"
            )
        )
    else:
        figure_output_path = args.output

    data_output_path = (
        Path(__file__).resolve().parent
        / "data"
        / (
            "order_recovery_success_vs_order_summary_"
            f"{algorithm_prefix}t{args.bits}{filename_suffix}.txt"
        )
    )

    plot_order_recovery_success(
        output_path=figure_output_path,
        order_summaries=order_summaries,
        algorithms=algorithms,
        num_phase_bits=args.bits,
        trials_per_instance=args.trials_per_instance,
        error_config=error_config,
        measurement_error_summary=measurement_error_summary,
        log_x=args.log_x,
    )
    write_data_summary(
        output_path=data_output_path,
        results=results,
        order_summaries=order_summaries,
        instance_source=instance_source,
        num_phase_bits=args.bits,
        num_instances=len(instances),
        trials_per_instance=args.trials_per_instance,
        min_N=min_N,
        max_N=max_N,
        min_order=args.min_order,
        max_order=args.max_order,
        error_config=error_config,
        seed=args.seed,
        measurement_error_summary=measurement_error_summary,
        analytic_window_diagnostics=analytic_window_diagnostics,
    )

    seed_label = "random" if args.seed is None else str(args.seed)
    print(f"Saved figure to: {figure_output_path}")
    print(f"Saved data summary to: {data_output_path}")
    print(f"Algorithm: {algorithms_display_label(algorithms)}")
    if STANDARD_ALGORITHM in algorithms:
        print(f"Standard backend: {args.standard_backend}")
        if args.standard_backend != STATEVECTOR_STANDARD_BACKEND:
            print(f"Analytic window: {args.analytic_window}")
            print(
                "Analytic window captured mass: "
                f"{analytic_window_diagnostics_console_label(analytic_window_diagnostics)}"
            )
    print(f"Fixed phase-estimation bits: {args.bits}")
    print(f"Instance source: {instance_source}")
    print(f"Instances used: {len(instances)}")
    print(f"Trials per instance and algorithm: {args.trials_per_instance}")
    print(f"N range: {min_N} to {max_N}")
    print(f"Measurement bit-flip error: {measurement_error_summary}")
    print(f"Seed: {seed_label}")
    print(f"Elapsed time: {time.perf_counter() - start_time:.2f} s")


if __name__ == "__main__":
    main()
