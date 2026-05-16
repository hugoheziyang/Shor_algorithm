from __future__ import annotations

import argparse
import json
import math
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


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
    error_config_for_model,
    error_model_display_label,
    normalize_error_model,
)
from shor_one_qubit.shor import shor_order_finding_attempt as kitaev_order_attempt
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STATEVECTOR_STANDARD_BACKEND,
    STANDARD_BACKEND_CHOICES,
)
from shor_standard.analytic_sampling import multiplicative_order
from shor_standard.shor import shor_order_finding_attempt as standard_order_attempt


KITAEV_ALGORITHM = "kitaev"
STANDARD_ALGORITHM = "standard"
ALGORITHM_CHOICES = (KITAEV_ALGORITHM, STANDARD_ALGORITHM)

DEFAULT_INSTANCES_FILE = REPO_ROOT / "graphs" / "config" / "order_recovery_instances.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "graphs" / "figures"
DEFAULT_DATA_DIR = REPO_ROOT / "graphs" / "data"
DEFAULT_ERROR_PROBABILITIES = [0.0, 0.01, 0.05, 0.1]


@dataclass(frozen=True)
class OrderRecoveryInstance:
    N: int
    a: int
    true_order: int


@dataclass(frozen=True)
class OrderRecoveryResult:
    instance: OrderRecoveryInstance
    algorithm: str
    error_model: str
    error_probability: float
    successful_trials: int
    trials: int
    num_phase_bits: int

    @property
    def success_probability(self) -> float:
        return self.successful_trials / self.trials


def normalize_algorithms(algorithms: list[str]) -> list[str]:
    if not algorithms:
        raise ValueError("at least one algorithm is required")
    normalized = []
    for algorithm in algorithms:
        label = algorithm.strip().lower()
        if label not in ALGORITHM_CHOICES:
            choices = ", ".join(ALGORITHM_CHOICES)
            raise ValueError(f"algorithm must be one of: {choices}")
        if label not in normalized:
            normalized.append(label)
    return normalized


def validate_error_probabilities(probabilities: list[float]) -> list[float]:
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


def probability_token(probability: float) -> str:
    mantissa, exponent = format(probability, ".0e").split("e")
    return f"{mantissa}e{int(exponent)}".replace("-", "m")


def load_instances(path: Path) -> list[OrderRecoveryInstance]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    raw_instances = payload["instances"] if isinstance(payload, dict) else payload
    if not isinstance(raw_instances, list):
        raise ValueError("instances file must contain a list of instances")

    instances = []
    seen_orders = set()
    for raw in raw_instances:
        N = int(raw["N"])
        a = int(raw["a"])
        true_order = multiplicative_order(a, N)

        expected_order = raw.get("r", raw.get("order"))
        if expected_order is not None and int(expected_order) != true_order:
            raise ValueError(
                f"instance (N={N}, a={a}) has configured order "
                f"{expected_order}, but computed order is {true_order}"
            )
        if true_order in seen_orders:
            raise ValueError("instances file must have at most one pair per order")
        seen_orders.add(true_order)
        instances.append(OrderRecoveryInstance(N=N, a=a, true_order=true_order))

    if not instances:
        raise ValueError("instances file must contain at least one instance")
    return sorted(instances, key=lambda instance: instance.true_order)


def simulate_order_recovery_for_instance(
    instance: OrderRecoveryInstance,
    algorithm: str,
    error_model: str,
    error_probability: float,
    trials: int,
    rng: np.random.Generator,
    num_phase_bits: int,
    standard_backend: str,
) -> OrderRecoveryResult:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    error_config = error_config_for_model(error_model, error_probability)
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
        elif algorithm == STANDARD_ALGORITHM:
            result = standard_order_attempt(
                N=instance.N,
                a=instance.a,
                num_control_qubits=num_phase_bits,
                rng=rng,
                postprocess_mode="one_shot",
                error_config=error_config,
                standard_backend=standard_backend,
            )
        else:
            raise ValueError("unknown algorithm")

        if result["order"] == instance.true_order:
            successful_trials += 1

    return OrderRecoveryResult(
        instance=instance,
        algorithm=algorithm,
        error_model=normalize_error_model(error_model),
        error_probability=float(error_probability),
        successful_trials=successful_trials,
        trials=trials,
        num_phase_bits=num_phase_bits,
    )


def simulate_order_recovery(
    instances: list[OrderRecoveryInstance],
    algorithms: list[str],
    error_model: str,
    error_probabilities: list[float],
    trials_per_instance: int,
    num_phase_bits: int,
    master_rng: np.random.Generator,
    standard_backend: str,
) -> list[OrderRecoveryResult]:
    results = []
    for probability in error_probabilities:
        for algorithm in algorithms:
            for instance in instances:
                rng = np.random.default_rng(int(master_rng.integers(0, 2 ** 63 - 1)))
                results.append(
                    simulate_order_recovery_for_instance(
                        instance=instance,
                        algorithm=algorithm,
                        error_model=error_model,
                        error_probability=probability,
                        trials=trials_per_instance,
                        rng=rng,
                        num_phase_bits=num_phase_bits,
                        standard_backend=standard_backend,
                    )
                )
    return results


def algorithm_label(algorithm: str) -> str:
    return "Kitaev" if algorithm == KITAEV_ALGORITHM else "Standard"


def plot_order_recovery_success(
    results: list[OrderRecoveryResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    probabilities = sorted({result.error_probability for result in results})
    algorithms = list(dict.fromkeys(result.algorithm for result in results))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(probabilities), 1)))
    color_for_probability = dict(zip(probabilities, colors))
    markers = {
        KITAEV_ALGORITHM: "o",
        STANDARD_ALGORITHM: "s",
    }

    for probability in probabilities:
        for algorithm in algorithms:
            selected = [
                result
                for result in results
                if result.error_probability == probability
                and result.algorithm == algorithm
            ]
            if not selected:
                continue
            selected.sort(key=lambda result: result.instance.true_order)
            orders = [result.instance.true_order for result in selected]
            success = [result.success_probability for result in selected]
            ax.plot(
                orders,
                success,
                marker=markers.get(algorithm, "o"),
                color=color_for_probability[probability],
                label=f"{algorithm_label(algorithm)}, p={probability:g}",
            )

    [first] = {result.error_model for result in results}
    ax.set_title(f"Order recovery under {error_model_display_label(first)}")
    ax.set_xlabel("True order r")
    ax.set_ylabel("One-shot order recovery probability")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary(results: list[OrderRecoveryResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Depolarizing error order-recovery success summary",
        f"Error model = {error_model_display_label(results[0].error_model)}",
        "",
        " N | a | r | algorithm | t | p | successes | trials | success_probability",
        "---|---|---|-----------|---|---|-----------|--------|--------------------",
    ]
    for result in sorted(
        results,
        key=lambda item: (
            item.error_probability,
            item.algorithm,
            item.instance.true_order,
        ),
    ):
        lines.append(
            f"{result.instance.N:>3} | {result.instance.a:>3} | "
            f"{result.instance.true_order:>3} | {algorithm_label(result.algorithm):>8} | "
            f"{result.num_phase_bits:>2} | {result.error_probability:.6f} | "
            f"{result.successful_trials:>9} | {result.trials:>6} | "
            f"{result.success_probability:.6f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one-shot order-recovery success versus true order under "
            "controlled-unitary depolarizing trajectory errors."
        )
    )
    parser.add_argument(
        "--instances-file",
        type=Path,
        default=DEFAULT_INSTANCES_FILE,
        help="JSON file containing representative N,a instances",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        default=[KITAEV_ALGORITHM],
        help="Algorithms to overlay: kitaev, standard",
    )
    parser.add_argument(
        "--error-model",
        default=BLOCK_DEPOLARIZING_MODEL,
        choices=(BLOCK_DEPOLARIZING_MODEL, PER_QUBIT_DEPOLARIZING_MODEL),
        help="Depolarizing model to sweep",
    )
    parser.add_argument(
        "--error-probabilities",
        type=float,
        nargs="+",
        default=list(DEFAULT_ERROR_PROBABILITIES),
        help="Depolarizing probabilities to overlay",
    )
    parser.add_argument("--bits", type=int, default=11, help="Phase/control bits")
    parser.add_argument(
        "--trials-per-instance",
        type=int,
        default=100,
        help="Trials per instance, algorithm, and probability",
    )
    parser.add_argument(
        "--standard-backend",
        choices=STANDARD_BACKEND_CHOICES,
        default=STATEVECTOR_STANDARD_BACKEND,
        help="Standard backend. Depolarizing errors require statevector.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
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

    if args.standard_backend == ANALYTIC_STANDARD_BACKEND:
        parser.error(
            "depolarizing errors require the standard statevector backend; the "
            "analytic backend samples ideal outcomes from the true order and has "
            "no control/work state or work-register trajectory to perturb"
        )

    try:
        algorithms = normalize_algorithms(args.algorithm)
        error_model = normalize_error_model(args.error_model)
        error_probabilities = validate_error_probabilities(args.error_probabilities)
        instances = load_instances(args.instances_file)
    except ValueError as exc:
        parser.error(str(exc))

    master_rng = np.random.default_rng(args.seed)
    results = simulate_order_recovery(
        instances=instances,
        algorithms=algorithms,
        error_model=error_model,
        error_probabilities=error_probabilities,
        trials_per_instance=args.trials_per_instance,
        num_phase_bits=args.bits,
        master_rng=master_rng,
        standard_backend=args.standard_backend,
    )

    probability_suffix = "_".join(
        probability_token(probability) for probability in error_probabilities
    )
    algorithm_suffix = "_".join(algorithms)
    model_suffix = error_model.replace("-", "_")
    output_path = args.output or (
        DEFAULT_OUTPUT_DIR
        / (
            "depolarizing_order_recovery_success_vs_order_"
            f"{algorithm_suffix}_{model_suffix}_p_{probability_suffix}_t{args.bits}.png"
        )
    )
    summary_path = (
        DEFAULT_DATA_DIR
        / (
            "depolarizing_order_recovery_success_vs_order_summary_"
            f"{algorithm_suffix}_{model_suffix}_p_{probability_suffix}_t{args.bits}.txt"
        )
    )

    plot_order_recovery_success(results, output_path)
    write_summary(results, summary_path)

    print(f"Wrote figure: {output_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"time_taken_seconds: {time.perf_counter() - start_time:.3f}")


if __name__ == "__main__":
    main()
