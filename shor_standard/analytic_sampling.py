import math
from dataclasses import dataclass

import numpy as np

from shor_common.errors import apply_measurement_bitflip, validate_error_config

from .phase_estimation import bits_to_phase_estimate, integer_to_bits


@dataclass(frozen=True)
class QPEWindowMassDiagnostics:
    order: int
    num_control_qubits: int
    window: int
    num_eigenphases: int
    min_mass: float
    mean_mass: float
    max_mass: float
    worst_eigenphase_numerator: int
    best_eigenphase_numerator: int


def multiplicative_order(a: int, N: int) -> int:
    """
    Return the multiplicative order of a modulo N by direct repeated
    multiplication.
    """
    if not (1 <= a < N):
        raise ValueError("a must satisfy 1 <= a < N")
    if math.gcd(a, N) != 1:
        raise ValueError("a and N must be coprime")

    residue = 1
    for order in range(1, N + 1):
        residue = (residue * a) % N
        if residue == 1:
            return order

    raise RuntimeError("failed to find multiplicative order")


def qpe_outcome_probability(
    y: int,
    eigenphase_numerator: int,
    order: int,
    num_control_qubits: int,
) -> float:
    """
    Return P(y | eigenphase_numerator / order) for ideal standard QPE.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if not (0 <= eigenphase_numerator < order):
        raise ValueError("eigenphase_numerator must satisfy 0 <= s < order")
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

    Q = 2 ** num_control_qubits
    y = int(y) % Q
    delta = (eigenphase_numerator / order) - (y / Q)
    delta -= round(delta)

    if abs(delta) < 1e-15:
        return 1.0

    ratio = np.sinc(Q * delta) / np.sinc(delta)
    probability = float((ratio * ratio).real)

    if probability < 0.0 and probability > -1e-15:
        return 0.0
    return probability


def qpe_window_candidates(
    eigenphase_numerator: int,
    order: int,
    num_control_qubits: int,
    window: int,
) -> list[int]:
    """
    Return control-register outcome candidates near Q * s / r.
    """
    if window < 0:
        raise ValueError("window must be >= 0")
    if order < 1:
        raise ValueError("order must be >= 1")
    if not (0 <= eigenphase_numerator < order):
        raise ValueError("eigenphase_numerator must satisfy 0 <= s < order")
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

    Q = 2 ** num_control_qubits
    center = (Q * eigenphase_numerator) / order
    nearest = int(math.floor(center + 0.5))

    candidates = []
    seen = set()
    for offset in range(-window, window + 1):
        y = (nearest + offset) % Q
        if y not in seen:
            seen.add(y)
            candidates.append(y)

    return candidates


def qpe_window_probability_mass(
    eigenphase_numerator: int,
    order: int,
    num_control_qubits: int,
    window: int,
) -> float:
    """
    Return the exact standard-QPE probability mass captured by one local window.
    """
    candidates = qpe_window_candidates(
        eigenphase_numerator=eigenphase_numerator,
        order=order,
        num_control_qubits=num_control_qubits,
        window=window,
    )
    mass = sum(
        qpe_outcome_probability(
            y=y,
            eigenphase_numerator=eigenphase_numerator,
            order=order,
            num_control_qubits=num_control_qubits,
        )
        for y in candidates
    )

    if mass < 0.0 and mass > -1e-12:
        return 0.0
    if mass > 1.0 and mass < 1.0 + 1e-12:
        return 1.0
    return float(mass)


def qpe_window_mass_diagnostics(
    order: int,
    num_control_qubits: int,
    window: int,
) -> QPEWindowMassDiagnostics:
    """
    Summarize captured local-window mass over all eigenphases s/r.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")
    if window < 0:
        raise ValueError("window must be >= 0")

    masses = [
        qpe_window_probability_mass(
            eigenphase_numerator=s,
            order=order,
            num_control_qubits=num_control_qubits,
            window=window,
        )
        for s in range(order)
    ]
    min_index = min(range(order), key=lambda s: masses[s])
    max_index = max(range(order), key=lambda s: masses[s])

    return QPEWindowMassDiagnostics(
        order=order,
        num_control_qubits=num_control_qubits,
        window=window,
        num_eigenphases=order,
        min_mass=float(masses[min_index]),
        mean_mass=float(np.mean(masses)),
        max_mass=float(masses[max_index]),
        worst_eigenphase_numerator=min_index,
        best_eigenphase_numerator=max_index,
    )


def sample_qpe_outcome_for_eigenphase(
    eigenphase_numerator: int,
    order: int,
    num_control_qubits: int,
    rng=None,
    window: int = 100,
) -> int:
    """
    Sample a standard-QPE outcome from a local window around the closest peak.
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = qpe_window_candidates(
        eigenphase_numerator=eigenphase_numerator,
        order=order,
        num_control_qubits=num_control_qubits,
        window=window,
    )
    weights = np.array(
        [
            qpe_outcome_probability(
                y=y,
                eigenphase_numerator=eigenphase_numerator,
                order=order,
                num_control_qubits=num_control_qubits,
            )
            for y in candidates
        ],
        dtype=float,
    )
    total = float(np.sum(weights))
    if total <= 0.0:
        return candidates[len(candidates) // 2]

    probabilities = weights / total
    return int(rng.choice(candidates, p=probabilities))


def sample_qpe_outcome_from_order(
    order: int,
    num_control_qubits: int,
    rng=None,
    window: int = 100,
) -> tuple[int, int]:
    """
    Sample an ideal standard-Shor control-register outcome using the order r.

    Returns (y, s), where s/r is the sampled eigenphase.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")
    if rng is None:
        rng = np.random.default_rng()

    eigenphase_numerator = int(rng.integers(0, order))
    y = sample_qpe_outcome_for_eigenphase(
        eigenphase_numerator=eigenphase_numerator,
        order=order,
        num_control_qubits=num_control_qubits,
        rng=rng,
        window=window,
    )
    return y, eigenphase_numerator


def sample_standard_phase_bits_from_order(
    order: int,
    num_control_qubits: int,
    rng=None,
    window: int = 100,
    measurement_bitflip_mask: list[bool] | None = None,
) -> tuple[list[int], float, int, int]:
    """
    Sample standard-QPE bits from the analytic order-based distribution.

    Returns (bits, phase_estimate, y, s).
    """
    if rng is None:
        rng = np.random.default_rng()
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

    if measurement_bitflip_mask is None:
        measurement_bitflip_mask = [False] * num_control_qubits
    elif len(measurement_bitflip_mask) != num_control_qubits:
        raise ValueError(
            "measurement_bitflip_mask must have length num_control_qubits"
        )

    y, eigenphase_numerator = sample_qpe_outcome_from_order(
        order=order,
        num_control_qubits=num_control_qubits,
        rng=rng,
        window=window,
    )
    raw_bits = integer_to_bits(y, num_control_qubits)
    bits = [
        apply_measurement_bitflip(bit, should_flip=bool(should_flip))
        for bit, should_flip in zip(raw_bits, measurement_bitflip_mask)
    ]
    return bits, bits_to_phase_estimate(bits), y, eigenphase_numerator


def validate_analytic_error_config(error_config):
    """
    Require an error model supported by analytic standard-QPE sampling.
    """
    error_config = validate_error_config(error_config)
    if (
        error_config.controlled_unitary_depolarizing_probability is not None
        or error_config.controlled_unitary_per_qubit_depolarizing_probability is not None
    ):
        raise ValueError(
            "analytic standard backend supports measurement bit-flip errors only"
        )
    return error_config
