from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from shor_common.errors import (
    ErrorConfig,
    apply_controlled_unitary_depolarizing_error,
    apply_measurement_bitflip,
    choose_measurement_bitflip_mask,
    validate_error_config,
)
from shor_common.modular_unitaries import work_dim
from shor_common.shor_utils import (
    default_num_phase_bits,
    make_basis_state,
    make_controlled_power_factory,
)
from shor_one_qubit.gates import H, phase
from shor_one_qubit.kitaev_pe import feedback_angle
from shor_one_qubit.state_ops import (
    apply_control_gate,
    apply_controlled_work_operator,
    make_joint_state,
    measure_control,
)

from .classical_error_correction import validate_phase_estimation_instance


@dataclass(frozen=True)
class PhaseEstimationHammingDistanceTrial:
    clean_bits: list[int]
    noisy_bits: list[int]
    clean_phase_estimate: float
    noisy_phase_estimate: float
    measurement_bitflip_mask: list[bool]
    error_config: dict[str, bool | float]

    @property
    def hamming_distance(self) -> int:
        return hamming_distance(self.clean_bits, self.noisy_bits)

    @property
    def normalized_hamming_distance(self) -> float:
        return normalized_hamming_distance(self.clean_bits, self.noisy_bits)

    @property
    def decimal_distance(self) -> float:
        return decimal_distance(self.clean_bits, self.noisy_bits)


@dataclass(frozen=True)
class AverageHammingDistanceEstimate:
    trials: int
    num_phase_bits: int
    total_hamming_distance: int
    error_config: dict[str, bool | float]

    @property
    def average_hamming_distance(self) -> float:
        return self.total_hamming_distance / self.trials

    @property
    def average_normalized_hamming_distance(self) -> float:
        return self.total_hamming_distance / (self.trials * self.num_phase_bits)


@dataclass(frozen=True)
class AverageDecimalDistanceEstimate:
    trials: int
    num_phase_bits: int
    total_decimal_distance: float
    error_config: dict[str, bool | float]

    @property
    def average_decimal_distance(self) -> float:
        return self.total_decimal_distance / self.trials


def _validate_bit_string(bits, name: str) -> list[int]:
    bit_list = list(bits)
    for bit in bit_list:
        if bit not in (0, 1):
            raise ValueError(f"{name} must contain only 0 or 1")
    return bit_list


def hamming_distance(left_bits, right_bits) -> int:
    """
    Return the Hamming distance between two equal-length bit strings.
    """
    left = _validate_bit_string(left_bits, "left_bits")
    right = _validate_bit_string(right_bits, "right_bits")

    if len(left) != len(right):
        raise ValueError("bit strings must have the same length")

    return int(
        sum(left_bit != right_bit for left_bit, right_bit in zip(left, right))
    )


def normalized_hamming_distance(left_bits, right_bits) -> float:
    """
    Return the Hamming distance divided by the bit-string length.
    """
    left = _validate_bit_string(left_bits, "left_bits")
    right = _validate_bit_string(right_bits, "right_bits")

    if len(left) != len(right):
        raise ValueError("bit strings must have the same length")
    if len(left) == 0:
        raise ValueError("bit strings must be nonempty")

    return hamming_distance(left, right) / len(left)


def bits_to_decimal_fraction(bits) -> Fraction:
    """
    Convert a nonempty bit string to its exact binary fraction in [0, 1).
    """
    bit_list = _validate_bit_string(bits, "bits")
    if len(bit_list) == 0:
        raise ValueError("bits must be nonempty")

    numerator = 0
    for bit in bit_list:
        numerator = (2 * numerator) + bit

    return Fraction(numerator, 2 ** len(bit_list))


def bits_to_decimal(bits) -> float:
    """
    Convert a nonempty bit string to its decimal phase estimate in [0, 1).
    """
    return float(bits_to_decimal_fraction(bits))


def decimal_distance(left_bits, right_bits) -> float:
    """
    Return min(|x - y|, 1 - |x - y|) for two equal-length bit fractions.
    """
    left = _validate_bit_string(left_bits, "left_bits")
    right = _validate_bit_string(right_bits, "right_bits")

    if len(left) != len(right):
        raise ValueError("bit strings must have the same length")
    if len(left) == 0:
        raise ValueError("bit strings must be nonempty")

    left_fraction = bits_to_decimal_fraction(left)
    right_fraction = bits_to_decimal_fraction(right)
    distance = abs(left_fraction - right_fraction)
    return float(min(distance, 1 - distance))


def one_iterative_pe_diagnostic_step(
    work_state: np.ndarray,
    controlled_power_op,
    feedback: float = 0.0,
    measurement_rng=None,
    error_rng=None,
    error_config: ErrorConfig | None = None,
) -> tuple[int, np.ndarray]:
    """
    Perform one PE step with separate RNG streams for measurement and errors.
    """
    error_config = validate_error_config(error_config)

    if measurement_rng is None:
        measurement_rng = np.random.default_rng()
    if error_rng is None:
        error_rng = np.random.default_rng()

    psi = make_joint_state(work_state)
    psi = apply_control_gate(psi, H())
    psi = apply_controlled_work_operator(psi, controlled_power_op)
    psi = apply_controlled_unitary_depolarizing_error(
        psi,
        error_config=error_config,
        rng=error_rng,
    )

    if feedback != 0.0:
        psi = apply_control_gate(psi, phase(feedback))

    psi = apply_control_gate(psi, H())

    bit, collapsed = measure_control(psi, rng=measurement_rng)
    new_work_state = collapsed[bit, :].copy()

    return bit, new_work_state


def iterative_phase_estimation_for_diagnostics(
    initial_work_state: np.ndarray,
    controlled_power_factory,
    num_bits: int,
    measurement_bitflip_mask: list[bool] | None = None,
    measurement_rng=None,
    error_rng=None,
    error_config: ErrorConfig | None = None,
) -> tuple[list[int], float, np.ndarray]:
    """
    Run iterative PE with explicit measurement and trajectory-error RNGs.

    The measurement bit-flip mask is supplied explicitly so paired clean/noisy
    diagnostic trials can share measurement randomness while sampling errors
    from independent streams.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    error_config = validate_error_config(error_config)

    if measurement_rng is None:
        measurement_rng = np.random.default_rng()
    if error_rng is None:
        error_rng = np.random.default_rng()

    if measurement_bitflip_mask is None:
        measurement_bitflip_mask = [False] * num_bits
    elif len(measurement_bitflip_mask) != num_bits:
        raise ValueError("measurement_bitflip_mask must have length num_bits")

    work_state = np.asarray(initial_work_state, dtype=complex)
    bits = [None] * num_bits

    for j in range(num_bits, 0, -1):
        power = 2 ** (j - 1)
        work_op = controlled_power_factory(power)

        if j == num_bits:
            angle = 0.0
        else:
            angle = feedback_angle(bits, j)

        measured_bit, work_state = one_iterative_pe_diagnostic_step(
            work_state=work_state,
            controlled_power_op=work_op,
            feedback=angle,
            measurement_rng=measurement_rng,
            error_rng=error_rng,
            error_config=error_config,
        )

        bits[j - 1] = apply_measurement_bitflip(
            measured_bit,
            should_flip=bool(measurement_bitflip_mask[j - 1]),
        )

    phase_estimate = sum(bits[j - 1] / (2 ** j) for j in range(1, num_bits + 1))

    return bits, phase_estimate, work_state


def run_phase_estimation_hamming_distance_trial(
    N: int,
    a: int,
    num_phase_bits: int | None = None,
    error_config: ErrorConfig | None = None,
    measurement_bitflip_mask: list[bool] | None = None,
    rng=None,
) -> PhaseEstimationHammingDistanceTrial:
    """
    Run paired clean/noisy Kitaev PE and compare the measured bit strings.
    """
    validate_phase_estimation_instance(N, a)
    error_config = validate_error_config(error_config)

    if rng is None:
        rng = np.random.default_rng()

    if num_phase_bits is None:
        num_phase_bits = default_num_phase_bits(N)
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    measurement_seed = int(rng.integers(0, 2 ** 63 - 1))
    measurement_error_seed = int(rng.integers(0, 2 ** 63 - 1))
    trajectory_error_seed = int(rng.integers(0, 2 ** 63 - 1))

    if measurement_bitflip_mask is None:
        measurement_bitflip_mask = choose_measurement_bitflip_mask(
            num_phase_bits,
            error_config=error_config,
            rng=np.random.default_rng(measurement_error_seed),
        )
    elif len(measurement_bitflip_mask) != num_phase_bits:
        raise ValueError("measurement_bitflip_mask must have length num_phase_bits")

    initial_work_state = make_basis_state(1, work_dim(N))
    controlled_power_factory = make_controlled_power_factory(a, N)

    clean_bits, clean_phase_estimate, _ = iterative_phase_estimation_for_diagnostics(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=num_phase_bits,
        measurement_bitflip_mask=[False] * num_phase_bits,
        measurement_rng=np.random.default_rng(measurement_seed),
        error_rng=np.random.default_rng(trajectory_error_seed),
        error_config=ErrorConfig(),
    )

    noisy_bits, noisy_phase_estimate, _ = iterative_phase_estimation_for_diagnostics(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=num_phase_bits,
        measurement_bitflip_mask=measurement_bitflip_mask,
        measurement_rng=np.random.default_rng(measurement_seed),
        error_rng=np.random.default_rng(trajectory_error_seed),
        error_config=error_config,
    )

    return PhaseEstimationHammingDistanceTrial(
        clean_bits=clean_bits,
        noisy_bits=noisy_bits,
        clean_phase_estimate=clean_phase_estimate,
        noisy_phase_estimate=noisy_phase_estimate,
        measurement_bitflip_mask=measurement_bitflip_mask,
        error_config=error_config.to_dict(),
    )


def estimate_average_hamming_distance(
    N: int,
    a: int,
    trials: int,
    num_phase_bits: int | None = None,
    error_config: ErrorConfig | None = None,
    rng=None,
) -> AverageHammingDistanceEstimate:
    """
    Estimate the average clean/noisy Hamming distance over many paired trials.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    error_config = validate_error_config(error_config)

    if rng is None:
        rng = np.random.default_rng()

    if num_phase_bits is None:
        num_phase_bits = default_num_phase_bits(N)
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    total_hamming_distance = 0
    for _ in range(trials):
        trial = run_phase_estimation_hamming_distance_trial(
            N=N,
            a=a,
            num_phase_bits=num_phase_bits,
            error_config=error_config,
            rng=rng,
        )
        total_hamming_distance += trial.hamming_distance

    return AverageHammingDistanceEstimate(
        trials=trials,
        num_phase_bits=num_phase_bits,
        total_hamming_distance=total_hamming_distance,
        error_config=error_config.to_dict(),
    )


def estimate_average_decimal_distance(
    N: int,
    a: int,
    trials: int,
    num_phase_bits: int | None = None,
    error_config: ErrorConfig | None = None,
    rng=None,
) -> AverageDecimalDistanceEstimate:
    """
    Estimate the average clean/noisy decimal distance over many trials.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    error_config = validate_error_config(error_config)

    if rng is None:
        rng = np.random.default_rng()

    if num_phase_bits is None:
        num_phase_bits = default_num_phase_bits(N)
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    total_decimal_distance = 0.0
    for _ in range(trials):
        trial = run_phase_estimation_hamming_distance_trial(
            N=N,
            a=a,
            num_phase_bits=num_phase_bits,
            error_config=error_config,
            rng=rng,
        )
        total_decimal_distance += trial.decimal_distance

    return AverageDecimalDistanceEstimate(
        trials=trials,
        num_phase_bits=num_phase_bits,
        total_decimal_distance=total_decimal_distance,
        error_config=error_config.to_dict(),
    )
