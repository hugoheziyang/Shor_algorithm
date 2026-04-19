from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from shor_one_qubit.errors import (
    ErrorConfig,
    apply_measurement_bitflip,
    choose_measurement_bitflip_mask,
    validate_error_config,
)
from shor_one_qubit.kitaev_pe import feedback_angle, one_iterative_pe_step
from shor_one_qubit.modular_unitaries import work_dim
from shor_one_qubit.shor import (
    default_num_phase_bits,
    make_basis_state,
    make_controlled_power_factory,
)


@dataclass(frozen=True)
class PairedPhaseEstimationTrial:
    clean_bits: list[int]
    noisy_bits: list[int]
    clean_phase_estimate: float
    noisy_phase_estimate: float
    measurement_bitflip_mask: list[bool]
    recovered: bool


@dataclass(frozen=True)
class RecoveryProbabilityEstimate:
    error_probability: float
    successful_trials: int
    trials: int
    naive_baseline_probability: float

    @property
    def recovery_probability(self) -> float:
        return self.successful_trials / self.trials


def validate_phase_estimation_instance(N: int, a: int) -> None:
    """
    Validate the fixed modular-multiplication instance used in the recovery experiment.
    """
    if N < 4:
        raise ValueError("N must satisfy N >= 4")

    if not (2 <= a <= N - 2):
        raise ValueError("a must satisfy 2 <= a <= N - 2")
    if math.gcd(a, N) != 1:
        raise ValueError("a must be coprime to N")


def flip_bit(bits: list[int], index: int) -> list[int]:
    """
    Return a copy of the bit string with one selected bit flipped.
    """
    if not (0 <= index < len(bits)):
        raise ValueError("index out of range")

    flipped = list(bits)
    bit = flipped[index]
    if bit not in (0, 1):
        raise ValueError("bits must contain only 0 or 1")
    flipped[index] = 1 - bit
    return flipped


def single_bit_flip_candidates(bits: list[int]) -> list[list[int]]:
    """
    Return the noisy bit string together with all one-bit-flip candidates.
    """
    return [list(bits)] + [flip_bit(bits, index) for index in range(len(bits))]


def can_recover_clean_bits_by_single_flip(
    noisy_bits: list[int],
    clean_bits: list[int],
) -> bool:
    """
    Return True when a single-bit classical search can recover the clean bits.
    """
    if len(noisy_bits) != len(clean_bits):
        raise ValueError("bit strings must have the same length")

    for candidate in single_bit_flip_candidates(noisy_bits):
        if candidate == list(clean_bits):
            return True

    return False


def naive_single_bit_recovery_baseline(
    num_bits: int,
    error_probability: float,
) -> float:
    """
    Exact recovery probability for a non-propagating independent bit-flip model.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    if not (0.0 <= error_probability <= 1.0):
        raise ValueError("error_probability must satisfy 0 <= p <= 1")

    p = float(error_probability)
    return ((1.0 - p) ** num_bits) + (
        num_bits * p * ((1.0 - p) ** (num_bits - 1))
    )


def iterative_phase_estimation_with_bitflip_mask(
    initial_work_state: np.ndarray,
    controlled_power_factory,
    num_bits: int,
    measurement_bitflip_mask: list[bool] | None = None,
    measurement_rng=None,
) -> tuple[list[int], float, np.ndarray]:
    """
    Run iterative Kitaev phase estimation with an explicitly supplied bit-flip mask.

    This keeps the quantum measurement randomness separate from the classical
    error-mask sampling so clean and noisy paired trials can share the same
    measurement RNG seed.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    if measurement_rng is None:
        measurement_rng = np.random.default_rng()

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

        measured_bit, work_state = one_iterative_pe_step(
            work_state=work_state,
            controlled_power_op=work_op,
            feedback=angle,
            rng=measurement_rng,
        )

        bits[j - 1] = apply_measurement_bitflip(
            measured_bit,
            should_flip=bool(measurement_bitflip_mask[j - 1]),
        )

    phase_estimate = sum(bits[j - 1] / (2 ** j) for j in range(1, num_bits + 1))

    return bits, phase_estimate, work_state


def run_paired_phase_estimation_trial(
    N: int,
    a: int,
    num_phase_bits: int | None = None,
    error_probability: float = 0.0,
    rng=None,
) -> PairedPhaseEstimationTrial:
    """
    Run one paired clean/noisy Kitaev PE trial for the same fixed (N, a).
    """
    validate_phase_estimation_instance(N, a)

    if rng is None:
        rng = np.random.default_rng()

    if num_phase_bits is None:
        num_phase_bits = default_num_phase_bits(N)
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    error_config = validate_error_config(
        ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=error_probability,
        )
    )

    measurement_seed = int(rng.integers(0, 2 ** 63 - 1))
    error_seed = int(rng.integers(0, 2 ** 63 - 1))

    measurement_rng_clean = np.random.default_rng(measurement_seed)
    measurement_rng_noisy = np.random.default_rng(measurement_seed)
    error_rng = np.random.default_rng(error_seed)

    measurement_bitflip_mask = choose_measurement_bitflip_mask(
        num_phase_bits,
        error_config=error_config,
        rng=error_rng,
    )

    initial_work_state = make_basis_state(1, work_dim(N))
    controlled_power_factory = make_controlled_power_factory(a, N)

    clean_bits, clean_phase_estimate, _ = iterative_phase_estimation_with_bitflip_mask(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=num_phase_bits,
        measurement_bitflip_mask=[False] * num_phase_bits,
        measurement_rng=measurement_rng_clean,
    )

    noisy_bits, noisy_phase_estimate, _ = iterative_phase_estimation_with_bitflip_mask(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=num_phase_bits,
        measurement_bitflip_mask=measurement_bitflip_mask,
        measurement_rng=measurement_rng_noisy,
    )

    return PairedPhaseEstimationTrial(
        clean_bits=clean_bits,
        noisy_bits=noisy_bits,
        clean_phase_estimate=clean_phase_estimate,
        noisy_phase_estimate=noisy_phase_estimate,
        measurement_bitflip_mask=measurement_bitflip_mask,
        recovered=can_recover_clean_bits_by_single_flip(noisy_bits, clean_bits),
    )


def estimate_classical_recovery_probability(
    N: int,
    a: int,
    num_phase_bits: int | None,
    error_probability: float,
    trials: int,
    rng=None,
) -> RecoveryProbabilityEstimate:
    """
    Estimate the empirical recovery probability for one fixed error rate p.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    if rng is None:
        rng = np.random.default_rng()

    if num_phase_bits is None:
        num_phase_bits = default_num_phase_bits(N)
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")

    successful_trials = 0

    for _ in range(trials):
        trial = run_paired_phase_estimation_trial(
            N=N,
            a=a,
            num_phase_bits=num_phase_bits,
            error_probability=error_probability,
            rng=rng,
        )
        if trial.recovered:
            successful_trials += 1

    return RecoveryProbabilityEstimate(
        error_probability=error_probability,
        successful_trials=successful_trials,
        trials=trials,
        naive_baseline_probability=naive_single_bit_recovery_baseline(
            num_phase_bits,
            error_probability,
        ),
    )
