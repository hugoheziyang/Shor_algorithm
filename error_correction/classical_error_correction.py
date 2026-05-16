from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from shor_common.errors import (
    ErrorConfig,
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
from shor_one_qubit.kitaev_pe import feedback_angle, one_iterative_pe_step
from shor_standard.analytic_sampling import (
    multiplicative_order,
    sample_standard_phase_bits_from_order,
)
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STATEVECTOR_STANDARD_BACKEND,
    validate_standard_backend,
)
from shor_standard.phase_estimation import standard_phase_estimation_with_bitflip_mask


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


def classical_recovery_lower_bound(
    num_bits: int,
    error_probability: float,
) -> float:
    """
    Guaranteed recovery probability from no raw errors plus a final-bit error.

    A single measurement bit-flip on the last measured phase bit cannot
    propagate into later feedback, so it is always recoverable by a single-bit
    classical search.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    if not (0.0 <= error_probability <= 1.0):
        raise ValueError("error_probability must satisfy 0 <= p <= 1")

    p = float(error_probability)
    return ((1.0 - p) ** num_bits) + (p * ((1.0 - p) ** (num_bits - 1)))


def classical_recovery_first_order_upper_bound(
    num_bits: int,
    error_probability: float,
) -> float:
    """
    First-order upper envelope where every one-error location is recoverable.

    This is exact for a non-propagating independent bit-flip model. For the
    propagating Kitaev feedback model it upper-bounds the zero-error plus
    one-error sector, but it does not include possible recoverable multi-error
    cancellations.
    """
    return naive_single_bit_recovery_baseline(num_bits, error_probability)


def kitaev_single_error_recoverability_coefficient(num_bits: int) -> float:
    """
    Return the ideal-Kitaev first-order coefficient C_t.

    C_t = 1 + sum_{k=2}^t prod_{j=1}^{k-1} cos^2(pi / 2^{j+1}).
    The first term corresponds to an error on the final measured bit, whose
    empty product is recoverable with probability one.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    coefficient = 1.0
    product = 1.0
    for k in range(2, num_bits + 1):
        j = k - 1
        product *= math.cos(math.pi / (2 ** (j + 1))) ** 2
        coefficient += product

    return coefficient


def kitaev_first_order_recovery_estimate(
    num_bits: int,
    error_probability: float,
) -> float:
    """
    First-order recovery estimate using the ideal-Kitaev C_t coefficient.

    The retained sectors are no raw bit-flips and recoverable single raw
    measurement bit-flips:
        (1-p)^t + C_t p (1-p)^(t-1).
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    if not (0.0 <= error_probability <= 1.0):
        raise ValueError("error_probability must satisfy 0 <= p <= 1")

    p = float(error_probability)
    coefficient = kitaev_single_error_recoverability_coefficient(num_bits)
    return ((1.0 - p) ** num_bits) + (
        coefficient * p * ((1.0 - p) ** (num_bits - 1))
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


def run_paired_standard_phase_estimation_trial(
    N: int,
    a: int,
    num_control_qubits: int | None = None,
    error_probability: float = 0.0,
    rng=None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
    order: int | None = None,
) -> PairedPhaseEstimationTrial:
    """
    Run one paired clean/noisy standard PE trial for the same fixed (N, a).
    """
    validate_phase_estimation_instance(N, a)
    standard_backend = validate_standard_backend(standard_backend)

    if rng is None:
        rng = np.random.default_rng()

    if num_control_qubits is None:
        num_control_qubits = default_num_phase_bits(N)
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

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
        num_control_qubits,
        error_config=error_config,
        rng=error_rng,
    )

    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        initial_work_state = make_basis_state(1, work_dim(N))
        controlled_power_factory = make_controlled_power_factory(a, N)

        clean_bits, clean_phase_estimate, _ = standard_phase_estimation_with_bitflip_mask(
            initial_work_state=initial_work_state,
            controlled_power_factory=controlled_power_factory,
            num_control_qubits=num_control_qubits,
            measurement_bitflip_mask=[False] * num_control_qubits,
            measurement_rng=measurement_rng_clean,
        )

        noisy_bits, noisy_phase_estimate, _ = standard_phase_estimation_with_bitflip_mask(
            initial_work_state=initial_work_state,
            controlled_power_factory=controlled_power_factory,
            num_control_qubits=num_control_qubits,
            measurement_bitflip_mask=measurement_bitflip_mask,
            measurement_rng=measurement_rng_noisy,
        )
    else:
        if order is None:
            order = multiplicative_order(a, N)

        clean_bits, clean_phase_estimate, _, _ = sample_standard_phase_bits_from_order(
            order=order,
            num_control_qubits=num_control_qubits,
            rng=measurement_rng_clean,
            window=analytic_window,
            measurement_bitflip_mask=[False] * num_control_qubits,
        )

        noisy_bits, noisy_phase_estimate, _, _ = sample_standard_phase_bits_from_order(
            order=order,
            num_control_qubits=num_control_qubits,
            rng=measurement_rng_noisy,
            window=analytic_window,
            measurement_bitflip_mask=measurement_bitflip_mask,
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


def estimate_standard_classical_recovery_probability(
    N: int,
    a: int,
    num_control_qubits: int | None,
    error_probability: float,
    trials: int,
    rng=None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> RecoveryProbabilityEstimate:
    """
    Estimate empirical single-bit recovery probability for standard PE.
    """
    standard_backend = validate_standard_backend(standard_backend)
    if trials < 1:
        raise ValueError("trials must be >= 1")

    if rng is None:
        rng = np.random.default_rng()

    if num_control_qubits is None:
        num_control_qubits = default_num_phase_bits(N)
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

    successful_trials = 0
    order = (
        multiplicative_order(a, N)
        if standard_backend == ANALYTIC_STANDARD_BACKEND
        else None
    )

    for _ in range(trials):
        trial = run_paired_standard_phase_estimation_trial(
            N=N,
            a=a,
            num_control_qubits=num_control_qubits,
            error_probability=error_probability,
            rng=rng,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
            order=order,
        )
        if trial.recovered:
            successful_trials += 1

    return RecoveryProbabilityEstimate(
        error_probability=error_probability,
        successful_trials=successful_trials,
        trials=trials,
        naive_baseline_probability=naive_single_bit_recovery_baseline(
            num_control_qubits,
            error_probability,
        ),
    )
