from fractions import Fraction

import numpy as np

from shor_common.errors import (
    apply_controlled_unitary_depolarizing_error,
    apply_measurement_bitflip,
    choose_measurement_bitflip_mask,
    controlled_unitary_depolarizing_allowed_for_phase_bit,
    validate_error_config,
)

from .qft import apply_inverse_qft_to_control


def integer_to_bits(value: int, num_bits: int) -> list[int]:
    """
    Convert an integer to a big-endian bit list of length `num_bits`.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")
    if not (0 <= value < 2 ** num_bits):
        raise ValueError("value out of range for num_bits")

    return [(value >> shift) & 1 for shift in range(num_bits - 1, -1, -1)]


def bits_to_phase_estimate(bits: list[int]) -> float:
    """
    Convert bits [b1, ..., bt] to the binary fraction 0.b1...bt.
    """
    if len(bits) < 1:
        raise ValueError("bits must be nonempty")

    numerator = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("bits must contain only 0 or 1")
        numerator = (2 * numerator) + bit

    return float(Fraction(numerator, 2 ** len(bits)))


def _apply_work_operator(work_op, work_state: np.ndarray) -> np.ndarray:
    if callable(work_op):
        out = work_op(work_state)
    else:
        work_op = np.asarray(work_op, dtype=complex)
        if work_op.shape != (work_state.shape[0], work_state.shape[0]):
            raise ValueError(
                f"work_op must have shape ({work_state.shape[0]}, {work_state.shape[0]})"
            )
        out = work_op @ work_state

    out = np.asarray(out, dtype=complex)
    if out.shape != work_state.shape:
        raise ValueError(f"work_op must return shape {work_state.shape}")
    return out


def apply_controlled_power_superposition(
    initial_work_state: np.ndarray,
    controlled_power_factory,
    num_control_qubits: int,
) -> np.ndarray:
    """
    Build the post-modular-exponentiation state used by standard QPE.

    Starting from H^{otimes t}|0^t> |work>, this computes

        1/sqrt(Q) sum_x |x> U^x |work>,

    where Q = 2**t.
    """
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

    work_state = np.asarray(initial_work_state, dtype=complex)
    if work_state.ndim != 1:
        raise ValueError("initial_work_state must be a 1D array")

    Q = 2 ** num_control_qubits
    psi = np.empty((Q, work_state.shape[0]), dtype=complex)

    for x in range(Q):
        work_op = controlled_power_factory(x)
        psi[x, :] = _apply_work_operator(work_op, work_state)

    return psi / np.sqrt(Q)


def apply_controlled_work_operator_to_control_qubit(
    psi: np.ndarray,
    work_op,
    control_qubit: int,
    num_control_qubits: int,
) -> np.ndarray:
    """
    Apply one controlled work-register operator to a full control/work state.

    `control_qubit` uses big-endian control-register indexing: qubit 0 is the
    most significant bit of the integer label on the first axis.
    """
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")
    if not (0 <= control_qubit < num_control_qubits):
        raise ValueError("control_qubit out of range")

    psi = np.asarray(psi, dtype=complex)
    Q = 2 ** num_control_qubits
    if psi.ndim != 2 or psi.shape[0] != Q:
        raise ValueError(f"psi must have shape ({Q}, M)")

    bit_mask = 1 << (num_control_qubits - 1 - control_qubit)
    active_labels = np.flatnonzero((np.arange(Q) & bit_mask) != 0)

    out = psi.copy()
    for label in active_labels:
        out[label, :] = _apply_work_operator(work_op, psi[label, :])
    return out


def _has_controlled_unitary_depolarizing_error(error_config) -> bool:
    error_config = validate_error_config(error_config)
    return (
        (
            error_config.controlled_unitary_depolarizing_probability is not None
            and error_config.controlled_unitary_depolarizing_probability != 0.0
        )
        or (
            error_config.controlled_unitary_per_qubit_depolarizing_probability
            is not None
            and error_config.controlled_unitary_per_qubit_depolarizing_probability != 0.0
        )
    )


def apply_controlled_power_sequence(
    initial_work_state: np.ndarray,
    controlled_power_factory,
    num_control_qubits: int,
    error_config=None,
    rng=None,
) -> np.ndarray:
    """
    Build the standard-QPE controlled-power state sequentially.

    This follows the same ideal convention as
    `apply_controlled_power_superposition`, but exposes each controlled
    U**(2**k) block so trajectory errors can be sampled after every block.
    """
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")
    if rng is None:
        rng = np.random.default_rng()

    error_config = validate_error_config(error_config)
    work_state = np.asarray(initial_work_state, dtype=complex)
    if work_state.ndim != 1:
        raise ValueError("initial_work_state must be a 1D array")

    Q = 2 ** num_control_qubits
    psi = np.empty((Q, work_state.shape[0]), dtype=complex)
    psi[:, :] = work_state
    psi /= np.sqrt(Q)

    for control_qubit in range(num_control_qubits):
        power = 2 ** (num_control_qubits - 1 - control_qubit)
        work_op = controlled_power_factory(power)
        psi = apply_controlled_work_operator_to_control_qubit(
            psi,
            work_op=work_op,
            control_qubit=control_qubit,
            num_control_qubits=num_control_qubits,
        )
        phase_bit_number = control_qubit + 1
        if controlled_unitary_depolarizing_allowed_for_phase_bit(
            error_config,
            phase_bit_number,
        ):
            psi = apply_controlled_unitary_depolarizing_error(
                psi,
                error_config=error_config,
                rng=rng,
            )

    return psi


def control_register_probabilities(psi: np.ndarray) -> np.ndarray:
    """
    Return measurement probabilities for the first axis of a control/work state.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 2:
        raise ValueError("psi must have shape (control_dim, work_dim)")

    probabilities = np.sum(np.abs(psi) ** 2, axis=1).real
    probabilities = np.maximum(probabilities, 0.0)
    total = float(np.sum(probabilities))

    if total <= 0.0:
        raise ValueError("state has zero norm")

    return probabilities / total


def measure_control_register(psi: np.ndarray, rng=None) -> tuple[int, np.ndarray]:
    """
    Measure the full control register and return the outcome integer and
    collapsed work-register state.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 2:
        raise ValueError("psi must have shape (control_dim, work_dim)")

    if rng is None:
        rng = np.random.default_rng()

    probabilities = control_register_probabilities(psi)
    outcome = int(rng.choice(np.arange(psi.shape[0]), p=probabilities))

    p = probabilities[outcome]
    if p <= 0.0:
        raise ValueError("sampled a zero-probability outcome")

    collapsed_work_state = psi[outcome, :].copy() / np.sqrt(p)
    return outcome, collapsed_work_state


def standard_phase_estimation_with_bitflip_mask(
    initial_work_state: np.ndarray,
    controlled_power_factory,
    num_control_qubits: int,
    measurement_bitflip_mask: list[bool] | None = None,
    measurement_rng=None,
) -> tuple[list[int], float, np.ndarray]:
    """
    Run standard phase estimation with an explicitly supplied measurement
    bit-flip mask.

    This is useful for paired clean/noisy diagnostics, where the clean and noisy
    runs should share the same control-register measurement randomness while
    differing only by the sampled classical bit-flip mask.
    """
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

    if measurement_rng is None:
        measurement_rng = np.random.default_rng()

    if measurement_bitflip_mask is None:
        measurement_bitflip_mask = [False] * num_control_qubits
    elif len(measurement_bitflip_mask) != num_control_qubits:
        raise ValueError(
            "measurement_bitflip_mask must have length num_control_qubits"
        )

    psi = apply_controlled_power_superposition(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_control_qubits=num_control_qubits,
    )
    psi = apply_inverse_qft_to_control(psi, num_control_qubits)

    measured_integer, final_work_state = measure_control_register(
        psi,
        rng=measurement_rng,
    )
    raw_bits = integer_to_bits(measured_integer, num_control_qubits)
    bits = [
        apply_measurement_bitflip(bit, should_flip=bool(should_flip))
        for bit, should_flip in zip(raw_bits, measurement_bitflip_mask)
    ]
    phase_estimate = bits_to_phase_estimate(bits)

    return bits, phase_estimate, final_work_state


def standard_phase_estimation(
    initial_work_state: np.ndarray,
    controlled_power_factory,
    num_control_qubits: int,
    rng=None,
    error_config=None,
) -> tuple[list[int], float, np.ndarray]:
    """
    Run textbook phase estimation with a t-qubit control register.

    The simulator constructs the whole control/work state, applies inverse QFT
    to the control register, measures all control qubits, and returns measured
    bits in most-significant-bit first order.
    """
    if num_control_qubits < 1:
        raise ValueError("num_control_qubits must be >= 1")

    if rng is None:
        rng = np.random.default_rng()

    error_config = validate_error_config(error_config)
    if _has_controlled_unitary_depolarizing_error(error_config):
        psi = apply_controlled_power_sequence(
            initial_work_state=initial_work_state,
            controlled_power_factory=controlled_power_factory,
            num_control_qubits=num_control_qubits,
            error_config=error_config,
            rng=rng,
        )
    else:
        psi = apply_controlled_power_superposition(
            initial_work_state=initial_work_state,
            controlled_power_factory=controlled_power_factory,
            num_control_qubits=num_control_qubits,
        )
    psi = apply_inverse_qft_to_control(psi, num_control_qubits)

    measured_integer, final_work_state = measure_control_register(psi, rng=rng)
    raw_bits = integer_to_bits(measured_integer, num_control_qubits)
    measurement_bitflip_mask = choose_measurement_bitflip_mask(
        num_control_qubits,
        error_config=error_config,
        rng=rng,
    )
    bits = [
        apply_measurement_bitflip(bit, should_flip=should_flip)
        for bit, should_flip in zip(raw_bits, measurement_bitflip_mask)
    ]
    phase_estimate = bits_to_phase_estimate(bits)

    return bits, phase_estimate, final_work_state
