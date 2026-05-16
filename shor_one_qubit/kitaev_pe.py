import numpy as np

from shor_common.errors import (
    apply_measurement_bitflip,
    apply_controlled_unitary_depolarizing_error,
    choose_measurement_bitflip_mask,
    validate_error_config,
)
from .gates import H, phase
from .state_ops import (
    make_joint_state,
    apply_control_gate,
    apply_controlled_work_operator,
    measure_control,
)


def feedback_angle(bits, j):
    """
    Compute the semiclassical inverse-QFT feedback angle used when extracting
    bit b_j of the phase.

    Convention:
      - bits[k] stores b_{k+1}
      - j is 1-indexed, so j=1 means the most significant bit

    To determine b_j, the less-significant bits b_{j+1}, ..., b_t must
    already have been measured.
    """
    t = len(bits)

    if not (1 <= j <= t):
        raise ValueError("j must satisfy 1 <= j <= len(bits)")

    frac = 0.0
    for l in range(j + 1, t + 1):
        b = bits[l - 1]
        if b is None:
            raise ValueError("feedback requires lower-significance bits to be known")
        frac += b / (2 ** (l - j + 1))

    return -2.0 * np.pi * frac


def one_iterative_pe_step(
    work_state,
    controlled_power_op,
    feedback=0.0,
    rng=None,
    error_config=None,
):
    """
    Perform one single-control-qubit phase-estimation step.
    """
    error_config = validate_error_config(error_config)
    psi = make_joint_state(work_state)

    psi = apply_control_gate(psi, H())
    psi = apply_controlled_work_operator(psi, controlled_power_op)
    psi = apply_controlled_unitary_depolarizing_error(
        psi,
        error_config=error_config,
        rng=rng,
    )

    if feedback != 0.0:
        psi = apply_control_gate(psi, phase(feedback))

    psi = apply_control_gate(psi, H())

    bit, collapsed = measure_control(psi, rng=rng)
    new_work_state = collapsed[bit, :].copy()

    return bit, new_work_state


def iterative_phase_estimation(
    initial_work_state,
    controlled_power_factory,
    num_bits,
    rng=None,
    error_config=None,
):
    """
    Estimate `num_bits` binary digits of the eigenphase using one reusable
    control qubit and semiclassical feedback.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    if rng is None:
        rng = np.random.default_rng()

    error_config = validate_error_config(error_config)
    work_state = np.asarray(initial_work_state, dtype=complex)
    bits = [None] * num_bits
    measurement_bitflip_mask = choose_measurement_bitflip_mask(
        num_bits,
        error_config=error_config,
        rng=rng,
    )

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
            rng=rng,
            error_config=error_config,
        )

        bit = apply_measurement_bitflip(
            measured_bit,
            should_flip=measurement_bitflip_mask[j - 1],
        )
        bits[j - 1] = bit

    phase_estimate = sum(bits[j - 1] / (2 ** j) for j in range(1, num_bits + 1))

    return bits, phase_estimate, work_state
