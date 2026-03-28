import numpy as np

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


def one_iterative_pe_step(work_state, controlled_power_op, feedback=0.0, rng=None):
    """
    Perform one single-control-qubit phase-estimation step.
    """
    psi = make_joint_state(work_state)

    psi = apply_control_gate(psi, H())
    psi = apply_controlled_work_operator(psi, controlled_power_op)

    if feedback != 0.0:
        psi = apply_control_gate(psi, phase(feedback))

    psi = apply_control_gate(psi, H())

    bit, collapsed = measure_control(psi, rng=rng)
    new_work_state = collapsed[bit, :].copy()

    return bit, new_work_state


def iterative_phase_estimation(initial_work_state, controlled_power_factory, num_bits, rng=None):
    """
    Estimate `num_bits` binary digits of the eigenphase using one reusable
    control qubit and semiclassical feedback.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    work_state = np.asarray(initial_work_state, dtype=complex)
    bits = [None] * num_bits

    for j in range(num_bits, 0, -1):
        power = 2 ** (j - 1)
        work_op = controlled_power_factory(power)

        if j == num_bits:
            angle = 0.0
        else:
            angle = feedback_angle(bits, j)

        bit, work_state = one_iterative_pe_step(
            work_state=work_state,
            controlled_power_op=work_op,
            feedback=angle,
            rng=rng,
        )

        bits[j - 1] = bit

    phase_estimate = sum(bits[j - 1] / (2 ** j) for j in range(1, num_bits + 1))

    return bits, phase_estimate, work_state