import numpy as np

from shor_one_qubit.gates import H
from shor_one_qubit.state_ops import (
    make_joint_state,
    apply_control_gate,
    apply_controlled_work_operator,
    control_probabilities,
    measure_control,
)


def test_make_joint_state_places_work_state_in_control_zero_branch():
    work = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi = make_joint_state(work)

    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(psi, expected)


def test_apply_control_gate_hadamard():
    work = np.array([0.0, 1.0], dtype=complex)
    psi = make_joint_state(work)
    out = apply_control_gate(psi, H())

    expected = np.array(
        [
            [0.0, 1 / np.sqrt(2)],
            [0.0, 1 / np.sqrt(2)],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(out, expected)


def test_apply_controlled_work_operator_only_changes_control_one_branch():
    psi = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=complex,
    )

    def work_op(v):
        return 2.0 * v

    out = apply_controlled_work_operator(psi, work_op)

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [8.0, 10.0, 12.0],
        ],
        dtype=complex,
    )

    np.testing.assert_allclose(out, expected)


def test_control_probabilities():
    psi = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=complex,
    ) / np.sqrt(2)

    p0, p1 = control_probabilities(psi)
    assert np.isclose(p0, 0.5)
    assert np.isclose(p1, 0.5)


def test_measure_control_deterministic_zero():
    psi = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    bit, collapsed = measure_control(psi, rng=np.random.default_rng(123))
    assert bit == 0
    np.testing.assert_allclose(collapsed, psi)


def test_measure_control_deterministic_one():
    psi = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=complex,
    )

    bit, collapsed = measure_control(psi, rng=np.random.default_rng(123))
    assert bit == 1
    np.testing.assert_allclose(collapsed, psi)