import numpy as np
import pytest

from shor_standard.qft import (
    apply_inverse_qft_to_control,
    apply_qft_to_control,
    inverse_qft_matrix,
    qft_matrix,
)


def test_qft_matrix_is_unitary():
    qft = qft_matrix(3)
    np.testing.assert_allclose(qft.conj().T @ qft, np.eye(8), atol=1e-12)


def test_inverse_qft_matrix_undoes_qft_matrix():
    qft = qft_matrix(2)
    inv_qft = inverse_qft_matrix(2)

    np.testing.assert_allclose(inv_qft @ qft, np.eye(4), atol=1e-12)


def test_apply_qft_and_inverse_qft_round_trip_control_axis():
    psi = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 1.0 + 0.0j],
            [0.5 + 0.0j, 0.0 - 0.5j],
            [0.25 + 0.0j, 0.25 + 0.0j],
        ],
        dtype=complex,
    )

    out = apply_inverse_qft_to_control(apply_qft_to_control(psi, 2), 2)
    np.testing.assert_allclose(out, psi, atol=1e-12)


def test_inverse_qft_maps_exact_phase_state_to_integer_label():
    num_control_qubits = 3
    Q = 2 ** num_control_qubits
    target_label = 3
    labels = np.arange(Q)
    phase_state = np.exp(2j * np.pi * labels * target_label / Q) / np.sqrt(Q)

    out = apply_inverse_qft_to_control(phase_state, num_control_qubits)

    expected = np.zeros(Q, dtype=complex)
    expected[target_label] = 1.0
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_qft_rejects_nonpositive_qubit_count():
    with pytest.raises(ValueError, match="num_qubits"):
        qft_matrix(0)


def test_apply_inverse_qft_rejects_wrong_control_dimension():
    with pytest.raises(ValueError, match="first dimension"):
        apply_inverse_qft_to_control(np.zeros((3, 2), dtype=complex), 2)
