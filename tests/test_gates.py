import numpy as np

from shor_one_qubit.gates import I2, H, Z, phase, R_k, R_k_dagger


def test_identity_gate():
    np.testing.assert_allclose(I2(), np.eye(2, dtype=complex))


def test_hadamard_squared_is_identity():
    np.testing.assert_allclose(H() @ H(), np.eye(2, dtype=complex))


def test_pauli_z_squared_is_identity():
    np.testing.assert_allclose(Z() @ Z(), np.eye(2, dtype=complex))


def test_phase_gate():
    theta = np.pi / 3
    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, np.exp(1j * theta)],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(phase(theta), expected)


def test_rk_matches_definition():
    k = 3
    expected = np.array(
        [
            [1.0, 0.0],
            [0.0, np.exp(2j * np.pi / (2**k))],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(R_k(k), expected)


def test_rk_dagger_is_conjugate_transpose():
    k = 4
    np.testing.assert_allclose(R_k_dagger(k), R_k(k).conj().T)


def test_r1_is_z():
    np.testing.assert_allclose(R_k(1), Z())