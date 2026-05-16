import numpy as np


def _control_dim(num_qubits: int) -> int:
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    return 2 ** num_qubits


def qft_matrix(num_qubits: int) -> np.ndarray:
    """
    Return the QFT matrix for a register with `num_qubits` qubits.

    The convention is

        |x> -> 1/sqrt(Q) sum_y exp(2*pi*i*x*y/Q) |y>,

    where Q = 2**num_qubits and integer labels use the usual big-endian
    binary representation.
    """
    Q = _control_dim(num_qubits)
    labels = np.arange(Q)
    exponents = np.outer(labels, labels)
    return np.exp(2j * np.pi * exponents / Q) / np.sqrt(Q)


def inverse_qft_matrix(num_qubits: int) -> np.ndarray:
    """
    Return the inverse-QFT matrix for a register with `num_qubits` qubits.
    """
    return qft_matrix(num_qubits).conj().T


def _validate_control_state(psi: np.ndarray, num_control_qubits: int) -> np.ndarray:
    Q = _control_dim(num_control_qubits)
    psi = np.asarray(psi, dtype=complex)

    if psi.ndim < 1 or psi.shape[0] != Q:
        raise ValueError(f"psi must have first dimension {Q}")

    return psi


def apply_qft_to_control(psi: np.ndarray, num_control_qubits: int) -> np.ndarray:
    """
    Apply the QFT to the first axis of a state array.

    For Shor, the state has shape (2**t, 2**L), with the first axis being the
    control register and the second axis being the work register.
    """
    psi = _validate_control_state(psi, num_control_qubits)
    Q = psi.shape[0]
    return np.fft.ifft(psi, axis=0) * np.sqrt(Q)


def apply_inverse_qft_to_control(psi: np.ndarray, num_control_qubits: int) -> np.ndarray:
    """
    Apply the inverse QFT to the first axis of a state array.
    """
    psi = _validate_control_state(psi, num_control_qubits)
    Q = psi.shape[0]
    return np.fft.fft(psi, axis=0) / np.sqrt(Q)
