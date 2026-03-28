import numpy as np

SQRT2_INV = 1.0 / np.sqrt(2.0)


def I2() -> np.ndarray:
    """2x2 identity."""
    return np.eye(2, dtype=complex)


def H() -> np.ndarray:
    """Hadamard gate."""
    return SQRT2_INV * np.array([
        [1.0,  1.0],
        [1.0, -1.0]
    ], dtype=complex)


def Z() -> np.ndarray:
    """Pauli-Z gate."""
    return np.array([
        [1.0,  0.0],
        [0.0, -1.0]
    ], dtype=complex)


def phase(theta: float) -> np.ndarray:
    """
    General phase gate:
        diag(1, e^{i theta})

    This is the most useful primitive for semiclassical/Kitaev feedback.
    """
    return np.array([
        [1.0, 0.0],
        [0.0, np.exp(1j * theta)]
    ], dtype=complex)


def R_k(k: int) -> np.ndarray:
    """
    QFT rotation gate:
        diag(1, exp(2πi / 2^k))

    Example:
        k=1 -> Z
        k=2 -> S
        k=3 -> T
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    return phase(2.0 * np.pi / (2 ** k))


def R_k_dagger(k: int) -> np.ndarray:
    """
    Inverse-QFT rotation gate:
        diag(1, exp(-2πi / 2^k))
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    return phase(-2.0 * np.pi / (2 ** k))