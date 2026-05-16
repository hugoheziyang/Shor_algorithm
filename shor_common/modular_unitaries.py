from functools import lru_cache
from math import gcd

import numpy as np


def register_size(N: int) -> int:
    """Number of qubits needed to store 0, ..., N-1."""
    if N < 2:
        raise ValueError("N must be >= 2")
    return int(np.ceil(np.log2(N)))


def work_dim(N: int) -> int:
    """Hilbert-space dimension of the work register."""
    L = register_size(N)
    return 2 ** L


def validate_multiplier(a: int, N: int) -> None:
    """Require gcd(a, N) = 1 so modular multiplication is a permutation."""
    if not (1 <= a < N):
        raise ValueError("a must satisfy 1 <= a < N")
    if gcd(a, N) != 1:
        raise ValueError("a and N must be coprime")


def modmul_label(y: int, a: int, N: int) -> int:
    """
    Basis-state action:
        |y> -> |a y mod N>   for y < N
        |y> -> |y>           for y >= N  (unused computational states to complete unitary)
    """
    M = work_dim(N)
    if not (0 <= y < M):
        raise ValueError("basis label y out of range")

    if y < N:
        return (a * y) % N
    return y


def modmul_power_label(y: int, a: int, power: int, N: int) -> int:
    """
    Basis-state action of U_a^power.
    Since U_a^power = U_{a^power mod N},
    we compute a_eff = a^power mod N first.
    """
    if power < 0:
        raise ValueError("power must be >= 0")
    a_eff = pow(a, power, N)
    return modmul_label(y, a_eff, N)


@lru_cache(maxsize=128)
def modmul_permutation(a: int, N: int) -> np.ndarray:
    """
    Return the permutation array implementing U_a on work-register labels.

    The returned array `perm` satisfies out[perm[y]] = psi[y].
    """
    validate_multiplier(a, N)
    M = work_dim(N)

    perm = np.arange(M, dtype=np.int64)
    active = perm < N
    perm[active] = (a * perm[active]) % N
    perm.setflags(write=False)
    return perm


def apply_modmul_state(psi: np.ndarray, a: int, N: int) -> np.ndarray:
    """
    Apply U_a to a work-register state vector psi.

    psi has shape (M,), where M = 2^ceil(log2 N).

    |psi> = sum_y psi[y] |y> --> U_a |psi> = sum_y psi[y] |a y mod N>
    """
    perm = modmul_permutation(a, N)
    M = perm.shape[0]

    psi = np.asarray(psi, dtype=complex)
    if psi.shape != (M,):
        raise ValueError(f"psi must have shape ({M},)")

    out = np.empty_like(psi)
    out[perm] = psi
    return out


def apply_modmul_power_state(psi: np.ndarray, a: int, power: int, N: int) -> np.ndarray:
    """
    Apply U_a^power to a work-register state vector psi.
    """
    if power < 0:
        raise ValueError("power must be >= 0")
    a_eff = pow(a, power, N)
    return apply_modmul_state(psi, a_eff, N)
