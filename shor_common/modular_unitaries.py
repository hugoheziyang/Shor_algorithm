from functools import lru_cache
from math import gcd

import numpy as np


def register_size(N: int) -> int:
    """Number of qubits needed to store 0, ..., N-1."""
    if N < 2:
        raise ValueError("N must be >= 2")
    return int(np.ceil(np.log2(N)))


def validate_work_register_size(N: int, num_work_qubits: int | None = None) -> int:
    """
    Return a valid work-register qubit count for storing labels 0, ..., N-1.

    If `num_work_qubits` is omitted, use the minimal register size ceil(log2 N).
    A supplied value may pad the work register, but it cannot be smaller than
    the minimal size needed to represent N.
    """
    minimum_num_work_qubits = register_size(N)
    if num_work_qubits is None:
        return minimum_num_work_qubits

    effective_num_work_qubits = int(num_work_qubits)
    if effective_num_work_qubits < minimum_num_work_qubits:
        raise ValueError(
            "num_work_qubits must be at least ceil(log2 N) = "
            f"{minimum_num_work_qubits} for N = {N}"
        )
    return effective_num_work_qubits


def work_dim(N: int, num_work_qubits: int | None = None) -> int:
    """Hilbert-space dimension of the work register."""
    L = validate_work_register_size(N, num_work_qubits)
    return 2 ** L


def validate_multiplier(a: int, N: int) -> None:
    """Require gcd(a, N) = 1 so modular multiplication is a permutation."""
    if not (1 <= a < N):
        raise ValueError("a must satisfy 1 <= a < N")
    if gcd(a, N) != 1:
        raise ValueError("a and N must be coprime")


def validate_padding_phase_radians(padding_phase_radians: float) -> float:
    """Return a finite phase used to extend modular multiplication for y >= N."""
    padding_phase_radians = float(padding_phase_radians)
    if not np.isfinite(padding_phase_radians):
        raise ValueError("padding_phase_radians must be finite")
    return padding_phase_radians


def modmul_label(
    y: int,
    a: int,
    N: int,
    num_work_qubits: int | None = None,
) -> int:
    """
    Basis-state action:
        |y> -> |a y mod N>   for y < N
        |y> -> |y>           for y >= N  (unused computational states to complete unitary)
    """
    M = work_dim(N, num_work_qubits=num_work_qubits)
    if not (0 <= y < M):
        raise ValueError("basis label y out of range")

    if y < N:
        return (a * y) % N
    return y


def modmul_power_label(
    y: int,
    a: int,
    power: int,
    N: int,
    num_work_qubits: int | None = None,
) -> int:
    """
    Basis-state action of U_a^power.
    Since U_a^power = U_{a^power mod N},
    we compute a_eff = a^power mod N first.
    """
    if power < 0:
        raise ValueError("power must be >= 0")
    a_eff = pow(a, power, N)
    return modmul_label(y, a_eff, N, num_work_qubits=num_work_qubits)


@lru_cache(maxsize=128)
def modmul_permutation(
    a: int,
    N: int,
    num_work_qubits: int | None = None,
) -> np.ndarray:
    """
    Return the permutation array implementing U_a on work-register labels.

    The returned array `perm` satisfies out[perm[y]] = psi[y].
    """
    validate_multiplier(a, N)
    M = work_dim(N, num_work_qubits=num_work_qubits)

    perm = np.arange(M, dtype=np.int64)
    active = perm < N
    perm[active] = (a * perm[active]) % N
    perm.setflags(write=False)
    return perm


def apply_modmul_state(
    psi: np.ndarray,
    a: int,
    N: int,
    num_work_qubits: int | None = None,
    padding_phase_radians: float = 0.0,
) -> np.ndarray:
    """
    Apply U_a to a work-register state vector psi.

    psi has shape (M,), where M = 2^L and L is either ceil(log2 N) or the
    supplied padded work-register size.

    For y < N, U_a performs modular multiplication. For y >= N, it applies
    exp(i * padding_phase_radians) without changing the basis label. The
    default phase zero recovers the conventional identity extension.
    """
    perm = modmul_permutation(a, N, num_work_qubits=num_work_qubits)
    M = perm.shape[0]

    psi = np.asarray(psi, dtype=complex)
    if psi.shape != (M,):
        raise ValueError(f"psi must have shape ({M},)")
    padding_phase_radians = validate_padding_phase_radians(
        padding_phase_radians
    )

    out = np.empty_like(psi)
    out[perm] = psi
    if padding_phase_radians != 0.0 and N < M:
        out[N:] *= np.exp(1j * np.remainder(padding_phase_radians, 2.0 * np.pi))
    return out


def apply_modmul_power_state(
    psi: np.ndarray,
    a: int,
    power: int,
    N: int,
    num_work_qubits: int | None = None,
    padding_phase_radians: float = 0.0,
) -> np.ndarray:
    """
    Apply U_a^power to a work-register state vector psi.

    If padded states have base-unitary eigenvalue exp(i * theta), U_a^power
    applies exp(i * power * theta) to those states.
    """
    if power < 0:
        raise ValueError("power must be >= 0")
    padding_phase_radians = validate_padding_phase_radians(
        padding_phase_radians
    )
    a_eff = pow(a, power, N)
    return apply_modmul_state(
        psi,
        a_eff,
        N,
        num_work_qubits=num_work_qubits,
        padding_phase_radians=power * padding_phase_radians,
    )
