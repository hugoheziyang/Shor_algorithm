import math

import numpy as np

from .classical_post import (
    recover_order_from_bits,
    recover_order_from_bits_one_shot,
)
from .modular_unitaries import apply_modmul_power_state


VALID_POSTPROCESS_MODES = ("one_shot", "rescue")


def is_trivial_case(N: int) -> bool:
    """
    Return True for values of N where running Shor's algorithm is not meaningful.
    """
    return N < 2


def is_even(N: int) -> bool:
    """
    Return True if N is even.
    """
    return (N % 2) == 0


def integer_nth_root(n: int, k: int) -> int:
    """
    Return floor(n^(1/k)) for integers n >= 0, k >= 1.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if k < 1:
        raise ValueError("k must be >= 1")
    if n in (0, 1):
        return n
    if k == 1:
        return n

    lo, hi = 1, n
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid ** k
        if p == n:
            return mid
        if p < n:
            lo = mid + 1
        else:
            hi = mid - 1

    return hi


def perfect_power_decomposition(N: int):
    """
    If N = b^k with integers b > 1 and k > 1, return (b, k).
    Otherwise return None.
    """
    if N < 2:
        return None

    max_k = N.bit_length() - 1

    for k in range(max_k, 1, -1):
        b = integer_nth_root(N, k)
        if b > 1 and b ** k == N:
            return b, k

    return None


def choose_random_a(N: int, rng=None) -> int:
    """
    Choose a random integer a with 2 <= a <= N-2 and gcd(a, N) = 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    while True:
        a = int(rng.integers(2, N - 1))
        if math.gcd(a, N) == 1:
            return a


def make_basis_state(index: int, dim: int) -> np.ndarray:
    """
    Return the state vector for the computational basis state |index>.
    `dim` is the Hilbert-space dimension, and index must satisfy 0 <= index < dim.
    """
    if not (0 <= index < dim):
        raise ValueError("basis index out of range")

    amplitudes = np.zeros(dim, dtype=complex)
    amplitudes[index] = 1.0 + 0.0j
    return amplitudes


def default_num_phase_bits(N: int) -> int:
    """
    Default number of phase-estimation bits.

    A conservative standard choice is twice the work-register size plus one.
    """
    L = math.ceil(math.log2(N))
    return (2 * L) + 1


def make_controlled_power_factory(
    a: int,
    N: int,
    num_work_qubits: int | None = None,
    padding_phase_radians: float = 0.0,
):
    """
    Return a factory such that factory(power) gives a callable implementing
    U_a^power on the work-register state.
    """
    def factory(power: int):
        def work_op(work_state: np.ndarray) -> np.ndarray:
            return apply_modmul_power_state(
                work_state,
                a=a,
                power=power,
                N=N,
                num_work_qubits=num_work_qubits,
                padding_phase_radians=padding_phase_radians,
            )
        return work_op

    return factory


def validate_postprocess_mode(postprocess_mode: str) -> str:
    """
    Validate the classical post-processing mode used after phase estimation.
    """
    if postprocess_mode not in VALID_POSTPROCESS_MODES:
        choices = ", ".join(VALID_POSTPROCESS_MODES)
        raise ValueError(f"postprocess_mode must be one of: {choices}")

    return postprocess_mode


def default_max_attempts(postprocess_mode: str) -> int:
    """
    Return the default retry budget for the chosen post-processing mode.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)

    if postprocess_mode == "one_shot":
        return 1

    return 10


def recover_order_with_mode(
    bits: list[int],
    a: int,
    N: int,
    postprocess_mode: str,
) -> int | None:
    """
    Recover the order using the requested classical post-processing mode.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)

    if postprocess_mode == "one_shot":
        return recover_order_from_bits_one_shot(bits, a, N)

    return recover_order_from_bits(bits, a, N)
