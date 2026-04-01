import math
import numpy as np

from .modular_unitaries import work_dim, apply_modmul_power_state
from .kitaev_pe import iterative_phase_estimation
from .classical_post import (
    recover_order_from_bits,
    recover_order_from_bits_one_shot,
    shor_factors_from_order,
)


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

    A simple standard choice is twice the work-register size.
    """
    L = math.ceil(math.log2(N))
    return 2 * L


def make_controlled_power_factory(a: int, N: int):
    """
    Return a factory such that factory(power) gives a callable implementing
    U_a^power on the work-register state.
    """
    def factory(power: int):
        def work_op(work_state: np.ndarray) -> np.ndarray:
            return apply_modmul_power_state(work_state, a=a, power=power, N=N)
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


def recover_order_with_mode(bits: list[int], a: int, N: int, postprocess_mode: str) -> int | None:
    """
    Recover the order using the requested classical post-processing mode.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)

    if postprocess_mode == "one_shot":
        return recover_order_from_bits_one_shot(bits, a, N)

    return recover_order_from_bits(bits, a, N)


def shor_order_finding_attempt(
    N: int,
    a: int,
    num_phase_bits: int,
    rng=None,
    postprocess_mode: str = "one_shot",
) -> dict:
    """
    Run one iterative phase-estimation attempt for the modular multiplication
    unitary U_a and try to recover the order of a mod N.

    Returns a dictionary containing the measured bits, phase estimate,
    recovered order, and final work-register state.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)

    M = work_dim(N)
    initial_work_state = make_basis_state(1, M)

    controlled_power_factory = make_controlled_power_factory(a, N)

    bits, phase_estimate, final_work_state = iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=num_phase_bits,
        rng=rng,
    )

    order = recover_order_with_mode(bits, a, N, postprocess_mode=postprocess_mode)

    return {
        "a": a,
        "bits": bits,
        "phase_estimate": phase_estimate,
        "order": order,
        "postprocess_mode": postprocess_mode,
        "final_work_state": final_work_state,
    }


def shor_attempt(
    N: int,
    a: int | None = None,
    num_phase_bits: int | None = None,
    rng=None,
    postprocess_mode: str = "one_shot",
) -> dict:
    """
    Perform one top-level Shor attempt.

    Logic:
      - handle easy classical cases,
      - choose or validate a,
      - if gcd(a, N) > 1, we already found a factor,
      - otherwise run order finding,
      - then try to extract factors from the recovered order.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)

    if is_trivial_case(N):
        raise ValueError("N must be >= 2")

    if is_even(N):
        return {
            "N": N,
            "postprocess_mode": postprocess_mode,
            "success": True,
            "reason": "N is even",
            "factors": (2, N // 2),
        }
    
    pp = perfect_power_decomposition(N)
    if pp is not None:
        b, k = pp
        return {
            "N": N,
            "postprocess_mode": postprocess_mode,
            "success": True,
            "reason": f"N is a perfect power: {b}^{k}",
            "factors": (b, N // b),
        }

    if rng is None:
        rng = np.random.default_rng()

    if num_phase_bits is None:
        num_phase_bits = default_num_phase_bits(N)

    if a is None:
        a = choose_random_a(N, rng=rng)
    else:
        if not (2 <= a <= N - 2):
            raise ValueError("a must satisfy 2 <= a <= N - 2")

    g = math.gcd(a, N)
    if g > 1:
        return {
            "N": N,
            "a": a,
            "postprocess_mode": postprocess_mode,
            "success": True,
            "reason": "nontrivial gcd found before quantum step",
            "factors": (g, N // g),
        }

    order_result = shor_order_finding_attempt(
        N=N,
        a=a,
        num_phase_bits=num_phase_bits,
        rng=rng,
        postprocess_mode=postprocess_mode,
    )

    r = order_result["order"]
    factors = shor_factors_from_order(a, r, N)

    success = factors is not None

    return {
        "N": N,
        "a": a,
        "num_phase_bits": num_phase_bits,
        "postprocess_mode": postprocess_mode,
        "bits": order_result["bits"],
        "phase_estimate": order_result["phase_estimate"],
        "order": r,
        "factors": factors,
        "success": success,
        "reason": "success" if success else "order/factor extraction failed",
        "final_work_state": order_result["final_work_state"],
    }


def shor_factor(
    N: int,
    max_attempts: int | None = None,
    num_phase_bits: int | None = None,
    rng=None,
    postprocess_mode: str = "one_shot",
) -> dict:
    """
    Repeat Shor attempts until factors are found or the attempt budget is exhausted.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)
    if max_attempts is None:
        max_attempts = default_max_attempts(postprocess_mode)

    if rng is None:
        rng = np.random.default_rng()

    last_result = None

    for _ in range(max_attempts):
        result = shor_attempt(
            N=N,
            a=None,
            num_phase_bits=num_phase_bits,
            rng=rng,
            postprocess_mode=postprocess_mode,
        )

        if result["success"]:
            return result

        last_result = result

    return {
        "N": N,
        "postprocess_mode": postprocess_mode,
        "success": False,
        "reason": "all attempts failed",
        "last_result": last_result,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(1234)

    N = 15
    result = shor_factor(N, max_attempts=10, rng=rng)

    print("Result:")
    for key, value in result.items():
        if key == "final_work_state":
            print(f"{key}: <state vector omitted>")
        else:
            print(f"{key}: {value}")
