import math
import numpy as np

from shor_common.classical_post import shor_factors_from_order
from shor_common.errors import validate_error_config
from shor_common.modular_unitaries import work_dim
from shor_common.shor_utils import (
    VALID_POSTPROCESS_MODES,
    choose_random_a,
    default_max_attempts,
    default_num_phase_bits,
    integer_nth_root,
    is_even,
    is_trivial_case,
    make_basis_state,
    make_controlled_power_factory,
    perfect_power_decomposition,
    recover_order_with_mode,
    validate_postprocess_mode,
)

from .kitaev_pe import iterative_phase_estimation


def shor_order_finding_attempt(
    N: int,
    a: int,
    num_phase_bits: int,
    rng=None,
    postprocess_mode: str = "one_shot",
    error_config=None,
) -> dict:
    """
    Run one iterative phase-estimation attempt for the modular multiplication
    unitary U_a and try to recover the order of a mod N.

    Returns a dictionary containing the measured bits, phase estimate,
    recovered order, and final work-register state.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)
    error_config = validate_error_config(error_config)

    M = work_dim(N)
    initial_work_state = make_basis_state(1, M)

    controlled_power_factory = make_controlled_power_factory(a, N)

    bits, phase_estimate, final_work_state = iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=num_phase_bits,
        rng=rng,
        error_config=error_config,
    )

    order = recover_order_with_mode(bits, a, N, postprocess_mode=postprocess_mode)

    return {
        "a": a,
        "bits": bits,
        "phase_estimate": phase_estimate,
        "order": order,
        "postprocess_mode": postprocess_mode,
        "error_config": error_config.to_dict(),
        "final_work_state": final_work_state,
    }


def shor_attempt(
    N: int,
    a: int | None = None,
    num_phase_bits: int | None = None,
    rng=None,
    postprocess_mode: str = "one_shot",
    error_config=None,
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
    error_config = validate_error_config(error_config)

    if is_trivial_case(N):
        raise ValueError("N must be >= 2")

    if is_even(N):
        return {
            "N": N,
            "postprocess_mode": postprocess_mode,
            "error_config": error_config.to_dict(),
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
            "error_config": error_config.to_dict(),
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
            "error_config": error_config.to_dict(),
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
        error_config=error_config,
    )

    r = order_result["order"]
    factors = shor_factors_from_order(a, r, N)

    success = factors is not None

    return {
        "N": N,
        "a": a,
        "num_phase_bits": num_phase_bits,
        "postprocess_mode": postprocess_mode,
        "error_config": error_config.to_dict(),
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
    error_config=None,
) -> dict:
    """
    Repeat Shor attempts until factors are found or the attempt budget is exhausted.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)
    error_config = validate_error_config(error_config)
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
            error_config=error_config,
        )

        if result["success"]:
            return result

        last_result = result

    return {
        "N": N,
        "postprocess_mode": postprocess_mode,
        "error_config": error_config.to_dict(),
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
