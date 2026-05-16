import math

import numpy as np

from shor_common.classical_post import shor_factors_from_order
from shor_common.errors import choose_measurement_bitflip_mask, validate_error_config
from shor_common.modular_unitaries import register_size, work_dim
from shor_common.shor_utils import (
    choose_random_a,
    default_max_attempts,
    default_num_phase_bits,
    is_even,
    is_trivial_case,
    make_basis_state,
    make_controlled_power_factory,
    perfect_power_decomposition,
    recover_order_with_mode,
    validate_postprocess_mode,
)

from .analytic_sampling import (
    multiplicative_order,
    sample_standard_phase_bits_from_order,
    validate_analytic_error_config,
)
from .backends import (
    STATEVECTOR_STANDARD_BACKEND,
    validate_standard_backend,
)
from .phase_estimation import standard_phase_estimation


def default_num_control_qubits(N: int) -> int:
    """
    Default number t of control qubits for standard phase estimation.
    """
    return default_num_phase_bits(N)


def shor_order_finding_attempt(
    N: int,
    a: int,
    num_control_qubits: int,
    rng=None,
    postprocess_mode: str = "one_shot",
    error_config=None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> dict:
    """
    Run one textbook phase-estimation attempt for U_a and recover the order.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)
    standard_backend = validate_standard_backend(standard_backend)
    error_config = validate_error_config(error_config)
    if rng is None:
        rng = np.random.default_rng()

    result_metadata = {}
    if standard_backend == STATEVECTOR_STANDARD_BACKEND:
        M = work_dim(N)
        initial_work_state = make_basis_state(1, M)
        controlled_power_factory = make_controlled_power_factory(a, N)

        bits, phase_estimate, final_work_state = standard_phase_estimation(
            initial_work_state=initial_work_state,
            controlled_power_factory=controlled_power_factory,
            num_control_qubits=num_control_qubits,
            rng=rng,
            error_config=error_config,
        )
    else:
        error_config = validate_analytic_error_config(error_config)
        true_order = multiplicative_order(a, N)
        measurement_bitflip_mask = choose_measurement_bitflip_mask(
            num_control_qubits,
            error_config=error_config,
            rng=rng,
        )
        bits, phase_estimate, sampled_integer, eigenphase_numerator = (
            sample_standard_phase_bits_from_order(
                order=true_order,
                num_control_qubits=num_control_qubits,
                rng=rng,
                window=analytic_window,
                measurement_bitflip_mask=measurement_bitflip_mask,
            )
        )
        final_work_state = None
        result_metadata = {
            "analytic_window": analytic_window,
            "sampling_order": true_order,
            "sampled_integer": sampled_integer,
            "sampled_eigenphase_numerator": eigenphase_numerator,
        }

    order = recover_order_with_mode(bits, a, N, postprocess_mode=postprocess_mode)

    result = {
        "algorithm": "standard",
        "standard_backend": standard_backend,
        "a": a,
        "num_control_qubits": num_control_qubits,
        "num_work_qubits": register_size(N),
        "bits": bits,
        "phase_estimate": phase_estimate,
        "order": order,
        "postprocess_mode": postprocess_mode,
        "error_config": error_config.to_dict(),
        "final_work_state": final_work_state,
    }
    result.update(result_metadata)
    return result


def shor_attempt(
    N: int,
    a: int | None = None,
    num_control_qubits: int | None = None,
    rng=None,
    postprocess_mode: str = "one_shot",
    error_config=None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> dict:
    """
    Perform one top-level standard Shor attempt.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)
    standard_backend = validate_standard_backend(standard_backend)
    error_config = validate_error_config(error_config)

    if is_trivial_case(N):
        raise ValueError("N must be >= 2")

    if is_even(N):
        return {
            "algorithm": "standard",
            "standard_backend": standard_backend,
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
            "algorithm": "standard",
            "standard_backend": standard_backend,
            "N": N,
            "postprocess_mode": postprocess_mode,
            "error_config": error_config.to_dict(),
            "success": True,
            "reason": f"N is a perfect power: {b}^{k}",
            "factors": (b, N // b),
        }

    if rng is None:
        rng = np.random.default_rng()

    if num_control_qubits is None:
        num_control_qubits = default_num_control_qubits(N)

    if a is None:
        a = choose_random_a(N, rng=rng)
    else:
        if not (2 <= a <= N - 2):
            raise ValueError("a must satisfy 2 <= a <= N - 2")

    g = math.gcd(a, N)
    if g > 1:
        return {
            "algorithm": "standard",
            "standard_backend": standard_backend,
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
        num_control_qubits=num_control_qubits,
        rng=rng,
        postprocess_mode=postprocess_mode,
        error_config=error_config,
        standard_backend=standard_backend,
        analytic_window=analytic_window,
    )

    r = order_result["order"]
    factors = shor_factors_from_order(a, r, N)
    success = factors is not None

    result = {
        "algorithm": "standard",
        "standard_backend": standard_backend,
        "N": N,
        "a": a,
        "num_control_qubits": num_control_qubits,
        "num_work_qubits": register_size(N),
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
    if standard_backend != STATEVECTOR_STANDARD_BACKEND:
        result.update(
            {
                "analytic_window": order_result["analytic_window"],
                "sampling_order": order_result["sampling_order"],
                "sampled_integer": order_result["sampled_integer"],
                "sampled_eigenphase_numerator": order_result[
                    "sampled_eigenphase_numerator"
                ],
            }
        )
    return result


def shor_factor(
    N: int,
    max_attempts: int | None = None,
    num_control_qubits: int | None = None,
    rng=None,
    postprocess_mode: str = "one_shot",
    error_config=None,
    standard_backend: str = STATEVECTOR_STANDARD_BACKEND,
    analytic_window: int = 100,
) -> dict:
    """
    Repeat standard Shor attempts until factors are found or attempts run out.
    """
    postprocess_mode = validate_postprocess_mode(postprocess_mode)
    standard_backend = validate_standard_backend(standard_backend)
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
            num_control_qubits=num_control_qubits,
            rng=rng,
            postprocess_mode=postprocess_mode,
            error_config=error_config,
            standard_backend=standard_backend,
            analytic_window=analytic_window,
        )

        if result["success"]:
            return result

        last_result = result

    return {
        "algorithm": "standard",
        "standard_backend": standard_backend,
        "N": N,
        "postprocess_mode": postprocess_mode,
        "error_config": error_config.to_dict(),
        "success": False,
        "reason": "all attempts failed",
        "last_result": last_result,
    }
