from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

from shor_common.modular_unitaries import validate_work_register_size
from shor_common.shor_utils import default_num_phase_bits
from shor_standard.backends import STATEVECTOR_STANDARD_BACKEND


STANDARD_ALGORITHM = "standard"


def standard_statevector_allows_subdefault_phase_bits(
    algorithms: Iterable[str],
    standard_backend: str,
) -> bool:
    """
    Return True when standard Shor is selected with the explicit statevector backend.
    """
    return (
        STANDARD_ALGORITHM in set(algorithms)
        and standard_backend == STATEVECTOR_STANDARD_BACKEND
    )


def _N_from_item(item: Any) -> int:
    return int(getattr(item, "N", item))


def warn_if_statevector_phase_bits_below_default(
    algorithms: Iterable[str],
    standard_backend: str,
    num_phase_bits: int,
    instances_or_N_values: Iterable[Any],
    num_work_qubits: int | None = None,
    stacklevel: int = 2,
) -> list[tuple[int, int]]:
    """
    Warn, without failing, when standard statevector Shor uses t < 2L+1.

    Returns the affected (N, default_t) pairs for tests and callers that want
    to include the warning status in summaries.
    """
    if not standard_statevector_allows_subdefault_phase_bits(
        algorithms,
        standard_backend,
    ):
        return []

    affected_by_N: dict[int, int] = {}
    for item in instances_or_N_values:
        N = _N_from_item(item)
        default_t = (
            default_num_phase_bits(N)
            if num_work_qubits is None
            else (2 * validate_work_register_size(N, num_work_qubits)) + 1
        )
        if num_phase_bits < default_t:
            affected_by_N[N] = default_t

    affected = sorted(affected_by_N.items())
    if not affected:
        return []

    preview = ", ".join(
        f"N={N} requires t>={default_t}"
        for N, default_t in affected[:5]
    )
    if len(affected) > 5:
        preview += f", and {len(affected) - 5} more"

    L_description = (
        f"fixed work-register L={num_work_qubits}"
        if num_work_qubits is not None
        else "L=ceil(log2(N))"
    )
    warnings.warn(
        "standard statevector backend is running with phase-estimation bits "
        f"t={num_phase_bits} below the default 2L+1 ({L_description}) "
        f"for {preview}. "
        "This is allowed for statevector simulations, but order-recovery "
        "success probabilities may be below the textbook-precision setting.",
        RuntimeWarning,
        stacklevel=stacklevel,
    )
    return affected
