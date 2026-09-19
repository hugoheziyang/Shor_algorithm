from __future__ import annotations

import argparse
from collections.abc import Iterable

from shor_common.modular_unitaries import register_size, validate_work_register_size


def _extract_N_values(instances_or_N_values: Iterable[object]) -> list[int]:
    N_values = [
        int(getattr(item, "N", item))
        for item in instances_or_N_values
    ]
    if not N_values:
        raise ValueError("at least one modulus N is required")
    return N_values


def max_required_work_qubits(instances_or_N_values: Iterable[object]) -> int:
    """
    Return the smallest common work-register size covering every supplied N.
    """
    return max(register_size(N) for N in _extract_N_values(instances_or_N_values))


def resolve_work_qubits(
    instances_or_N_values: Iterable[object],
    requested_work_qubits: int | None = None,
    fixed_work_register: bool = False,
) -> int | None:
    """
    Resolve the work-register mode for a depolarizing-error sweep.

    Returning None means each instance uses its own minimal ceil(log2 N) work
    register. Returning an integer means every instance uses that common L.
    """
    N_values = _extract_N_values(instances_or_N_values)
    if requested_work_qubits is None and not fixed_work_register:
        return None

    num_work_qubits = (
        max(register_size(N) for N in N_values)
        if requested_work_qubits is None
        else int(requested_work_qubits)
    )
    for N in N_values:
        validate_work_register_size(N, num_work_qubits)
    return num_work_qubits


def effective_work_qubits_for_N(
    N: int,
    num_work_qubits: int | None = None,
) -> int:
    return validate_work_register_size(N, num_work_qubits)


def work_register_display_label(num_work_qubits: int | None) -> str:
    if num_work_qubits is None:
        return "minimal L = ceil(log2 N) per instance"
    return f"fixed L = {num_work_qubits} work qubits"


def work_register_filename_segment(num_work_qubits: int | None) -> str:
    if num_work_qubits is None:
        return ""
    return f"_fixed_work_L{num_work_qubits}"


def add_work_register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-qubits",
        type=int,
        default=None,
        help=(
            "Use a fixed work-register size L for every instance. "
            "L must satisfy 2^L >= N for all analyzed moduli."
        ),
    )
    parser.add_argument(
        "--fixed-work-register",
        action="store_true",
        help=(
            "Use one common work register sized by the largest analyzed N. "
            "Ignored when --work-qubits is supplied."
        ),
    )
