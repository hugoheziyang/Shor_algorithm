"""Shared utilities for the Shor simulators."""

from .errors import ErrorConfig
from .modular_unitaries import register_size, work_dim
from .shor_utils import default_num_phase_bits

__all__ = [
    "ErrorConfig",
    "default_num_phase_bits",
    "register_size",
    "work_dim",
]
