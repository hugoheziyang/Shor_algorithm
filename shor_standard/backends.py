STATEVECTOR_STANDARD_BACKEND = "statevector"
ANALYTIC_STANDARD_BACKEND = "analytic"

STANDARD_BACKEND_CHOICES = (
    STATEVECTOR_STANDARD_BACKEND,
    ANALYTIC_STANDARD_BACKEND,
)


def validate_standard_backend(standard_backend: str) -> str:
    """
    Validate the backend used by the standard Shor simulator.
    """
    if standard_backend not in STANDARD_BACKEND_CHOICES:
        choices = ", ".join(STANDARD_BACKEND_CHOICES)
        raise ValueError(f"standard_backend must be one of: {choices}")

    return standard_backend

