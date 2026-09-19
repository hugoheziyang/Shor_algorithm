from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MEASUREMENT_BITFLIP_SINGLE_TOKEN = "single"
CONTROLLED_UNITARY_DEPOLARIZING_ALL_RANGE = "all"
CONTROLLED_UNITARY_DEPOLARIZING_LEADING_RANGE = "leading"
CONTROLLED_UNITARY_DEPOLARIZING_DANGEROUS_RANGE = "dangerous"
CONTROLLED_UNITARY_DEPOLARIZING_SAFE_TAIL_RANGE = "safe-tail"
CONTROLLED_UNITARY_DEPOLARIZING_RANGE_CHOICES = (
    CONTROLLED_UNITARY_DEPOLARIZING_ALL_RANGE,
    CONTROLLED_UNITARY_DEPOLARIZING_LEADING_RANGE,
    CONTROLLED_UNITARY_DEPOLARIZING_DANGEROUS_RANGE,
    CONTROLLED_UNITARY_DEPOLARIZING_SAFE_TAIL_RANGE,
)
ControlledUnitaryDepolarizingRangeSpec = (
    str | tuple[str, ...] | list[str] | None
)
CONTROLLED_UNITARY_DEPOLARIZING_PAULI_CHOICES = ("X", "Y", "Z")
ControlledUnitaryDepolarizingPauliSpec = str | tuple[str, ...] | list[str] | None


@dataclass(frozen=True)
class ErrorConfig:
    """
    Configuration for optional error models used by the simulator.

    For now, `measurement_bitflip=True` injects exactly one classical readout
    bit-flip into a uniformly random phase-estimation bit during each
    iterative phase-estimation run.

    If `measurement_bitflip_probability` is set to a value p in [0, 1], then
    each phase-estimation bit detection is independently flipped with
    probability p instead.

    If `controlled_unitary_depolarizing_probability` is set to a value p in
    [0, 1], then after each controlled-U block the simulator samples a
    trajectory error: with probability p, choose one work-register qubit
    uniformly and apply one of X, Y, Z uniformly to that qubit.

    If `controlled_unitary_per_qubit_depolarizing_probability` is set to a
    value p in [0, 1], then after each controlled-U block the simulator
    independently samples each work-register qubit and applies one of X, Y, Z
    uniformly to every selected qubit.

    If `controlled_unitary_depolarizing_allowed_phase_bits` is not None, then
    work-register depolarizing errors are sampled only after controlled-U
    blocks associated with those 1-indexed phase-bit numbers b_j.

    If `controlled_unitary_depolarizing_paulis` is not None, then each sampled
    depolarizing hit chooses uniformly from that nonempty subset of X, Y, Z.
    """

    measurement_bitflip: bool = False
    measurement_bitflip_probability: float | None = None
    controlled_unitary_depolarizing_probability: float | None = None
    controlled_unitary_per_qubit_depolarizing_probability: float | None = None
    controlled_unitary_depolarizing_allowed_phase_bits: tuple[int, ...] | None = None
    controlled_unitary_depolarizing_paulis: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        """
        Return a plain dictionary representation for result payloads.
        """
        payload = {"measurement_bitflip": self.measurement_bitflip}
        if self.measurement_bitflip_probability is not None:
            payload["measurement_bitflip_probability"] = self.measurement_bitflip_probability
        if self.controlled_unitary_depolarizing_probability is not None:
            payload[
                "controlled_unitary_depolarizing_probability"
            ] = self.controlled_unitary_depolarizing_probability
        if self.controlled_unitary_per_qubit_depolarizing_probability is not None:
            payload[
                "controlled_unitary_per_qubit_depolarizing_probability"
            ] = self.controlled_unitary_per_qubit_depolarizing_probability
        if self.controlled_unitary_depolarizing_allowed_phase_bits is not None:
            payload["controlled_unitary_depolarizing_allowed_phase_bits"] = list(
                self.controlled_unitary_depolarizing_allowed_phase_bits
            )
        if self.controlled_unitary_depolarizing_paulis is not None:
            payload["controlled_unitary_depolarizing_paulis"] = list(
                self.controlled_unitary_depolarizing_paulis
            )
        return payload


def normalize_controlled_unitary_depolarizing_allowed_phase_bits(
    allowed_phase_bits,
) -> tuple[int, ...] | None:
    """
    Normalize optional 1-indexed phase-bit numbers for depolarizing locations.
    """
    if allowed_phase_bits is None:
        return None

    normalized = []
    for raw_bit in allowed_phase_bits:
        bit = int(raw_bit)
        if bit < 1:
            raise ValueError(
                "controlled_unitary_depolarizing_allowed_phase_bits must be "
                "positive 1-indexed bit numbers"
            )
        if bit not in normalized:
            normalized.append(bit)
    if not normalized:
        raise ValueError(
            "controlled_unitary_depolarizing_allowed_phase_bits must be nonempty"
        )
    return tuple(sorted(normalized))


def normalize_controlled_unitary_depolarizing_paulis(
    paulis: ControlledUnitaryDepolarizingPauliSpec,
) -> tuple[str, ...] | None:
    """
    Normalize an optional nonempty subset of Pauli labels X, Y, Z.
    """
    if paulis is None:
        return None

    raw_paulis = [paulis] if isinstance(paulis, str) else list(paulis)
    if not raw_paulis:
        raise ValueError("controlled_unitary_depolarizing_paulis must be nonempty")

    normalized = []
    for raw_pauli in raw_paulis:
        pauli = str(raw_pauli).upper()
        if pauli not in CONTROLLED_UNITARY_DEPOLARIZING_PAULI_CHOICES:
            raise ValueError(
                "controlled_unitary_depolarizing_paulis must contain only X, Y, Z"
            )
        if pauli not in normalized:
            normalized.append(pauli)
    return tuple(normalized)


def order_resolution_bits(true_order: int) -> int:
    """
    Return m = ceil(log2(r)) for the true multiplicative order r.
    """
    if true_order < 1:
        raise ValueError("true_order must be >= 1")
    return max(1, math.ceil(math.log2(true_order)))


def normalize_controlled_unitary_depolarizing_ranges(
    range_names: ControlledUnitaryDepolarizingRangeSpec,
) -> tuple[str, ...] | None:
    """
    Normalize one or more controlled-unitary depolarizing phase-bit ranges.
    """
    if range_names is None:
        return None
    raw_names = [range_names] if isinstance(range_names, str) else list(range_names)
    if not raw_names:
        raise ValueError("at least one controlled-unitary range is required")

    normalized = []
    for range_name in raw_names:
        if range_name not in CONTROLLED_UNITARY_DEPOLARIZING_RANGE_CHOICES:
            raise ValueError("unknown controlled-unitary depolarizing range")
        if range_name not in normalized:
            normalized.append(range_name)
    return tuple(normalized)


def _controlled_unitary_depolarizing_phase_bits_for_one_range(
    num_phase_bits: int,
    true_order: int,
    range_name: str,
) -> tuple[int, ...]:
    if num_phase_bits < 1:
        raise ValueError("num_phase_bits must be >= 1")
    if range_name not in CONTROLLED_UNITARY_DEPOLARIZING_RANGE_CHOICES:
        raise ValueError("unknown controlled-unitary depolarizing range")

    if range_name == CONTROLLED_UNITARY_DEPOLARIZING_ALL_RANGE:
        start = 1
        end = num_phase_bits
    else:
        m = order_resolution_bits(true_order)
        if range_name == CONTROLLED_UNITARY_DEPOLARIZING_LEADING_RANGE:
            start = 1
            end = min(m, num_phase_bits)
        elif range_name == CONTROLLED_UNITARY_DEPOLARIZING_DANGEROUS_RANGE:
            start = m + 1
            end = min(2 * m, num_phase_bits)
        else:
            start = (2 * m) + 1
            end = num_phase_bits

    if start > end:
        raise ValueError(
            f"controlled-unitary depolarizing range {range_name!r} is empty for "
            f"r={true_order}, t={num_phase_bits}"
        )
    return tuple(range(start, end + 1))


def controlled_unitary_depolarizing_phase_bits(
    num_phase_bits: int,
    true_order: int,
    range_names: ControlledUnitaryDepolarizingRangeSpec,
) -> tuple[int, ...] | None:
    """
    Return 1-indexed phase-bit numbers eligible for depolarizing errors.

    Phase bit number j refers to the coefficient of 2^-j in 0.b1...bt.
    For r=2^m, the ranges are exact: leading bits 1..m, dangerous bits
    m+1..2m, and safe-tail bits 2m+1..t. Multiple ranges mean the union.
    A None range means all controlled-U blocks remain eligible.
    """
    normalized_ranges = normalize_controlled_unitary_depolarizing_ranges(
        range_names
    )
    if normalized_ranges is None:
        return None

    phase_bits: set[int] = set()
    for range_name in normalized_ranges:
        phase_bits.update(
            _controlled_unitary_depolarizing_phase_bits_for_one_range(
                num_phase_bits=num_phase_bits,
                true_order=true_order,
                range_name=range_name,
            )
        )
    return tuple(sorted(phase_bits))


def validate_error_config(error_config: ErrorConfig | None) -> ErrorConfig:
    """
    Normalize an optional error configuration.
    """
    if error_config is None:
        return ErrorConfig()

    if not isinstance(error_config, ErrorConfig):
        raise TypeError("error_config must be an ErrorConfig instance or None")

    depolarizing_probability = error_config.controlled_unitary_depolarizing_probability

    if depolarizing_probability is not None:
        p_depolarizing = float(depolarizing_probability)
        if not math.isfinite(p_depolarizing) or not (0.0 <= p_depolarizing <= 1.0):
            raise ValueError(
                "controlled_unitary_depolarizing_probability must satisfy 0 <= p <= 1"
            )
    else:
        p_depolarizing = None

    per_qubit_depolarizing_probability = (
        error_config.controlled_unitary_per_qubit_depolarizing_probability
    )

    if per_qubit_depolarizing_probability is not None:
        p_per_qubit_depolarizing = float(per_qubit_depolarizing_probability)
        if not math.isfinite(p_per_qubit_depolarizing) or not (
            0.0 <= p_per_qubit_depolarizing <= 1.0
        ):
            raise ValueError(
                "controlled_unitary_per_qubit_depolarizing_probability must satisfy 0 <= p <= 1"
            )
    else:
        p_per_qubit_depolarizing = None

    allowed_phase_bits = (
        normalize_controlled_unitary_depolarizing_allowed_phase_bits(
            error_config.controlled_unitary_depolarizing_allowed_phase_bits
        )
    )
    depolarizing_paulis = normalize_controlled_unitary_depolarizing_paulis(
        error_config.controlled_unitary_depolarizing_paulis
    )

    if error_config.measurement_bitflip_probability is None:
        return ErrorConfig(
            measurement_bitflip=error_config.measurement_bitflip,
            measurement_bitflip_probability=None,
            controlled_unitary_depolarizing_probability=p_depolarizing,
            controlled_unitary_per_qubit_depolarizing_probability=p_per_qubit_depolarizing,
            controlled_unitary_depolarizing_allowed_phase_bits=allowed_phase_bits,
            controlled_unitary_depolarizing_paulis=depolarizing_paulis,
        )

    if not error_config.measurement_bitflip:
        raise ValueError(
            "measurement_bitflip_probability requires measurement_bitflip=True"
        )

    p = float(error_config.measurement_bitflip_probability)
    if not math.isfinite(p) or not (0.0 <= p <= 1.0):
        raise ValueError("measurement_bitflip_probability must satisfy 0 <= p <= 1")

    return ErrorConfig(
        measurement_bitflip=True,
        measurement_bitflip_probability=p,
        controlled_unitary_depolarizing_probability=p_depolarizing,
        controlled_unitary_per_qubit_depolarizing_probability=p_per_qubit_depolarizing,
        controlled_unitary_depolarizing_allowed_phase_bits=allowed_phase_bits,
        controlled_unitary_depolarizing_paulis=depolarizing_paulis,
    )


def controlled_unitary_depolarizing_allowed_for_phase_bit(
    error_config: ErrorConfig | None,
    phase_bit_number: int,
) -> bool:
    """
    Return whether depolarizing errors may occur after phase bit b_j's block.
    """
    if phase_bit_number < 1:
        raise ValueError("phase_bit_number must be a positive 1-indexed bit number")

    error_config = validate_error_config(error_config)
    allowed_phase_bits = (
        error_config.controlled_unitary_depolarizing_allowed_phase_bits
    )
    if allowed_phase_bits is None:
        return True
    return int(phase_bit_number) in allowed_phase_bits


def parse_measurement_bitflip_error_arg(raw_value) -> ErrorConfig:
    """
    Parse the command-line value for --measurement-bitflip-error.

    Behavior:
      - omitted flag -> no measurement bit-flip error
      - flag with no value -> exactly one random bit-flip per PE run
      - flag with value p -> independent per-detection bit-flips with
        probability p
    """
    if raw_value is None:
        return ErrorConfig()

    if raw_value == MEASUREMENT_BITFLIP_SINGLE_TOKEN:
        return ErrorConfig(measurement_bitflip=True)

    try:
        p = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--measurement-bitflip-error must be used either without a value "
            "or with a probability p in [0, 1]"
        ) from exc

    return validate_error_config(
        ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=p,
        )
    )


def measurement_bitflip_label(error_config: ErrorConfig | None) -> str:
    """
    Return a human-readable description of the configured measurement error.
    """
    error_config = validate_error_config(error_config)
    if not error_config.measurement_bitflip:
        return "disabled"

    if error_config.measurement_bitflip_probability is None:
        return "single random bit-flip per phase-estimation run"

    return (
        "independent per-detection bit-flip probability "
        f"p={format_measurement_bitflip_probability(error_config.measurement_bitflip_probability)}"
    )


def measurement_bitflip_filename_suffix(error_config: ErrorConfig | None) -> str:
    """
    Return a filename suffix describing the measurement-error setting.
    """
    error_config = validate_error_config(error_config)
    if not error_config.measurement_bitflip:
        return ""

    if error_config.measurement_bitflip_probability is None:
        return "_measurement_bitflip_error"

    probability_text = format_measurement_bitflip_probability(
        error_config.measurement_bitflip_probability
    )
    return f"_measurement_bitflip_error_{probability_text}"


def format_measurement_bitflip_probability(probability: float) -> str:
    """
    Return a compact one-significant-figure scientific-notation string.

    Examples:
      - 0.1 -> 1e-1
      - 0.05 -> 5e-2
      - 1.0 -> 1e0
    """
    if not math.isfinite(probability) or not (0.0 <= probability <= 1.0):
        raise ValueError("measurement bit-flip probability must satisfy 0 <= p <= 1")

    mantissa, exponent = format(probability, ".0e").split("e")
    return f"{mantissa}e{int(exponent)}"


def measurement_bitflip_title_suffix(error_config: ErrorConfig | None) -> str:
    """
    Return a short title suffix describing the configured measurement error.
    """
    error_config = validate_error_config(error_config)
    if not error_config.measurement_bitflip:
        return ""

    if error_config.measurement_bitflip_probability is None:
        return " (single random measurement bit-flip error)"

    probability_text = format_measurement_bitflip_probability(
        error_config.measurement_bitflip_probability
    )
    return f" (measurement bit-flip error, p = {probability_text})"


def choose_measurement_bitflip_index(
    num_bits: int,
    error_config: ErrorConfig | None = None,
    rng=None,
) -> int | None:
    """
    Return the single bit index to flip in the reported phase-estimation output.

    The returned index uses the same convention as the final bits list:
    index 0 is the most significant phase bit.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    error_config = validate_error_config(error_config)
    if (
        not error_config.measurement_bitflip
        or error_config.measurement_bitflip_probability is not None
    ):
        return None

    if rng is None:
        rng = np.random.default_rng()

    return int(rng.integers(0, num_bits))


def choose_measurement_bitflip_mask(
    num_bits: int,
    error_config: ErrorConfig | None = None,
    rng=None,
) -> list[bool]:
    """
    Return the per-bit measurement-flip pattern for one PE run.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be >= 1")

    error_config = validate_error_config(error_config)
    if rng is None:
        rng = np.random.default_rng()

    if not error_config.measurement_bitflip:
        return [False] * num_bits

    if error_config.measurement_bitflip_probability is None:
        flip_index = choose_measurement_bitflip_index(
            num_bits,
            error_config=error_config,
            rng=rng,
        )
        mask = [False] * num_bits
        mask[flip_index] = True
        return mask

    random_draws = rng.random(num_bits)
    return [bool(draw < error_config.measurement_bitflip_probability) for draw in random_draws]


def apply_measurement_bitflip(bit: int, should_flip: bool) -> int:
    """
    Flip a measured classical bit when the error model requests it.
    """
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")

    if should_flip:
        return 1 - bit

    return bit


def num_qubits_from_dimension(dim: int) -> int:
    """
    Return log2(dim), requiring dim to be a positive power of two.
    """
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError("dimension must be a positive power of two")

    return dim.bit_length() - 1


def choose_work_depolarizing_error(
    num_work_qubits: int,
    error_config: ErrorConfig | None = None,
    rng=None,
) -> tuple[int, str] | None:
    """
    Sample a trajectory depolarizing error on the work register.

    Returns None if no error occurs, otherwise returns (work_qubit, pauli),
    where work_qubit uses label-bit indexing: qubit 0 is the least significant
    bit of the work-register basis label.
    """
    if num_work_qubits < 0:
        raise ValueError("num_work_qubits must be nonnegative")

    error_config = validate_error_config(error_config)
    p = error_config.controlled_unitary_depolarizing_probability
    if p is None or p == 0.0 or num_work_qubits == 0:
        return None

    if rng is None:
        rng = np.random.default_rng()

    if rng.random() >= p:
        return None

    work_qubit = int(rng.integers(0, num_work_qubits))
    paulis = (
        error_config.controlled_unitary_depolarizing_paulis
        or CONTROLLED_UNITARY_DEPOLARIZING_PAULI_CHOICES
    )
    pauli = str(rng.choice(paulis))
    return work_qubit, pauli


def choose_work_per_qubit_depolarizing_errors(
    num_work_qubits: int,
    error_config: ErrorConfig | None = None,
    rng=None,
) -> list[tuple[int, str]]:
    """
    Sample independent trajectory depolarizing errors on all work qubits.

    Returns a list of (work_qubit, pauli) pairs. Qubit 0 is the least
    significant bit of the work-register basis label.
    """
    if num_work_qubits < 0:
        raise ValueError("num_work_qubits must be nonnegative")

    error_config = validate_error_config(error_config)
    p = error_config.controlled_unitary_per_qubit_depolarizing_probability
    if p is None or p == 0.0 or num_work_qubits == 0:
        return []

    if rng is None:
        rng = np.random.default_rng()

    hit_mask = rng.random(num_work_qubits) < p
    hit_qubits = np.flatnonzero(hit_mask)
    if hit_qubits.size == 0:
        return []

    pauli_choices = (
        error_config.controlled_unitary_depolarizing_paulis
        or CONTROLLED_UNITARY_DEPOLARIZING_PAULI_CHOICES
    )
    paulis = rng.choice(pauli_choices, size=hit_qubits.size)
    return [
        (int(work_qubit), str(pauli))
        for work_qubit, pauli in zip(hit_qubits, paulis)
    ]


def apply_work_qubit_pauli(psi: np.ndarray, work_qubit: int, pauli: str) -> np.ndarray:
    """
    Apply a Pauli operator to one work-register qubit of a joint state.

    The joint state has shape (control_dim, M), where the first axis is the
    control register and M is the work-register dimension. Qubit 0 is the least
    significant bit of the work-register basis label.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 2:
        raise ValueError("psi must have shape (control_dim, M)")

    M = psi.shape[1]
    num_work_qubits = num_qubits_from_dimension(M)
    if not (0 <= work_qubit < num_work_qubits):
        raise ValueError("work_qubit out of range")

    pauli = pauli.upper()
    if pauli not in ("X", "Y", "Z"):
        raise ValueError("pauli must be one of: X, Y, Z")

    labels = np.arange(M)
    bit_mask = 1 << work_qubit

    if pauli == "Z":
        out = psi.copy()
        out[:, (labels & bit_mask) != 0] *= -1
        return out

    target_labels = labels ^ bit_mask
    out = np.empty_like(psi)

    if pauli == "X":
        out[:, target_labels] = psi
        return out

    phases = np.where((labels & bit_mask) == 0, 1j, -1j)
    out[:, target_labels] = psi * phases
    return out


def apply_controlled_unitary_depolarizing_error(
    psi: np.ndarray,
    error_config: ErrorConfig | None = None,
    rng=None,
) -> np.ndarray:
    """
    Maybe apply one sampled work-register depolarizing error to a joint state.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 2:
        raise ValueError("psi must have shape (control_dim, M)")

    num_work_qubits = num_qubits_from_dimension(psi.shape[1])
    block_error = choose_work_depolarizing_error(
        num_work_qubits,
        error_config=error_config,
        rng=rng,
    )
    if block_error is not None:
        work_qubit, pauli = block_error
        psi = apply_work_qubit_pauli(psi, work_qubit=work_qubit, pauli=pauli)

    per_qubit_errors = choose_work_per_qubit_depolarizing_errors(
        num_work_qubits,
        error_config=error_config,
        rng=rng,
    )
    for work_qubit, pauli in per_qubit_errors:
        psi = apply_work_qubit_pauli(psi, work_qubit=work_qubit, pauli=pauli)

    return psi
