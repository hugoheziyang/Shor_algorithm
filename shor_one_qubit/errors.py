from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MEASUREMENT_BITFLIP_SINGLE_TOKEN = "single"


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
    """

    measurement_bitflip: bool = False
    measurement_bitflip_probability: float | None = None
    controlled_unitary_depolarizing_probability: float | None = None
    controlled_unitary_per_qubit_depolarizing_probability: float | None = None

    def to_dict(self) -> dict[str, bool | float]:
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
        return payload


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

    if error_config.measurement_bitflip_probability is None:
        return ErrorConfig(
            measurement_bitflip=error_config.measurement_bitflip,
            measurement_bitflip_probability=None,
            controlled_unitary_depolarizing_probability=p_depolarizing,
            controlled_unitary_per_qubit_depolarizing_probability=p_per_qubit_depolarizing,
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
    )


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
    pauli = str(rng.choice(["X", "Y", "Z"]))
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

    paulis = rng.choice(["X", "Y", "Z"], size=hit_qubits.size)
    return [
        (int(work_qubit), str(pauli))
        for work_qubit, pauli in zip(hit_qubits, paulis)
    ]


def apply_work_qubit_pauli(psi: np.ndarray, work_qubit: int, pauli: str) -> np.ndarray:
    """
    Apply a Pauli operator to one work-register qubit of a joint state.

    The joint state has shape (2, M), where the first axis is the control qubit
    and M is the work-register dimension. Qubit 0 is the least significant bit
    of the work-register basis label.
    """
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError("psi must have shape (2, M)")

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
    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError("psi must have shape (2, M)")

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
