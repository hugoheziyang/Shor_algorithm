from fractions import Fraction

import numpy as np
import pytest

import error_correction.code_distance as code_distance_module
from error_correction.code_distance import (
    bits_to_decimal,
    bits_to_decimal_fraction,
    decimal_distance,
    estimate_average_decimal_distance,
    estimate_average_hamming_distance,
    hamming_distance,
    iterative_phase_estimation_for_diagnostics,
    normalized_hamming_distance,
    run_phase_estimation_hamming_distance_trial,
)
from shor_one_qubit.errors import ErrorConfig, apply_work_qubit_pauli


def make_z_power_factory():
    def factory(power):
        def work_op(work_state):
            work_state = np.asarray(work_state, dtype=complex)
            if (power % 2) == 0:
                return work_state.copy()
            return np.array([work_state[0], -work_state[1]], dtype=complex)

        return work_op

    return factory


def make_x_power_factory():
    def factory(power):
        def work_op(work_state):
            work_state = np.asarray(work_state, dtype=complex)
            if (power % 2) == 0:
                return work_state.copy()
            return np.array([work_state[1], work_state[0]], dtype=complex)

        return work_op

    return factory


def test_hamming_distance_counts_different_bit_positions():
    assert hamming_distance([0, 1, 0, 1], [1, 1, 0, 0]) == 2
    assert normalized_hamming_distance([0, 1, 0, 1], [1, 1, 0, 0]) == 0.5


def test_bits_to_decimal_converts_binary_fraction():
    assert bits_to_decimal_fraction([1, 0, 1]) == Fraction(5, 8)
    assert bits_to_decimal([1, 0, 1]) == 0.625


def test_decimal_distance_uses_unit_circle_shortcut():
    assert decimal_distance([0, 0, 0], [1, 1, 1]) == 0.125
    assert decimal_distance([0, 1, 0], [1, 0, 0]) == 0.25


def test_hamming_distance_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        hamming_distance([0, 1], [0])


def test_hamming_distance_rejects_non_bits():
    with pytest.raises(ValueError, match="only 0 or 1"):
        hamming_distance([0, 2], [0, 1])


def test_normalized_hamming_distance_rejects_empty_bit_strings():
    with pytest.raises(ValueError, match="nonempty"):
        normalized_hamming_distance([], [])


def test_decimal_distance_rejects_empty_bit_strings():
    with pytest.raises(ValueError, match="nonempty"):
        decimal_distance([], [])


def test_diagnostic_phase_estimation_applies_explicit_measurement_mask():
    initial_work_state = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    def controlled_power_factory(power):
        return lambda work_state: work_state

    bits, phase_estimate, _ = iterative_phase_estimation_for_diagnostics(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=3,
        measurement_bitflip_mask=[True, False, True],
        measurement_rng=np.random.default_rng(123),
        error_rng=np.random.default_rng(456),
    )

    assert bits == [1, 0, 1]
    assert phase_estimate == 0.625


def test_hamming_distance_trial_is_zero_without_errors():
    trial = run_phase_estimation_hamming_distance_trial(
        N=21,
        a=11,
        num_phase_bits=6,
        error_config=ErrorConfig(),
        rng=np.random.default_rng(1234),
    )

    assert trial.clean_bits == trial.noisy_bits
    assert trial.hamming_distance == 0
    assert trial.normalized_hamming_distance == 0.0
    assert trial.decimal_distance == 0.0
    assert trial.measurement_bitflip_mask == [False] * 6


def test_hamming_distance_trial_detects_single_measurement_flip():
    trial = run_phase_estimation_hamming_distance_trial(
        N=21,
        a=11,
        num_phase_bits=6,
        error_config=ErrorConfig(measurement_bitflip=True),
        rng=np.random.default_rng(1234),
    )

    assert sum(trial.measurement_bitflip_mask) == 1
    assert trial.hamming_distance >= 1


def test_average_hamming_distance_is_zero_without_errors():
    estimate = estimate_average_hamming_distance(
        N=21,
        a=11,
        trials=5,
        num_phase_bits=6,
        error_config=ErrorConfig(),
        rng=np.random.default_rng(1234),
    )

    assert estimate.total_hamming_distance == 0
    assert estimate.average_hamming_distance == 0.0
    assert estimate.average_normalized_hamming_distance == 0.0


def test_average_decimal_distance_is_zero_without_errors():
    estimate = estimate_average_decimal_distance(
        N=21,
        a=11,
        trials=5,
        num_phase_bits=6,
        error_config=ErrorConfig(),
        rng=np.random.default_rng(1234),
    )

    assert estimate.total_decimal_distance == 0.0
    assert estimate.average_decimal_distance == 0.0


@pytest.mark.parametrize(
    ("pauli", "initial_work_state", "controlled_power_factory"),
    [
        ("X", np.array([1.0 + 0.0j, 0.0 + 0.0j]), make_z_power_factory()),
        ("Y", np.array([1.0 + 0.0j, 0.0 + 0.0j]), make_z_power_factory()),
        (
            "Z",
            np.array([1.0 + 0.0j, 1.0 + 0.0j]) / np.sqrt(2.0),
            make_x_power_factory(),
        ),
    ],
)
def test_fixed_work_pauli_error_can_change_later_phase_bit(
    monkeypatch,
    pauli,
    initial_work_state,
    controlled_power_factory,
):
    injected = {"count": 0}

    def inject_once_after_controlled_unitary(psi, error_config=None, rng=None):
        if error_config == ErrorConfig():
            return psi
        if injected["count"] == 0:
            injected["count"] += 1
            return apply_work_qubit_pauli(psi, work_qubit=0, pauli=pauli)
        return psi

    monkeypatch.setattr(
        code_distance_module,
        "apply_controlled_unitary_depolarizing_error",
        inject_once_after_controlled_unitary,
    )

    clean_bits, _, _ = iterative_phase_estimation_for_diagnostics(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=3,
        measurement_rng=np.random.default_rng(123),
        error_config=ErrorConfig(),
    )
    noisy_bits, _, _ = iterative_phase_estimation_for_diagnostics(
        initial_work_state=initial_work_state,
        controlled_power_factory=controlled_power_factory,
        num_bits=3,
        measurement_rng=np.random.default_rng(123),
        error_config=ErrorConfig(controlled_unitary_depolarizing_probability=1.0),
    )

    assert injected["count"] == 1
    assert clean_bits == [0, 0, 0]
    assert noisy_bits == [1, 0, 0]
    assert hamming_distance(clean_bits, noisy_bits) == 1
