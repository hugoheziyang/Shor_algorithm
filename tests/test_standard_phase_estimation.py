import numpy as np

import shor_standard.phase_estimation as phase_estimation_module
from shor_common.errors import ErrorConfig
from shor_standard.phase_estimation import (
    apply_controlled_power_superposition,
    apply_controlled_power_sequence,
    bits_to_phase_estimate,
    integer_to_bits,
    standard_phase_estimation,
    standard_phase_estimation_with_bitflip_mask,
)


def make_phase_operator_factory(phi: float):
    """
    Return a factory for a 1D work register with eigenphase phi.
    U^power acts by multiplication with exp(2*pi*i*power*phi).
    """
    def factory(power: int):
        def work_op(work_state: np.ndarray) -> np.ndarray:
            return np.exp(2j * np.pi * power * phi) * work_state
        return work_op

    return factory


def test_integer_to_bits_big_endian():
    assert integer_to_bits(6, 4) == [0, 1, 1, 0]


def test_bits_to_phase_estimate():
    assert np.isclose(bits_to_phase_estimate([0, 1, 0]), 0.25)
    assert np.isclose(bits_to_phase_estimate([1, 0, 0]), 0.5)


def test_apply_controlled_power_superposition_for_eigenstate():
    initial_work_state = np.array([1.0 + 0.0j])
    phi = 0.25

    psi = apply_controlled_power_superposition(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(phi),
        num_control_qubits=2,
    )

    labels = np.arange(4)
    expected = np.exp(2j * np.pi * labels * phi).reshape(4, 1) / 2
    np.testing.assert_allclose(psi, expected)


def test_apply_controlled_power_sequence_matches_direct_without_errors():
    initial_work_state = np.array([1.0 + 0.0j])
    phi = 0.25

    direct = apply_controlled_power_superposition(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(phi),
        num_control_qubits=3,
    )
    sequential = apply_controlled_power_sequence(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(phi),
        num_control_qubits=3,
        error_config=ErrorConfig(),
        rng=np.random.default_rng(123),
    )

    np.testing.assert_allclose(sequential, direct)


def test_standard_phase_estimation_phi_zero():
    initial_work_state = np.array([1.0 + 0.0j])

    bits, phase_estimate, final_work_state = standard_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(0.0),
        num_control_qubits=3,
        rng=np.random.default_rng(123),
    )

    assert bits == [0, 0, 0]
    assert np.isclose(phase_estimate, 0.0)
    np.testing.assert_allclose(final_work_state, initial_work_state)


def test_standard_phase_estimation_phi_one_half():
    initial_work_state = np.array([1.0 + 0.0j])

    bits, phase_estimate, _ = standard_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(0.5),
        num_control_qubits=3,
        rng=np.random.default_rng(123),
    )

    assert bits == [1, 0, 0]
    assert np.isclose(phase_estimate, 0.5)


def test_standard_phase_estimation_phi_one_quarter():
    initial_work_state = np.array([1.0 + 0.0j])

    bits, phase_estimate, _ = standard_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(0.25),
        num_control_qubits=3,
        rng=np.random.default_rng(123),
    )

    assert bits == [0, 1, 0]
    assert np.isclose(phase_estimate, 0.25)


def test_standard_phase_estimation_applies_measurement_bitflip(monkeypatch):
    initial_work_state = np.array([1.0 + 0.0j])

    monkeypatch.setattr(
        phase_estimation_module,
        "choose_measurement_bitflip_mask",
        lambda num_bits, error_config=None, rng=None: [True, False, False],
    )

    bits, phase_estimate, final_work_state = standard_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(0.0),
        num_control_qubits=3,
        rng=np.random.default_rng(123),
        error_config=ErrorConfig(measurement_bitflip=True),
    )

    assert bits == [1, 0, 0]
    assert np.isclose(phase_estimate, 0.5)
    np.testing.assert_allclose(final_work_state, initial_work_state)


def test_standard_phase_estimation_with_explicit_bitflip_mask():
    initial_work_state = np.array([1.0 + 0.0j])

    bits, phase_estimate, final_work_state = standard_phase_estimation_with_bitflip_mask(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(0.0),
        num_control_qubits=3,
        measurement_bitflip_mask=[False, True, False],
        measurement_rng=np.random.default_rng(123),
    )

    assert bits == [0, 1, 0]
    assert np.isclose(phase_estimate, 0.25)
    np.testing.assert_allclose(final_work_state, initial_work_state)


def test_standard_phase_estimation_applies_depolarizing_error_after_each_block(
    monkeypatch,
):
    initial_work_state = np.array([1.0 + 0.0j])
    calls = []

    def tracking_depolarizing_error(psi, error_config=None, rng=None):
        calls.append(error_config)
        return psi

    monkeypatch.setattr(
        phase_estimation_module,
        "apply_controlled_unitary_depolarizing_error",
        tracking_depolarizing_error,
    )

    error_config = ErrorConfig(controlled_unitary_depolarizing_probability=0.25)
    bits, phase_estimate, final_work_state = standard_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=make_phase_operator_factory(0.0),
        num_control_qubits=3,
        rng=np.random.default_rng(123),
        error_config=error_config,
    )

    assert len(calls) == 3
    assert all(call == error_config for call in calls)
    assert bits == [0, 0, 0]
    assert np.isclose(phase_estimate, 0.0)
    np.testing.assert_allclose(final_work_state, initial_work_state)
