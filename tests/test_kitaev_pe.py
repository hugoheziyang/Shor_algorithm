import numpy as np

import shor_one_qubit.kitaev_pe as kitaev_module
from shor_common.errors import ErrorConfig
from shor_one_qubit.kitaev_pe import feedback_angle, iterative_phase_estimation


def make_phase_operator_factory(phi: float):
    """
    Return a factory for a 1D work register with eigenphase phi.
    U^power acts by multiplication with exp(2πi * power * phi).
    """
    def factory(power: int):
        def work_op(work_state: np.ndarray) -> np.ndarray:
            return np.exp(2j * np.pi * power * phi) * work_state
        return work_op

    return factory


def test_feedback_angle_simple_case():
    # bits = [b1, b2, b3] and extracting j=2 should subtract only b3/4
    bits = [1, 0, 1]
    angle = feedback_angle(bits, 2)
    expected = -2 * np.pi * (1 / 4)
    assert np.isclose(angle, expected)


def test_iterative_phase_estimation_phi_zero():
    initial_work_state = np.array([1.0 + 0.0j])
    factory = make_phase_operator_factory(0.0)

    bits, phase_estimate, final_work_state = iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=factory,
        num_bits=3,
        rng=np.random.default_rng(123),
    )

    assert bits == [0, 0, 0]
    assert np.isclose(phase_estimate, 0.0)
    np.testing.assert_allclose(final_work_state, np.array([1.0 + 0.0j]))


def test_iterative_phase_estimation_phi_one_half():
    # 0.1 in binary
    initial_work_state = np.array([1.0 + 0.0j])
    factory = make_phase_operator_factory(0.5)

    bits, phase_estimate, _ = iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=factory,
        num_bits=3,
        rng=np.random.default_rng(123),
    )

    assert bits == [1, 0, 0]
    assert np.isclose(phase_estimate, 0.5)


def test_iterative_phase_estimation_phi_one_quarter():
    # 0.01 in binary
    initial_work_state = np.array([1.0 + 0.0j])
    factory = make_phase_operator_factory(0.25)

    bits, phase_estimate, _ = iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=factory,
        num_bits=3,
        rng=np.random.default_rng(123),
    )

    assert bits == [0, 1, 0]
    assert np.isclose(phase_estimate, 0.25)


def test_iterative_phase_estimation_with_single_measurement_bitflip_on_msb(monkeypatch):
    initial_work_state = np.array([1.0 + 0.0j])
    factory = make_phase_operator_factory(0.0)

    monkeypatch.setattr(
        kitaev_module,
        "choose_measurement_bitflip_mask",
        lambda num_bits, error_config=None, rng=None: [True, False, False],
    )

    bits, phase_estimate, final_work_state = kitaev_module.iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=factory,
        num_bits=3,
        rng=np.random.default_rng(123),
        error_config=ErrorConfig(measurement_bitflip=True),
    )

    assert bits == [1, 0, 0]
    assert np.isclose(phase_estimate, 0.5)
    np.testing.assert_allclose(final_work_state, np.array([1.0 + 0.0j]))


def test_iterative_phase_estimation_applies_measurement_bitflip_to_exactly_one_index(monkeypatch):
    initial_work_state = np.array([1.0 + 0.0j])
    factory = make_phase_operator_factory(0.0)
    flip_flags = []

    monkeypatch.setattr(
        kitaev_module,
        "choose_measurement_bitflip_mask",
        lambda num_bits, error_config=None, rng=None: [False, False, True, False],
    )

    original_apply = kitaev_module.apply_measurement_bitflip

    def tracking_apply_measurement_bitflip(bit, should_flip):
        flip_flags.append(should_flip)
        return original_apply(bit, should_flip)

    monkeypatch.setattr(
        kitaev_module,
        "apply_measurement_bitflip",
        tracking_apply_measurement_bitflip,
    )

    bits, phase_estimate, final_work_state = kitaev_module.iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=factory,
        num_bits=4,
        rng=np.random.default_rng(123),
        error_config=ErrorConfig(measurement_bitflip=True),
    )

    assert flip_flags.count(True) == 1
    assert flip_flags.count(False) == 3
    assert bits == [0, 0, 1, 0]
    assert np.isclose(phase_estimate, 0.125)
    assert final_work_state.shape == (1,)
    np.testing.assert_allclose(np.abs(final_work_state), np.array([1.0]))


def test_iterative_phase_estimation_applies_depolarizing_error_after_controlled_unitary(monkeypatch):
    initial_work_state = np.array([1.0 + 0.0j])
    factory = make_phase_operator_factory(0.0)
    calls = []

    def tracking_depolarizing_error(psi, error_config=None, rng=None):
        calls.append(error_config)
        return psi

    monkeypatch.setattr(
        kitaev_module,
        "apply_controlled_unitary_depolarizing_error",
        tracking_depolarizing_error,
    )

    error_config = ErrorConfig(controlled_unitary_depolarizing_probability=0.25)
    bits, phase_estimate, final_work_state = kitaev_module.iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=factory,
        num_bits=3,
        rng=np.random.default_rng(123),
        error_config=error_config,
    )

    assert len(calls) == 3
    assert all(call == error_config for call in calls)
    assert bits == [0, 0, 0]
    assert np.isclose(phase_estimate, 0.0)
    np.testing.assert_allclose(final_work_state, np.array([1.0 + 0.0j]))


def test_iterative_phase_estimation_filters_depolarizing_error_by_phase_bit(monkeypatch):
    initial_work_state = np.array([1.0 + 0.0j])
    factory = make_phase_operator_factory(0.0)
    calls = []

    def tracking_depolarizing_error(psi, error_config=None, rng=None):
        calls.append(error_config)
        return psi

    monkeypatch.setattr(
        kitaev_module,
        "apply_controlled_unitary_depolarizing_error",
        tracking_depolarizing_error,
    )

    error_config = ErrorConfig(
        controlled_unitary_depolarizing_probability=0.25,
        controlled_unitary_depolarizing_allowed_phase_bits=(1, 3),
    )
    bits, phase_estimate, final_work_state = kitaev_module.iterative_phase_estimation(
        initial_work_state=initial_work_state,
        controlled_power_factory=factory,
        num_bits=3,
        rng=np.random.default_rng(123),
        error_config=error_config,
    )

    assert len(calls) == 2
    assert all(call == error_config for call in calls)
    assert bits == [0, 0, 0]
    assert np.isclose(phase_estimate, 0.0)
    np.testing.assert_allclose(final_work_state, np.array([1.0 + 0.0j]))
