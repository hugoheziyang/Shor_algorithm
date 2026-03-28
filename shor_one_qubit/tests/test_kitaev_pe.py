import numpy as np

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