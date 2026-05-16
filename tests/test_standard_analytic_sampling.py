import numpy as np
import pytest

from shor_common.errors import ErrorConfig
from shor_standard.analytic_sampling import (
    multiplicative_order,
    qpe_outcome_probability,
    qpe_window_mass_diagnostics,
    qpe_window_probability_mass,
    sample_qpe_outcome_from_order,
    sample_standard_phase_bits_from_order,
    validate_analytic_error_config,
)


def test_multiplicative_order_for_small_instance():
    assert multiplicative_order(2, 15) == 4


def test_qpe_outcome_probability_is_one_at_exact_peak():
    assert np.isclose(
        qpe_outcome_probability(
            y=4,
            eigenphase_numerator=1,
            order=4,
            num_control_qubits=4,
        ),
        1.0,
    )


def test_qpe_outcome_probability_is_zero_off_exact_peak():
    assert np.isclose(
        qpe_outcome_probability(
            y=5,
            eigenphase_numerator=1,
            order=4,
            num_control_qubits=4,
        ),
        0.0,
        atol=1e-12,
    )


def test_qpe_window_mass_is_one_for_exact_integer_peaks():
    diagnostics = qpe_window_mass_diagnostics(
        order=4,
        num_control_qubits=4,
        window=0,
    )

    assert diagnostics.min_mass == 1.0
    assert diagnostics.mean_mass == 1.0
    assert diagnostics.max_mass == 1.0
    assert diagnostics.worst_eigenphase_numerator == 0


def test_qpe_window_mass_increases_with_window_size():
    narrow = qpe_window_probability_mass(
        eigenphase_numerator=1,
        order=3,
        num_control_qubits=5,
        window=0,
    )
    wide = qpe_window_probability_mass(
        eigenphase_numerator=1,
        order=3,
        num_control_qubits=5,
        window=4,
    )

    assert 0.0 < narrow < wide <= 1.0


def test_qpe_window_mass_diagnostics_identify_valid_range():
    diagnostics = qpe_window_mass_diagnostics(
        order=3,
        num_control_qubits=5,
        window=4,
    )

    assert diagnostics.order == 3
    assert diagnostics.num_control_qubits == 5
    assert diagnostics.window == 4
    assert diagnostics.num_eigenphases == 3
    assert 0.0 < diagnostics.min_mass <= diagnostics.mean_mass <= diagnostics.max_mass <= 1.0
    assert 0 <= diagnostics.worst_eigenphase_numerator < 3
    assert 0 <= diagnostics.best_eigenphase_numerator < 3


def test_sample_qpe_outcome_from_order_hits_exact_peaks_when_order_divides_Q():
    rng = np.random.default_rng(123)

    for _ in range(20):
        y, s = sample_qpe_outcome_from_order(
            order=4,
            num_control_qubits=4,
            rng=rng,
            window=3,
        )
        assert y == 4 * s


def test_sample_standard_phase_bits_applies_explicit_mask():
    bits, phase_estimate, y, s = sample_standard_phase_bits_from_order(
        order=4,
        num_control_qubits=4,
        rng=np.random.default_rng(123),
        window=3,
        measurement_bitflip_mask=[True, False, False, False],
    )

    assert len(bits) == 4
    assert 0 <= y < 16
    assert 0 <= s < 4
    assert np.isclose(phase_estimate, sum(bit / (2 ** index) for index, bit in enumerate(bits, start=1)))


def test_multiplicative_order_rejects_non_coprime_base():
    with pytest.raises(ValueError, match="coprime"):
        multiplicative_order(3, 15)


def test_analytic_backend_rejects_depolarizing_error_config():
    with pytest.raises(ValueError, match="measurement bit-flip errors only"):
        validate_analytic_error_config(
            ErrorConfig(controlled_unitary_depolarizing_probability=0.1)
        )
