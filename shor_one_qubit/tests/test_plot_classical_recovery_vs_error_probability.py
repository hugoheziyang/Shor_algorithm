import numpy as np
import pytest

from graphs.plot_classical_recovery_vs_error_probability import (
    build_probability_values,
)


def test_build_probability_values_linear_spacing():
    values = build_probability_values(0.0, 1.0, 3, log_p_grid=False)
    assert values == [0.0, 0.5, 1.0]


def test_build_probability_values_log_spacing():
    values = build_probability_values(1e-3, 1e-1, 3, log_p_grid=True)
    np.testing.assert_allclose(values, [1e-3, 1e-2, 1e-1])


def test_build_probability_values_log_spacing_requires_positive_minimum():
    with pytest.raises(ValueError, match="min-p > 0"):
        build_probability_values(0.0, 1.0, 5, log_p_grid=True)
