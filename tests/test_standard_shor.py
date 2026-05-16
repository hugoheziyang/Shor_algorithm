import numpy as np
import pytest

from shor_common.errors import ErrorConfig
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND
import shor_standard.shor as standard_shor_module
from shor_standard.shor import (
    default_num_control_qubits,
    shor_attempt,
    shor_factor,
    shor_order_finding_attempt,
)


def test_default_num_control_qubits_matches_textbook_precision_choice():
    assert default_num_control_qubits(15) == 9


def test_standard_order_finding_attempt_returns_expected_keys():
    result = shor_order_finding_attempt(
        N=15,
        a=2,
        num_control_qubits=8,
        rng=np.random.default_rng(1234),
        postprocess_mode="rescue",
    )

    assert result["algorithm"] == "standard"
    assert result["a"] == 2
    assert result["num_control_qubits"] == 8
    assert result["num_work_qubits"] == 4
    assert isinstance(result["bits"], list)
    assert len(result["bits"]) == 8
    assert "phase_estimate" in result
    assert "order" in result
    assert result["postprocess_mode"] == "rescue"
    assert "final_work_state" in result


def test_standard_order_finding_attempt_passes_error_config(monkeypatch):
    captured = {}

    def fake_standard_phase_estimation(
        initial_work_state,
        controlled_power_factory,
        num_control_qubits,
        rng=None,
        error_config=None,
    ):
        captured["num_control_qubits"] = num_control_qubits
        captured["error_config"] = error_config
        return [0, 1, 0, 0], 0.25, np.array([1.0 + 0.0j])

    monkeypatch.setattr(
        standard_shor_module,
        "standard_phase_estimation",
        fake_standard_phase_estimation,
    )
    monkeypatch.setattr(
        standard_shor_module,
        "recover_order_with_mode",
        lambda bits, a, N, postprocess_mode: 4,
    )

    error_config = ErrorConfig(measurement_bitflip=True)
    result = shor_order_finding_attempt(
        N=15,
        a=2,
        num_control_qubits=4,
        postprocess_mode="one_shot",
        rng=np.random.default_rng(1234),
        error_config=error_config,
    )

    assert captured["num_control_qubits"] == 4
    assert captured["error_config"] == error_config
    assert result["order"] == 4
    assert result["error_config"] == {"measurement_bitflip": True}


def test_standard_shor_attempt_factors_15_with_fixed_base():
    result = shor_attempt(
        N=15,
        a=2,
        num_control_qubits=8,
        rng=np.random.default_rng(2),
        postprocess_mode="one_shot",
    )

    assert result["success"] is True
    assert result["algorithm"] == "standard"
    assert result["factors"] in [(3, 5), (5, 3)]
    assert result["num_control_qubits"] == 8
    assert result["num_work_qubits"] == 4


def test_standard_shor_attempt_supports_analytic_backend():
    result = shor_attempt(
        N=15,
        a=2,
        num_control_qubits=8,
        rng=np.random.default_rng(2),
        postprocess_mode="rescue",
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result["standard_backend"] == ANALYTIC_STANDARD_BACKEND
    assert result["analytic_window"] == 20
    assert result["sampling_order"] == 4
    assert len(result["bits"]) == 8


def test_standard_even_number_shortcuts_immediately():
    result = shor_attempt(N=12, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] == (2, 6)
    assert result["algorithm"] == "standard"


def test_standard_perfect_power_shortcuts_before_quantum_step():
    result = shor_attempt(N=49, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] == (7, 7)


def test_standard_nontrivial_gcd_shortcuts_before_quantum_step():
    result = shor_attempt(N=15, a=3, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] in [(3, 5), (5, 3)]


def test_standard_shor_attempt_rejects_invalid_postprocess_mode():
    with pytest.raises(ValueError, match="postprocess_mode"):
        shor_attempt(N=15, postprocess_mode="invalid", rng=np.random.default_rng(123))


def test_standard_shor_factor_one_shot_defaults_to_single_attempt(monkeypatch):
    calls = []

    def fake_shor_attempt(
        N,
        a=None,
        num_control_qubits=None,
        rng=None,
        postprocess_mode="one_shot",
        error_config=None,
        standard_backend="statevector",
        analytic_window=100,
    ):
        calls.append((postprocess_mode, error_config, standard_backend, analytic_window))
        return {
            "algorithm": "standard",
            "N": N,
            "postprocess_mode": postprocess_mode,
            "error_config": {"measurement_bitflip": False},
            "success": False,
            "reason": "order/factor extraction failed",
        }

    monkeypatch.setattr(standard_shor_module, "shor_attempt", fake_shor_attempt)

    result = shor_factor(N=21, postprocess_mode="one_shot", rng=np.random.default_rng(123))

    assert result["success"] is False
    assert result["algorithm"] == "standard"
    assert len(calls) == 1
    assert calls[0][0] == "one_shot"
    assert calls[0][1] == ErrorConfig()
    assert calls[0][2] == "statevector"
    assert calls[0][3] == 100
