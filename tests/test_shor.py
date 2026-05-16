import numpy as np
import pytest

from shor_common.errors import ErrorConfig
import shor_one_qubit.shor as shor_module
from shor_one_qubit.shor import (
    is_trivial_case,
    is_even,
    make_basis_state,
    default_num_phase_bits,
    default_max_attempts,
    shor_attempt,
    shor_factor,
    shor_order_finding_attempt,
)


def test_is_trivial_case():
    assert is_trivial_case(1) is True
    assert is_trivial_case(15) is False


def test_is_even():
    assert is_even(14) is True
    assert is_even(15) is False


def test_perfect_power_shortcuts_before_quantum_step():
    result = shor_attempt(N=49, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] == (7, 7)
    

def test_make_basis_state():
    psi = make_basis_state(1, 4)
    expected = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_allclose(psi, expected)


def test_default_num_phase_bits():
    # ceil(log2(15)) = 4, so default should be 2 * 4 + 1 = 9
    assert default_num_phase_bits(15) == 9


def test_default_max_attempts_depends_on_postprocess_mode():
    assert default_max_attempts("one_shot") == 1
    assert default_max_attempts("rescue") == 10


def test_even_number_shortcuts_immediately():
    result = shor_attempt(N=12, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] == (2, 6)


def test_nontrivial_gcd_shortcuts_before_quantum_step():
    # gcd(3, 15) = 3, so the algorithm should succeed immediately.
    result = shor_attempt(N=15, a=3, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] in [(3, 5), (5, 3)]
    assert result["postprocess_mode"] == "one_shot"


def test_shor_order_finding_attempt_returns_expected_keys():
    result = shor_order_finding_attempt(
        N=15,
        a=2,
        num_phase_bits=8,
        rng=np.random.default_rng(1234),
    )

    assert result["a"] == 2
    assert isinstance(result["bits"], list)
    assert len(result["bits"]) == 8
    assert "phase_estimate" in result
    assert "order" in result
    assert result["postprocess_mode"] == "one_shot"
    assert "final_work_state" in result


def test_shor_order_finding_attempt_dispatches_one_shot(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        shor_module,
        "iterative_phase_estimation",
        lambda initial_work_state, controlled_power_factory, num_bits, rng=None, error_config=None: (
            [0, 1, 0, 0],
            0.25,
            np.array([1.0 + 0.0j]),
        ),
    )

    def fake_recover_order_with_mode(bits, a, N, postprocess_mode):
        captured["postprocess_mode"] = postprocess_mode
        return 4

    monkeypatch.setattr(shor_module, "recover_order_with_mode", fake_recover_order_with_mode)

    result = shor_module.shor_order_finding_attempt(
        N=15,
        a=2,
        num_phase_bits=4,
        postprocess_mode="one_shot",
        rng=np.random.default_rng(1234),
    )

    assert result["order"] == 4
    assert result["postprocess_mode"] == "one_shot"
    assert captured["postprocess_mode"] == "one_shot"


def test_shor_order_finding_attempt_dispatches_rescue(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        shor_module,
        "iterative_phase_estimation",
        lambda initial_work_state, controlled_power_factory, num_bits, rng=None, error_config=None: (
            [1, 0, 0, 0],
            0.5,
            np.array([1.0 + 0.0j]),
        ),
    )

    def fake_recover_order_with_mode(bits, a, N, postprocess_mode):
        captured["postprocess_mode"] = postprocess_mode
        return 4

    monkeypatch.setattr(shor_module, "recover_order_with_mode", fake_recover_order_with_mode)

    result = shor_module.shor_order_finding_attempt(
        N=15,
        a=2,
        num_phase_bits=4,
        postprocess_mode="rescue",
        rng=np.random.default_rng(1234),
    )

    assert result["order"] == 4
    assert result["postprocess_mode"] == "rescue"
    assert captured["postprocess_mode"] == "rescue"


def test_shor_order_finding_attempt_passes_error_config(monkeypatch):
    captured = {}

    def fake_iterative_phase_estimation(
        initial_work_state,
        controlled_power_factory,
        num_bits,
        rng=None,
        error_config=None,
    ):
        captured["error_config"] = error_config
        return [0, 1, 0, 0], 0.25, np.array([1.0 + 0.0j])

    monkeypatch.setattr(
        shor_module,
        "iterative_phase_estimation",
        fake_iterative_phase_estimation,
    )
    monkeypatch.setattr(
        shor_module,
        "recover_order_with_mode",
        lambda bits, a, N, postprocess_mode: 4,
    )

    error_config = ErrorConfig(measurement_bitflip=True)
    result = shor_module.shor_order_finding_attempt(
        N=15,
        a=2,
        num_phase_bits=4,
        postprocess_mode="one_shot",
        rng=np.random.default_rng(1234),
        error_config=error_config,
    )

    assert captured["error_config"] == error_config
    assert result["error_config"] == {"measurement_bitflip": True}


def test_shor_attempt_rejects_invalid_postprocess_mode():
    with pytest.raises(ValueError, match="postprocess_mode"):
        shor_attempt(N=15, postprocess_mode="invalid", rng=np.random.default_rng(123))


def test_shor_factor_one_shot_defaults_to_single_attempt(monkeypatch):
    calls = []

    def fake_shor_attempt(
        N,
        a=None,
        num_phase_bits=None,
        rng=None,
        postprocess_mode="one_shot",
        error_config=None,
    ):
        calls.append((postprocess_mode, error_config))
        return {
            "N": N,
            "postprocess_mode": postprocess_mode,
            "error_config": {"measurement_bitflip": False},
            "success": False,
            "reason": "order/factor extraction failed",
        }

    monkeypatch.setattr(shor_module, "shor_attempt", fake_shor_attempt)

    result = shor_factor(N=21, postprocess_mode="one_shot", rng=np.random.default_rng(123))

    assert result["success"] is False
    assert len(calls) == 1
    assert calls[0][0] == "one_shot"
    assert calls[0][1] == ErrorConfig()
