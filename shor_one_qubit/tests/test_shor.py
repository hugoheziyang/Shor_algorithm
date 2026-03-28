import numpy as np

from shor_one_qubit.shor import (
    is_trivial_case,
    is_even,
    make_basis_state,
    default_num_phase_bits,
    shor_attempt,
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
    # ceil(log2(15)) = 4, so default should be 8
    assert default_num_phase_bits(15) == 8


def test_even_number_shortcuts_immediately():
    result = shor_attempt(N=12, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] == (2, 6)


def test_nontrivial_gcd_shortcuts_before_quantum_step():
    # gcd(3, 15) = 3, so the algorithm should succeed immediately.
    result = shor_attempt(N=15, a=3, rng=np.random.default_rng(123))
    assert result["success"] is True
    assert result["factors"] in [(3, 5), (5, 3)]


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
    assert "final_work_state" in result