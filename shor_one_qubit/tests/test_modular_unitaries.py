import numpy as np
import pytest

from shor_one_qubit.modular_unitaries import (
    register_size,
    work_dim,
    validate_multiplier,
    modmul_label,
    modmul_power_label,
    apply_modmul_state,
    apply_modmul_power_state,
)


def basis_state(index: int, dim: int) -> np.ndarray:
    psi = np.zeros(dim, dtype=complex)
    psi[index] = 1.0
    return psi


def test_register_size_and_work_dim_for_15():
    assert register_size(15) == 4
    assert work_dim(15) == 16


def test_register_size_and_work_dim_for_21():
    assert register_size(21) == 5
    assert work_dim(21) == 32


def test_validate_multiplier_accepts_coprime():
    validate_multiplier(2, 15)


def test_validate_multiplier_rejects_non_coprime():
    with pytest.raises(ValueError):
        validate_multiplier(3, 15)


def test_modmul_label_basic_orbit_for_N15_a2():
    assert modmul_label(1, 2, 15) == 2
    assert modmul_label(2, 2, 15) == 4
    assert modmul_label(4, 2, 15) == 8
    assert modmul_label(8, 2, 15) == 1


def test_modmul_label_leaves_unused_state_fixed():
    # For N=15 the work dimension is 16, so |15> is unused and should be fixed.
    assert modmul_label(15, 2, 15) == 15


def test_modmul_power_label_matches_expected_power():
    # 2^2 mod 15 = 4, so U_2^2 should act like multiplication by 4 mod 15.
    assert modmul_power_label(1, 2, 2, 15) == 4
    assert modmul_power_label(2, 2, 2, 15) == 8


def test_apply_modmul_state_on_basis_state():
    M = work_dim(15)
    psi = basis_state(1, M)
    out = apply_modmul_state(psi, 2, 15)

    expected = basis_state(2, M)
    np.testing.assert_allclose(out, expected)


def test_apply_modmul_power_state_on_basis_state():
    M = work_dim(15)
    psi = basis_state(1, M)

    # U_2^2 |1> = |4>
    out = apply_modmul_power_state(psi, a=2, power=2, N=15)
    expected = basis_state(4, M)

    np.testing.assert_allclose(out, expected)


def test_apply_modmul_power_matches_repeated_application():
    M = work_dim(15)
    psi = basis_state(1, M)

    repeated = apply_modmul_state(apply_modmul_state(psi, 2, 15), 2, 15)
    powered = apply_modmul_power_state(psi, a=2, power=2, N=15)

    np.testing.assert_allclose(powered, repeated)