import numpy as np
import pytest

from shor_common.errors import (
    ErrorConfig,
    apply_measurement_bitflip,
    apply_controlled_unitary_depolarizing_error,
    apply_work_qubit_pauli,
    choose_work_depolarizing_error,
    choose_work_per_qubit_depolarizing_errors,
    choose_measurement_bitflip_mask,
    choose_measurement_bitflip_index,
    format_measurement_bitflip_probability,
    measurement_bitflip_filename_suffix,
    measurement_bitflip_label,
    measurement_bitflip_title_suffix,
    parse_measurement_bitflip_error_arg,
    validate_error_config,
)


def test_validate_error_config_defaults_to_no_errors():
    config = validate_error_config(None)
    assert config == ErrorConfig()


def test_validate_error_config_rejects_wrong_type():
    with pytest.raises(TypeError, match="error_config"):
        validate_error_config({"measurement_bitflip": True})


def test_validate_error_config_rejects_probability_without_enabled_flag():
    with pytest.raises(ValueError, match="measurement_bitflip_probability"):
        validate_error_config(
            ErrorConfig(
                measurement_bitflip=False,
                measurement_bitflip_probability=0.1,
            )
        )


def test_validate_error_config_rejects_probability_out_of_range():
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        validate_error_config(
            ErrorConfig(
                measurement_bitflip=True,
                measurement_bitflip_probability=1.5,
            )
        )


def test_validate_error_config_accepts_controlled_unitary_depolarizing_probability():
    config = validate_error_config(
        ErrorConfig(controlled_unitary_depolarizing_probability=0.25)
    )
    assert config.controlled_unitary_depolarizing_probability == 0.25
    assert config.to_dict() == {
        "measurement_bitflip": False,
        "controlled_unitary_depolarizing_probability": 0.25,
    }


def test_validate_error_config_accepts_per_qubit_depolarizing_probability():
    config = validate_error_config(
        ErrorConfig(controlled_unitary_per_qubit_depolarizing_probability=0.25)
    )
    assert config.controlled_unitary_per_qubit_depolarizing_probability == 0.25
    assert config.to_dict() == {
        "measurement_bitflip": False,
        "controlled_unitary_per_qubit_depolarizing_probability": 0.25,
    }


def test_validate_error_config_rejects_depolarizing_probability_out_of_range():
    with pytest.raises(ValueError, match="controlled_unitary_depolarizing_probability"):
        validate_error_config(
            ErrorConfig(controlled_unitary_depolarizing_probability=1.5)
        )


def test_validate_error_config_rejects_per_qubit_depolarizing_probability_out_of_range():
    with pytest.raises(
        ValueError,
        match="controlled_unitary_per_qubit_depolarizing_probability",
    ):
        validate_error_config(
            ErrorConfig(controlled_unitary_per_qubit_depolarizing_probability=1.5)
        )


def test_choose_measurement_bitflip_index_returns_none_when_disabled():
    index = choose_measurement_bitflip_index(
        5,
        error_config=ErrorConfig(measurement_bitflip=False),
        rng=np.random.default_rng(123),
    )
    assert index is None


def test_choose_measurement_bitflip_index_returns_valid_index_when_enabled():
    index = choose_measurement_bitflip_index(
        5,
        error_config=ErrorConfig(measurement_bitflip=True),
        rng=np.random.default_rng(123),
    )
    assert 0 <= index < 5


def test_choose_measurement_bitflip_index_returns_none_in_probability_mode():
    index = choose_measurement_bitflip_index(
        5,
        error_config=ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=0.25,
        ),
        rng=np.random.default_rng(123),
    )
    assert index is None


def test_choose_measurement_bitflip_mask_has_exactly_one_flip_in_single_mode():
    mask = choose_measurement_bitflip_mask(
        5,
        error_config=ErrorConfig(measurement_bitflip=True),
        rng=np.random.default_rng(123),
    )
    assert len(mask) == 5
    assert mask.count(True) == 1


def test_choose_measurement_bitflip_mask_matches_probability_extremes():
    mask_zero = choose_measurement_bitflip_mask(
        4,
        error_config=ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=0.0,
        ),
        rng=np.random.default_rng(123),
    )
    mask_one = choose_measurement_bitflip_mask(
        4,
        error_config=ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=1.0,
        ),
        rng=np.random.default_rng(123),
    )
    assert mask_zero == [False, False, False, False]
    assert mask_one == [True, True, True, True]


def test_apply_measurement_bitflip_flips_requested_bit():
    assert apply_measurement_bitflip(0, should_flip=False) == 0
    assert apply_measurement_bitflip(0, should_flip=True) == 1
    assert apply_measurement_bitflip(1, should_flip=True) == 0


def test_choose_work_depolarizing_error_respects_probability_extremes():
    assert (
        choose_work_depolarizing_error(
            3,
            error_config=ErrorConfig(controlled_unitary_depolarizing_probability=0.0),
            rng=np.random.default_rng(123),
        )
        is None
    )

    sampled_error = choose_work_depolarizing_error(
        3,
        error_config=ErrorConfig(controlled_unitary_depolarizing_probability=1.0),
        rng=np.random.default_rng(123),
    )
    assert sampled_error is not None
    work_qubit, pauli = sampled_error
    assert 0 <= work_qubit < 3
    assert pauli in ("X", "Y", "Z")


def test_choose_work_per_qubit_depolarizing_errors_respects_probability_extremes():
    assert (
        choose_work_per_qubit_depolarizing_errors(
            3,
            error_config=ErrorConfig(
                controlled_unitary_per_qubit_depolarizing_probability=0.0
            ),
            rng=np.random.default_rng(123),
        )
        == []
    )

    sampled_errors = choose_work_per_qubit_depolarizing_errors(
        3,
        error_config=ErrorConfig(
            controlled_unitary_per_qubit_depolarizing_probability=1.0
        ),
        rng=np.random.default_rng(123),
    )
    assert len(sampled_errors) == 3
    assert [work_qubit for work_qubit, _ in sampled_errors] == [0, 1, 2]
    assert all(pauli in ("X", "Y", "Z") for _, pauli in sampled_errors)


def test_apply_work_qubit_pauli_x_on_joint_state():
    psi = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=complex,
    )

    out = apply_work_qubit_pauli(psi, work_qubit=0, pauli="X")
    expected = np.array(
        [
            [2.0, 1.0, 4.0, 3.0],
            [6.0, 5.0, 8.0, 7.0],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(out, expected)


def test_apply_work_qubit_pauli_z_on_joint_state():
    psi = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=complex,
    )

    out = apply_work_qubit_pauli(psi, work_qubit=1, pauli="Z")
    expected = np.array(
        [
            [1.0, 2.0, -3.0, -4.0],
            [5.0, 6.0, -7.0, -8.0],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(out, expected)


def test_apply_work_qubit_pauli_y_on_joint_state():
    psi = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=complex,
    )

    out = apply_work_qubit_pauli(psi, work_qubit=0, pauli="Y")
    expected = np.array(
        [
            [-2.0j, 1.0j],
            [-4.0j, 3.0j],
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(out, expected)


def test_apply_controlled_unitary_depolarizing_error_changes_state_when_forced():
    psi = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=complex,
    )

    out = apply_controlled_unitary_depolarizing_error(
        psi,
        error_config=ErrorConfig(controlled_unitary_depolarizing_probability=1.0),
        rng=np.random.default_rng(4),
    )

    assert out.shape == psi.shape
    np.testing.assert_allclose(np.linalg.norm(out), np.linalg.norm(psi))


def test_apply_controlled_unitary_per_qubit_depolarizing_error_when_forced():
    psi = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=complex,
    )

    out = apply_controlled_unitary_depolarizing_error(
        psi,
        error_config=ErrorConfig(
            controlled_unitary_per_qubit_depolarizing_probability=1.0
        ),
        rng=np.random.default_rng(4),
    )

    assert out.shape == psi.shape
    np.testing.assert_allclose(np.linalg.norm(out), np.linalg.norm(psi))


def test_parse_measurement_bitflip_error_arg_supports_both_modes():
    assert parse_measurement_bitflip_error_arg(None) == ErrorConfig()
    assert parse_measurement_bitflip_error_arg("single") == ErrorConfig(
        measurement_bitflip=True
    )
    assert parse_measurement_bitflip_error_arg("0.2") == ErrorConfig(
        measurement_bitflip=True,
        measurement_bitflip_probability=0.2,
    )


def test_measurement_bitflip_label_and_suffix_reflect_mode():
    assert measurement_bitflip_label(ErrorConfig()) == "disabled"
    assert measurement_bitflip_label(ErrorConfig(measurement_bitflip=True)).startswith(
        "single random bit-flip"
    )
    assert measurement_bitflip_label(
        ErrorConfig(
            measurement_bitflip=True,
            measurement_bitflip_probability=0.2,
        )
    ).endswith("p=2e-1")
    assert measurement_bitflip_filename_suffix(ErrorConfig()) == ""
    assert (
        measurement_bitflip_filename_suffix(ErrorConfig(measurement_bitflip=True))
        == "_measurement_bitflip_error"
    )
    assert (
        measurement_bitflip_filename_suffix(
            ErrorConfig(
                measurement_bitflip=True,
                measurement_bitflip_probability=0.2,
            )
        )
        == "_measurement_bitflip_error_2e-1"
    )


def test_measurement_bitflip_probability_format_and_title_suffix():
    assert format_measurement_bitflip_probability(0.1) == "1e-1"
    assert format_measurement_bitflip_probability(0.05) == "5e-2"
    assert format_measurement_bitflip_probability(1.0) == "1e0"
    assert (
        measurement_bitflip_title_suffix(ErrorConfig(measurement_bitflip=True))
        == " (single random measurement bit-flip error)"
    )
    assert (
        measurement_bitflip_title_suffix(
            ErrorConfig(
                measurement_bitflip=True,
                measurement_bitflip_probability=0.1,
            )
        )
        == " (measurement bit-flip error, p = 1e-1)"
    )
