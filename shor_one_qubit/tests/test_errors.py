import numpy as np
import pytest

from shor_one_qubit.errors import (
    ErrorConfig,
    apply_measurement_bitflip,
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
