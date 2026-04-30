import numpy as np
import pytest

from graphs.plot_decimal_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    DecimalDistanceInstance,
    ErrorModelDecimalDistanceSweep,
    MEASUREMENT_BITFLIP_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
    build_argument_parser,
    build_instances,
    error_models_filename_suffix,
    instance_display_label,
    instance_display_label_with_bits,
    instance_filename_suffix,
    instance_filename_suffix_with_bits,
    normalize_error_models,
    simulate_for_error_probability,
    validate_overlay_mode,
)


def test_argument_parser_accepts_multiple_error_models():
    parser = build_argument_parser()
    args = parser.parse_args([
        "21",
        "--a",
        "11",
        "--error-model",
        "measurement",
        "block",
    ])

    assert args.N == [21]
    assert args.a == [11]
    assert args.error_model == ["measurement", "block"]


def test_build_instances_supports_multiple_N_values_with_matching_bases():
    assert build_instances([21, 105], [11, 17]) == [
        DecimalDistanceInstance(N=21, a=11),
        DecimalDistanceInstance(N=105, a=17),
    ]


def test_build_instances_supports_more_than_five_bases():
    assert build_instances([21], [2, 4, 5, 8, 10, 11]) == [
        DecimalDistanceInstance(N=21, a=2),
        DecimalDistanceInstance(N=21, a=4),
        DecimalDistanceInstance(N=21, a=5),
        DecimalDistanceInstance(N=21, a=8),
        DecimalDistanceInstance(N=21, a=10),
        DecimalDistanceInstance(N=21, a=11),
    ]


def test_instance_filename_suffix_supports_multiple_instances():
    assert instance_filename_suffix([DecimalDistanceInstance(N=21, a=11)]) == "N21_a11"
    assert (
        instance_filename_suffix([
            DecimalDistanceInstance(N=21, a=11),
            DecimalDistanceInstance(N=105, a=17),
        ])
        == "instances_N21_a11_N105_a17"
    )


def test_instance_filename_suffix_with_bits_places_t_after_base():
    sweeps = [
        ErrorModelDecimalDistanceSweep(
            N=21,
            a=11,
            error_model=MEASUREMENT_BITFLIP_MODEL,
            num_phase_bits=11,
            results=[],
        ),
        ErrorModelDecimalDistanceSweep(
            N=105,
            a=17,
            error_model=MEASUREMENT_BITFLIP_MODEL,
            num_phase_bits=15,
            results=[],
        ),
    ]

    assert (
        instance_filename_suffix_with_bits(sweeps)
        == "instances_N21_a11_t11_N105_a17_t15"
    )


def test_instance_display_label_supports_multiple_instances():
    assert instance_display_label([
        DecimalDistanceInstance(N=21, a=11),
        DecimalDistanceInstance(N=105, a=17),
    ]) == "instances = (N = 21, a = 11); (N = 105, a = 17)"


def test_instance_display_label_with_bits_places_t_after_base():
    sweeps = [
        ErrorModelDecimalDistanceSweep(
            N=21,
            a=11,
            error_model=MEASUREMENT_BITFLIP_MODEL,
            num_phase_bits=11,
            results=[],
        ),
    ]

    assert instance_display_label_with_bits(sweeps) == "N = 21, a = 11, t = 11"


def test_normalize_error_models_defaults_to_all_models():
    assert normalize_error_models(None) == [
        MEASUREMENT_BITFLIP_MODEL,
        BLOCK_DEPOLARIZING_MODEL,
        PER_QUBIT_DEPOLARIZING_MODEL,
    ]
    assert (
        error_models_filename_suffix(["measurement", "per-qubit"])
        == "measurement_bitflip_per_qubit_depolarizing"
    )


def test_validate_overlay_mode_rejects_multiple_instances_and_models():
    with pytest.raises(ValueError, match="exactly one --error-model"):
        validate_overlay_mode(
            [
                DecimalDistanceInstance(N=21, a=11),
                DecimalDistanceInstance(N=105, a=17),
            ],
            [MEASUREMENT_BITFLIP_MODEL, BLOCK_DEPOLARIZING_MODEL],
        )


def test_simulate_for_error_probability_is_zero_at_p_zero():
    result = simulate_for_error_probability(
        N=21,
        a=11,
        num_phase_bits=4,
        error_model=MEASUREMENT_BITFLIP_MODEL,
        error_probability=0.0,
        trials=3,
        rng=np.random.default_rng(1234),
    )

    assert result.total_decimal_distance == 0.0
    assert result.average_decimal_distance == 0.0
