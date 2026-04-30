import numpy as np
import pytest

from graphs.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    ErrorModelHammingDistanceSweep,
    HammingDistanceInstance,
    MEASUREMENT_BITFLIP_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
    build_argument_parser,
    build_instances,
    build_probability_values,
    error_config_for_model,
    error_model_display_label,
    error_models_filename_suffix,
    instance_display_label,
    instance_display_label_with_bits,
    instance_filename_suffix,
    instance_filename_suffix_with_bits,
    normalize_error_model,
    normalize_error_models,
    simulate_for_error_probability,
    validate_overlay_mode,
)
from shor_one_qubit.errors import ErrorConfig


def test_build_probability_values_linear_spacing():
    values = build_probability_values(0.0, 1.0, 3, log_p_grid=False)
    assert values == [0.0, 0.5, 1.0]


def test_build_probability_values_log_spacing():
    values = build_probability_values(1e-3, 1e-1, 3, log_p_grid=True)
    np.testing.assert_allclose(values, [1e-3, 1e-2, 1e-1])


def test_build_probability_values_log_spacing_requires_positive_minimum():
    with pytest.raises(ValueError, match="min-p > 0"):
        build_probability_values(0.0, 1.0, 5, log_p_grid=True)


def test_build_probability_values_one_log_point_requires_positive_minimum():
    with pytest.raises(ValueError, match="min-p > 0"):
        build_probability_values(0.0, 1.0, 1, log_p_grid=True)


def test_normalize_error_model_accepts_aliases():
    assert normalize_error_model("measurement") == MEASUREMENT_BITFLIP_MODEL
    assert normalize_error_model("block") == BLOCK_DEPOLARIZING_MODEL
    assert normalize_error_model("per-qubit") == PER_QUBIT_DEPOLARIZING_MODEL


def test_normalize_error_models_defaults_to_all_models():
    assert normalize_error_models(None) == [
        MEASUREMENT_BITFLIP_MODEL,
        BLOCK_DEPOLARIZING_MODEL,
        PER_QUBIT_DEPOLARIZING_MODEL,
    ]


def test_normalize_error_models_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="distinct"):
        normalize_error_models(["measurement", "measurement-bitflip"])


def test_error_config_for_model_sets_expected_probability_field():
    assert error_config_for_model("measurement", 0.2) == ErrorConfig(
        measurement_bitflip=True,
        measurement_bitflip_probability=0.2,
    )
    assert error_config_for_model("block", 0.3) == ErrorConfig(
        controlled_unitary_depolarizing_probability=0.3,
    )
    assert error_config_for_model("per-qubit", 0.4) == ErrorConfig(
        controlled_unitary_per_qubit_depolarizing_probability=0.4,
    )


def test_error_model_labels_and_filename_suffixes():
    assert error_model_display_label("measurement") == "Measurement bit-flip"
    assert (
        error_models_filename_suffix([
            MEASUREMENT_BITFLIP_MODEL,
            BLOCK_DEPOLARIZING_MODEL,
            PER_QUBIT_DEPOLARIZING_MODEL,
        ])
        == "all_error_models"
    )
    assert (
        error_models_filename_suffix(["measurement", "per-qubit"])
        == "measurement_bitflip_per_qubit_depolarizing"
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


def test_argument_parser_accepts_multiple_instances():
    parser = build_argument_parser()
    args = parser.parse_args(["21", "105", "--a", "11", "17"])

    assert args.N == [21, 105]
    assert args.a == [11, 17]


def test_build_instances_supports_single_N_with_multiple_bases():
    assert build_instances([21], [11, 13]) == [
        HammingDistanceInstance(N=21, a=11),
        HammingDistanceInstance(N=21, a=13),
    ]


def test_build_instances_supports_more_than_five_bases():
    assert build_instances([21], [2, 4, 5, 8, 10, 11]) == [
        HammingDistanceInstance(N=21, a=2),
        HammingDistanceInstance(N=21, a=4),
        HammingDistanceInstance(N=21, a=5),
        HammingDistanceInstance(N=21, a=8),
        HammingDistanceInstance(N=21, a=10),
        HammingDistanceInstance(N=21, a=11),
    ]


def test_build_instances_supports_multiple_N_values_with_matching_bases():
    assert build_instances([21, 105], [11, 17]) == [
        HammingDistanceInstance(N=21, a=11),
        HammingDistanceInstance(N=105, a=17),
    ]


def test_build_instances_supports_multiple_N_values_with_one_shared_base():
    assert build_instances([21, 33], [5]) == [
        HammingDistanceInstance(N=21, a=5),
        HammingDistanceInstance(N=33, a=5),
    ]


def test_build_instances_rejects_mismatched_N_and_base_counts():
    with pytest.raises(ValueError, match="same number of bases"):
        build_instances([21, 105], [11, 17, 19])


def test_instance_filename_suffix_supports_multiple_instances():
    assert instance_filename_suffix([HammingDistanceInstance(N=21, a=11)]) == "N21_a11"
    assert (
        instance_filename_suffix([
            HammingDistanceInstance(N=21, a=11),
            HammingDistanceInstance(N=21, a=13),
        ])
        == "N21_bases11_13"
    )
    assert (
        instance_filename_suffix([
            HammingDistanceInstance(N=21, a=11),
            HammingDistanceInstance(N=105, a=17),
        ])
        == "instances_N21_a11_N105_a17"
    )


def test_instance_filename_suffix_with_bits_places_t_after_base():
    sweeps = [
        ErrorModelHammingDistanceSweep(
            N=21,
            a=11,
            error_model=MEASUREMENT_BITFLIP_MODEL,
            num_phase_bits=11,
            results=[],
        ),
        ErrorModelHammingDistanceSweep(
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
        HammingDistanceInstance(N=21, a=11),
        HammingDistanceInstance(N=105, a=17),
    ]) == "instances = (N = 21, a = 11); (N = 105, a = 17)"


def test_instance_display_label_with_bits_places_t_after_base():
    sweeps = [
        ErrorModelHammingDistanceSweep(
            N=21,
            a=11,
            error_model=MEASUREMENT_BITFLIP_MODEL,
            num_phase_bits=11,
            results=[],
        ),
    ]

    assert instance_display_label_with_bits(sweeps) == "N = 21, a = 11, t = 11"


def test_validate_overlay_mode_rejects_multiple_instances_and_models():
    with pytest.raises(ValueError, match="exactly one --error-model"):
        validate_overlay_mode(
            [
                HammingDistanceInstance(N=21, a=11),
                HammingDistanceInstance(N=105, a=17),
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

    assert result.total_hamming_distance == 0
    assert result.average_hamming_distance == 0.0
    assert result.average_normalized_hamming_distance == 0.0
