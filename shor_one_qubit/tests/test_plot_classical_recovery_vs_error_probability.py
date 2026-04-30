import numpy as np
import pytest

from graphs.plot_classical_recovery_vs_error_probability import (
    InstanceRecoverySweep,
    RecoveryInstance,
    base_display_label,
    base_filename_suffix,
    build_argument_parser,
    build_instances,
    build_probability_values,
    instance_display_label,
    instance_display_label_with_bits,
    instance_filename_suffix,
    instance_filename_suffix_with_bits,
    kitaev_first_order_recovery_estimate,
    propagating_error_lower_bound,
    validate_bases,
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


def test_propagating_error_lower_bound_counts_no_error_and_final_bit_error():
    assert np.isclose(propagating_error_lower_bound(4, 0.1), 0.9 ** 3)
    assert np.isclose(propagating_error_lower_bound(1, 0.5), 1.0)


def test_kitaev_first_order_recovery_estimate_uses_ct_coefficient():
    estimate = kitaev_first_order_recovery_estimate(2, 0.1)
    assert np.isclose(estimate, (0.9 ** 2) + (1.5 * 0.1 * 0.9))


def test_validate_bases_accepts_up_to_five_distinct_coprime_bases():
    assert validate_bases(21, [2, 4, 5, 8, 10]) == [2, 4, 5, 8, 10]


def test_validate_bases_rejects_more_than_five_bases():
    with pytest.raises(ValueError, match="at most 5"):
        validate_bases(21, [2, 4, 5, 8, 10, 11])


def test_validate_bases_rejects_duplicate_bases():
    with pytest.raises(ValueError, match="distinct"):
        validate_bases(21, [11, 11])


def test_base_labels_preserve_single_base_filename_compatibility():
    assert base_filename_suffix([11]) == "a11"
    assert base_display_label([11]) == "a = 11"


def test_base_labels_support_multiple_bases():
    assert base_filename_suffix([11, 13, 17]) == "bases11_13_17"
    assert base_display_label([11, 13, 17]) == "bases = 11, 13, 17"


def test_argument_parser_accepts_multiple_bases():
    parser = build_argument_parser()
    args = parser.parse_args(["21", "--a", "11", "13"])
    assert args.N == [21]
    assert args.a == [11, 13]


def test_argument_parser_accepts_multiple_N_values():
    parser = build_argument_parser()
    args = parser.parse_args(["21", "105", "--a", "11", "17"])
    assert args.N == [21, 105]
    assert args.a == [11, 17]


def test_argument_parser_accepts_lower_bound_flag():
    parser = build_argument_parser()
    args = parser.parse_args(["21", "--a", "11", "--show-lower-bound"])
    assert args.show_lower_bound is True


def test_argument_parser_accepts_first_order_flag():
    parser = build_argument_parser()
    args = parser.parse_args(["21", "--a", "11", "--show-first-order"])
    assert args.show_first_order is True


def test_build_instances_supports_single_N_with_multiple_bases():
    assert build_instances([21], [11, 13]) == [
        RecoveryInstance(N=21, a=11),
        RecoveryInstance(N=21, a=13),
    ]


def test_build_instances_supports_multiple_N_values_with_matching_bases():
    assert build_instances([21, 105], [11, 17]) == [
        RecoveryInstance(N=21, a=11),
        RecoveryInstance(N=105, a=17),
    ]


def test_build_instances_supports_multiple_N_values_with_one_shared_base():
    assert build_instances([21, 33], [5]) == [
        RecoveryInstance(N=21, a=5),
        RecoveryInstance(N=33, a=5),
    ]


def test_build_instances_rejects_mismatched_N_and_base_counts():
    with pytest.raises(ValueError, match="same number of bases"):
        build_instances([21, 105], [11, 17, 19])


def test_instance_filename_suffix_preserves_single_N_compatibility():
    assert instance_filename_suffix([RecoveryInstance(N=21, a=11)]) == "N21_a11"
    assert (
        instance_filename_suffix([
            RecoveryInstance(N=21, a=11),
            RecoveryInstance(N=21, a=13),
        ])
        == "N21_bases11_13"
    )


def test_instance_filename_suffix_supports_multiple_N_values():
    assert (
        instance_filename_suffix([
            RecoveryInstance(N=21, a=11),
            RecoveryInstance(N=105, a=17),
        ])
        == "instances_N21_a11_N105_a17"
    )


def test_instance_filename_suffix_with_bits_places_t_after_base():
    sweeps = [
        InstanceRecoverySweep(N=21, a=11, num_phase_bits=11, results=[]),
        InstanceRecoverySweep(N=105, a=17, num_phase_bits=15, results=[]),
    ]

    assert (
        instance_filename_suffix_with_bits(sweeps)
        == "instances_N21_a11_t11_N105_a17_t15"
    )


def test_instance_display_label_supports_multiple_N_values():
    assert instance_display_label([
        RecoveryInstance(N=21, a=11),
        RecoveryInstance(N=105, a=17),
    ]) == "instances = (N = 21, a = 11); (N = 105, a = 17)"


def test_instance_display_label_with_bits_places_t_after_base():
    sweeps = [
        InstanceRecoverySweep(N=21, a=11, num_phase_bits=11, results=[]),
    ]

    assert instance_display_label_with_bits(sweeps) == "N = 21, a = 11, t = 11"
