import numpy as np
import pytest

from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_error_probability import (
    build_fixed_instance,
)
from graphs.measurement_error.plot_noisy_cf_membership_vs_order import (
    adaptive_probability_ylim,
    bit_numbers_display_label,
    build_argument_parser,
    normalize_reference_curves,
    output_filename_segment,
    plot_noisy_cf_membership_vs_order,
    reference_curve_values,
    safe_tail_bit_count,
    safe_tail_propagation_reference_values,
    simulate_noisy_cf_membership_vs_order,
    single_bitflip_bits_for_instance,
)
from graphs.measurement_error.plot_noisy_cf_membership_bar import NoisyCFMembershipResult
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_single_bitflip_bits_for_instance_uses_safe_tail_range():
    instance = build_fixed_instance(15, 2)

    assert single_bitflip_bits_for_instance(
        instance=instance,
        num_phase_bits=10,
        single_measurement_bitflip_range="safe-tail",
    ) == (5, 6, 7, 8, 9, 10)


def test_argument_parser_accepts_membership_vs_order_options():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--algorithm",
        "kitaev",
        "standard",
        "--single-measurement-bitflip-range",
        "safe-tail",
        "--classical-error-correction",
        "--reference-curves",
        "mp",
        "2mp",
        "--include-safe-tail-propagation-reference",
        "--log-x",
    ])

    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.single_measurement_bitflip_range == ["safe-tail"]
    assert args.reference_curves == ["mp", "2mp"]
    assert args.include_safe_tail_propagation_reference
    assert args.classical_error_correction
    assert args.log_x


def test_argument_parser_accepts_multiple_bitflip_ranges():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--single-measurement-bitflip-range",
        "leading",
        "dangerous",
    ])

    assert args.single_measurement_bitflip_range == ["leading", "dangerous"]


def test_output_filename_segment_marks_safe_tail_range():
    assert (
        output_filename_segment(
            algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
            standard_backend=ANALYTIC_STANDARD_BACKEND,
            instances_file=build_argument_parser().parse_args([]).instances_file,
            num_phase_bits=29,
            measurement_error_probability=0.1,
            single_measurement_bitflip_range="safe-tail",
        )
        == "compare_algorithms_analytic_instances_order_recovery_instances_single_measurement_bitflip_safe_tail_t29"
    )


def test_output_filename_segment_marks_combined_ranges():
    assert "single_measurement_bitflip_leading_dangerous" in output_filename_segment(
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=build_argument_parser().parse_args([]).instances_file,
        num_phase_bits=29,
        measurement_error_probability=0.1,
        single_measurement_bitflip_range=("leading", "dangerous"),
    )


def test_output_filename_segment_marks_reference_curves():
    assert output_filename_segment(
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=build_argument_parser().parse_args([]).instances_file,
        num_phase_bits=29,
        measurement_error_probability=0.1,
        reference_curves=("mp", "2mp"),
    ).endswith("_refs_mp_two_mp")


def test_output_filename_segment_marks_safe_tail_propagation_reference():
    assert output_filename_segment(
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=build_argument_parser().parse_args([]).instances_file,
        num_phase_bits=29,
        measurement_error_probability=0.1,
        reference_curves=("mp",),
        include_safe_tail_propagation_reference=True,
    ).endswith("_p_1em1_t29_refs_mp_safe_tail_prop_ref")


def test_normalize_reference_curves_deduplicates_in_order():
    assert normalize_reference_curves(["2mp", "mp", "2mp"]) == ("2mp", "mp")


def test_reference_curve_values_use_exponential_order_resolution_reference():
    instances = [
        build_fixed_instance(15, 2),
        build_fixed_instance(21, 11),
    ]

    assert reference_curve_values(instances, 0.1, "mp") == pytest.approx(
        [1 - np.exp(-0.2), 1 - np.exp(-0.3)]
    )
    assert reference_curve_values(instances, 0.5, "2mp") == pytest.approx(
        [1 - np.exp(-2.0), 1 - np.exp(-3.0)]
    )


def test_safe_tail_propagation_reference_values_use_tail_probability():
    instance = build_fixed_instance(15, 2)
    result = NoisyCFMembershipResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        measurement_error_probability=0.1,
        trials=10,
        num_phase_bits=10,
        inside_trials=7,
    )

    assert safe_tail_bit_count(num_phase_bits=10, true_order=4) == 6
    assert safe_tail_propagation_reference_values(
        series=[result],
        num_phase_bits=10,
        measurement_error_probability=0.1,
        reference_curve="mp",
    ) == pytest.approx([1 - np.exp(-(0.2 + 0.1 * 6 * 0.3))])


def test_bit_numbers_display_label_marks_disjoint_runs():
    assert bit_numbers_display_label((1, 2, 5, 6, 9)) == "1..2,5..6,9"


def test_adaptive_probability_ylim_uses_relative_padding():
    assert adaptive_probability_ylim([0.2]) == (0.0, 0.22400000000000003)
    assert adaptive_probability_ylim([0.0]) == (0.0, 0.05)


def test_simulate_membership_vs_order_runs_standard_analytic_safe_tail():
    instance = build_fixed_instance(15, 2)

    results = simulate_noisy_cf_membership_vs_order(
        instances=[instance],
        algorithms=[STANDARD_ALGORITHM],
        num_phase_bits=8,
        measurement_error_probability=0.1,
        trials_per_instance=3,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        single_measurement_bitflip_range="safe-tail",
    )

    assert len(results) == 1
    assert results[0].trials == 3


def test_plot_noisy_cf_membership_vs_order_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    results = [
        NoisyCFMembershipResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            measurement_error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            inside_trials=6,
        ),
        NoisyCFMembershipResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            inside_trials=8,
        ),
    ]
    output_path = tmp_path / "membership_vs_order.png"
    safe_tail_reference_results = [
        NoisyCFMembershipResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            measurement_error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            inside_trials=7,
        ),
        NoisyCFMembershipResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            inside_trials=9,
        ),
    ]

    plot_noisy_cf_membership_vs_order(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        num_phase_bits=8,
        measurement_error_probability=0.1,
        trials_per_instance=10,
        reference_curves=("mp",),
        include_safe_tail_propagation_reference=True,
        safe_tail_reference_results=safe_tail_reference_results,
    )

    assert output_path.exists()
