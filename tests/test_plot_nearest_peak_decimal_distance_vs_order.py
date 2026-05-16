from fractions import Fraction
from pathlib import Path

import numpy as np

from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_order import (
    CLEAN_PHASE,
    CORRECTED_PHASE,
    NOISY_PHASE,
    NearestPeakDistanceResult,
    adaptive_distance_ylim,
    build_argument_parser,
    cf_sufficient_bound_distance,
    figure_output_path_for_probability,
    inverse_order_reference_distance,
    minimum_single_bit_flip_nearest_peak_distance,
    nearest_peak_distance,
    phase_choices,
    plot_nearest_peak_distance_for_probability,
    result_phase_distance,
    simulate_nearest_peak_distance_for_instance,
    truncated_cf_bound_curve,
    truncated_inverse_order_curve,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_nearest_peak_distance_uses_own_nearest_order_lattice_point():
    distance, numerator = nearest_peak_distance([0, 1, 0, 0], true_order=4)
    assert distance == 0
    assert numerator == 1

    distance, numerator = nearest_peak_distance([0, 1, 0, 1], true_order=4)
    assert distance == Fraction(1, 16)
    assert numerator == 1

    distance, numerator = nearest_peak_distance([1, 1, 1, 1], true_order=4)
    assert distance == Fraction(1, 16)
    assert numerator == 0


def test_minimum_single_bit_flip_nearest_peak_distance_uses_candidate_peak():
    distance, numerator, bits = minimum_single_bit_flip_nearest_peak_distance(
        [0, 1, 0, 1],
        true_order=4,
    )

    assert distance == 0
    assert numerator == 1
    assert bits == [0, 1, 0, 0]


def test_minimum_single_bit_flip_nearest_peak_distance_keeps_noisy_on_tie():
    distance, numerator, bits = minimum_single_bit_flip_nearest_peak_distance(
        [0, 1, 0, 0],
        true_order=4,
    )

    assert distance == 0
    assert numerator == 1
    assert bits == [0, 1, 0, 0]


def test_cf_sufficient_bound_distance_matches_one_over_two_r_squared():
    assert cf_sufficient_bound_distance(4) == Fraction(1, 32)


def test_inverse_order_reference_distance_matches_one_over_two_r():
    assert inverse_order_reference_distance(4) == Fraction(1, 8)


def test_truncated_cf_bound_curve_starts_where_bound_fits_y_axis():
    instances = [
        OrderRecoveryInstance(
            N=15,
            a=2,
            true_order=true_order,
            num_work_qubits=4,
            default_phase_bits=9,
        )
        for true_order in (2, 4, 8)
    ]

    x_values, y_values = truncated_cf_bound_curve(instances, y_upper=0.04)

    assert x_values == [4, 8]
    assert y_values == [1 / 32, 1 / 128]


def test_truncated_inverse_order_curve_starts_where_reference_fits_y_axis():
    instances = [
        OrderRecoveryInstance(
            N=15,
            a=2,
            true_order=true_order,
            num_work_qubits=4,
            default_phase_bits=9,
        )
        for true_order in (2, 4, 8)
    ]

    x_values, y_values = truncated_inverse_order_curve(instances, y_upper=0.20)

    assert x_values == [4, 8]
    assert y_values == [1 / 8, 1 / 16]


def test_adaptive_distance_ylim_uses_relative_padding_for_small_values():
    lower, upper = adaptive_distance_ylim([4.5e-4])

    assert lower == 0.0
    assert upper == 4.5e-4 * 1.12


def test_result_phase_distance_selects_clean_or_noisy_average():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = NearestPeakDistanceResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        measurement_error_probability=0.1,
        trials=4,
        num_phase_bits=8,
        total_clean_distance=Fraction(1, 8),
        total_noisy_distance=Fraction(1, 4),
        same_peak_trials=2,
    )

    assert result_phase_distance(result, CLEAN_PHASE) == 1 / 32
    assert result_phase_distance(result, NOISY_PHASE) == 1 / 16
    assert result.same_peak_probability == 0.5


def test_result_phase_distance_selects_corrected_average():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = NearestPeakDistanceResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        measurement_error_probability=0.1,
        trials=4,
        num_phase_bits=8,
        total_clean_distance=Fraction(1, 8),
        total_noisy_distance=Fraction(1, 4),
        same_peak_trials=2,
        total_corrected_distance=Fraction(1, 16),
        corrected_same_peak_trials=3,
    )

    assert result_phase_distance(result, CORRECTED_PHASE) == 1 / 64
    assert result.corrected_same_peak_probability == 0.75
    assert phase_choices(include_classical_error_correction=True) == (CORRECTED_PHASE,)


def test_argument_parser_accepts_error_probabilities():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--algorithm",
        "kitaev",
        "standard",
        "--error-probabilities",
        "0",
        "0.1",
        "--classical-error-correction",
        "--include-half-inverse-order-reference",
    ])

    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.error_probabilities == [0.0, 0.1]
    assert args.classical_error_correction
    assert args.include_inverse_order_reference
    assert args.standard_backend == ANALYTIC_STANDARD_BACKEND


def test_output_path_appends_probability_to_custom_path_when_needed():
    output_path = figure_output_path_for_probability(
        output_path=Path("nearest.png"),
        probability=0.1,
        all_probabilities=[0.0, 0.1],
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=Path("instances.json"),
        num_phase_bits=8,
    )

    assert output_path == Path("nearest_p_1em1.png")


def test_output_path_marks_inverse_order_reference_for_default_path():
    output_path = figure_output_path_for_probability(
        output_path=None,
        probability=0.1,
        all_probabilities=[0.1],
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=Path("instances.json"),
        num_phase_bits=8,
        include_inverse_order_reference=True,
    )

    assert output_path.name.endswith("_with_half_inverse_order.png")


def test_standard_analytic_zero_error_has_same_clean_and_noisy_peak_distance():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_nearest_peak_distance_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.total_clean_distance == result.total_noisy_distance
    assert result.same_peak_trials == result.trials


def test_standard_analytic_zero_error_correction_does_not_increase_distance():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_nearest_peak_distance_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        include_classical_error_correction=True,
    )

    assert result.total_corrected_distance <= result.total_clean_distance


def test_plot_nearest_peak_distance_for_probability_writes_file(tmp_path):
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    results = [
        NearestPeakDistanceResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            measurement_error_probability=0.1,
            trials=4,
            num_phase_bits=8,
            total_clean_distance=Fraction(1, 8),
            total_noisy_distance=Fraction(1, 4),
            same_peak_trials=2,
        ),
        NearestPeakDistanceResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trials=4,
            num_phase_bits=8,
            total_clean_distance=Fraction(1, 16),
            total_noisy_distance=Fraction(3, 16),
            same_peak_trials=3,
        ),
    ]
    output_path = tmp_path / "nearest.png"

    plot_nearest_peak_distance_for_probability(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        num_phase_bits=8,
        measurement_error_probability=0.1,
        trials_per_instance=4,
    )

    assert output_path.exists()


def test_plot_nearest_peak_distance_for_probability_corrected_only_writes_file(tmp_path):
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    results = [
        NearestPeakDistanceResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            measurement_error_probability=0.1,
            trials=4,
            num_phase_bits=8,
            total_clean_distance=Fraction(1, 8),
            total_noisy_distance=Fraction(1, 4),
            same_peak_trials=2,
            total_corrected_distance=Fraction(1, 32),
            corrected_same_peak_trials=3,
        ),
        NearestPeakDistanceResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trials=4,
            num_phase_bits=8,
            total_clean_distance=Fraction(1, 16),
            total_noisy_distance=Fraction(3, 16),
            same_peak_trials=3,
            total_corrected_distance=Fraction(1, 64),
            corrected_same_peak_trials=4,
        ),
    ]
    output_path = tmp_path / "nearest_corrected.png"

    plot_nearest_peak_distance_for_probability(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        num_phase_bits=8,
        measurement_error_probability=0.1,
        trials_per_instance=4,
        include_classical_error_correction=True,
    )

    assert output_path.exists()
