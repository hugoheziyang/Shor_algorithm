from pathlib import Path

import numpy as np

from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_error_probability import (
    build_argument_parser,
    build_fixed_instance,
    output_filename_segment,
    plot_nearest_peak_distance_vs_error_probability,
    simulate_nearest_peak_distance_vs_error_probability,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_build_fixed_instance_computes_true_order():
    instance = build_fixed_instance(15, 2)

    assert instance.N == 15
    assert instance.a == 2
    assert instance.true_order == 4


def test_argument_parser_accepts_fixed_instance_sweep_options():
    parser = build_argument_parser()
    args = parser.parse_args([
        "21",
        "--a",
        "11",
        "--algorithm",
        "kitaev",
        "standard",
        "--min-p",
        "0",
        "--max-p",
        "0.2",
        "--num-points",
        "3",
        "--classical-error-correction",
        "--include-half-inverse-order-reference",
    ])

    assert args.N == 21
    assert args.a == 11
    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.min_p == 0.0
    assert args.max_p == 0.2
    assert args.num_points == 3
    assert args.classical_error_correction
    assert args.include_inverse_order_reference


def test_output_filename_segment_marks_analytic_backend():
    assert (
        output_filename_segment(
            algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
            standard_backend=ANALYTIC_STANDARD_BACKEND,
            N=21,
            a=11,
            num_phase_bits=11,
        )
        == "compare_algorithms_analytic_N21_a11_t11"
    )


def test_output_filename_segment_marks_classical_error_correction():
    assert output_filename_segment(
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        N=21,
        a=11,
        num_phase_bits=11,
        include_classical_error_correction=True,
    ).endswith("_classical_ec")


def test_output_filename_segment_marks_inverse_order_reference():
    assert output_filename_segment(
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        N=21,
        a=11,
        num_phase_bits=11,
        include_inverse_order_reference=True,
    ).endswith("_with_half_inverse_order")


def test_standard_analytic_zero_error_has_identical_clean_noisy_distances():
    instance = build_fixed_instance(15, 2)

    results = simulate_nearest_peak_distance_vs_error_probability(
        instance=instance,
        algorithms=[STANDARD_ALGORITHM],
        probability_values=[0.0],
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert len(results) == 1
    assert results[0].total_clean_distance == results[0].total_noisy_distance
    assert results[0].same_peak_trials == results[0].trials


def test_standard_analytic_zero_error_correction_does_not_increase_distances():
    instance = build_fixed_instance(15, 2)

    results = simulate_nearest_peak_distance_vs_error_probability(
        instance=instance,
        algorithms=[STANDARD_ALGORITHM],
        probability_values=[0.0],
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        include_classical_error_correction=True,
    )

    assert len(results) == 1
    assert results[0].total_corrected_distance <= results[0].total_clean_distance


def test_plot_nearest_peak_distance_vs_error_probability_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    results = simulate_nearest_peak_distance_vs_error_probability(
        instance=instance,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        probability_values=[0.0, 0.1],
        trials=2,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )
    output_path = tmp_path / "nearest_vs_p.png"

    plot_nearest_peak_distance_vs_error_probability(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        probability_values=[0.0, 0.1],
        num_phase_bits=8,
        trials_per_probability=2,
    )

    assert output_path.exists()


def test_plot_nearest_peak_distance_vs_error_probability_corrected_only_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    results = simulate_nearest_peak_distance_vs_error_probability(
        instance=instance,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        probability_values=[0.0, 0.1],
        trials=2,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        include_classical_error_correction=True,
    )
    output_path = tmp_path / "nearest_vs_p_corrected.png"

    plot_nearest_peak_distance_vs_error_probability(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        probability_values=[0.0, 0.1],
        num_phase_bits=8,
        trials_per_probability=2,
        include_classical_error_correction=True,
    )

    assert output_path.exists()
