import math
from pathlib import Path

import numpy as np

from graphs.measurement_error.plot_conditional_cf_bound_comparison_vs_order import (
    ConditionalCFBoundComparisonResult,
    KITAEV_SAME_PEAK_CONDITION,
    KITAEV_SHIFTED_PEAK_CONDITION,
    STANDARD_SAME_PEAK_CONDITION,
    STANDARD_SHIFTED_PEAK_CONDITION,
    build_argument_parser,
    condition_algorithm,
    condition_display_label,
    condition_is_eligible,
    output_path_for_plot,
    plot_conditional_cf_bound_comparison,
    simulate_conditional_cf_bound_for_instance,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_condition_helpers():
    assert condition_algorithm(STANDARD_SAME_PEAK_CONDITION) == STANDARD_ALGORITHM
    assert condition_algorithm(STANDARD_SHIFTED_PEAK_CONDITION) == STANDARD_ALGORITHM
    assert condition_algorithm(KITAEV_SAME_PEAK_CONDITION) == KITAEV_ALGORITHM
    assert condition_algorithm(KITAEV_SHIFTED_PEAK_CONDITION) == KITAEV_ALGORITHM
    assert "standard" in condition_display_label(STANDARD_SAME_PEAK_CONDITION)
    assert "Kitaev" in condition_display_label(KITAEV_SAME_PEAK_CONDITION)
    assert condition_is_eligible(STANDARD_SAME_PEAK_CONDITION, 2, 2)
    assert not condition_is_eligible(STANDARD_SAME_PEAK_CONDITION, 2, 3)
    assert condition_is_eligible(KITAEV_SAME_PEAK_CONDITION, 2, 2)
    assert not condition_is_eligible(KITAEV_SAME_PEAK_CONDITION, 2, 3)
    assert condition_is_eligible(STANDARD_SHIFTED_PEAK_CONDITION, 2, 3)
    assert not condition_is_eligible(STANDARD_SHIFTED_PEAK_CONDITION, 2, 2)
    assert condition_is_eligible(KITAEV_SHIFTED_PEAK_CONDITION, 2, 3)
    assert not condition_is_eligible(KITAEV_SHIFTED_PEAK_CONDITION, 2, 2)


def test_conditional_probability_is_nan_without_denominator():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = ConditionalCFBoundComparisonResult(
        instance=instance,
        condition=KITAEV_SHIFTED_PEAK_CONDITION,
        measurement_error_probability=0.0,
        trials=4,
        num_phase_bits=8,
        eligible_trials=0,
        within_bound_trials=0,
    )

    assert math.isnan(result.conditional_probability)
    assert result.eligible_probability == 0.0


def test_argument_parser_accepts_probability_overlay():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--error-probabilities",
        "0",
        "0.1",
        "--bits",
        "29",
    ])

    assert args.error_probabilities == [0.0, 0.1]
    assert args.bits == 29


def test_output_path_appends_condition_group_to_custom_path():
    output_path = output_path_for_plot(
        output_path=Path("comparison.png"),
        condition_group="same_peak",
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=Path("instances.json"),
        num_phase_bits=8,
        measurement_error_probabilities=[0.0, 0.1],
    )

    assert output_path == Path("comparison_same_peak.png")


def test_standard_same_peak_zero_error_has_full_denominator():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_conditional_cf_bound_for_instance(
        instance=instance,
        condition=STANDARD_SAME_PEAK_CONDITION,
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.eligible_trials == 4
    assert 0 <= result.within_bound_trials <= 4


def test_kitaev_shifted_zero_error_has_empty_denominator():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_conditional_cf_bound_for_instance(
        instance=instance,
        condition=KITAEV_SHIFTED_PEAK_CONDITION,
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.eligible_trials == 0
    assert math.isnan(result.conditional_probability)


def test_standard_shifted_zero_error_has_empty_denominator():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_conditional_cf_bound_for_instance(
        instance=instance,
        condition=STANDARD_SHIFTED_PEAK_CONDITION,
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.eligible_trials == 0
    assert math.isnan(result.conditional_probability)


def test_plot_conditional_cf_bound_comparison_writes_file(tmp_path):
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    results = [
        ConditionalCFBoundComparisonResult(
            instance=instance,
            condition=STANDARD_SAME_PEAK_CONDITION,
            measurement_error_probability=0.1,
            trials=5,
            num_phase_bits=8,
            eligible_trials=4,
            within_bound_trials=3,
        ),
        ConditionalCFBoundComparisonResult(
            instance=instance,
            condition=KITAEV_SAME_PEAK_CONDITION,
            measurement_error_probability=0.1,
            trials=5,
            num_phase_bits=8,
            eligible_trials=3,
            within_bound_trials=2,
        ),
        ConditionalCFBoundComparisonResult(
            instance=instance,
            condition=STANDARD_SHIFTED_PEAK_CONDITION,
            measurement_error_probability=0.1,
            trials=5,
            num_phase_bits=8,
            eligible_trials=1,
            within_bound_trials=1,
        ),
        ConditionalCFBoundComparisonResult(
            instance=instance,
            condition=KITAEV_SHIFTED_PEAK_CONDITION,
            measurement_error_probability=0.1,
            trials=5,
            num_phase_bits=8,
            eligible_trials=2,
            within_bound_trials=1,
        ),
    ]
    output_path = tmp_path / "conditional.png"

    plot_conditional_cf_bound_comparison(
        output_path=output_path,
        results=results,
        condition_group="same_peak",
        num_phase_bits=8,
        measurement_error_probabilities=[0.1],
        trials_per_instance=5,
    )

    assert output_path.exists()
