from fractions import Fraction
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphs.measurement_error.plot_cf_bound_order_recovery_vs_order import (
    A_PROBABILITY,
    CFBoundOrderRecoveryResult,
    adaptive_probability_ylim,
    apply_adaptive_probability_yaxis,
    build_argument_parser,
    circular_fraction_distance,
    figure_output_path_for_probability,
    nearest_order_phase_distance,
    normalized_cf_error,
    plot_cf_bound_quantity_overlay,
    probability_tick_decimals,
    quantity_overlay_output_path,
    simulate_cf_bound_order_recovery_for_instance,
    within_cf_sufficient_bound,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_circular_fraction_distance_wraps_mod_one():
    assert circular_fraction_distance(Fraction(15, 16), Fraction(0, 1)) == Fraction(1, 16)
    assert circular_fraction_distance(Fraction(1, 4), Fraction(1, 2)) == Fraction(1, 4)


def test_normalized_cf_error_matches_sufficient_bound():
    assert normalized_cf_error(Fraction(1, 4), true_order=4) == 0
    assert normalized_cf_error(Fraction(1, 4) + Fraction(1, 32), true_order=4) == 1
    assert normalized_cf_error(Fraction(1, 4) + Fraction(1, 16), true_order=4) == 2


def test_nearest_order_phase_distance_handles_wraparound():
    distance, numerator = nearest_order_phase_distance(
        Fraction(15, 16),
        true_order=4,
    )

    assert distance == Fraction(1, 16)
    assert numerator == 0


def test_within_cf_sufficient_bound_uses_measured_bits():
    assert within_cf_sufficient_bound([0, 1, 0, 0, 0], true_order=4)
    assert not within_cf_sufficient_bound([0, 1, 1, 0, 0], true_order=4)


def test_result_conditional_probability_is_nan_without_outside_samples():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = CFBoundOrderRecoveryResult(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        measurement_error_probability=0.0,
        trials=3,
        num_phase_bits=8,
        within_bound_trials=3,
        recovered_trials=2,
        recovered_within_bound_trials=2,
        recovered_outside_bound_trials=0,
    )

    assert result.outside_bound_trials == 0
    assert math.isnan(result.recovered_given_outside_bound_probability)
    assert result.recovered_given_within_bound_probability == 2 / 3


def test_adaptive_probability_ylim_expands_small_probabilities():
    lower, upper = adaptive_probability_ylim([0.0, 0.01, 0.02])

    assert lower == 0.0
    assert 0.02 < upper < 0.10


def test_adaptive_probability_ylim_handles_values_near_one():
    lower, upper = adaptive_probability_ylim([0.98, 1.0])

    assert 0.9 < lower < 0.98
    assert upper == 1.0


def test_adaptive_probability_ylim_ignores_nan_values():
    assert adaptive_probability_ylim([math.nan]) == (0.0, 1.0)


def test_probability_tick_decimals_prevents_repeated_labels():
    assert probability_tick_decimals(0.0, 0.33) == 2
    assert probability_tick_decimals(0.73, 1.0) == 2
    assert probability_tick_decimals(0.0, 1.0) == 1


def test_adaptive_probability_yaxis_uses_distinct_visible_labels():
    fig, ax = plt.subplots()

    apply_adaptive_probability_yaxis(ax, [0.0, 0.30])
    fig.canvas.draw()
    labels = [
        label.get_text()
        for label in ax.get_yticklabels()
        if label.get_visible() and label.get_text()
    ]
    plt.close(fig)

    assert len(labels) == len(set(labels))


def test_standard_analytic_simulation_tracks_counts():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_cf_bound_order_recovery_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        trials=5,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.trials == 5
    assert result.within_bound_trials + result.outside_bound_trials == 5
    assert result.recovered_trials <= 5
    assert (
        result.recovered_within_bound_trials
        + result.recovered_outside_bound_trials
        == result.recovered_trials
    )


def test_argument_parser_accepts_multiple_probabilities():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--algorithm",
        "kitaev",
        "standard",
        "--error-probabilities",
        "0",
        "0.1",
        "--overlay-error-probabilities",
    ])

    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.error_probabilities == [0.0, 0.1]
    assert args.overlay_error_probabilities is True


def test_custom_output_path_gets_probability_suffix_for_multi_probability_run():
    output_path = figure_output_path_for_probability(
        output_path=Path("figure.png"),
        probability=0.1,
        all_probabilities=[0.0, 0.1],
        algorithms=[STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=Path("instances.json"),
        num_phase_bits=8,
    )

    assert output_path == Path("figure_p_1em1.png")


def test_custom_output_path_gets_quantity_overlay_suffix():
    output_path = quantity_overlay_output_path(
        output_path=Path("figure.png"),
        quantity=A_PROBABILITY,
        algorithms=[STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=Path("instances.json"),
        num_phase_bits=8,
        measurement_error_probabilities=[0.0, 0.1],
    )

    assert output_path == Path("figure_overlay_within_bound.png")


def test_plot_cf_bound_quantity_overlay_writes_file(tmp_path):
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    results = [
        CFBoundOrderRecoveryResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.0,
            trials=4,
            num_phase_bits=8,
            within_bound_trials=4,
            recovered_trials=3,
            recovered_within_bound_trials=3,
            recovered_outside_bound_trials=0,
        ),
        CFBoundOrderRecoveryResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trials=4,
            num_phase_bits=8,
            within_bound_trials=2,
            recovered_trials=2,
            recovered_within_bound_trials=1,
            recovered_outside_bound_trials=1,
        ),
    ]
    output_path = tmp_path / "overlay.png"

    plot_cf_bound_quantity_overlay(
        output_path=output_path,
        results=results,
        algorithms=[STANDARD_ALGORITHM],
        num_phase_bits=8,
        measurement_error_probabilities=[0.0, 0.1],
        quantity=A_PROBABILITY,
        trials_per_instance=4,
    )

    assert output_path.exists()
