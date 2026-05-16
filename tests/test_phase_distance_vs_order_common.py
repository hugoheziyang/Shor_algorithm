import numpy as np
import pytest

from graphs.measurement_error.phase_distance_vs_order_common import (
    DECIMAL_METRIC,
    HAMMING_METRIC,
    build_argument_parser,
    choose_single_measurement_bitflip_mask,
    metric_filename_label,
    minimum_single_bit_flip_candidate,
    normalize_single_measurement_bitflip_ranges,
    single_measurement_bitflip_bit_numbers,
    result_y_value,
    run_kitaev_phase_distance_trial,
    run_standard_phase_distance_trial,
    simulate_phase_distance_for_instance,
    validate_measurement_error_probabilities,
    validate_measurement_error_probability,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_validate_measurement_error_probability_accepts_unit_interval():
    assert validate_measurement_error_probability(0.0) == 0.0
    assert validate_measurement_error_probability(0.5) == 0.5
    assert validate_measurement_error_probability(1.0) == 1.0


def test_validate_measurement_error_probability_rejects_invalid_values():
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        validate_measurement_error_probability(-0.1)
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        validate_measurement_error_probability(1.1)


def test_validate_measurement_error_probabilities_deduplicates_in_order():
    assert validate_measurement_error_probabilities([0.1, 0.01, 0.1]) == [
        0.1,
        0.01,
    ]


def test_hamming_parser_accepts_normalized_flag():
    parser = build_argument_parser(HAMMING_METRIC)
    args = parser.parse_args([
        "--p",
        "0.2",
        "--normalized",
        "--max-order",
        "8",
    ])

    assert args.p == 0.2
    assert args.normalized is True
    assert args.max_order == 8


def test_hamming_parser_accepts_error_probability_overlay():
    parser = build_argument_parser(HAMMING_METRIC)
    args = parser.parse_args([
        "--error-probabilities",
        "0",
        "0.01",
        "0.1",
    ])

    assert args.error_probabilities == [0.0, 0.01, 0.1]


def test_decimal_parser_uses_default_config_file():
    parser = build_argument_parser(DECIMAL_METRIC)
    args = parser.parse_args([])

    assert args.instances_file.name == "order_recovery_instances.json"
    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.standard_backend == ANALYTIC_STANDARD_BACKEND


def test_standard_analytic_p_zero_has_zero_distance():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_phase_distance_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.total_hamming_distance == 0
    assert result.total_decimal_distance == 0.0
    assert result_y_value(result, HAMMING_METRIC) == 0.0
    assert result_y_value(result, DECIMAL_METRIC) == 0.0


def test_standard_analytic_p_one_flips_every_final_bit():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    trial = run_standard_phase_distance_trial(
        instance=instance,
        num_phase_bits=8,
        measurement_error_probability=1.0,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert trial.measurement_bitflip_mask == [True] * 8
    assert trial.hamming_distance == 8
    assert 0.0 <= trial.decimal_distance <= 0.5


def test_standard_analytic_single_measurement_bitflip_flips_one_final_bit():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    trial = run_standard_phase_distance_trial(
        instance=instance,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        single_measurement_bitflip=True,
    )

    assert sum(trial.measurement_bitflip_mask) == 1
    assert trial.hamming_distance == 1


def test_single_measurement_bitflip_ranges_use_order_resolution_bits():
    assert single_measurement_bitflip_bit_numbers(
        num_phase_bits=10,
        true_order=8,
        range_name="leading",
    ) == (1, 2, 3)
    assert single_measurement_bitflip_bit_numbers(
        num_phase_bits=10,
        true_order=8,
        range_name="dangerous",
    ) == (4, 5, 6)
    assert single_measurement_bitflip_bit_numbers(
        num_phase_bits=10,
        true_order=8,
        range_name="safe-tail",
    ) == (7, 8, 9, 10)
    assert single_measurement_bitflip_bit_numbers(
        num_phase_bits=10,
        true_order=8,
        range_name=("leading", "dangerous"),
    ) == (1, 2, 3, 4, 5, 6)


def test_normalize_single_measurement_bitflip_ranges_deduplicates_in_order():
    assert normalize_single_measurement_bitflip_ranges(
        ["dangerous", "leading", "dangerous"]
    ) == ("dangerous", "leading")


def test_choose_single_measurement_bitflip_mask_respects_allowed_bits():
    mask = choose_single_measurement_bitflip_mask(
        6,
        rng=np.random.default_rng(123),
        bit_numbers=(5, 6),
    )

    assert sum(mask) == 1
    assert mask[:4] == [False] * 4


def test_kitaev_p_zero_has_zero_distance():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_phase_distance_for_instance(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        trials=3,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
    )

    assert result.total_hamming_distance == 0
    assert result.total_decimal_distance == 0.0


def test_kitaev_single_measurement_bitflip_uses_one_error_location():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    trial = run_kitaev_phase_distance_trial(
        instance=instance,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        rng=np.random.default_rng(123),
        single_measurement_bitflip=True,
    )

    assert sum(trial.measurement_bitflip_mask) == 1
    assert trial.hamming_distance >= 1


def test_metric_filename_labels():
    assert metric_filename_label(HAMMING_METRIC) == "hamming"
    assert metric_filename_label(HAMMING_METRIC, normalized=True) == "normalized_hamming"
    assert metric_filename_label(DECIMAL_METRIC) == "decimal"


def test_minimum_single_bit_flip_candidate_updates_only_on_lower_score():
    best_bits, best_score = minimum_single_bit_flip_candidate(
        [1, 0, 1],
        score_candidate=lambda bits: sum(bits),
    )

    assert best_bits == [0, 0, 1]
    assert best_score == 1


def test_minimum_single_bit_flip_candidate_keeps_original_on_tie():
    best_bits, best_score = minimum_single_bit_flip_candidate(
        [1, 0],
        score_candidate=lambda bits: 0,
    )

    assert best_bits == [1, 0]
    assert best_score == 0
