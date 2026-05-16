from pathlib import Path

import numpy as np
import pytest

from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
    OrderRecoveryResult,
    aggregate_results_by_order,
    algorithm_filename_prefix,
    analytic_window_diagnostics_console_label,
    analytic_window_diagnostics_summary_lines,
    analytic_window_mass_diagnostics_for_instances,
    build_argument_parser,
    error_config_for_probability,
    error_probabilities_filename_segment,
    error_probability_display_label,
    instances_file_filename_segment,
    load_order_recovery_instances_from_file,
    max_modulus_for_phase_bits,
    normalize_algorithms,
    sample_order_recovery_instances,
    simulate_order_recovery_for_instance,
    standard_backend_filename_segment,
    valid_moduli_for_fixed_bits,
    validate_error_probabilities,
    validate_statevector_budget,
)
from shor_common.errors import ErrorConfig
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_max_modulus_for_phase_bits_matches_two_L_plus_one_rule():
    assert max_modulus_for_phase_bits(9) == 16
    assert max_modulus_for_phase_bits(11) == 32


def test_valid_moduli_exclude_perfect_powers_and_values_exceeding_bits():
    assert valid_moduli_for_fixed_bits(9, min_N=9) == [15]
    assert valid_moduli_for_fixed_bits(11, min_N=9, max_N=32) == [15, 21]


def test_valid_moduli_reject_explicit_max_N_beyond_bit_budget():
    with pytest.raises(ValueError, match="exceeds the largest modulus"):
        valid_moduli_for_fixed_bits(9, min_N=9, max_N=21)


def test_sample_instances_returns_coprime_instances_with_matching_default_bits():
    instances = sample_order_recovery_instances(
        num_instances=4,
        num_phase_bits=11,
        rng=np.random.default_rng(123),
        min_N=9,
    )

    assert len(instances) == 4
    assert all(instance.default_phase_bits <= 11 for instance in instances)
    assert all(instance.true_order >= 1 for instance in instances)


def test_argument_parser_accepts_algorithm_comparison_and_analytic_backend():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--algorithm",
        "kitaev",
        "standard",
        "--standard-backend",
        "analytic",
        "--bits",
        "29",
    ])

    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.standard_backend == ANALYTIC_STANDARD_BACKEND
    assert args.bits == 29


def test_argument_parser_accepts_instances_file():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--instances-file",
        "graphs/config/order_recovery_instances.json",
    ])

    assert args.instances_file.name == "order_recovery_instances.json"


def test_argument_parser_accepts_error_probabilities():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--error-probabilities",
        "0",
        "0.01",
        "0.1",
    ])

    assert args.error_probabilities == [0.0, 0.01, 0.1]


def test_error_probability_helpers():
    assert validate_error_probabilities([0.1, 0.01, 0.1]) == [0.1, 0.01]
    assert error_probability_display_label(0.01) == "p=1e-2"
    assert error_probabilities_filename_segment([0.0, 0.1]) == "p_0e0_1em1_"
    assert error_config_for_probability(0.0) == ErrorConfig()
    assert error_config_for_probability(0.1) == ErrorConfig(
        measurement_bitflip=True,
        measurement_bitflip_probability=0.1,
    )


def test_load_instances_file_computes_orders_and_sorts_by_order(tmp_path):
    instances_file = tmp_path / "instances.json"
    instances_file.write_text(
        """
        {
          "description": "test representatives",
          "instances": [
            {"N": 21, "a": 2, "r": 6},
            {"N": 15, "a": 2, "r": 4}
          ]
        }
        """,
        encoding="utf-8",
    )

    instances = load_order_recovery_instances_from_file(
        instances_file,
        num_phase_bits=11,
    )

    assert [(instance.N, instance.a, instance.true_order) for instance in instances] == [
        (15, 2, 4),
        (21, 2, 6),
    ]


def test_load_instances_file_rejects_duplicate_true_orders(tmp_path):
    instances_file = tmp_path / "instances.json"
    instances_file.write_text(
        """
        [
          {"N": 15, "a": 2},
          {"N": 15, "a": 7}
        ]
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one representative per true order"):
        load_order_recovery_instances_from_file(instances_file, num_phase_bits=11)


def test_load_instances_file_verifies_optional_order_field(tmp_path):
    instances_file = tmp_path / "instances.json"
    instances_file.write_text(
        """
        [
          {"N": 15, "a": 2, "r": 5}
        ]
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected r=5"):
        load_order_recovery_instances_from_file(instances_file, num_phase_bits=11)


def test_load_instances_file_applies_order_filter_before_bit_budget_check(tmp_path):
    instances_file = tmp_path / "instances.json"
    instances_file.write_text(
        """
        [
          {"N": 15, "a": 2, "r": 4},
          {"N": 95, "a": 2, "r": 36}
        ]
        """,
        encoding="utf-8",
    )

    instances = load_order_recovery_instances_from_file(
        instances_file,
        num_phase_bits=11,
        max_order=4,
    )

    assert [(instance.N, instance.a, instance.true_order) for instance in instances] == [
        (15, 2, 4),
    ]


def test_instances_file_filename_segment_uses_file_stem():
    assert (
        instances_file_filename_segment(Path("graphs/config/My Orders.json"))
        == "instances_my_orders_"
    )


def test_normalize_algorithms_removes_duplicates_preserving_order():
    assert normalize_algorithms([
        KITAEV_ALGORITHM,
        STANDARD_ALGORITHM,
        KITAEV_ALGORITHM,
    ]) == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]


def test_filename_helpers_mark_algorithm_and_backend():
    assert algorithm_filename_prefix([KITAEV_ALGORITHM]) == "kitaev_"
    assert (
        algorithm_filename_prefix([KITAEV_ALGORITHM, STANDARD_ALGORITHM])
        == "compare_algorithms_"
    )
    assert standard_backend_filename_segment([KITAEV_ALGORITHM], "analytic") == ""
    assert (
        standard_backend_filename_segment([STANDARD_ALGORITHM], ANALYTIC_STANDARD_BACKEND)
        == "analytic_"
    )


def test_aggregate_results_by_order_groups_trials_by_algorithm_and_order():
    instance_a = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    instance_b = OrderRecoveryInstance(
        N=21,
        a=2,
        true_order=6,
        num_work_qubits=5,
        default_phase_bits=11,
    )

    summaries = aggregate_results_by_order([
        OrderRecoveryResult(
            instance=instance_a,
            algorithm=KITAEV_ALGORITHM,
            successful_trials=2,
            trials=5,
            num_phase_bits=11,
        ),
        OrderRecoveryResult(
            instance=instance_b,
            algorithm=KITAEV_ALGORITHM,
            successful_trials=3,
            trials=5,
            num_phase_bits=11,
        ),
        OrderRecoveryResult(
            instance=instance_a,
            algorithm=STANDARD_ALGORITHM,
            successful_trials=4,
            trials=5,
            num_phase_bits=11,
            standard_backend=ANALYTIC_STANDARD_BACKEND,
        ),
    ])

    assert [(summary.algorithm, summary.true_order) for summary in summaries] == [
        (KITAEV_ALGORITHM, 4),
        (KITAEV_ALGORITHM, 6),
        (STANDARD_ALGORITHM, 4),
    ]
    assert summaries[0].success_rate == 0.4
    assert summaries[2].standard_backend == ANALYTIC_STANDARD_BACKEND


def test_aggregate_results_by_order_separates_measurement_error_labels():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    summaries = aggregate_results_by_order([
        OrderRecoveryResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            successful_trials=4,
            trials=5,
            num_phase_bits=11,
            measurement_error_label="p=0e0",
            error_probability=0.0,
        ),
        OrderRecoveryResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            successful_trials=2,
            trials=5,
            num_phase_bits=11,
            measurement_error_label="p=1e-1",
            error_probability=0.1,
        ),
    ])

    assert [summary.measurement_error_label for summary in summaries] == [
        "p=0e0",
        "p=1e-1",
    ]
    assert [summary.success_rate for summary in summaries] == [0.8, 0.4]


def test_analytic_window_diagnostics_helpers_report_worst_mass():
    instances = [
        OrderRecoveryInstance(
            N=15,
            a=2,
            true_order=4,
            num_work_qubits=4,
            default_phase_bits=9,
        ),
        OrderRecoveryInstance(
            N=21,
            a=2,
            true_order=6,
            num_work_qubits=5,
            default_phase_bits=11,
        ),
    ]

    diagnostics = analytic_window_mass_diagnostics_for_instances(
        instances=instances,
        num_phase_bits=11,
        analytic_window=5,
    )
    lines = analytic_window_diagnostics_summary_lines(diagnostics)
    label = analytic_window_diagnostics_console_label(diagnostics)

    assert [diagnostic.order for diagnostic in diagnostics] == [4, 6]
    assert lines[0] == "Analytic window captured-mass diagnostics:"
    assert "Worst captured mass" in "\n".join(lines)
    assert label.startswith("min=")


def test_simulate_standard_order_recovery_supports_analytic_backend():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_order_recovery_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        trials=5,
        rng=np.random.default_rng(123),
        num_phase_bits=11,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.algorithm == STANDARD_ALGORITHM
    assert result.standard_backend == ANALYTIC_STANDARD_BACKEND
    assert 0.0 <= result.success_rate <= 1.0


def test_statevector_budget_recommends_analytic_backend_for_large_sweeps():
    with pytest.raises(ValueError, match="standard statevector backend"):
        validate_statevector_budget(
            algorithms=[STANDARD_ALGORITHM],
            standard_backend="statevector",
            num_phase_bits=29,
            max_N=1000,
            max_statevector_amplitudes=8_000_000,
        )
