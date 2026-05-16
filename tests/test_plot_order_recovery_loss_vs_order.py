import numpy as np

from graphs.measurement_error.plot_order_recovery_loss_vs_order import (
    DEFAULT_ERROR_PROBABILITIES,
    OrderRecoveryLossResult,
    build_argument_parser,
    error_probabilities_filename_segment,
    simulate_loss_for_instance,
    validate_error_probabilities,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_validate_error_probabilities_removes_duplicates_preserving_order():
    assert validate_error_probabilities([0.1, 0.01, 0.1]) == [0.1, 0.01]


def test_error_probabilities_filename_segment_marks_default_values():
    assert (
        error_probabilities_filename_segment(list(DEFAULT_ERROR_PROBABILITIES))
        == "default_p_"
    )
    assert error_probabilities_filename_segment([0.01, 0.1]) == "p_1em2_1em1_"


def test_argument_parser_requires_instances_file():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--instances-file",
        "graphs/config/order_recovery_instances.json",
        "--error-probabilities",
        "0.01",
        "0.1",
    ])

    assert args.instances_file.name == "order_recovery_instances.json"
    assert args.error_probabilities == [0.01, 0.1]


def test_loss_result_uses_clean_minus_noisy_rate():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = OrderRecoveryLossResult(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        error_probability=0.1,
        clean_successful_trials=8,
        noisy_successful_trials=5,
        trials=10,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
    )

    assert result.clean_success_rate == 0.8
    assert result.noisy_success_rate == 0.5
    assert result.recovery_loss == 0.30000000000000004


def test_zero_error_probability_reuses_clean_success_count():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = simulate_loss_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        clean_successful_trials=3,
        error_probability=0.0,
        trials=5,
        rng=np.random.default_rng(123),
        num_phase_bits=11,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
    )

    assert result.noisy_successful_trials == 3
    assert result.recovery_loss == 0.0
    assert result.standard_error == 0.0


def test_simulate_standard_loss_supports_analytic_backend():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = simulate_loss_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        clean_successful_trials=5,
        error_probability=0.1,
        trials=5,
        rng=np.random.default_rng(123),
        num_phase_bits=11,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.algorithm == STANDARD_ALGORITHM
    assert result.standard_backend == ANALYTIC_STANDARD_BACKEND
    assert -1.0 <= result.recovery_loss <= 1.0
