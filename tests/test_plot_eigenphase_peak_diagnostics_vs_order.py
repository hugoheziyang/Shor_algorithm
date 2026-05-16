import math
from pathlib import Path

import numpy as np

from graphs.measurement_error.plot_eigenphase_peak_diagnostics_vs_order import (
    EigenphasePeakDiagnosticResult,
    RECOVERY_GIVEN_SHIFTED_WITHIN_BOUND_PROBABILITY,
    SAME_PEAK_PROBABILITY,
    SHIFTED_WITHIN_BOUND_PROBABILITY,
    build_argument_parser,
    diagnostic_output_path,
    nearest_peak_numerator,
    noisy_phase_within_cf_bound,
    plot_peak_diagnostic_vs_order,
    result_diagnostic_value,
    simulate_peak_diagnostics_for_instance,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    OrderRecoveryInstance,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_nearest_peak_numerator_uses_order_lattice():
    assert nearest_peak_numerator([0, 1, 0, 0], true_order=4) == 1
    assert nearest_peak_numerator([1, 1, 1, 1], true_order=4) == 0


def test_noisy_phase_within_cf_bound_checks_nearest_noisy_peak():
    assert noisy_phase_within_cf_bound([0, 1, 0, 0, 0], true_order=4)
    assert not noisy_phase_within_cf_bound([0, 1, 1, 0, 0], true_order=4)


def test_result_diagnostic_values_and_nan_conditional():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    result = EigenphasePeakDiagnosticResult(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        measurement_error_probability=0.0,
        trials=4,
        num_phase_bits=8,
        same_peak_trials=4,
        shifted_within_bound_trials=0,
        noisy_peak_coprime_trials=2,
        recovered_shifted_within_bound_trials=0,
        recovered_trials=3,
    )

    assert result_diagnostic_value(result, SAME_PEAK_PROBABILITY) == 1.0
    assert result_diagnostic_value(result, SHIFTED_WITHIN_BOUND_PROBABILITY) == 0.0
    assert math.isnan(
        result_diagnostic_value(
            result,
            RECOVERY_GIVEN_SHIFTED_WITHIN_BOUND_PROBABILITY,
        )
    )


def test_argument_parser_accepts_error_probability_overlay():
    parser = build_argument_parser()
    args = parser.parse_args([
        "--algorithm",
        "kitaev",
        "standard",
        "--error-probabilities",
        "0",
        "0.1",
    ])

    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.error_probabilities == [0.0, 0.1]


def test_diagnostic_output_path_appends_diagnostic_to_custom_stem():
    output_path = diagnostic_output_path(
        output_path=Path("figure.png"),
        diagnostic=SAME_PEAK_PROBABILITY,
        algorithms=[STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        instances_file=Path("instances.json"),
        num_phase_bits=8,
        measurement_error_probabilities=[0.0, 0.1],
    )

    assert output_path == Path("figure_same_peak.png")


def test_standard_analytic_zero_error_stays_on_same_peak():
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )

    result = simulate_peak_diagnostics_for_instance(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        trials=4,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        measurement_error_probability=0.0,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.same_peak_trials == 4
    assert result.shifted_within_bound_trials == 0
    assert result.trials == 4


def test_plot_peak_diagnostic_vs_order_writes_file(tmp_path):
    instance = OrderRecoveryInstance(
        N=15,
        a=2,
        true_order=4,
        num_work_qubits=4,
        default_phase_bits=9,
    )
    results = [
        EigenphasePeakDiagnosticResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.0,
            trials=4,
            num_phase_bits=8,
            same_peak_trials=4,
            shifted_within_bound_trials=0,
            noisy_peak_coprime_trials=2,
            recovered_shifted_within_bound_trials=0,
            recovered_trials=3,
        ),
        EigenphasePeakDiagnosticResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trials=4,
            num_phase_bits=8,
            same_peak_trials=2,
            shifted_within_bound_trials=1,
            noisy_peak_coprime_trials=2,
            recovered_shifted_within_bound_trials=1,
            recovered_trials=2,
        ),
    ]
    output_path = tmp_path / "peaks.png"

    plot_peak_diagnostic_vs_order(
        output_path=output_path,
        results=results,
        algorithms=[STANDARD_ALGORITHM],
        num_phase_bits=8,
        measurement_error_probabilities=[0.0, 0.1],
        diagnostic=SAME_PEAK_PROBABILITY,
        trials_per_instance=4,
    )

    assert output_path.exists()
