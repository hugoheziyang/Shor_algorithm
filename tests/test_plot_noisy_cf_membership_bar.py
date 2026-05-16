from types import SimpleNamespace

import numpy as np

from graphs.measurement_error.plot_noisy_cf_membership_bar import (
    S_CATEGORY,
    S_COMPLEMENT_CATEGORY,
    NoisyCFMembershipResult,
    build_argument_parser,
    output_filename_segment,
    plot_noisy_cf_membership_bar,
    result_category_probability,
    run_noisy_membership_trial,
    simulate_noisy_cf_membership,
)
from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_error_probability import (
    build_fixed_instance,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_result_category_probability_selects_inside_or_outside():
    instance = build_fixed_instance(15, 2)
    result = NoisyCFMembershipResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        measurement_error_probability=0.1,
        trials=10,
        num_phase_bits=8,
        inside_trials=7,
    )

    assert result_category_probability(result, S_CATEGORY) == 0.7
    assert result_category_probability(result, S_COMPLEMENT_CATEGORY) == 0.3
    assert result.outside_trials == 3


def test_argument_parser_accepts_fixed_probability_options():
    parser = build_argument_parser()
    args = parser.parse_args([
        "21",
        "--a",
        "11",
        "--algorithm",
        "kitaev",
        "standard",
        "--p",
        "0.05",
        "--trials",
        "200",
        "--classical-error-correction",
    ])

    assert args.N == 21
    assert args.a == 11
    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.p == 0.05
    assert args.trials == 200
    assert args.classical_error_correction


def test_output_filename_segment_includes_probability():
    assert (
        output_filename_segment(
            algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
            standard_backend=ANALYTIC_STANDARD_BACKEND,
            N=21,
            a=11,
            num_phase_bits=11,
            measurement_error_probability=0.05,
        )
        == "compare_algorithms_analytic_N21_a11_t11_p_5em2"
    )


def test_output_filename_segment_marks_classical_error_correction():
    assert output_filename_segment(
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        N=21,
        a=11,
        num_phase_bits=11,
        measurement_error_probability=0.05,
        include_classical_error_correction=True,
    ).endswith("_classical_ec")


def test_standard_analytic_zero_error_membership_counts_all_trials():
    instance = build_fixed_instance(15, 2)

    result = simulate_noisy_cf_membership(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        trials=5,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.inside_trials == 5
    assert result.inside_probability == 1.0
    assert result.outside_probability == 0.0


def test_standard_analytic_zero_error_corrected_membership_counts_all_trials():
    instance = build_fixed_instance(15, 2)

    result = simulate_noisy_cf_membership(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        trials=5,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        classical_error_correction=True,
    )

    assert result.classical_error_correction
    assert result.inside_trials == 5


def test_corrected_membership_uses_minimum_distance_candidate(monkeypatch):
    instance = build_fixed_instance(15, 2)

    def fake_standard_trial(**_kwargs):
        return SimpleNamespace(
            clean_bits=[1, 1, 1, 1],
            noisy_bits=[0, 1, 0, 1],
        )

    monkeypatch.setattr(
        "graphs.measurement_error.plot_noisy_cf_membership_bar.run_standard_phase_distance_trial",
        fake_standard_trial,
    )

    noisy_inside = run_noisy_membership_trial(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=4,
        measurement_error_probability=0.1,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        classical_error_correction=False,
    )
    corrected_inside = run_noisy_membership_trial(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=4,
        measurement_error_probability=0.1,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        classical_error_correction=True,
    )

    assert not noisy_inside
    assert corrected_inside


def test_plot_noisy_cf_membership_bar_writes_file(tmp_path):
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
    output_path = tmp_path / "membership.png"

    plot_noisy_cf_membership_bar(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
    )

    assert output_path.exists()
