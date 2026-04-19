import numpy as np

from error_correction.classical_error_correction import (
    can_recover_clean_bits_by_single_flip,
    estimate_classical_recovery_probability,
    flip_bit,
    naive_single_bit_recovery_baseline,
    run_paired_phase_estimation_trial,
    single_bit_flip_candidates,
)


def test_flip_bit_flips_selected_index_only():
    assert flip_bit([0, 1, 0], 0) == [1, 1, 0]
    assert flip_bit([0, 1, 0], 1) == [0, 0, 0]
    assert flip_bit([0, 1, 0], 2) == [0, 1, 1]


def test_single_bit_flip_candidates_include_original_and_neighbors():
    assert single_bit_flip_candidates([0, 1, 0]) == [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
        [0, 1, 1],
    ]


def test_can_recover_clean_bits_by_single_flip_matches_search_rule():
    assert can_recover_clean_bits_by_single_flip([0, 1, 0], [0, 1, 0]) is True
    assert can_recover_clean_bits_by_single_flip([0, 1, 0], [0, 0, 0]) is True
    assert can_recover_clean_bits_by_single_flip([0, 1, 0], [1, 0, 0]) is False


def test_naive_single_bit_recovery_baseline_matches_expected_values():
    assert np.isclose(naive_single_bit_recovery_baseline(4, 0.0), 1.0)
    assert np.isclose(naive_single_bit_recovery_baseline(4, 1.0), 0.0)
    assert np.isclose(
        naive_single_bit_recovery_baseline(3, 0.1),
        (0.9 ** 3) + (3 * 0.1 * (0.9 ** 2)),
    )


def test_paired_phase_estimation_trial_matches_clean_run_when_p_is_zero():
    trial = run_paired_phase_estimation_trial(
        N=21,
        a=11,
        num_phase_bits=6,
        error_probability=0.0,
        rng=np.random.default_rng(1234),
    )

    assert trial.clean_bits == trial.noisy_bits
    assert trial.measurement_bitflip_mask == [False] * 6
    assert np.isclose(trial.clean_phase_estimate, trial.noisy_phase_estimate)
    assert trial.recovered is True


def test_estimated_classical_recovery_probability_is_one_when_p_is_zero():
    estimate = estimate_classical_recovery_probability(
        N=21,
        a=11,
        num_phase_bits=6,
        error_probability=0.0,
        trials=10,
        rng=np.random.default_rng(1234),
    )

    assert estimate.successful_trials == 10
    assert np.isclose(estimate.recovery_probability, 1.0)
    assert np.isclose(estimate.naive_baseline_probability, 1.0)
