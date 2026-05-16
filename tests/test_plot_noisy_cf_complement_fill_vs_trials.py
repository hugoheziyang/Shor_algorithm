import numpy as np
import pytest

from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_error_probability import (
    build_fixed_instance,
)
from graphs.measurement_error.plot_noisy_cf_complement_fill import (
    build_cf_membership_cache,
    cache_output_path,
    load_or_build_cf_membership_cache,
)
from graphs.measurement_error.plot_noisy_cf_complement_fill_vs_trials import (
    NoisyCFComplementFillTrialSweepResult,
    build_argument_parser,
    output_filename_segment,
    plot_noisy_cf_complement_fill_vs_trials,
    simulate_noisy_cf_complement_fill_vs_trials,
    validate_trial_counts,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_bitset_cache_tracks_small_exact_case():
    cache = build_cf_membership_cache(
        true_order=2,
        num_phase_bits=3,
        max_cache_bitstrings=8,
    )

    assert cache.inside_count == 6
    assert cache.outside_count == 2
    assert cache.is_inside_index(0)
    assert not cache.is_inside_index(2)
    assert cache.is_outside_index(6)


def test_bitset_cache_loads_existing_file_without_rebuild(tmp_path):
    cache = load_or_build_cf_membership_cache(
        cache_dir=tmp_path,
        true_order=2,
        num_phase_bits=3,
        max_cache_bitstrings=8,
    )
    loaded = load_or_build_cf_membership_cache(
        cache_dir=tmp_path,
        true_order=2,
        num_phase_bits=3,
        max_cache_bitstrings=1,
    )

    assert cache.path == cache_output_path(tmp_path, 2, 3)
    assert loaded.from_cache
    assert loaded.inside_count == cache.inside_count
    assert loaded.outside_count == cache.outside_count


def test_bitset_cache_guard_rejects_large_universe(tmp_path):
    with pytest.raises(ValueError, match="enumerate"):
        load_or_build_cf_membership_cache(
            cache_dir=tmp_path,
            true_order=6,
            num_phase_bits=29,
            max_cache_bitstrings=5_000_000,
        )


def test_validate_trial_counts_sorts_and_deduplicates():
    assert validate_trial_counts([100, 10, 10, 50]) == [10, 50, 100]
    with pytest.raises(ValueError):
        validate_trial_counts([0])


def test_argument_parser_accepts_trial_count_options():
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
        "--trial-counts",
        "10",
        "50",
        "100",
        "--max-cache-bitstrings",
        "100000",
        "--classical-error-correction",
        "--single-measurement-bitflip",
        "--single-measurement-bitflip-range",
        "leading",
        "dangerous",
        "--log-x",
    ])

    assert args.N == 21
    assert args.a == 11
    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.p == 0.05
    assert args.trial_counts == [10, 50, 100]
    assert args.max_cache_bitstrings == 100000
    assert args.classical_error_correction
    assert args.single_measurement_bitflip
    assert args.single_measurement_bitflip_range == ["leading", "dangerous"]
    assert args.log_x


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


def test_output_filename_segment_marks_single_measurement_bitflip():
    assert (
        output_filename_segment(
            algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
            standard_backend=ANALYTIC_STANDARD_BACKEND,
            N=21,
            a=11,
            num_phase_bits=11,
            measurement_error_probability=0.05,
            single_measurement_bitflip=True,
        )
        == "compare_algorithms_analytic_N21_a11_t11_single_measurement_bitflip"
    )


def test_output_filename_segment_marks_combined_single_measurement_bitflip_ranges():
    assert (
        output_filename_segment(
            algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
            standard_backend=ANALYTIC_STANDARD_BACKEND,
            N=21,
            a=11,
            num_phase_bits=11,
            measurement_error_probability=0.05,
            single_measurement_bitflip_range=("leading", "dangerous"),
        )
        == "compare_algorithms_analytic_N21_a11_t11_single_measurement_bitflip_leading_dangerous"
    )


def test_standard_analytic_zero_error_sweep_has_no_sc_hits():
    instance = build_fixed_instance(15, 2)
    cache = build_cf_membership_cache(
        true_order=instance.true_order,
        num_phase_bits=8,
        max_cache_bitstrings=256,
    )

    results = simulate_noisy_cf_complement_fill_vs_trials(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        trial_counts=[2, 5],
        rng=np.random.default_rng(123),
        membership_cache=cache,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert [result.trial_count for result in results] == [2, 5]
    assert all(result.outside_unique_hits == 0 for result in results)
    assert all(result.outside_trial_hits == 0 for result in results)
    assert all(result.outside_fill_fraction == 0.0 for result in results)


def test_standard_analytic_zero_error_corrected_sweep_has_no_sc_hits():
    instance = build_fixed_instance(15, 2)
    cache = build_cf_membership_cache(
        true_order=instance.true_order,
        num_phase_bits=8,
        max_cache_bitstrings=256,
    )

    results = simulate_noisy_cf_complement_fill_vs_trials(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        trial_counts=[2, 5],
        rng=np.random.default_rng(123),
        membership_cache=cache,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        classical_error_correction=True,
    )

    assert all(result.classical_error_correction for result in results)
    assert all(result.outside_unique_hits == 0 for result in results)
    assert all(result.outside_trial_hits == 0 for result in results)


def test_trial_count_points_are_independent_runs(monkeypatch):
    instance = build_fixed_instance(15, 2)
    cache = build_cf_membership_cache(
        true_order=instance.true_order,
        num_phase_bits=2,
        max_cache_bitstrings=4,
    )
    # Treat only index 0 as S and every other bitstring as S^c. The mocked
    # second run returns only index 0, so it proves the 2-trial point did not
    # inherit the first run's outside hit.
    cache.inside_bits_packed[:] = np.array([0b00000001], dtype=np.uint8)
    cache = cache.__class__(
        true_order=cache.true_order,
        num_phase_bits=cache.num_phase_bits,
        inside_count=1,
        outside_count=3,
        inside_bits_packed=cache.inside_bits_packed,
    )
    mocked_bits = iter([[0, 1], [0, 0], [0, 0]])

    def fake_noisy_bits_trial(**_kwargs):
        return next(mocked_bits)

    monkeypatch.setattr(
        "graphs.measurement_error.plot_noisy_cf_complement_fill_vs_trials.run_phase_bits_for_fill_trial",
        fake_noisy_bits_trial,
    )

    results = simulate_noisy_cf_complement_fill_vs_trials(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=2,
        measurement_error_probability=0.1,
        trial_counts=[1, 2],
        rng=np.random.default_rng(123),
        membership_cache=cache,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert [result.outside_unique_hits for result in results] == [1, 0]
    assert [result.outside_trial_hits for result in results] == [1, 0]


def test_trial_sweep_passes_single_measurement_bitflip(monkeypatch):
    instance = build_fixed_instance(15, 2)
    cache = build_cf_membership_cache(
        true_order=instance.true_order,
        num_phase_bits=2,
        max_cache_bitstrings=4,
    )
    captured = []

    def fake_noisy_bits_trial(**kwargs):
        captured.append(kwargs["single_measurement_bitflip"])
        return [0, 0]

    monkeypatch.setattr(
        "graphs.measurement_error.plot_noisy_cf_complement_fill_vs_trials.run_phase_bits_for_fill_trial",
        fake_noisy_bits_trial,
    )

    simulate_noisy_cf_complement_fill_vs_trials(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=2,
        measurement_error_probability=0.0,
        trial_counts=[3],
        rng=np.random.default_rng(123),
        membership_cache=cache,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        single_measurement_bitflip=True,
    )

    assert captured == [True, True, True]


def test_plot_noisy_cf_complement_fill_vs_trials_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    cache = build_cf_membership_cache(
        true_order=instance.true_order,
        num_phase_bits=8,
        max_cache_bitstrings=256,
    )
    results = [
        NoisyCFComplementFillTrialSweepResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            measurement_error_probability=0.1,
            trial_count=10,
            num_phase_bits=8,
            outside_unique_hits=3,
            outside_trial_hits=4,
            total_inside_bitstrings=cache.inside_count,
            total_outside_bitstrings=cache.outside_count,
        ),
        NoisyCFComplementFillTrialSweepResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            measurement_error_probability=0.1,
            trial_count=20,
            num_phase_bits=8,
            outside_unique_hits=5,
            outside_trial_hits=7,
            total_inside_bitstrings=cache.inside_count,
            total_outside_bitstrings=cache.outside_count,
        ),
        NoisyCFComplementFillTrialSweepResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trial_count=10,
            num_phase_bits=8,
            outside_unique_hits=2,
            outside_trial_hits=3,
            total_inside_bitstrings=cache.inside_count,
            total_outside_bitstrings=cache.outside_count,
        ),
        NoisyCFComplementFillTrialSweepResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trial_count=20,
            num_phase_bits=8,
            outside_unique_hits=4,
            outside_trial_hits=6,
            total_inside_bitstrings=cache.inside_count,
            total_outside_bitstrings=cache.outside_count,
        ),
    ]
    output_path = tmp_path / "fill_vs_trials.png"

    plot_noisy_cf_complement_fill_vs_trials(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        trial_counts=[10, 20],
        membership_cache=cache,
    )

    assert output_path.exists()
