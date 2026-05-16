from types import SimpleNamespace

import numpy as np

from graphs.measurement_error.plot_nearest_peak_decimal_distance_vs_error_probability import (
    build_fixed_instance,
)
from graphs.measurement_error.plot_noisy_cf_complement_fill import (
    CFMembershipCache,
    NoisyCFComplementFillResult,
    bits_to_index,
    build_argument_parser,
    build_cf_membership_cache,
    cache_output_path,
    load_or_build_cf_membership_cache,
    output_filename_segment,
    plot_noisy_cf_complement_fill,
    run_phase_bits_for_fill_trial,
    simulate_noisy_cf_complement_fill,
)
from graphs.measurement_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_bits_to_index_uses_binary_fraction_order():
    assert bits_to_index([1, 0, 1]) == 5
    assert bits_to_index([0, 1, 0, 0]) == 4


def test_membership_cache_counts_small_exact_case():
    cache = build_cf_membership_cache(
        true_order=2,
        num_phase_bits=3,
        max_cache_bitstrings=8,
    )

    assert cache.inside_count == 6
    assert cache.outside_count == 2
    assert cache.is_inside_index(0)
    assert cache.is_inside_index(1)
    assert not cache.is_inside_index(2)
    assert cache.is_inside_index(3)
    assert cache.is_inside_index(4)
    assert cache.is_inside_index(5)
    assert not cache.is_inside_index(6)
    assert cache.is_inside_index(7)


def test_membership_cache_round_trip_loads_existing_file(tmp_path):
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
    assert loaded.is_outside_index(2)


def test_argument_parser_accepts_cache_options():
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
        "--single-measurement-bitflip",
        "--single-measurement-bitflip-range",
        "leading",
        "dangerous",
        "--rebuild-cache",
        "--max-cache-bitstrings",
        "100000",
    ])

    assert args.N == 21
    assert args.a == 11
    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.p == 0.05
    assert args.trials == 200
    assert args.classical_error_correction
    assert args.single_measurement_bitflip
    assert args.single_measurement_bitflip_range == ["leading", "dangerous"]
    assert args.rebuild_cache
    assert args.max_cache_bitstrings == 100000


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


def test_standard_analytic_zero_error_has_no_sc_complement_hits():
    instance = build_fixed_instance(15, 2)
    cache = build_cf_membership_cache(
        true_order=instance.true_order,
        num_phase_bits=8,
        max_cache_bitstrings=256,
    )

    result = simulate_noisy_cf_complement_fill(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        trials=5,
        rng=np.random.default_rng(123),
        membership_cache=cache,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.outside_unique_hits == 0
    assert result.outside_trial_hits == 0
    assert result.outside_fill_fraction == 0.0
    assert result.outside_trial_probability == 0.0


def test_standard_analytic_zero_error_corrected_has_no_sc_complement_hits():
    instance = build_fixed_instance(15, 2)
    cache = build_cf_membership_cache(
        true_order=instance.true_order,
        num_phase_bits=8,
        max_cache_bitstrings=256,
    )

    result = simulate_noisy_cf_complement_fill(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=8,
        measurement_error_probability=0.0,
        trials=5,
        rng=np.random.default_rng(123),
        membership_cache=cache,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
        classical_error_correction=True,
    )

    assert result.classical_error_correction
    assert result.outside_unique_hits == 0
    assert result.outside_trial_hits == 0


def test_corrected_fill_trial_uses_minimum_distance_candidate(monkeypatch):
    instance = build_fixed_instance(15, 2)

    def fake_standard_trial(**_kwargs):
        return SimpleNamespace(
            clean_bits=[1, 1, 1, 1],
            noisy_bits=[0, 1, 0, 1],
        )

    monkeypatch.setattr(
        "graphs.measurement_error.plot_noisy_cf_complement_fill.run_standard_phase_distance_trial",
        fake_standard_trial,
    )

    bits = run_phase_bits_for_fill_trial(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=4,
        measurement_error_probability=0.1,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        classical_error_correction=True,
    )

    assert bits == [0, 1, 0, 0]


def test_fill_trial_passes_single_measurement_bitflip_to_standard(monkeypatch):
    instance = build_fixed_instance(15, 2)
    captured = {}

    def fake_standard_trial(**kwargs):
        captured["single"] = kwargs["single_measurement_bitflip"]
        return SimpleNamespace(
            clean_bits=[0, 0, 0, 0],
            noisy_bits=[0, 0, 0, 1],
        )

    monkeypatch.setattr(
        "graphs.measurement_error.plot_noisy_cf_complement_fill.run_standard_phase_distance_trial",
        fake_standard_trial,
    )

    bits = run_phase_bits_for_fill_trial(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=4,
        measurement_error_probability=0.0,
        rng=np.random.default_rng(123),
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        single_measurement_bitflip=True,
    )

    assert captured["single"] is True
    assert bits == [0, 0, 0, 1]


def test_plot_noisy_cf_complement_fill_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    cache = CFMembershipCache(
        true_order=instance.true_order,
        num_phase_bits=8,
        inside_count=20,
        outside_count=236,
        inside_bits_packed=np.zeros(32, dtype=np.uint8),
    )
    results = [
        NoisyCFComplementFillResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            measurement_error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            outside_unique_hits=4,
            outside_trial_hits=6,
            total_inside_bitstrings=20,
            total_outside_bitstrings=236,
        ),
        NoisyCFComplementFillResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            measurement_error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            outside_unique_hits=2,
            outside_trial_hits=4,
            total_inside_bitstrings=20,
            total_outside_bitstrings=236,
        ),
    ]
    output_path = tmp_path / "fill.png"

    plot_noisy_cf_complement_fill(
        output_path=output_path,
        results=results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        membership_cache=cache,
    )

    assert output_path.exists()
