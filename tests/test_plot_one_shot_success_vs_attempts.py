import numpy as np
import pytest

from graphs.measurement_error.plot_one_shot_success_vs_attempts import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
    algorithm_filename_prefix,
    algorithms_display_label,
    build_argument_parser,
    normalize_algorithms,
    simulate_for_attempt_budget,
    standard_backend_filename_segment,
)
from shor_standard.backends import ANALYTIC_STANDARD_BACKEND


def test_argument_parser_accepts_algorithm_comparison():
    parser = build_argument_parser()
    args = parser.parse_args([
        "21",
        "--algorithm",
        "kitaev",
        "standard",
    ])

    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]


def test_argument_parser_accepts_standard_backend():
    parser = build_argument_parser()
    args = parser.parse_args([
        "21",
        "--algorithm",
        "standard",
        "--standard-backend",
        "analytic",
        "--analytic-window",
        "50",
    ])

    assert args.algorithm == [STANDARD_ALGORITHM]
    assert args.standard_backend == ANALYTIC_STANDARD_BACKEND
    assert args.analytic_window == 50


def test_normalize_algorithms_removes_duplicates_preserving_order():
    assert normalize_algorithms([
        KITAEV_ALGORITHM,
        STANDARD_ALGORITHM,
        KITAEV_ALGORITHM,
    ]) == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]


def test_normalize_algorithms_rejects_empty_list():
    with pytest.raises(ValueError, match="at least one algorithm"):
        normalize_algorithms([])


def test_algorithm_display_helpers_support_comparison():
    assert algorithms_display_label([KITAEV_ALGORITHM]) == "Kitaev"
    assert (
        algorithms_display_label([KITAEV_ALGORITHM, STANDARD_ALGORITHM])
        == "Kitaev vs standard Shor"
    )


def test_algorithm_filename_prefix_preserves_default_kitaev_name():
    assert algorithm_filename_prefix([KITAEV_ALGORITHM]) == ""
    assert algorithm_filename_prefix([STANDARD_ALGORITHM]) == "standard_"
    assert (
        algorithm_filename_prefix([KITAEV_ALGORITHM, STANDARD_ALGORITHM])
        == "compare_algorithms_"
    )


def test_standard_backend_filename_segment_marks_analytic_backend():
    assert standard_backend_filename_segment([KITAEV_ALGORITHM], "analytic") == ""
    assert (
        standard_backend_filename_segment([STANDARD_ALGORITHM], ANALYTIC_STANDARD_BACKEND)
        == "analytic_"
    )


def test_simulate_standard_attempt_budget_supports_analytic_backend():
    result = simulate_for_attempt_budget(
        N=15,
        max_attempts=1,
        trials=5,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        algorithm=STANDARD_ALGORITHM,
        standard_backend=ANALYTIC_STANDARD_BACKEND,
        analytic_window=20,
    )

    assert result.algorithm == STANDARD_ALGORITHM
    assert result.standard_backend == ANALYTIC_STANDARD_BACKEND
    assert 0.0 <= result.success_rate <= 1.0
