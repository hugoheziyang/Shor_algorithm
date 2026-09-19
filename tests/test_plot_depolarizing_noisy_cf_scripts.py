from pathlib import Path

import pytest
import numpy as np

import graphs.depolarizing_error.plot_phase_estimation_distribution as phase_distribution_module
from graphs.depolarizing_error.cf_common import (
    CFMembershipCache,
    NoisyCFComplementFillResult,
    build_cf_membership_cache,
    build_fixed_instance,
    compact_filename_segment,
    depolarizing_error_filename_segment,
    simulate_noisy_cf_complement_fill,
    simulate_noisy_coprime_cf_membership,
    simulate_noisy_cf_membership,
    validate_depolarizing_standard_backend,
    validate_statevector_budget,
    within_coprime_cf_sufficient_bound,
)
from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
)
from graphs.depolarizing_error.plot_phase_estimation_distribution import (
    DISTRIBUTION_STYLE_BOTH,
    DISTRIBUTION_STYLE_SMOOTH,
    Y_AXIS_FORMAT_SCIENTIFIC,
    build_argument_parser as build_phase_distribution_argument_parser,
    depolarizing_pauli_display_label,
    depolarizing_pauli_filename_segment,
    normalize_depolarizing_pauli_arg,
    normalize_padding_phase_radians,
    output_filename_segment as phase_distribution_output_filename_segment,
    plot_phase_estimation_distribution,
    smoothed_phase_distribution_curve,
    simulate_clean_noisy_phase_distributions,
    write_data_summary as write_phase_distribution_data_summary,
)
from graphs.depolarizing_error.plot_noisy_cf_complement_fill import (
    output_filename_segment as fill_output_filename_segment,
    plot_noisy_cf_complement_fill,
)
from graphs.depolarizing_error.plot_noisy_cf_complement_fill_vs_trials import (
    build_argument_parser as build_fill_vs_trials_argument_parser,
    validate_trial_counts,
)
from graphs.depolarizing_error.plot_noisy_cf_membership_bar import (
    S_CATEGORY,
    S_COMPLEMENT_CATEGORY,
    NoisyCFMembershipResult,
    build_argument_parser as build_membership_bar_argument_parser,
    output_filename_segment as membership_bar_output_filename_segment,
    plot_noisy_cf_membership_bar,
    result_category_probability,
)
from graphs.depolarizing_error.plot_noisy_cf_membership_vs_order import (
    output_filename_segment as membership_vs_order_output_filename_segment,
)
from graphs.depolarizing_error.plot_noisy_cf_membership_vs_order_by_instance_file import (
    InstanceFileMembershipSweep,
    build_argument_parser as build_membership_vs_order_by_file_argument_parser,
    output_filename_segment as membership_vs_order_by_file_output_filename_segment,
    plot_noisy_cf_membership_vs_order_by_instance_file,
    resolve_work_qubits_by_file,
    validate_instance_file_labels,
)
from graphs.depolarizing_error.plot_noisy_cf_membership_excess_vs_order_by_instance_file import (
    COPRIME_MEMBERSHIP,
    InstanceFileMembershipExcessSweep,
    build_argument_parser as build_membership_excess_by_file_argument_parser,
    output_filename_segment as membership_excess_by_file_output_filename_segment,
    plot_noisy_cf_membership_excess_vs_order_by_instance_file,
)
from graphs.depolarizing_error.plot_noisy_cf_membership_normalized_excess_vs_order_by_instance_file import (
    build_argument_parser as build_membership_normalized_excess_by_file_argument_parser,
    effective_depolarizing_error_probability,
    normalized_excess_failure,
    output_filename_segment as membership_normalized_excess_by_file_output_filename_segment,
    plot_noisy_cf_membership_normalized_excess_vs_order_by_instance_file,
    simulate_clean_baseline_until_inside,
    simulate_membership_for_fixed_trials,
    validate_depolarizing_error_scale,
)
from graphs.depolarizing_error.plot_noisy_coprime_cf_membership_vs_order_by_instance_file import (
    build_argument_parser as build_coprime_membership_vs_order_by_file_argument_parser,
    output_filename_segment as coprime_membership_vs_order_by_file_output_filename_segment,
    plot_noisy_coprime_cf_membership_vs_order_by_instance_file,
)
from graphs.depolarizing_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
    STANDARD_ALGORITHM,
)
from shor_standard.backends import (
    ANALYTIC_STANDARD_BACKEND,
    STATEVECTOR_STANDARD_BACKEND,
)


def test_depolarizing_cf_filename_segment_marks_model_and_probability():
    assert (
        depolarizing_error_filename_segment(BLOCK_DEPOLARIZING_MODEL, 0.05)
        == "dp_p_5em2"
    )
    assert (
        depolarizing_error_filename_segment(PER_QUBIT_DEPOLARIZING_MODEL, 0.1)
        == "pqdp_p_1em1"
    )


def test_depolarizing_phase_distribution_parser_accepts_pauli_choice():
    parser = build_phase_distribution_argument_parser()
    args = parser.parse_args([
        "21",
        "--a",
        "11",
        "--algorithm",
        "kitaev",
        "standard",
        "--error-model",
        "per-qubit-depolarizing",
        "--p",
        "0.05",
        "--padding-phase",
        str(np.pi / 2.0),
        "--depolarizing-pauli",
        "X",
        "--depolarizing-region",
        "dangerous",
        "--work-qubits",
        "6",
        "--trials",
        "20",
        "--distribution-style",
        "smooth",
        "--smooth-sigma-bins",
        "3.5",
        "--y-max",
        "0.4",
        "--y-format",
        "scientific",
    ])

    assert args.N == 21
    assert args.a == 11
    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.error_model == PER_QUBIT_DEPOLARIZING_MODEL
    assert args.p == pytest.approx(0.05)
    assert normalize_padding_phase_radians(args.padding_phase_radians) == pytest.approx(
        np.pi / 2.0
    )
    assert args.depolarizing_pauli == "X"
    assert normalize_depolarizing_pauli_arg(args.depolarizing_pauli) == ("X",)
    assert depolarizing_pauli_filename_segment(("X",)) == "_pauli_X"
    assert depolarizing_pauli_display_label(("X",)) == "X only"
    assert args.depolarizing_region == ["dangerous"]
    assert args.work_qubits == 6
    assert args.trials == 20
    assert args.distribution_style == DISTRIBUTION_STYLE_SMOOTH
    assert args.smooth_sigma_bins == pytest.approx(3.5)
    assert args.y_max == pytest.approx(0.4)
    assert args.y_format == Y_AXIS_FORMAT_SCIENTIFIC


def test_depolarizing_phase_distribution_uses_distinct_algorithm_implementations(
    monkeypatch,
):
    calls = []

    def fake_kitaev_order_attempt(**kwargs):
        calls.append(("kitaev", kwargs))
        return {"bits": [0, 0, 0, 1]}

    def fake_standard_order_attempt(**kwargs):
        calls.append(("standard", kwargs))
        return {"bits": [1, 0, 0, 0]}

    monkeypatch.setattr(
        phase_distribution_module,
        "kitaev_order_attempt",
        fake_kitaev_order_attempt,
    )
    monkeypatch.setattr(
        phase_distribution_module,
        "standard_order_attempt",
        fake_standard_order_attempt,
    )

    samples = simulate_clean_noisy_phase_distributions(
        N=15,
        a=2,
        algorithms=[STANDARD_ALGORITHM, KITAEV_ALGORITHM],
        num_phase_bits=4,
        trials=1,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        rng=np.random.default_rng(123),
        true_order=4,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        num_work_qubits=5,
        padding_phase_radians=np.pi / 2.0,
    )

    assert [sample.algorithm for sample in samples] == [
        STANDARD_ALGORITHM,
        STANDARD_ALGORITHM,
        KITAEV_ALGORITHM,
        KITAEV_ALGORITHM,
    ]
    assert [algorithm for algorithm, _ in calls] == [
        "standard",
        "standard",
        "kitaev",
        "kitaev",
    ]
    standard_kwargs = calls[0][1]
    kitaev_kwargs = calls[2][1]
    assert "num_control_qubits" in standard_kwargs
    assert "standard_backend" in standard_kwargs
    assert "num_phase_bits" not in standard_kwargs
    assert standard_kwargs["padding_phase_radians"] == pytest.approx(np.pi / 2.0)
    assert "num_phase_bits" in kitaev_kwargs
    assert "standard_backend" not in kitaev_kwargs
    assert "num_control_qubits" not in kitaev_kwargs
    assert kitaev_kwargs["padding_phase_radians"] == pytest.approx(np.pi / 2.0)


def test_compact_filename_segment_shortens_long_generated_names():
    short_name = "kitaev_instances_small_t8"
    assert compact_filename_segment(short_name) == short_name

    long_name = "kitaev_" + ("order_recover_instances_" * 20)
    compacted = compact_filename_segment(long_name, max_length=80)

    assert len(compacted) <= 80
    assert compacted.startswith("kitaev_ori")
    assert "order_recover_instances" not in compacted


def test_membership_bar_parser_accepts_depolarizing_options():
    parser = build_membership_bar_argument_parser()
    args = parser.parse_args([
        "21",
        "--a",
        "11",
        "--algorithm",
        "kitaev",
        "standard",
        "--error-model",
        "per-qubit-depolarizing",
        "--p",
        "0.05",
        "--trials",
        "200",
        "--classical-error-correction",
        "--work-qubits",
        "6",
        "--depolarizing-region",
        "leading",
        "dangerous",
    ])

    assert args.N == 21
    assert args.a == 11
    assert args.algorithm == [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    assert args.error_model == PER_QUBIT_DEPOLARIZING_MODEL
    assert args.p == 0.05
    assert args.trials == 200
    assert args.classical_error_correction
    assert args.work_qubits == 6
    assert args.depolarizing_region == ["leading", "dangerous"]


def test_standard_depolarizing_rejects_analytic_backend():
    with pytest.raises(ValueError, match="statevector backend"):
        validate_depolarizing_standard_backend(
            [STANDARD_ALGORITHM],
            ANALYTIC_STANDARD_BACKEND,
        )


def test_depolarizing_statevector_budget_warns_when_bits_below_default():
    with pytest.warns(RuntimeWarning, match="below the default"):
        validate_statevector_budget(
            algorithms=[STANDARD_ALGORITHM],
            standard_backend=STATEVECTOR_STANDARD_BACKEND,
            num_phase_bits=9,
            max_N=21,
            max_statevector_amplitudes=8_000_000,
        )


def test_output_filename_segments_are_depolarizing_scoped():
    assert (
        membership_bar_output_filename_segment(
            algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
            standard_backend=STATEVECTOR_STANDARD_BACKEND,
            N=21,
            a=11,
            num_phase_bits=11,
            error_model=BLOCK_DEPOLARIZING_MODEL,
            error_probability=0.05,
            num_work_qubits=6,
        )
        == "compare_algorithms_N21_a11_t11_dp_p_5em2_fixed_work_L6"
    )
    assert (
        fill_output_filename_segment(
            algorithms=[KITAEV_ALGORITHM],
            standard_backend=STATEVECTOR_STANDARD_BACKEND,
            N=21,
            a=11,
            num_phase_bits=11,
            error_model=PER_QUBIT_DEPOLARIZING_MODEL,
            error_probability=0.1,
            include_classical_error_correction=True,
        )
        == "kitaev_N21_a11_t11_pqdp_p_1em1_classical_ec"
    )
    assert (
        fill_output_filename_segment(
            algorithms=[KITAEV_ALGORITHM],
            standard_backend=STATEVECTOR_STANDARD_BACKEND,
            N=21,
            a=11,
            num_phase_bits=11,
            error_model=PER_QUBIT_DEPOLARIZING_MODEL,
            error_probability=0.1,
            depolarizing_region="safe-tail",
        )
        == "kitaev_N21_a11_t11_pqdp_p_1em1_region_safe_tail"
    )
    assert membership_vs_order_output_filename_segment(
        algorithms=[KITAEV_ALGORITHM],
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        instances_file=Path("graphs/config/order_recovery_instances.json"),
        num_phase_bits=11,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
    ).startswith("kitaev_instances_order_recovery_instances_dp_p_0e0")
    assert phase_distribution_output_filename_segment(
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        N=15,
        a=2,
        num_phase_bits=4,
        error_model=PER_QUBIT_DEPOLARIZING_MODEL,
        error_probability=0.1,
        num_work_qubits=5,
        depolarizing_region="dangerous",
        depolarizing_paulis=("X",),
    ) == (
        "compare_algorithms_N15_a2_t4_pqdp_p_1em1_pauli_X_"
        "region_dangerous_fixed_work_L5"
    )
    assert phase_distribution_output_filename_segment(
        algorithms=[KITAEV_ALGORITHM],
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        N=15,
        a=2,
        num_phase_bits=4,
        error_model=PER_QUBIT_DEPOLARIZING_MODEL,
        error_probability=0.1,
        distribution_style=DISTRIBUTION_STYLE_SMOOTH,
        smooth_sigma_bins=3.5,
    ) == "kitaev_N15_a2_t4_pqdp_p_1em1_smooth_s3p5"
    assert "_padtheta_1p570796327rad" in phase_distribution_output_filename_segment(
        algorithms=[KITAEV_ALGORITHM],
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        N=15,
        a=2,
        num_phase_bits=4,
        error_model=PER_QUBIT_DEPOLARIZING_MODEL,
        error_probability=0.1,
        padding_phase_radians=np.pi / 2.0,
    )


def test_result_category_probability_selects_inside_or_outside():
    instance = build_fixed_instance(15, 2)
    result = NoisyCFMembershipResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        trials=10,
        num_phase_bits=8,
        inside_trials=7,
    )

    assert result_category_probability(result, S_CATEGORY) == 0.7
    assert result_category_probability(result, S_COMPLEMENT_CATEGORY) == 0.3


def test_smoothed_phase_distribution_curve_is_circular():
    rows = [
        phase_distribution_module.PhaseOutcomeRow(
            bits="0000",
            outcome_integer=0,
            phase_decimal=0.0,
            count=1,
            probability=1.0,
        )
    ]

    x_values, y_values = smoothed_phase_distribution_curve(
        rows=rows,
        num_phase_bits=4,
        sigma_bins=1.0,
    )

    assert len(x_values) == 16
    assert y_values[0] > y_values[2]
    assert y_values[-1] > y_values[2]
    assert y_values[1] == pytest.approx(y_values[-1])


def test_depolarizing_phase_distribution_plot_writes_outputs(tmp_path):
    instance = build_fixed_instance(15, 2)
    algorithms = [KITAEV_ALGORITHM, STANDARD_ALGORITHM]
    samples = simulate_clean_noisy_phase_distributions(
        N=15,
        a=2,
        algorithms=algorithms,
        num_phase_bits=4,
        trials=2,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        rng=np.random.default_rng(123),
        true_order=instance.true_order,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        num_work_qubits=5,
        depolarizing_paulis=("Z",),
    )

    assert len(samples) == 4
    assert all(sum(row.count for row in sample.rows) == 2 for sample in samples)

    figure_path = tmp_path / "distribution.png"
    summary_path = tmp_path / "distribution.txt"
    plot_phase_estimation_distribution(
        N=15,
        a=2,
        true_order=instance.true_order,
        algorithms=algorithms,
        num_phase_bits=4,
        trials=2,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        samples=samples,
        output_path=figure_path,
        num_work_qubits=5,
        depolarizing_paulis=("Z",),
        y_max=0.8,
        y_axis_format=Y_AXIS_FORMAT_SCIENTIFIC,
        distribution_style=DISTRIBUTION_STYLE_BOTH,
        smooth_sigma_bins=2.5,
        padding_phase_radians=np.pi / 2.0,
    )
    write_phase_distribution_data_summary(
        output_path=summary_path,
        N=15,
        a=2,
        true_order=instance.true_order,
        algorithms=algorithms,
        num_phase_bits=4,
        trials=2,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        samples=samples,
        seed=123,
        num_work_qubits=5,
        depolarizing_paulis=("Z",),
        y_max=0.8,
        y_axis_format=Y_AXIS_FORMAT_SCIENTIFIC,
        distribution_style=DISTRIBUTION_STYLE_BOTH,
        smooth_sigma_bins=2.5,
        padding_phase_radians=np.pi / 2.0,
    )

    assert figure_path.exists()
    assert summary_path.exists()
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Depolarizing Pauli = Z only" in summary_text
    assert "Distribution style = both" in summary_text
    assert "Smooth sigma bins = 2.5" in summary_text
    assert "Y-axis upper limit = 0.8" in summary_text
    assert "Y-axis tick format = scientific" in summary_text
    assert "eigenphase = 0.25" in summary_text


def test_depolarizing_phase_distribution_main_accepts_y_max(tmp_path, monkeypatch):
    figure_path = tmp_path / "phase_distribution.png"
    monkeypatch.setattr(phase_distribution_module, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_phase_estimation_distribution.py",
            "15",
            "--a",
            "2",
            "--algorithm",
            "standard",
            "kitaev",
            "--bits",
            "9",
            "--trials",
            "1",
            "--p",
            "0",
            "--y-max",
            "0.4",
            "--distribution-style",
            "smooth",
            "--smooth-sigma-bins",
            "2.5",
            "--y-format",
            "scientific",
            "--output",
            str(figure_path),
            "--seed",
            "123",
        ],
    )

    phase_distribution_module.main()

    summaries = list(
        tmp_path.glob("depolarizing_phase_estimation_distribution_summary_*.txt")
    )
    assert figure_path.exists()
    assert len(summaries) == 1
    summary_text = summaries[0].read_text(encoding="utf-8")
    assert "Algorithms = standard Shor, Kitaev" in summary_text
    assert "Distribution style = smooth" in summary_text
    assert "Smooth sigma bins = 2.5" in summary_text
    assert "Y-axis upper limit = 0.4" in summary_text
    assert "Y-axis tick format = scientific" in summary_text


def test_coprime_cf_membership_excludes_nonrecovering_peaks():
    assert within_coprime_cf_sufficient_bound([0, 1, 0, 0], true_order=4)
    assert within_coprime_cf_sufficient_bound([1, 1, 0, 0], true_order=4)
    assert not within_coprime_cf_sufficient_bound([0, 0, 0, 0], true_order=4)
    assert not within_coprime_cf_sufficient_bound([1, 0, 0, 0], true_order=4)


def test_standard_statevector_zero_depolarizing_membership_stays_inside_s():
    instance = build_fixed_instance(15, 2)

    result = simulate_noisy_cf_membership(
        instance=instance,
        algorithm=STANDARD_ALGORITHM,
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        trials=3,
        rng=np.random.default_rng(123),
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        num_work_qubits=5,
    )

    assert result.inside_trials == 3
    assert result.outside_probability == 0.0
    assert result.num_work_qubits == 5


def test_zero_depolarizing_coprime_membership_excludes_zero_phase(monkeypatch):
    instance = build_fixed_instance(15, 2)
    bit_sequences = iter([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ])

    def fake_phase_bits_trial(**kwargs):
        return next(bit_sequences)

    monkeypatch.setattr(
        "graphs.depolarizing_error.cf_common.run_depolarizing_phase_bits_trial",
        fake_phase_bits_trial,
    )

    result = simulate_noisy_coprime_cf_membership(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        num_phase_bits=4,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        trials=2,
        rng=np.random.default_rng(123),
    )

    assert result.inside_trials == 1
    assert result.outside_probability == 0.5


def test_standard_statevector_zero_depolarizing_fill_has_no_sc_hits():
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
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        trials=3,
        rng=np.random.default_rng(123),
        membership_cache=cache,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        num_work_qubits=5,
    )

    assert result.outside_unique_hits == 0
    assert result.outside_trial_hits == 0
    assert result.num_work_qubits == 5


def test_fill_vs_trials_parser_and_trial_validation():
    parser = build_fill_vs_trials_argument_parser()
    args = parser.parse_args([
        "21",
        "--a",
        "11",
        "--trial-counts",
        "20",
        "10",
        "20",
    ])

    assert validate_trial_counts(args.trial_counts) == [10, 20]


def test_membership_vs_order_by_file_parser_accepts_overlay_files():
    parser = build_membership_vs_order_by_file_argument_parser()
    args = parser.parse_args([
        "--instances-files",
        "graphs/config/order_recover_instances_6.json",
        "graphs/config/order_recover_instances_7.json",
        "--labels",
        "L7",
        "L8",
        "--algorithm",
        "standard",
        "--fixed-work-register-per-file",
        "--error-model",
        "per-qubit-depolarizing",
    ])

    assert [path.name for path in args.instances_files] == [
        "order_recover_instances_6.json",
        "order_recover_instances_7.json",
    ]
    assert args.labels == ["L7", "L8"]
    assert args.algorithm == STANDARD_ALGORITHM
    assert args.fixed_work_register_per_file
    assert args.error_model == PER_QUBIT_DEPOLARIZING_MODEL


def test_coprime_membership_vs_order_by_file_parser_accepts_overlay_files():
    parser = build_coprime_membership_vs_order_by_file_argument_parser()
    args = parser.parse_args([
        "--instances-files",
        "graphs/config/order_recover_instances_6.json",
        "graphs/config/order_recover_instances_7.json",
        "--labels",
        "L7",
        "L8",
        "--algorithm",
        "kitaev",
        "--fixed-work-register-per-file",
        "--error-model",
        "per-qubit-depolarizing",
    ])

    assert [path.name for path in args.instances_files] == [
        "order_recover_instances_6.json",
        "order_recover_instances_7.json",
    ]
    assert args.labels == ["L7", "L8"]
    assert args.algorithm == KITAEV_ALGORITHM
    assert args.fixed_work_register_per_file
    assert args.error_model == PER_QUBIT_DEPOLARIZING_MODEL


def test_membership_excess_by_file_parser_accepts_overlay_files():
    parser = build_membership_excess_by_file_argument_parser()
    args = parser.parse_args([
        "--instances-files",
        "graphs/config/order_recover_instances_6.json",
        "graphs/config/order_recover_instances_7.json",
        "--labels",
        "L7",
        "L8",
        "--algorithm",
        "kitaev",
        "--membership",
        "coprime",
        "--fixed-work-register-per-file",
        "--error-model",
        "per-qubit-depolarizing",
    ])

    assert [path.name for path in args.instances_files] == [
        "order_recover_instances_6.json",
        "order_recover_instances_7.json",
    ]
    assert args.labels == ["L7", "L8"]
    assert args.algorithm == KITAEV_ALGORITHM
    assert args.membership == COPRIME_MEMBERSHIP
    assert args.fixed_work_register_per_file
    assert args.error_model == PER_QUBIT_DEPOLARIZING_MODEL


def test_membership_normalized_excess_by_file_parser_accepts_overlay_files():
    parser = build_membership_normalized_excess_by_file_argument_parser()
    args = parser.parse_args([
        "--instances-files",
        "graphs/config/order_recover_instances_6.json",
        "graphs/config/order_recover_instances_7.json",
        "--labels",
        "L7",
        "L8",
        "--algorithm",
        "kitaev",
        "--membership",
        "coprime",
        "--fixed-work-register-per-file",
        "--error-model",
        "per-qubit-depolarizing",
        "--p",
        "5",
        "--scale-error-by-tl",
        "--clean-good-trials",
        "200",
    ])

    assert [path.name for path in args.instances_files] == [
        "order_recover_instances_6.json",
        "order_recover_instances_7.json",
    ]
    assert args.labels == ["L7", "L8"]
    assert args.algorithm == KITAEV_ALGORITHM
    assert args.membership == COPRIME_MEMBERSHIP
    assert args.fixed_work_register_per_file
    assert args.error_model == PER_QUBIT_DEPOLARIZING_MODEL
    assert args.p == pytest.approx(5.0)
    assert args.scale_error_by_tl
    assert args.clean_good_trials == 200


def test_effective_depolarizing_error_probability_scales_by_tl():
    instance = build_fixed_instance(21, 11)

    assert effective_depolarizing_error_probability(
        error_probability=0.29,
        num_phase_bits=29,
        instance=instance,
        num_work_qubits=5,
        scale_error_by_tl=False,
    ) == pytest.approx(0.29)
    assert effective_depolarizing_error_probability(
        error_probability=0.29,
        num_phase_bits=29,
        instance=instance,
        num_work_qubits=5,
        scale_error_by_tl=True,
    ) == pytest.approx(0.29 / (29 * 5))
    assert effective_depolarizing_error_probability(
        error_probability=5.0,
        num_phase_bits=29,
        instance=instance,
        num_work_qubits=5,
        scale_error_by_tl=True,
    ) == pytest.approx(5.0 / (29 * 5))
    assert validate_depolarizing_error_scale(
        error_probability=5.0,
        scale_error_by_tl=True,
    ) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        validate_depolarizing_error_scale(
            error_probability=5.0,
            scale_error_by_tl=False,
        )
    with pytest.raises(ValueError, match="p_eff"):
        effective_depolarizing_error_probability(
            error_probability=200.0,
            num_phase_bits=29,
            instance=instance,
            num_work_qubits=5,
            scale_error_by_tl=True,
        )


def test_normalized_excess_failure_uses_clean_good_denominator():
    assert normalized_excess_failure(
        noisy_outside_probability=0.55,
        baseline_outside_probability=0.40,
    ) == pytest.approx(0.25)
    assert np.isnan(
        normalized_excess_failure(
            noisy_outside_probability=1.0,
            baseline_outside_probability=1.0,
        )
    )


def test_normalized_excess_sampler_uses_clean_good_target(monkeypatch):
    instance = build_fixed_instance(15, 2)
    trial_results = iter([
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
    ])

    def fake_trial(**kwargs):
        return next(trial_results)

    monkeypatch.setattr(
        "graphs.depolarizing_error.plot_noisy_cf_membership_normalized_excess_vs_order_by_instance_file.run_noisy_coprime_membership_trial",
        fake_trial,
    )

    baseline = simulate_clean_baseline_until_inside(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        num_phase_bits=4,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        clean_good_trials=2,
        rng=np.random.default_rng(123),
        membership=COPRIME_MEMBERSHIP,
    )
    noisy = simulate_membership_for_fixed_trials(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        num_phase_bits=4,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        trials=baseline.trials,
        rng=np.random.default_rng(456),
        membership=COPRIME_MEMBERSHIP,
    )

    assert baseline.inside_trials == 2
    assert baseline.trials == 4
    assert noisy.trials == baseline.trials
    assert noisy.inside_trials == 2


def test_membership_vs_order_by_file_validates_labels_and_work_modes():
    paths = [
        Path("graphs/config/order_recover_instances_6.json"),
        Path("graphs/config/order_recover_instances_8.json"),
    ]
    assert validate_instance_file_labels(paths, None) == [
        "order_recover_instances_6",
        "order_recover_instances_8",
    ]
    with pytest.raises(ValueError, match="same length"):
        validate_instance_file_labels(paths, ["only-one"])

    instances_by_file = [
        [build_fixed_instance(65, 14)],
        [build_fixed_instance(259, 36)],
    ]
    assert resolve_work_qubits_by_file(
        instances_by_file,
        fixed_work_register_per_file=True,
    ) == [7, 9]
    assert resolve_work_qubits_by_file(
        instances_by_file,
        requested_work_qubits=10,
    ) == [10, 10]
    with pytest.raises(ValueError, match="choose either"):
        resolve_work_qubits_by_file(
            instances_by_file,
            fixed_work_register=True,
            fixed_work_register_per_file=True,
        )


def test_depolarizing_noisy_cf_plots_write_files(tmp_path):
    instance = build_fixed_instance(15, 2)
    membership_results = [
        NoisyCFMembershipResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            error_model=BLOCK_DEPOLARIZING_MODEL,
            error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            inside_trials=6,
        ),
        NoisyCFMembershipResult(
            instance=instance,
            algorithm=STANDARD_ALGORITHM,
            error_model=BLOCK_DEPOLARIZING_MODEL,
            error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            inside_trials=8,
        ),
    ]
    membership_output = tmp_path / "membership.png"
    plot_noisy_cf_membership_bar(
        output_path=membership_output,
        results=membership_results,
        algorithms=[KITAEV_ALGORITHM, STANDARD_ALGORITHM],
    )

    cache = CFMembershipCache(
        true_order=instance.true_order,
        num_phase_bits=8,
        inside_count=20,
        outside_count=236,
        inside_bits_packed=np.zeros(32, dtype=np.uint8),
    )
    fill_results = [
        NoisyCFComplementFillResult(
            instance=instance,
            algorithm=KITAEV_ALGORITHM,
            error_model=BLOCK_DEPOLARIZING_MODEL,
            error_probability=0.1,
            trials=10,
            num_phase_bits=8,
            outside_unique_hits=4,
            outside_trial_hits=6,
            total_inside_bitstrings=20,
            total_outside_bitstrings=236,
        )
    ]
    fill_output = tmp_path / "fill.png"
    plot_noisy_cf_complement_fill(
        output_path=fill_output,
        results=fill_results,
        algorithms=[KITAEV_ALGORITHM],
        membership_cache=cache,
    )

    assert membership_output.exists()
    assert fill_output.exists()


def test_depolarizing_membership_vs_order_by_file_plot_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    sweep = InstanceFileMembershipSweep(
        instances_file=Path("graphs/config/order_recover_instances_6.json"),
        label="L6 group",
        fixed_num_work_qubits=5,
        results=[
            NoisyCFMembershipResult(
                instance=instance,
                algorithm=KITAEV_ALGORITHM,
                error_model=BLOCK_DEPOLARIZING_MODEL,
                error_probability=0.1,
                trials=10,
                num_phase_bits=8,
                inside_trials=6,
                num_work_qubits=5,
            )
        ],
    )
    output = tmp_path / "by_file.png"

    plot_noisy_cf_membership_vs_order_by_instance_file(
        output_path=output,
        sweeps=[sweep],
        algorithm=KITAEV_ALGORITHM,
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        trials_per_instance=10,
    )

    assert output.exists()
    assert membership_vs_order_by_file_output_filename_segment(
        algorithm=KITAEV_ALGORITHM,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        instances_files=[Path("graphs/config/order_recover_instances_6.json")],
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        sweeps=[sweep],
    ) == (
        "kitaev_instances_ori_6_"
        "dp_p_1em1_t8_fixed_work_L5"
    )


def test_depolarizing_coprime_membership_vs_order_by_file_plot_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    sweep = InstanceFileMembershipSweep(
        instances_file=Path("graphs/config/order_recover_instances_6.json"),
        label="L6 group",
        fixed_num_work_qubits=5,
        results=[
            NoisyCFMembershipResult(
                instance=instance,
                algorithm=KITAEV_ALGORITHM,
                error_model=BLOCK_DEPOLARIZING_MODEL,
                error_probability=0.1,
                trials=10,
                num_phase_bits=8,
                inside_trials=5,
                num_work_qubits=5,
            )
        ],
    )
    output = tmp_path / "coprime_by_file.png"

    plot_noisy_coprime_cf_membership_vs_order_by_instance_file(
        output_path=output,
        sweeps=[sweep],
        algorithm=KITAEV_ALGORITHM,
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        trials_per_instance=10,
    )

    assert output.exists()
    assert coprime_membership_vs_order_by_file_output_filename_segment(
        algorithm=KITAEV_ALGORITHM,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        instances_files=[Path("graphs/config/order_recover_instances_6.json")],
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        sweeps=[sweep],
    ) == (
        "kitaev_instances_ori_6_"
        "dp_p_1em1_t8_fixed_work_L5"
    )


def test_depolarizing_membership_excess_by_file_plot_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    noisy_result = NoisyCFMembershipResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        trials=10,
        num_phase_bits=8,
        inside_trials=5,
        num_work_qubits=5,
    )
    baseline_result = NoisyCFMembershipResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        trials=10,
        num_phase_bits=8,
        inside_trials=8,
        num_work_qubits=5,
    )
    sweep = InstanceFileMembershipExcessSweep(
        instances_file=Path("graphs/config/order_recover_instances_6.json"),
        label="L6 group",
        fixed_num_work_qubits=5,
        noisy_results=[noisy_result],
        baseline_results=[baseline_result],
    )
    output = tmp_path / "excess_by_file.png"

    plot_noisy_cf_membership_excess_vs_order_by_instance_file(
        output_path=output,
        sweeps=[sweep],
        algorithm=KITAEV_ALGORITHM,
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        trials_per_instance=10,
        membership=COPRIME_MEMBERSHIP,
    )

    assert output.exists()
    assert membership_excess_by_file_output_filename_segment(
        algorithm=KITAEV_ALGORITHM,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        instances_files=[Path("graphs/config/order_recover_instances_6.json")],
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        membership=COPRIME_MEMBERSHIP,
        sweeps=[sweep],
    ) == (
        "kitaev_instances_ori_6_scoprime_excess_"
        "dp_p_1em1_minus_p0_t8_fixed_work_L5"
    )


def test_depolarizing_membership_normalized_excess_by_file_plot_writes_file(tmp_path):
    instance = build_fixed_instance(15, 2)
    noisy_result = NoisyCFMembershipResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        trials=10,
        num_phase_bits=8,
        inside_trials=5,
        num_work_qubits=5,
    )
    baseline_result = NoisyCFMembershipResult(
        instance=instance,
        algorithm=KITAEV_ALGORITHM,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        trials=10,
        num_phase_bits=8,
        inside_trials=8,
        num_work_qubits=5,
    )
    sweep = InstanceFileMembershipExcessSweep(
        instances_file=Path("graphs/config/order_recover_instances_6.json"),
        label="L6 group",
        fixed_num_work_qubits=5,
        noisy_results=[noisy_result],
        baseline_results=[baseline_result],
    )
    output = tmp_path / "normalized_excess_by_file.png"

    plot_noisy_cf_membership_normalized_excess_vs_order_by_instance_file(
        output_path=output,
        sweeps=[sweep],
        algorithm=KITAEV_ALGORITHM,
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        clean_good_trials=8,
        membership=COPRIME_MEMBERSHIP,
    )

    assert output.exists()
    assert membership_normalized_excess_by_file_output_filename_segment(
        algorithm=KITAEV_ALGORITHM,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        instances_files=[Path("graphs/config/order_recover_instances_6.json")],
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        membership=COPRIME_MEMBERSHIP,
        clean_good_trials=8,
        sweeps=[sweep],
    ) == (
        "kitaev_instances_ori_6_scoprime_norm_excess_"
        "dp_p_1em1_minus_p0_t8_cg8_fixed_work_L5"
    )
    assert membership_normalized_excess_by_file_output_filename_segment(
        algorithm=KITAEV_ALGORITHM,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        instances_files=[Path("graphs/config/order_recover_instances_6.json")],
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        membership=COPRIME_MEMBERSHIP,
        clean_good_trials=8,
        sweeps=[sweep],
        scale_error_by_tl=True,
    ) == (
        "kitaev_instances_ori_6_scoprime_norm_excess_"
        "dp_p_1em1_scaled_tL_minus_p0_t8_cg8_fixed_work_L5"
    )
    assert membership_normalized_excess_by_file_output_filename_segment(
        algorithm=KITAEV_ALGORITHM,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
        instances_files=[Path("graphs/config/order_recover_instances_6.json")],
        num_phase_bits=8,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=5.0,
        membership=COPRIME_MEMBERSHIP,
        clean_good_trials=8,
        sweeps=[sweep],
        scale_error_by_tl=True,
    ) == (
        "kitaev_instances_ori_6_scoprime_norm_excess_"
        "dp_p_5e0_scaled_tL_minus_p0_t8_cg8_fixed_work_L5"
    )
