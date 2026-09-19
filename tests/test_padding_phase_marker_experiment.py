import numpy as np
import pytest

import graphs.depolarizing_error.plot_padding_phase_marker_experiment as marker_module
from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
)
from graphs.depolarizing_error.plot_padding_phase_marker_experiment import (
    marker_distance_to_order_phases,
    marker_phase_radians,
    marker_window_excess,
    random_marker_bins,
    simulate_padding_marker_experiment,
    validate_marker_bins,
)
from graphs.depolarizing_error.plot_order_recovery_success_vs_order import (
    KITAEV_ALGORITHM,
)
from graphs.depolarizing_error.plot_phase_estimation_distribution import (
    PhaseDistributionSample,
    PhaseOutcomeRow,
    validate_stem_linewidth,
)
from shor_standard.backends import STATEVECTOR_STANDARD_BACKEND


def test_random_marker_bins_are_distinct_nonzero_and_reproducible():
    marker_bins = random_marker_bins(
        num_phase_bits=11,
        rng=np.random.default_rng(123),
        marker_count=3,
    )
    repeated_bins = random_marker_bins(
        num_phase_bits=11,
        rng=np.random.default_rng(123),
        marker_count=3,
    )

    assert marker_bins == repeated_bins
    assert len(marker_bins) == len(set(marker_bins)) == 3
    assert all(0 < marker_bin < 2048 for marker_bin in marker_bins)


def test_random_marker_bins_reject_too_many_markers():
    with pytest.raises(ValueError, match="not enough nonzero phase bins"):
        random_marker_bins(
            num_phase_bits=2,
            rng=np.random.default_rng(123),
            marker_count=4,
        )


def test_marker_distance_to_order_phases_is_report_only_diagnostic():
    assert marker_distance_to_order_phases(171, 2048, 6) == pytest.approx(
        1.0 / 12.0,
        abs=1.0 / 2048.0,
    )


def test_validate_marker_bins_rejects_duplicates_and_endpoints():
    with pytest.raises(ValueError, match="distinct"):
        validate_marker_bins([3, 3], num_phase_bits=4)
    with pytest.raises(ValueError, match="0 < p < Q"):
        validate_marker_bins([0, 3], num_phase_bits=4)
    with pytest.raises(ValueError, match="0 < p < Q"):
        validate_marker_bins([3, 16], num_phase_bits=4)


def test_marker_phase_is_exact_selected_bin():
    theta = marker_phase_radians(marker_bin=13, num_phase_bits=5)
    assert theta / (2.0 * np.pi) == pytest.approx(13.0 / 32.0)


def test_validate_stem_linewidth_accepts_positive_values_only():
    assert validate_stem_linewidth(None) is None
    assert validate_stem_linewidth(2) == 2.0
    with pytest.raises(ValueError, match="stem-linewidth must be > 0"):
        validate_stem_linewidth(0)


def test_noisy_marker_runs_reuse_each_trial_random_stream(monkeypatch):
    calls = []

    def fake_run_phase_trial(**kwargs):
        random_value = kwargs["rng"].random()
        calls.append((kwargs["padding_phase_radians"], random_value))
        return [0, 0, 0, 0]

    monkeypatch.setattr(marker_module, "run_phase_trial", fake_run_phase_trial)

    experiments = simulate_padding_marker_experiment(
        N=15,
        a=2,
        algorithms=[KITAEV_ALGORITHM],
        num_phase_bits=4,
        trials=2,
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.1,
        marker_bins=[1, 5],
        rng=np.random.default_rng(123),
        true_order=4,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
    )

    assert len(experiments) == 1
    assert len(calls) == 8
    for trial_start in (0, 4):
        noisy_draws = [calls[index][1] for index in range(trial_start + 1, trial_start + 4)]
        assert noisy_draws[0] == noisy_draws[1] == noisy_draws[2]


def test_marker_window_excess_uses_marker_minus_identity():
    marker = PhaseDistributionSample(
        algorithm=KITAEV_ALGORITHM,
        noise_label="marker",
        rows=[PhaseOutcomeRow("0011", 3, 3 / 16, 7, 0.7)],
        trials=10,
        num_phase_bits=4,
        num_work_qubits=4,
    )
    identity = PhaseDistributionSample(
        algorithm=KITAEV_ALGORITHM,
        noise_label="identity",
        rows=[PhaseOutcomeRow("0011", 3, 3 / 16, 2, 0.2)],
        trials=10,
        num_phase_bits=4,
        num_work_qubits=4,
    )

    assert marker_window_excess(marker, identity, 3, 0) == pytest.approx(0.5)


def test_padding_marker_main_writes_figure_and_summary(tmp_path, monkeypatch):
    figure_path = tmp_path / "markers.png"
    monkeypatch.setattr(marker_module, "DEFAULT_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_padding_phase_marker_experiment.py",
            "15",
            "--a",
            "2",
            "--algorithm",
            "kitaev",
            "--bits",
            "4",
            "--trials",
            "2",
            "--p",
            "0",
            "--marker-bins",
            "1",
            "5",
            "--output",
            str(figure_path),
            "--seed",
            "123",
        ],
    )

    marker_module.main()

    summaries = list(tmp_path.glob("padding_phase_marker_experiment_summary_*.txt"))
    assert figure_path.exists()
    assert len(summaries) == 1
    summary_text = summaries[0].read_text(encoding="utf-8")
    assert "True order r = 4" in summary_text
    assert "Marker source = explicit --marker-bins" in summary_text
    assert "Probability differences are marker noisy minus theta=0 noisy" in summary_text
