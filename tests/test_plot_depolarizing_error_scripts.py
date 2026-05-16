import numpy as np

from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
    error_config_for_model,
    normalize_error_model,
    normalize_error_models,
)
from graphs.depolarizing_error.plot_order_recovery_success_vs_order import (
    OrderRecoveryInstance,
    simulate_order_recovery_for_instance,
)
from shor_common.errors import ErrorConfig
from shor_standard.backends import STATEVECTOR_STANDARD_BACKEND


def test_depolarizing_hamming_script_accepts_only_depolarizing_models():
    assert normalize_error_model("block") == BLOCK_DEPOLARIZING_MODEL
    assert normalize_error_model("per-qubit") == PER_QUBIT_DEPOLARIZING_MODEL
    assert normalize_error_models(None) == [
        BLOCK_DEPOLARIZING_MODEL,
        PER_QUBIT_DEPOLARIZING_MODEL,
    ]


def test_depolarizing_error_config_for_model_sets_expected_field():
    assert error_config_for_model("block", 0.2) == ErrorConfig(
        controlled_unitary_depolarizing_probability=0.2,
    )
    assert error_config_for_model("per-qubit", 0.3) == ErrorConfig(
        controlled_unitary_per_qubit_depolarizing_probability=0.3,
    )


def test_depolarizing_order_recovery_simulates_standard_statevector():
    result = simulate_order_recovery_for_instance(
        instance=OrderRecoveryInstance(N=15, a=2, true_order=4),
        algorithm="standard",
        error_model=BLOCK_DEPOLARIZING_MODEL,
        error_probability=0.0,
        trials=2,
        rng=np.random.default_rng(123),
        num_phase_bits=8,
        standard_backend=STATEVECTOR_STANDARD_BACKEND,
    )

    assert result.trials == 2
    assert result.error_probability == 0.0
