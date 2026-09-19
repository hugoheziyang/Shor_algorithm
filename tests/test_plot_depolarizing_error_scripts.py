import numpy as np

from graphs.depolarizing_error.plot_hamming_distance_vs_error_probability import (
    BLOCK_DEPOLARIZING_MODEL,
    PER_QUBIT_DEPOLARIZING_MODEL,
    build_argument_parser,
    depolarizing_region_phase_bits,
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
    assert error_config_for_model(
        "per-qubit",
        0.3,
        allowed_phase_bits=(1, 3),
        depolarizing_paulis=("Y",),
    ) == ErrorConfig(
        controlled_unitary_per_qubit_depolarizing_probability=0.3,
        controlled_unitary_depolarizing_allowed_phase_bits=(1, 3),
        controlled_unitary_depolarizing_paulis=("Y",),
    )


def test_depolarizing_region_argument_maps_to_phase_bits():
    parser = build_argument_parser()
    args = parser.parse_args([
        "21",
        "--a",
        "11",
        "--depolarizing-region",
        "dangerous",
        "safe-tail",
    ])

    assert args.depolarizing_region == ["dangerous", "safe-tail"]
    assert depolarizing_region_phase_bits(
        num_phase_bits=10,
        true_order=16,
        depolarizing_region=args.depolarizing_region,
    ) == (5, 6, 7, 8, 9, 10)


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
        num_work_qubits=5,
    )

    assert result.trials == 2
    assert result.error_probability == 0.0
    assert result.num_work_qubits == 5
