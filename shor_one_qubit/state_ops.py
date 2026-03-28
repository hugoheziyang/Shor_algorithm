import numpy as np


def make_joint_state(work_state: np.ndarray) -> np.ndarray:
    """
    Build the joint state |0> ⊗ |work_state>.

    Returns an array of shape (2, M), where M is the work-register dimension.
    """
    work_state = np.asarray(work_state, dtype=complex)

    if work_state.ndim != 1:
        raise ValueError("work_state must be a 1D array")

    psi = np.zeros((2, work_state.shape[0]), dtype=complex)
    psi[0, :] = work_state
    return psi


def apply_control_gate(psi: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Apply a 2x2 gate G to the control qubit only.

    psi has shape (2, M).
    G has shape (2, 2).
    """
    psi = np.asarray(psi, dtype=complex)
    G = np.asarray(G, dtype=complex)

    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError("psi must have shape (2, M)")
    if G.shape != (2, 2):
        raise ValueError("G must have shape (2, 2)")

    return G @ psi


def apply_controlled_work_operator(psi: np.ndarray, work_op) -> np.ndarray:
    """
    Apply the controlled operator

        |0><0| ⊗ I + |1><1| ⊗ work_op

    to the joint state psi.

    psi has shape (2, M).

    work_op can be either:
      - a callable taking a vector of shape (M,) and returning a vector of shape (M,)
      - an array of shape (M, M)
    """
    psi = np.asarray(psi, dtype=complex)

    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError("psi must have shape (2, M)")

    out = psi.copy()

    if callable(work_op):
        out[1, :] = work_op(psi[1, :])
    else:
        work_op = np.asarray(work_op, dtype=complex)
        M = psi.shape[1]
        if work_op.shape != (M, M):
            raise ValueError(f"work_op must have shape ({M}, {M})")
        out[1, :] = work_op @ psi[1, :]

    return out


def control_probabilities(psi: np.ndarray) -> tuple[float, float]:
    """
    Return the probabilities of measuring the control qubit in |0> or |1>.
    """
    psi = np.asarray(psi, dtype=complex)

    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError("psi must have shape (2, M)")

    p0 = float(np.vdot(psi[0, :], psi[0, :]).real)
    p1 = float(np.vdot(psi[1, :], psi[1, :]).real)

    total = p0 + p1
    if total <= 0.0:
        raise ValueError("state has zero norm")

    p0 /= total
    p1 /= total
    return p0, p1


def measure_control(psi: np.ndarray, rng=None) -> tuple[int, np.ndarray]:
    """
    Measure the control qubit in the computational basis.

    Returns:
      outcome : 0 or 1
      collapsed_state : post-measurement joint state, normalized
    """
    psi = np.asarray(psi, dtype=complex)

    if psi.ndim != 2 or psi.shape[0] != 2:
        raise ValueError("psi must have shape (2, M)")

    if rng is None:
        rng = np.random.default_rng()

    p0, p1 = control_probabilities(psi)
    outcome = int(rng.choice([0, 1], p=[p0, p1]))

    collapsed = np.zeros_like(psi)

    if outcome == 0:
        collapsed[0, :] = psi[0, :] / np.sqrt(p0)
    else:
        collapsed[1, :] = psi[1, :] / np.sqrt(p1)

    return outcome, collapsed