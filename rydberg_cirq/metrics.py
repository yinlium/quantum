"""
Entanglement and metrology metrics, expressed with Cirq primitives.

Collective spin observables are built as :class:`cirq.PauliSum` objects so they can
be evaluated directly from a Cirq statevector or density matrix via
``expectation_from_state_vector`` / ``expectation_from_density_matrix`` -- no manual
matrix construction or ``np.trace(rho @ J)`` bookkeeping.

Provides
--------
* collective spin operators as ``cirq.PauliSum``
* mean spin, covariance, minimum transverse variance
* Wineland ``xi_R^2`` and Kitagawa-Ueda ``xi_S^2`` squeezing parameters
* permutation-symmetric two-atom reduced density matrix and Wootters concurrence
* Quantum Fisher Information
* Dicke-state purity and ``g^(2)``

See :mod:`rydberg_cirq.dicke` for the ``|0> = |g>`` sign convention that fixes
``Jz = -(1/2) sum_j Z_j`` and ``Jy = -(1/2) sum_j Y_j``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la
import cirq

__all__ = [
    "collective_spin_paulisums",
    "spin_observables",
    "squeezing_parameters",
    "two_atom_reduced_dm",
    "concurrence",
    "quantum_fisher_information",
    "dicke_purity",
    "g2_from_populations",
]

_PAULI = {
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
}


def collective_spin_paulisums(qubits) -> dict:
    """Collective spin operators ``Jx, Jy, Jz`` as :class:`cirq.PauliSum` objects.

    Uses the ``|0> = |g>`` convention, so ``Jz|g...g> = -(N/2)|g...g>``:

        Jx = +(1/2) sum_j X_j,  Jy = -(1/2) sum_j Y_j,  Jz = -(1/2) sum_j Z_j

    These signs satisfy ``[Jx, Jy] = i Jz``.
    """
    qubits = list(qubits)
    jx = cirq.PauliSum.from_pauli_strings(
        [cirq.PauliString({q: cirq.X}, coefficient=0.5) for q in qubits]
    )
    jy = cirq.PauliSum.from_pauli_strings(
        [cirq.PauliString({q: cirq.Y}, coefficient=-0.5) for q in qubits]
    )
    jz = cirq.PauliSum.from_pauli_strings(
        [cirq.PauliString({q: cirq.Z}, coefficient=-0.5) for q in qubits]
    )
    return {"Jx": jx, "Jy": jy, "Jz": jz}


def _expectation(op: cirq.PauliSum, state, qubit_map, is_density_matrix: bool) -> float:
    if is_density_matrix:
        val = op.expectation_from_density_matrix(state, qubit_map)
    else:
        val = op.expectation_from_state_vector(state, qubit_map)
    return float(np.real(val))


def spin_observables(state, qubits, is_density_matrix: bool | None = None) -> dict:
    """First and symmetrised second moments of the collective spin.

    Args:
        state: a statevector of shape ``(2**N,)`` or a density matrix ``(2**N, 2**N)``.
        qubits: the qubit ordering matching the state's tensor factors.
        is_density_matrix: inferred from ``state.ndim`` when omitted.

    Returns:
        A dict with ``J_mean``, ``J_len``, ``Jx/Jy/Jz``, the squares
        ``Jx2/Jy2/Jz2`` and symmetrised cross terms ``JxJy/JxJz/JyJz``.
    """
    qubits = list(qubits)
    state = np.asarray(state)
    if is_density_matrix is None:
        is_density_matrix = state.ndim == 2
    state = state.astype(np.complex64 if state.dtype == np.complex64 else np.complex128)
    qubit_map = {q: i for i, q in enumerate(qubits)}

    J = collective_spin_paulisums(qubits)
    jx, jy, jz = J["Jx"], J["Jy"], J["Jz"]

    exp = {k: _expectation(v, state, qubit_map, is_density_matrix) for k, v in J.items()}

    # Symmetrised products; PauliSum multiplication handles the operator algebra.
    def sym(a, b):
        return _expectation(0.5 * (a * b + b * a), state, qubit_map, is_density_matrix)

    out = {
        "J_mean": np.array([exp["Jx"], exp["Jy"], exp["Jz"]]),
        "Jx": exp["Jx"],
        "Jy": exp["Jy"],
        "Jz": exp["Jz"],
        "Jx2": sym(jx, jx),
        "Jy2": sym(jy, jy),
        "Jz2": sym(jz, jz),
        "JxJy": sym(jx, jy),
        "JxJz": sym(jx, jz),
        "JyJz": sym(jy, jz),
    }
    out["J_len"] = float(np.linalg.norm(out["J_mean"]))
    return out


def squeezing_parameters(obs: dict, N: int) -> dict:
    """Minimum transverse variance and the Wineland / Kitagawa-Ueda parameters.

    ``xi_R^2 = N (Delta J_perp)^2 / |<J>|^2`` (metrological squeezing; < 1 beats the
    standard quantum limit) and ``xi_S^2 = 4 (Delta J_perp)^2 / N``.
    """
    J_mean, J_len = obs["J_mean"], obs["J_len"]
    if J_len < 1e-10:
        return {"xi_R2": np.nan, "xi_S2": np.nan, "var_perp_min": np.nan}

    n0 = J_mean / J_len
    ref = np.array([0.0, 0.0, 1.0]) if abs(n0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n1 = np.cross(n0, ref)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(n0, n1)

    cov = np.array(
        [
            [obs["Jx2"] - obs["Jx"] ** 2, obs["JxJy"] - obs["Jx"] * obs["Jy"], obs["JxJz"] - obs["Jx"] * obs["Jz"]],
            [obs["JxJy"] - obs["Jx"] * obs["Jy"], obs["Jy2"] - obs["Jy"] ** 2, obs["JyJz"] - obs["Jy"] * obs["Jz"]],
            [obs["JxJz"] - obs["Jx"] * obs["Jz"], obs["JyJz"] - obs["Jy"] * obs["Jz"], obs["Jz2"] - obs["Jz"] ** 2],
        ]
    )

    P = np.column_stack([n1, n2])
    var_perp_min = max(0.0, float(np.linalg.eigvalsh(P.T @ cov @ P)[0]))

    return {
        "var_perp_min": var_perp_min,
        "xi_R2": (N * var_perp_min) / (J_len**2),
        "xi_S2": (4.0 * var_perp_min) / N,
    }


def two_atom_reduced_dm(obs: dict, N: int) -> np.ndarray:
    """Permutation-symmetric two-atom reduced density matrix from collective moments.

    For a permutation-symmetric state of ``N`` atoms, ``rho_12`` is fully determined
    by the collective first and second moments, avoiding an explicit partial trace.
    Basis order: ``|00>, |01>, |10>, |11>``.
    """
    if N < 2:
        raise ValueError("two-atom reduced density matrix requires N >= 2")

    # This package defines J_a = sigma_a * (1/2) * sum_j P_a^(j) with
    # sigma_x = +1, sigma_y = -1, sigma_z = -1 (see rydberg_cirq.dicke).
    # Converting collective moments back into Pauli expectation values therefore
    # requires those signs, otherwise |gg> and |rr> come out swapped.
    sign = np.array([1.0, -1.0, -1.0])

    keys = ("Jx", "Jy", "Jz")
    s = np.array([sign[a] * 2.0 * obs[keys[a]] / N for a in range(3)])

    # Diagonal correlators are sign-independent (sigma_a^2 = 1) but must have the
    # self-term  sum_j P_a^(j) P_a^(j) = N  removed before normalising.
    c = np.zeros((3, 3))
    diag_keys = ("Jx2", "Jy2", "Jz2")
    for a in range(3):
        c[a, a] = (4.0 * obs[diag_keys[a]] - N) / (N * (N - 1))

    cross = {(0, 1): "JxJy", (0, 2): "JxJz", (1, 2): "JyJz"}
    for (a, b), key in cross.items():
        val = sign[a] * sign[b] * 4.0 * obs[key] / (N * (N - 1))
        c[a, b] = c[b, a] = val

    s0 = np.eye(2, dtype=complex)
    paulis = [cirq.unitary(cirq.X), cirq.unitary(cirq.Y), cirq.unitary(cirq.Z)]

    rho = np.kron(s0, s0).astype(complex)
    for a in range(3):
        rho += s[a] * (np.kron(paulis[a], s0) + np.kron(s0, paulis[a]))
        for b in range(3):
            rho += c[a, b] * np.kron(paulis[a], paulis[b])
    return 0.25 * rho


def concurrence(rho12: np.ndarray) -> float:
    """Wootters concurrence of a two-qubit density matrix."""
    sy = cirq.unitary(cirq.Y)
    sy_sy = np.kron(sy, sy)
    rho_tilde = sy_sy @ rho12.conj() @ sy_sy
    eigvals = la.eigvals(rho12 @ rho_tilde)
    lambdas = np.sort(np.sqrt(np.maximum(0.0, np.real(eigvals))))[::-1]
    return float(max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))


def quantum_fisher_information(rho: np.ndarray, op: np.ndarray) -> float:
    """QFI ``F_Q[rho, op] = 2 sum_{jk} (p_j - p_k)^2 / (p_j + p_k) |<j|op|k>|^2``.

    For a pure state this reduces to ``4 Var(op)``.  ``F_Q / N > 1`` certifies
    metrologically useful entanglement (beyond the standard quantum limit).
    """
    eigvals, eigvecs = la.eigh(rho)
    mat = eigvecs.conj().T @ op @ eigvecs
    p = eigvals[:, None]
    q = eigvals[None, :]
    denom = p + q
    num = (p - q) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(denom > 1e-14, num / np.where(denom > 1e-14, denom, 1.0), 0.0)
    return float(2.0 * np.sum(weights * np.abs(mat) ** 2))


def dicke_purity(p_m: np.ndarray) -> float:
    """Fraction of the *excited* population residing in the single-excitation sector.

    ``P_Dicke = p_1 / (1 - p_0)`` -> 1 as interaction-induced dephasing purifies the
    spin wave into ``|W>`` (Phys. Rev. A 106, L051701).
    """
    p0 = float(p_m[0])
    denom = 1.0 - p0
    return float(p_m[1] / denom) if denom > 1e-12 else np.nan


def g2_from_populations(p_m: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> float:
    """Second-order autocorrelation of the retrieved field.

        g2 = sum_m |c_m|^2 m(m-1) X_m  /  (sum_m |c_m|^2 m Y_m)^2

    ``X_m`` and ``Y_m`` are the storage-time-dependent overlap factors of
    Phys. Rev. A 106, L051701 Eqs. (S.7)-(S.8).
    """
    m = np.arange(len(p_m))
    num = np.sum(p_m * m * (m - 1) * x_m)
    den = np.sum(p_m * m * y_m)
    return float(num / den**2) if den > 1e-15 else np.nan
