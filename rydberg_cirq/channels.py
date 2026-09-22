"""
Quantum channels (``_kraus_`` gates) for Rydberg ensemble decoherence.

The centrepiece is :class:`CollectiveDephasingChannel`, which replaces the
hand-rolled Strang-splitting master-equation solver with a genuine CPTP Cirq
channel that can be fed to ``cirq.DensityMatrixSimulator``.

Collective dephasing
--------------------
The Lindblad master equation for collective dephasing is

    d(rho)/dt = -i [H, rho] + (gamma_c / 2) (2 Jz rho Jz - Jz^2 rho - rho Jz^2)

Integrated over a time ``tau`` (with H switched off) this acts on matrix elements
between states of excitation number ``m`` and ``m'`` as

    rho_{m m'}  ->  rho_{m m'} * exp(-gamma_c * tau * (m - m')^2 / 2)         (*)

which is exactly the effect of applying a *global* random phase rotation
``exp(-i phi Jz)`` with ``phi ~ Normal(0, gamma_c * tau)`` and averaging, since
``<exp(-i phi (m - m'))> = exp(-gamma_c tau (m - m')^2 / 2)``.

Exact finite Kraus decomposition
--------------------------------
Rather than approximating (*) by sampling trajectories, we build an **exact,
finite** Kraus set.  Let ``D`` be the real symmetric ``2^N x 2^N`` matrix

    D[s, s'] = exp(-gamma_c * tau * (m_s - m_s')^2 / 2)

``D`` is a Gaussian kernel Gram matrix evaluated at the integer points ``m_s``, hence
positive semi-definite.  Its eigendecomposition ``D = sum_k lambda_k v_k v_k^T``
yields the diagonal Kraus operators

    K_k = sqrt(lambda_k) * diag(v_k)

Then ``sum_k K_k rho K_k^dag`` reproduces (*) elementwise, and trace preservation
``sum_k K_k^dag K_k = diag(D_ss) = I`` holds because ``D_ss = 1``.  Eigenvalues
below ``tol`` are dropped and the remainder renormalised, so the channel stays
CPTP to machine precision.
"""

from __future__ import annotations

import numpy as np
import cirq

from .dicke import hamming_weights

__all__ = [
    "CollectiveDephasingChannel",
    "collective_dephasing_kraus",
    "dicke_dephasing_factors",
    "RydbergDecayChannel",
]


def dicke_dephasing_factors(N: int, gamma_c: float, tau: float) -> np.ndarray:
    """The ``(N+1, N+1)`` elementwise damping matrix of Eq. (*) in the Dicke basis."""
    m = np.arange(N + 1)
    dm = m[:, None] - m[None, :]
    return np.exp(-0.5 * gamma_c * tau * dm.astype(float) ** 2)


def collective_dephasing_kraus(
    N: int, gamma_c: float, tau: float, tol: float = 1e-12
) -> list[np.ndarray]:
    """Exact finite Kraus operators for collective dephasing on ``N`` qubits.

    Args:
        N: number of atoms (qubits).
        gamma_c: collective dephasing rate.
        tau: duration.
        tol: eigenvalues of the Gram matrix below this are discarded.

    Returns:
        A list of diagonal ``2^N x 2^N`` Kraus operators summing to the identity.
    """
    dim = 2**N
    if gamma_c * tau <= 0.0:
        return [np.eye(dim, dtype=complex)]

    m = hamming_weights(N).astype(float)
    dm = m[:, None] - m[None, :]
    D = np.exp(-0.5 * gamma_c * tau * dm**2)

    # D is a PSD Gaussian kernel -> exact eigen (Kraus) decomposition.
    eigvals, eigvecs = np.linalg.eigh(D)
    keep = eigvals > tol
    eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]

    # Diagonals of the (diagonal) Kraus operators, one row per operator.
    diags = np.sqrt(eigvals)[:, None] * eigvecs.T.astype(complex)

    # Renormalise away the truncation error so the channel is exactly trace preserving.
    total = np.sum(np.abs(diags) ** 2, axis=0)
    diags *= 1.0 / np.sqrt(np.maximum(total, 1e-300))[None, :]

    return [np.diag(d) for d in diags]


class CollectiveDephasingChannel(cirq.Gate):
    """``N``-qubit CPTP channel implementing collective (correlated) dephasing.

    Unlike ``cirq.phase_damp``, which acts independently on each qubit, this channel
    is *correlated across the whole ensemble*: it damps coherences between different
    total excitation numbers while leaving every fixed-``m`` sector untouched.  That
    distinction is the entire physics of Phys. Rev. A 106, L051701 -- it is what
    purifies a multi-excitation spin wave into the single-excitation Dicke state.

    Example:
        >>> import cirq, numpy as np
        >>> from rydberg_cirq.channels import CollectiveDephasingChannel
        >>> q = cirq.LineQubit.range(3)
        >>> ch = CollectiveDephasingChannel(3, gamma_c=1.0, tau=0.5)
        >>> circuit = cirq.Circuit(cirq.H.on_each(*q), ch.on(*q))
        >>> rho = cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix
        >>> float(np.round(np.trace(rho).real, 10))
        1.0
    """

    def __init__(self, num_qubits: int, gamma_c: float, tau: float, tol: float = 1e-12):
        super().__init__()
        self._n = int(num_qubits)
        self.gamma_c = float(gamma_c)
        self.tau = float(tau)
        self.tol = float(tol)
        self._cached_kraus: tuple | None = None

    def _num_qubits_(self) -> int:
        return self._n

    def _kraus_(self):
        if self._cached_kraus is None:
            self._cached_kraus = tuple(
                collective_dephasing_kraus(self._n, self.gamma_c, self.tau, self.tol)
            )
        return self._cached_kraus

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return [f"CollDeph(g={self.gamma_c:.3g},t={self.tau:.3g})"] + ["#"] * (self._n - 1)

    def __repr__(self) -> str:
        return f"CollectiveDephasingChannel({self._n}, {self.gamma_c!r}, {self.tau!r})"


class RydbergDecayChannel(cirq.Gate):
    """Single-qubit spontaneous decay ``|r> -> |g>`` over a duration ``tau``.

    Thin, self-documenting wrapper around ``cirq.amplitude_damp`` that converts a
    physical lifetime ``T1`` and a duration into the damping probability
    ``p = 1 - exp(-tau / T1)``.
    """

    def __init__(self, t1: float, tau: float):
        super().__init__()
        self.t1 = float(t1)
        self.tau = float(tau)

    @property
    def probability(self) -> float:
        return 1.0 - np.exp(-self.tau / self.t1)

    def _num_qubits_(self) -> int:
        return 1

    def _kraus_(self):
        return cirq.kraus(cirq.amplitude_damp(self.probability))

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        return f"Decay(p={self.probability:.3g})"

    def __repr__(self) -> str:
        return f"RydbergDecayChannel({self.t1!r}, {self.tau!r})"
