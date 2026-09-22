"""
Custom ``cirq.Gate`` subclasses modelling Rydberg-ensemble physics.

Every gate here implements ``_decompose_`` into *native* Cirq operations
(``XPowGate``, ``CZPowGate``, ``ZPowGate``) so that circuits can be compiled onto
the :class:`rydberg_cirq.device.RydbergTweezerDevice` gateset.

Units
-----
All rates (``omega``, ``v_vdw``, ``delta``) are angular frequencies and all times
are in the reciprocal unit, so phases are simply ``rate * time``.

References
----------
* Y. Mei et al., Phys. Rev. Lett. 128, 123601 (2022)  -- superatom Rabi, magic lattice
* Y. Li et al., Phys. Rev. A 106, L051701 (2022)      -- interaction-induced dephasing
"""

from __future__ import annotations

import numpy as np
import cirq

__all__ = [
    "CollectiveLaserDriveStep",
    "RydbergBlockadeStep",
    "PairwisePhaseGate",
    "MagicLatticeStorageGate",
]


class CollectiveLaserDriveStep(cirq.Gate):
    """Trotter step of uniform resonant laser driving of an ``N``-atom ensemble.

    Implements ``exp(-i H dt)`` for

        H_drive = (Omega_1 / 2) * sum_j X_j

    which, because every atom sees the same field, generates collective
    ``sqrt(N)``-enhanced Rabi flopping when combined with a strong blockade.

    Decomposes into one ``cirq.XPowGate`` per atom.
    """

    def __init__(self, num_qubits: int, omega_1: float, dt: float):
        super().__init__()
        self._n = int(num_qubits)
        self.omega_1 = float(omega_1)
        self.dt = float(dt)

    def _num_qubits_(self) -> int:
        return self._n

    def _decompose_(self, qubits):
        # X^e == exp(i pi e / 2) Rx(pi e); choose pi*e = Omega_1 * dt.
        exponent = (self.omega_1 * self.dt) / np.pi
        for q in qubits:
            yield cirq.XPowGate(exponent=exponent).on(q)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return ["Drive(O1*dt)"] * self._n

    def _value_equality_values_(self):
        return self._n, self.omega_1, self.dt

    def __repr__(self) -> str:
        return f"CollectiveLaserDriveStep({self._n}, {self.omega_1!r}, {self.dt!r})"


class RydbergBlockadeStep(cirq.Gate):
    """Trotter step of all-to-all van der Waals blockade interactions.

    Implements ``exp(-i H dt)`` for

        H_vdW = V_vdW * sum_{j < k} n_j n_k ,     n_j = |r><r|_j

    Decomposes into ``N(N-1)/2`` native ``cirq.CZPowGate`` operations.  Note that
    ``CZ**e = diag(1, 1, 1, exp(i pi e))``, hence ``e = -V dt / pi``.
    """

    def __init__(self, num_qubits: int, v_vdw: float, dt: float):
        super().__init__()
        self._n = int(num_qubits)
        self.v_vdw = float(v_vdw)
        self.dt = float(dt)

    def _num_qubits_(self) -> int:
        return self._n

    def _decompose_(self, qubits):
        exponent = -(self.v_vdw * self.dt) / np.pi
        for i in range(self._n):
            for j in range(i + 1, self._n):
                yield cirq.CZPowGate(exponent=exponent).on(qubits[i], qubits[j])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return ["Blockade(V)"] * self._n

    def _value_equality_values_(self):
        return self._n, self.v_vdw, self.dt

    def __repr__(self) -> str:
        return f"RydbergBlockadeStep({self._n}, {self.v_vdw!r}, {self.dt!r})"


class PairwisePhaseGate(cirq.Gate):
    """Exact storage-time evolution under a *position-dependent* interaction matrix.

    Implements the propagator of Phys. Rev. A 106, L051701 Eq. (4),

        U(T_s) = prod_{mu < nu} exp(-i Phi_{mu nu} n_mu n_nu),

    where ``phi_matrix[mu, nu] = Phi_{mu nu} = kappa_{mu nu} T_s`` is built from the
    sampled atomic positions (see :mod:`rydberg_cirq.cloud`).  Because each factor is
    diagonal the decomposition is exact -- no Trotter error.

    This gate is the microscopic origin of interaction-induced dephasing: the
    ``m = 0`` and ``m = 1`` sectors are untouched (a single Rydberg atom has no
    partner), while ``m >= 2`` sectors acquire random phases.
    """

    def __init__(self, phi_matrix: np.ndarray):
        super().__init__()
        phi = np.asarray(phi_matrix, dtype=float)
        if phi.ndim != 2 or phi.shape[0] != phi.shape[1]:
            raise ValueError("phi_matrix must be square")
        self.phi_matrix = phi
        self._n = phi.shape[0]

    def _num_qubits_(self) -> int:
        return self._n

    def _decompose_(self, qubits):
        for i in range(self._n):
            for j in range(i + 1, self._n):
                phi = self.phi_matrix[i, j]
                if phi != 0.0:
                    yield cirq.CZPowGate(exponent=-phi / np.pi).on(qubits[i], qubits[j])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return ["U(Ts)"] * self._n

    def __repr__(self) -> str:
        return f"PairwisePhaseGate(<{self._n}x{self._n} matrix>)"


class MagicLatticeStorageGate(cirq.Gate):
    """Idle storage interval of a Rydberg qubit in a magic-wavelength lattice (SILT).

    During an idle interval of duration ``t_storage`` (microseconds) the atom
    accumulates a phase from

    1. a static inhomogeneous differential AC-Stark shift ``delta_static``;
    2. a quasi-static thermal motional modulation ``delta_osc`` at trap frequency
       ``omega_trap``.

    Keeping ``t_start`` explicit is what makes dynamical decoupling meaningful: a
    pi pulse inserted midway only refocuses the phase if the two sub-intervals are
    evaluated at the correct absolute times.
    """

    def __init__(
        self,
        t_storage: float,
        delta_static: float,
        delta_osc: float = 0.25,
        omega_trap: float = 0.20,
        t_start: float = 0.0,
    ):
        super().__init__()
        self.t_storage = float(t_storage)
        self.delta_static = float(delta_static)
        self.delta_osc = float(delta_osc)
        self.omega_trap = float(omega_trap)
        self.t_start = float(t_start)

    def _num_qubits_(self) -> int:
        return 1

    def accumulated_phase(self) -> float:
        """Phase integral ``int_{t_start}^{t_start + t_storage} delta(t) dt``."""
        t1 = self.t_start
        t2 = self.t_start + self.t_storage
        phi_static = self.delta_static * (t2 - t1)
        if self.omega_trap > 0:
            phi_motional = (self.delta_osc / self.omega_trap) * (
                np.sin(self.omega_trap * t2) - np.sin(self.omega_trap * t1)
            )
        else:
            phi_motional = 0.0
        return phi_static + phi_motional

    def _unitary_(self) -> np.ndarray:
        return np.array([[1.0, 0.0], [0.0, np.exp(-1j * self.accumulated_phase())]], dtype=complex)

    def _decompose_(self, qubits):
        # Native Z rotation; global phase is irrelevant for the Ramsey observable.
        yield cirq.ZPowGate(exponent=-self.accumulated_phase() / np.pi).on(qubits[0])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> str:
        return f"Idle(t={self.t_storage:.1f}us)"

    def __repr__(self) -> str:
        return (
            f"MagicLatticeStorageGate({self.t_storage!r}, {self.delta_static!r}, "
            f"{self.delta_osc!r}, {self.omega_trap!r}, {self.t_start!r})"
        )
