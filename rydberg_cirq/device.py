"""
A hardware-aware ``cirq.Device`` for a Rydberg optical-tweezer array.

The defining physical constraint of a neutral-atom processor is the **blockade
radius** ``R_b``: two atoms can only be entangled if they sit close enough that the
van der Waals shift ``C6 / R^6`` exceeds the drive Rabi frequency.  Beyond ``R_b``
the interaction is negligible and an entangling gate is simply not available.

:class:`RydbergTweezerDevice` encodes exactly that as a ``validate_operation``
rule, so illegal circuits raise at construction time rather than silently
producing unphysical results.

Example:
    >>> import cirq
    >>> from rydberg_cirq.device import RydbergTweezerDevice
    >>> dev = RydbergTweezerDevice.square_array(3, blockade_radius=1.5)
    >>> qs = sorted(dev.metadata.qubit_set)
    >>> dev.validate_operation(cirq.CZ(qs[0], qs[1]))          # adjacent -> OK
    >>> dev.validate_operation(cirq.CZ(qs[0], qs[8]))          # far apart -> raises
    Traceback (most recent call last):
        ...
    ValueError: ...outside the blockade radius...
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import cirq

from .gates import (
    CollectiveLaserDriveStep,
    MagicLatticeStorageGate,
    PairwisePhaseGate,
    RydbergBlockadeStep,
)

__all__ = ["RydbergTweezerDevice", "RYDBERG_NATIVE_GATESET"]


#: Native operations of a neutral-atom Rydberg processor: arbitrary single-qubit
#: rotations delivered by the global/addressed laser, plus the CZ-type phase gate
#: produced by the blockade interaction.
RYDBERG_NATIVE_GATESET = cirq.Gateset(
    cirq.PhasedXZGate,
    cirq.XPowGate,
    cirq.YPowGate,
    cirq.ZPowGate,
    cirq.HPowGate,
    cirq.CZPowGate,
    cirq.MeasurementGate,
    cirq.IdentityGate,
    cirq.GlobalPhaseGate,
    unroll_circuit_op=True,
)


class RydbergTweezerDevice(cirq.Device):
    """Neutral-atom tweezer array with a finite Rydberg blockade radius.

    Args:
        qubit_positions: mapping from ``cirq.GridQubit`` (or any ``cirq.Qid``) to an
            ``(x, y)`` or ``(x, y, z)`` position in micrometres.
        blockade_radius: maximum separation (same units) at which a two-qubit
            entangling gate is physically available.
        gateset: allowed operations; defaults to :data:`RYDBERG_NATIVE_GATESET`.
        allow_composite_gates: if ``True`` (default) the ensemble-level composite
            gates from :mod:`rydberg_cirq.gates` are accepted and validated by
            decomposing them first.
    """

    def __init__(
        self,
        qubit_positions: dict,
        blockade_radius: float,
        gateset: cirq.Gateset | None = None,
        allow_composite_gates: bool = True,
    ):
        self._positions = {q: np.asarray(p, dtype=float) for q, p in qubit_positions.items()}
        self.blockade_radius = float(blockade_radius)
        self._gateset = gateset if gateset is not None else RYDBERG_NATIVE_GATESET
        self._allow_composite = allow_composite_gates

        qubits = frozenset(self._positions)
        graph = nx.Graph()
        graph.add_nodes_from(sorted(qubits))
        for a, b in itertools.combinations(sorted(qubits), 2):
            if self.distance(a, b) <= self.blockade_radius:
                graph.add_edge(a, b)
        self._graph = graph
        self._metadata = cirq.DeviceMetadata(qubits, graph)

    # ------------------------------------------------------------------ builders

    @classmethod
    def square_array(
        cls, side: int, spacing: float = 1.0, blockade_radius: float = 1.5, **kwargs
    ) -> "RydbergTweezerDevice":
        """A ``side x side`` square tweezer lattice with the given lattice ``spacing``."""
        positions = {
            cirq.GridQubit(r, c): (c * spacing, r * spacing)
            for r in range(side)
            for c in range(side)
        }
        return cls(positions, blockade_radius, **kwargs)

    @classmethod
    def chain(
        cls, n: int, spacing: float = 1.0, blockade_radius: float = 1.5, **kwargs
    ) -> "RydbergTweezerDevice":
        """A 1D chain of ``n`` atoms on ``cirq.LineQubit`` s."""
        positions = {cirq.LineQubit(i): (i * spacing, 0.0) for i in range(n)}
        return cls(positions, blockade_radius, **kwargs)

    @classmethod
    def from_cloud(
        cls, positions: np.ndarray, blockade_radius: float, **kwargs
    ) -> "RydbergTweezerDevice":
        """Build from an ``(N, 3)`` array of sampled atomic coordinates."""
        mapping = {cirq.LineQubit(i): positions[i] for i in range(len(positions))}
        return cls(mapping, blockade_radius, **kwargs)

    # ------------------------------------------------------------------ geometry

    @property
    def metadata(self) -> cirq.DeviceMetadata:
        return self._metadata

    @property
    def qubits(self) -> list:
        return sorted(self._positions)

    @property
    def connectivity_graph(self) -> nx.Graph:
        """Graph whose edges are the atom pairs within the blockade radius."""
        return self._graph

    def position(self, q: cirq.Qid) -> np.ndarray:
        return self._positions[q]

    def distance(self, a: cirq.Qid, b: cirq.Qid) -> float:
        return float(np.linalg.norm(self._positions[a] - self._positions[b]))

    def blockade_shift(self, a: cirq.Qid, b: cirq.Qid, c6: float) -> float:
        """van der Waals shift ``C6 / R^6`` between two atoms."""
        r = self.distance(a, b)
        return float("inf") if r == 0.0 else c6 / r**6

    # ---------------------------------------------------------------- validation

    def validate_operation(self, operation: cirq.Operation) -> None:
        for q in operation.qubits:
            if q not in self._positions:
                raise ValueError(f"Qubit {q} is not a tweezer site of this device.")

        gate = operation.gate

        # Ensemble-level composite gates are validated through their decomposition.
        if self._allow_composite and isinstance(
            gate,
            (
                CollectiveLaserDriveStep,
                RydbergBlockadeStep,
                PairwisePhaseGate,
                MagicLatticeStorageGate,
            ),
        ):
            for sub_op in cirq.decompose_once(operation):
                self.validate_operation(sub_op)
            return

        if gate is not None and operation not in self._gateset:
            raise ValueError(
                f"Operation {operation!r} uses gate {gate!r}, which is not in the "
                f"native Rydberg gateset."
            )

        if len(operation.qubits) == 2:
            a, b = operation.qubits
            r = self.distance(a, b)
            if r > self.blockade_radius:
                raise ValueError(
                    f"Cannot apply the entangling operation {operation!r}: atoms {a} and "
                    f"{b} are separated by R = {r:.3f}, which is outside the blockade "
                    f"radius R_b = {self.blockade_radius:.3f}. The van der Waals shift is "
                    f"too weak to entangle them."
                )
        elif len(operation.qubits) > 2:
            raise ValueError(
                f"Operation {operation!r} acts on {len(operation.qubits)} atoms; the "
                f"native Rydberg gateset supports at most pairwise entangling gates."
            )

    def __str__(self) -> str:
        return (
            f"RydbergTweezerDevice({len(self._positions)} atoms, "
            f"R_b={self.blockade_radius:g}, "
            f"{self._graph.number_of_edges()} blockaded pairs)"
        )

    def __repr__(self) -> str:
        return self.__str__()
