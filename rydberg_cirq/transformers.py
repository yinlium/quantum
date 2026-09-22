"""
Custom ``@cirq.transformer`` compiler passes for Rydberg qubit circuits.

Contains
--------
``compile_dynamical_decoupling``
    Scans a circuit for idle storage intervals
    (:class:`rydberg_cirq.gates.MagicLatticeStorageGate`) and rewrites each into a
    Hahn spin-echo or CPMG-N pulse train, refocusing inhomogeneous magic-lattice
    AC-Stark shifts.  This is what extends ``T_2`` by >10x in
    Phys. Rev. Lett. 128, 123601.

``compile_to_rydberg_hardware``
    Full lowering pipeline: decompose ensemble-level composite gates, merge
    single-qubit rotations into ``PhasedXZGate``, align and compact moments, and
    validate the result against a :class:`rydberg_cirq.device.RydbergTweezerDevice`.
"""

from __future__ import annotations

import cirq

from .gates import MagicLatticeStorageGate

__all__ = ["compile_dynamical_decoupling", "compile_to_rydberg_hardware", "circuit_stats"]


@cirq.transformer
def compile_dynamical_decoupling(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    num_pulses: int = 0,
) -> cirq.Circuit:
    """Insert CPMG-``num_pulses`` dynamical decoupling into every idle storage interval.

    The CPMG timing convention divides a storage interval ``T`` into ``2 num_pulses``
    sub-intervals of length ``tau = T / (2 num_pulses)``, applying a ``Y`` pi-pulse at
    the centre of each pair:  ``[tau - Y - tau] x num_pulses``.  An extra ``Y`` is
    appended when ``num_pulses`` is odd so the net rotation is the identity and the
    Ramsey readout basis is preserved.

    Crucially each sub-interval is re-instantiated with its correct absolute
    ``t_start``, so the time-dependent motional phase is integrated properly and
    genuinely refocuses.

    Args:
        circuit: the circuit to transform.
        context: standard Cirq transformer context (unused; present for the protocol).
        num_pulses: ``0`` leaves the circuit untouched (free Ramsey decay), ``1``
            gives a Hahn spin echo, ``>= 2`` gives CPMG-N.
    """
    if num_pulses <= 0:
        return cirq.Circuit(circuit)

    new_moments: list[cirq.Moment] = []
    for moment in circuit:
        replaced_ops = []
        has_storage_gate = False

        for op in moment.operations:
            if isinstance(op.gate, MagicLatticeStorageGate):
                has_storage_gate = True
                g = op.gate
                q = op.qubits[0]
                tau = g.t_storage / (2.0 * num_pulses)
                cur_t = g.t_start

                seq_ops = []
                for _ in range(num_pulses):
                    seq_ops.append(
                        MagicLatticeStorageGate(
                            tau, g.delta_static, g.delta_osc, g.omega_trap, cur_t
                        ).on(q)
                    )
                    cur_t += tau
                    seq_ops.append(cirq.Y(q))
                    seq_ops.append(
                        MagicLatticeStorageGate(
                            tau, g.delta_static, g.delta_osc, g.omega_trap, cur_t
                        ).on(q)
                    )
                    cur_t += tau

                # Restore the identity net rotation for odd pulse counts.
                if num_pulses % 2 == 1:
                    seq_ops.append(cirq.Y(q))

                replaced_ops.append(seq_ops)
            else:
                replaced_ops.append([op])

        if not has_storage_gate:
            new_moments.append(moment)
        else:
            max_len = max(len(seq) for seq in replaced_ops)
            for step_i in range(max_len):
                step_ops = [seq[step_i] for seq in replaced_ops if step_i < len(seq)]
                new_moments.append(cirq.Moment(step_ops))

    return cirq.Circuit(new_moments)


def circuit_stats(circuit: cirq.AbstractCircuit) -> dict:
    """Gate-count / depth summary used for before-vs-after compilation reports."""
    ops = list(circuit.all_operations())
    return {
        "moments": len(circuit),
        "total_ops": len(ops),
        "1q_ops": sum(1 for op in ops if len(op.qubits) == 1),
        "2q_ops": sum(1 for op in ops if len(op.qubits) == 2),
        "nq_ops": sum(1 for op in ops if len(op.qubits) > 2),
        "qubits": len(circuit.all_qubits()),
    }


def compile_to_rydberg_hardware(
    circuit: cirq.AbstractCircuit,
    device=None,
    gateset: cirq.Gateset | None = None,
    merge_single_qubit: bool = True,
) -> cirq.Circuit:
    """Lower an abstract ensemble circuit onto the native Rydberg gateset.

    Pipeline:
        1. ``cirq.expand_composite`` -- unroll ``CollectiveLaserDriveStep``,
           ``RydbergBlockadeStep``, ``PairwisePhaseGate`` into native operations.
        2. ``cirq.merge_single_qubit_moments_to_phxz`` -- fuse adjacent single-qubit
           rotations into one ``PhasedXZGate`` each.
        3. ``cirq.drop_negligible_operations`` + ``cirq.drop_empty_moments`` -- remove
           identity-like leftovers.
        4. ``cirq.align_left`` -- compact the schedule.
        5. Device validation, if a device is supplied.

    Args:
        circuit: input circuit, possibly containing composite ensemble gates.
        device: optional :class:`~rydberg_cirq.device.RydbergTweezerDevice` used to
            validate the compiled result.
        gateset: target gateset; defaults to the device's native gateset.
        merge_single_qubit: whether to run the PhasedXZ merge pass.

    Returns:
        The compiled circuit.
    """
    out = cirq.expand_composite(
        cirq.Circuit(circuit),
        no_decomp=lambda op: len(op.qubits) <= 2 and not isinstance(op.gate, MagicLatticeStorageGate),
    )

    if merge_single_qubit:
        out = cirq.merge_single_qubit_moments_to_phxz(out)

    out = cirq.drop_negligible_operations(out)
    out = cirq.drop_empty_moments(out)
    out = cirq.align_left(out)

    if device is not None:
        for op in out.all_operations():
            device.validate_operation(op)

    return out
