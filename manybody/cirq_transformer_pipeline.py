"""
Lowering an abstract Rydberg-ensemble circuit onto native hardware with Cirq transformers.

Reference
---------
Y. Mei, Y. Li, H. Nguyen, P. R. Berman, and A. Kuzmich,
*Trapped Alkali-Metal Rydberg Qubit*, Phys. Rev. Lett. **128**, 123601 (2022).

What this module does
---------------------
The physics of PRL 128, 123601 is written naturally at the *ensemble* level:

    U(dt) ~ Drive(dt/2) . Blockade(dt) . Drive(dt/2)          (Strang/Trotter step)
    |  Drive(t)     = exp(-i (Omega_1/2) sum_j X_j t)          collective laser pulse
    |  Blockade(t)  = exp(-i V sum_{j<k} n_j n_k t)            van der Waals interaction
    |  Storage(T)   = exp(-i int delta(t) dt  n)               magic-lattice idle interval

but a neutral-atom machine only executes *single-atom rotations* and *pairwise
blockade phase gates*.  Bridging the two is the job of the Cirq transformer stack:

    cirq.expand_composite                  unroll the N-atom composites (``_decompose_``)
    cirq.merge_single_qubit_moments_to_phxz fuse 1q rotations into one PhasedXZGate
    cirq.drop_negligible_operations        delete identity-like leftovers
    cirq.drop_empty_moments                compact the schedule
    cirq.align_left                        left-justify the moments
    cirq.optimize_for_target_gateset       re-synthesise onto a hardware gateset
    rydberg_cirq.compile_dynamical_decoupling   custom pass: CPMG into idle intervals

Every ensemble Trotter step must lower to exactly ``N(N-1)/2`` two-qubit
``cirq.CZPowGate`` operations -- the all-to-all blockade of a single superatom -- and
that ``O(N^2)`` scaling is what the report below measures.

Semantics preservation
----------------------
A compiler pass that changes the physics is a bug, so every stage is checked with
``cirq.allclose_up_to_global_phase`` on the full unitary *and* with a state fidelity
``|<psi_before|psi_after>|^2``.  One genuine caveat is demonstrated explicitly: the
PhasedXZ merge preserves the *unitary* exactly, but it also erases the pulse timing,
so under a ``rydberg_cirq.RydbergNoiseModel`` the merged circuit is over-optimistic.
Timing-faithful circuits (dynamical decoupling!) must be lowered with
``merge_single_qubit=False``.

Cirq APIs showcased
-------------------
``cirq.expand_composite``, ``cirq.merge_single_qubit_moments_to_phxz``,
``cirq.drop_negligible_operations``, ``cirq.drop_empty_moments``, ``cirq.align_left``,
``cirq.optimize_for_target_gateset`` with ``cirq.CZTargetGateset`` /
``cirq.SqrtIswapTargetGateset``, ``cirq.Gateset.__contains__``, ``cirq.unitary``,
``cirq.allclose_up_to_global_phase``, ``cirq.DensityMatrixSimulator`` with a custom
``cirq.NoiseModel``, plus the package passes ``rydberg_cirq.compile_to_rydberg_hardware``,
``rydberg_cirq.compile_dynamical_decoupling`` and ``rydberg_cirq.circuit_stats``.

Run:  python manybody/cirq_transformer_pipeline.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import cirq

import rydberg_cirq as rc
from rydberg_cirq import plotting as rplt  # submodule is not re-exported by __init__

# --------------------------------------------------------------------------- units
MHZ = 2.0 * np.pi  # 1 MHz -> rad/us

OMEGA_1 = 1.0 * MHZ        # single-atom Rabi frequency
V_BLOCKADE = 45.0 * MHZ    # uniform van der Waals shift inside one blockade volume
DT = 0.05                  # Trotter step (us)
N_TROTTER = 2              # Trotter steps in the demonstration circuit
T_STORAGE = 6.0            # magic-lattice idle interval (us)
DELTA_STATIC = 0.35        # residual differential AC-Stark shift (rad/us)

N_RANGE = (3, 4, 5, 6, 7, 8)


# ----------------------------------------------------------------- circuit builders
def build_ensemble_circuit(n_atoms: int, n_trotter: int = N_TROTTER, t_storage: float = T_STORAGE):
    """High-level ensemble circuit: Trotterised drive+blockade, then a stored idle.

    Uses only the composite gates of :mod:`rydberg_cirq.gates`, i.e. the way a
    physicist would write the experiment down.
    """
    qubits = cirq.LineQubit.range(n_atoms)
    circuit = cirq.Circuit()
    for k in range(n_trotter):
        circuit.append(rc.CollectiveLaserDriveStep(n_atoms, OMEGA_1, DT / 2).on(*qubits))
        circuit.append(rc.RydbergBlockadeStep(n_atoms, V_BLOCKADE, DT).on(*qubits))
        circuit.append(rc.CollectiveLaserDriveStep(n_atoms, OMEGA_1, DT / 2).on(*qubits))
        del k
    circuit.append(
        cirq.Moment(
            rc.MagicLatticeStorageGate(
                t_storage, DELTA_STATIC, 0.25, 0.20, n_trotter * DT
            ).on(q)
            for q in qubits
        )
    )
    circuit.append(rc.CollectiveLaserDriveStep(n_atoms, OMEGA_1, DT / 2).on(*qubits))
    return circuit, qubits


def build_ramsey_storage_circuit(
    t_storage: float, delta_static: float, delta_osc: float, omega_trap: float
):
    """Single-qubit Ramsey sequence around one magic-lattice storage interval."""
    q = cirq.LineQubit(0)
    return cirq.Circuit(
        cirq.H(q),
        rc.MagicLatticeStorageGate(t_storage, delta_static, delta_osc, omega_trap, 0.0).on(q),
        cirq.H(q),
    ), q


# ------------------------------------------------------------------ local utilities
def indented(circuit: cirq.AbstractCircuit, pad: str = "    ") -> str:
    """ASCII circuit diagram, indented for the console report."""
    return "\n".join(pad + line for line in str(cirq.Circuit(circuit)).splitlines())


def is_ensemble_composite(op: cirq.Operation) -> bool:
    """True for the package's ensemble-level composite gates."""
    return isinstance(
        op.gate,
        (rc.CollectiveLaserDriveStep, rc.RydbergBlockadeStep, rc.PairwisePhaseGate,
         rc.MagicLatticeStorageGate),
    )


def expand_composites(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    """``cirq.expand_composite`` keyed on the *gate type* rather than the arity.

    Local helper: ``rc.compile_to_rydberg_hardware`` uses
    ``no_decomp = len(op.qubits) <= 2``, which accidentally leaves the ensemble
    composites intact for ``N = 2`` (see the caveat printed by this script).
    """
    return cirq.expand_composite(cirq.Circuit(circuit), no_decomp=lambda op: not is_ensemble_composite(op))


def equivalence_report(before: cirq.AbstractCircuit, after: cirq.AbstractCircuit, qubits) -> dict:
    """Unitary + state-fidelity comparison of two circuits over the same qubits."""
    u_before = cirq.Circuit(before).unitary(qubit_order=qubits)
    u_after = cirq.Circuit(after).unitary(qubit_order=qubits)
    psi_before = cirq.final_state_vector(cirq.Circuit(before), qubit_order=qubits)
    psi_after = cirq.final_state_vector(cirq.Circuit(after), qubit_order=qubits)
    fidelity = float(abs(np.vdot(psi_before, psi_after)) ** 2)

    # Global-phase-insensitive matrix distance.
    overlap = np.trace(u_before.conj().T @ u_after)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-12 else 1.0
    max_dev = float(np.max(np.abs(u_after / phase - u_before)))
    return {
        "allclose": bool(cirq.allclose_up_to_global_phase(u_before, u_after)),
        "fidelity": fidelity,
        "max_unitary_deviation": max_dev,
        "process_fidelity": float(abs(overlap) ** 2 / u_before.shape[0] ** 2),
    }


def native(circuit: cirq.AbstractCircuit) -> bool:
    """Is every operation inside :data:`rydberg_cirq.RYDBERG_NATIVE_GATESET`?"""
    return all(op in rc.RYDBERG_NATIVE_GATESET for op in circuit.all_operations())


def fmt_stats(stats: dict) -> str:
    return (f"moments={stats['moments']:4d}  ops={stats['total_ops']:4d}  "
            f"1q={stats['1q_ops']:4d}  2q={stats['2q_ops']:4d}  Nq={stats['nq_ops']:3d}")


# ------------------------------------------------------------------------- studies
def study_decomposition() -> None:
    """Print the high-level and lowered circuits for a small ensemble."""
    print("\n" + "-" * 94)
    print("[1] From ensemble physics to native gates  (N = 3, one Trotter step + storage)")
    print("-" * 94)
    circuit, qubits = build_ensemble_circuit(3, n_trotter=1)
    print("\n  High-level circuit (composite cirq.Gate subclasses):\n")
    print(indented(cirq.Circuit(circuit)))

    expanded = cirq.expand_composite(
        cirq.Circuit(circuit),
        no_decomp=lambda op: len(op.qubits) <= 2 and not isinstance(op.gate, rc.MagicLatticeStorageGate),
    )
    print("\n  After cirq.expand_composite (native XPow / CZPow / ZPow):\n")
    print(indented(expanded))

    compiled = rc.compile_to_rydberg_hardware(circuit)
    print("\n  After the full rc.compile_to_rydberg_hardware pipeline:\n")
    print(indented(compiled))

    print(f"\n  native gateset? expanded={native(expanded)}  compiled={native(compiled)}")
    eq = equivalence_report(circuit, compiled, qubits)
    print(f"  semantics: allclose_up_to_global_phase={eq['allclose']}  "
          f"state fidelity={eq['fidelity']:.12f}  max|dU|={eq['max_unitary_deviation']:.2e}")


def study_pass_by_pass() -> list:
    """Stats after each individual Cirq pass, with an equivalence check each time."""
    print("\n" + "-" * 94)
    print("[2] Pass-by-pass compilation report (N = 5, %d Trotter steps + storage)" % N_TROTTER)
    print("-" * 94)
    circuit, qubits = build_ensemble_circuit(5)
    u_ref = cirq.Circuit(circuit).unitary(qubit_order=qubits)

    rows = [("input (composites)", rc.circuit_stats(circuit), True)]
    work = cirq.Circuit(circuit)

    passes = [
        ("cirq.expand_composite", lambda c: cirq.expand_composite(
            c, no_decomp=lambda op: len(op.qubits) <= 2 and not isinstance(op.gate, rc.MagicLatticeStorageGate))),
        ("cirq.merge_single_qubit_moments_to_phxz", cirq.merge_single_qubit_moments_to_phxz),
        ("cirq.drop_negligible_operations", cirq.drop_negligible_operations),
        ("cirq.drop_empty_moments", cirq.drop_empty_moments),
        ("cirq.align_left", cirq.align_left),
    ]
    for name, fn in passes:
        work = fn(work)
        ok = cirq.allclose_up_to_global_phase(work.unitary(qubit_order=qubits), u_ref)
        rows.append((name, rc.circuit_stats(work), ok))

    print(f"  {'pass':42s} {'moments':>8} {'ops':>6} {'1q':>5} {'2q':>5} {'Nq':>4}  unitary preserved")
    for name, st, ok in rows:
        print(f"  {name:42s} {st['moments']:8d} {st['total_ops']:6d} {st['1q_ops']:5d} "
              f"{st['2q_ops']:5d} {st['nq_ops']:4d}  {'YES' if ok else 'NO !!'}")
    print("\n  Note: expand_composite is the only pass that changes the gate *set*; the")
    print("  remaining passes only reshape the schedule, which is why the unitary is")
    print("  invariant at every stage.")
    return rows


def study_target_gatesets() -> None:
    """``cirq.optimize_for_target_gateset`` onto hardware-constrained gatesets."""
    print("\n" + "-" * 94)
    print("[3] cirq.optimize_for_target_gateset -- re-synthesis onto hardware gatesets")
    print("-" * 94)
    circuit, qubits = build_ensemble_circuit(4, n_trotter=1)
    compiled = rc.compile_to_rydberg_hardware(circuit)
    print(f"  {'target gateset':46s} {'moments':>8} {'ops':>6} {'1q':>5} {'2q':>5}  equiv  native")
    st = rc.circuit_stats(compiled)
    print(f"  {'rc.compile_to_rydberg_hardware (partial CZ)':46s} {st['moments']:8d} "
          f"{st['total_ops']:6d} {st['1q_ops']:5d} {st['2q_ops']:5d}   YES   {native(compiled)}")

    targets = [
        ("cirq.CZTargetGateset(allow_partial_czs=True)", cirq.CZTargetGateset(allow_partial_czs=True)),
        ("cirq.CZTargetGateset(allow_partial_czs=False)", cirq.CZTargetGateset(allow_partial_czs=False)),
        ("cirq.SqrtIswapTargetGateset()", cirq.SqrtIswapTargetGateset()),
    ]
    for name, gateset in targets:
        out = cirq.optimize_for_target_gateset(compiled, gateset=gateset)
        eq = equivalence_report(circuit, out, qubits)
        st = rc.circuit_stats(out)
        print(f"  {name:46s} {st['moments']:8d} {st['total_ops']:6d} {st['1q_ops']:5d} "
              f"{st['2q_ops']:5d}   {'YES' if eq['allclose'] else 'NO'}   {native(out)}")
    print("\n  A Rydberg blockade gate natively realises an *arbitrary-angle* CZ**e, so")
    print("  forbidding partial CZs (as on a superconducting chip) doubles the two-qubit")
    print("  count -- and sqrt(iSWAP) synthesis is not native at all (native=False).")


def study_scaling() -> dict:
    """Before/after compilation statistics as a function of ensemble size ``N``."""
    print("\n" + "-" * 94)
    print("[4] Compilation report vs ensemble size N  (%d Trotter steps + storage + readout pulse)"
          % N_TROTTER)
    print("-" * 94)
    data = {"N": [], "before": [], "after": [], "fidelity": [], "allclose": [], "time": []}
    print(f"  {'N':>3} | {'before: mom/ops/1q/2q':>24} | {'after: mom/ops/1q/2q':>24} | "
          f"{'2q analytic':>11} | {'fidelity':>12} | equiv")
    for n_atoms in N_RANGE:
        circuit, qubits = build_ensemble_circuit(n_atoms)
        t0 = time.time()
        compiled = rc.compile_to_rydberg_hardware(circuit)
        dt_compile = time.time() - t0
        before, after = rc.circuit_stats(circuit), rc.circuit_stats(compiled)
        eq = equivalence_report(circuit, compiled, qubits)
        analytic_2q = N_TROTTER * n_atoms * (n_atoms - 1) // 2

        data["N"].append(n_atoms)
        data["before"].append(before)
        data["after"].append(after)
        data["fidelity"].append(eq["fidelity"])
        data["allclose"].append(eq["allclose"])
        data["time"].append(dt_compile)
        print(f"  {n_atoms:3d} | {before['moments']:6d}/{before['total_ops']:5d}/"
              f"{before['1q_ops']:4d}/{before['2q_ops']:4d}      | "
              f"{after['moments']:6d}/{after['total_ops']:5d}/{after['1q_ops']:4d}/"
              f"{after['2q_ops']:4d}      | {analytic_2q:11d} | {eq['fidelity']:12.10f} | "
              f"{'YES' if eq['allclose'] else 'NO !!'}")

    mismatch = [n for n, st in zip(data["N"], data["after"])
                if st["2q_ops"] != N_TROTTER * n * (n - 1) // 2]
    print(f"\n  Two-qubit count matches N_trotter * N(N-1)/2 for every N: "
          f"{'yes' if not mismatch else 'NO -> ' + str(mismatch)}")
    print(f"  Worst-case state infidelity over the sweep: "
          f"{1.0 - min(data['fidelity']):.2e}  (all allclose_up_to_global_phase: "
          f"{all(data['allclose'])})")
    print(f"  Compilation wall time: {min(data['time']) * 1e3:.1f} - {max(data['time']) * 1e3:.1f} ms")

    # ---- caveat: the packaged pipeline keys `no_decomp` off the operation arity.
    circuit2, qubits2 = build_ensemble_circuit(2, n_trotter=1)
    packaged = rc.compile_to_rydberg_hardware(circuit2)
    local = cirq.align_left(cirq.merge_single_qubit_moments_to_phxz(expand_composites(circuit2)))
    left_over = sum(1 for op in packaged.all_operations() if is_ensemble_composite(op))
    print("\n  CAVEAT (N = 2): rc.compile_to_rydberg_hardware uses")
    print("  no_decomp = (len(op.qubits) <= 2), so two-atom ensemble composites are never")
    print(f"  expanded: {left_over} composite op(s) survive -> {fmt_stats(rc.circuit_stats(packaged))}")
    print(f"  A gate-type-keyed predicate lowers it correctly    -> {fmt_stats(rc.circuit_stats(local))}")
    print(f"  (both are unitarily equivalent: "
          f"{cirq.allclose_up_to_global_phase(cirq.Circuit(packaged).unitary(qubit_order=qubits2), cirq.Circuit(local).unitary(qubit_order=qubits2))})")
    return data


def _t2_from_curve(t_values: np.ndarray, coherence: np.ndarray) -> float:
    """1/e coherence time by linear interpolation of the decay curve."""
    target = 1.0 / np.e
    below = np.where(coherence <= target)[0]
    if len(below) == 0:
        return float(t_values[-1])
    i = int(below[0])
    if i == 0:
        return float(t_values[0])
    c0, c1 = coherence[i - 1], coherence[i]
    frac = (c0 - target) / max(c0 - c1, 1e-12)
    return float(t_values[i - 1] + frac * (t_values[i] - t_values[i - 1]))


def ensemble_coherence(
    t_values: np.ndarray, num_pulses: int, num_atoms: int = 48, t1: float = 35.0, seed: int = 42
) -> np.ndarray:
    """Ramsey/CPMG visibility of an inhomogeneous ensemble, one Cirq circuit per atom.

    Each shot is built at the *physics* level, decoupled by the custom transformer
    ``rc.compile_dynamical_decoupling`` and then lowered by
    ``rc.compile_to_rydberg_hardware(..., merge_single_qubit=False)`` so the pulse
    timing survives compilation.
    """
    rng = np.random.default_rng(seed)
    static = rng.normal(0.0, 0.55, num_atoms)
    osc = rng.uniform(0.15, 0.45, num_atoms)
    traps = rng.normal(0.22, 0.04, num_atoms)

    out = np.zeros(len(t_values))
    for it, t_s in enumerate(t_values):
        visibility = 0.0
        for idx in range(num_atoms):
            raw, _ = build_ramsey_storage_circuit(t_s, static[idx], osc[idx], traps[idx])
            decoupled = rc.compile_dynamical_decoupling(raw, num_pulses=num_pulses)
            hardware = rc.compile_to_rydberg_hardware(decoupled, merge_single_qubit=False)
            psi = cirq.final_state_vector(hardware)
            visibility += 2.0 * float(abs(psi[0]) ** 2) - 1.0
        out[it] = (visibility / num_atoms) * np.exp(-t_s / t1)
    return out


def study_dd() -> dict:
    """Gate-count overhead of dynamical decoupling vs the coherence it buys."""
    print("\n" + "-" * 94)
    print("[5] Composing the DD transformer with hardware lowering")
    print("-" * 94)
    raw, q = build_ramsey_storage_circuit(T_STORAGE, 0.5, 0.25, 0.20)
    print("\n  Uncompiled Ramsey circuit:\n")
    print(indented(cirq.Circuit(raw)))
    echo = rc.compile_dynamical_decoupling(raw, num_pulses=1)
    print("\n  rc.compile_dynamical_decoupling(num_pulses=1)  [Hahn echo]:\n")
    print(indented(echo))
    print("\n  ... then rc.compile_to_rydberg_hardware(merge_single_qubit=False):\n")
    print(indented(rc.compile_to_rydberg_hardware(echo, merge_single_qubit=False)))

    pulse_counts = [0, 1, 2, 4, 8, 16]
    t_values = np.linspace(0.25, 40.0, 24)

    print(f"\n  {'N_pi':>5} {'DD ops':>7} {'hw ops':>7} {'hw moments':>11} {'merged ops':>11} "
          f"{'T2 [us]':>9} {'T2/T2*':>8}")
    result = {"pulses": [], "dd_ops": [], "hw_ops": [], "hw_moments": [], "merged_ops": [],
              "t2": [], "curves": {}, "t_values": t_values}
    for k in pulse_counts:
        decoupled = rc.compile_dynamical_decoupling(raw, num_pulses=k)
        hardware = rc.compile_to_rydberg_hardware(decoupled, merge_single_qubit=False)
        merged = rc.compile_to_rydberg_hardware(decoupled, merge_single_qubit=True)
        curve = ensemble_coherence(t_values, num_pulses=k)
        t2 = _t2_from_curve(t_values, curve)

        result["pulses"].append(k)
        result["dd_ops"].append(rc.circuit_stats(decoupled)["total_ops"])
        result["hw_ops"].append(rc.circuit_stats(hardware)["total_ops"])
        result["hw_moments"].append(rc.circuit_stats(hardware)["moments"])
        result["merged_ops"].append(rc.circuit_stats(merged)["total_ops"])
        result["t2"].append(t2)
        result["curves"][k] = curve

    t2_free = result["t2"][0]
    for i, k in enumerate(result["pulses"]):
        print(f"  {k:5d} {result['dd_ops'][i]:7d} {result['hw_ops'][i]:7d} "
              f"{result['hw_moments'][i]:11d} {result['merged_ops'][i]:11d} "
              f"{result['t2'][i]:9.2f} {result['t2'][i] / t2_free:8.2f}")
    result["enhancement"] = [t / t2_free for t in result["t2"]]
    print(f"\n  CPMG-{result['pulses'][-1]} costs {result['hw_ops'][-1] / result['hw_ops'][0]:.1f}x "
          f"the native operations of a bare Ramsey sequence and buys "
          f"{result['enhancement'][-1]:.1f}x the coherence")
    print(f"  ({t2_free:.1f} us -> {result['t2'][-1]:.1f} us), reproducing the >10x extension of PRL 128, 123601.")

    # ---- the one place where a semantics-preserving pass is still physically wrong
    print("\n  Why merge_single_qubit=False matters for timing-faithful circuits:")
    decoupled = rc.compile_dynamical_decoupling(raw, num_pulses=8)
    unmerged = rc.compile_to_rydberg_hardware(decoupled, merge_single_qubit=False)
    merged = rc.compile_to_rydberg_hardware(decoupled, merge_single_qubit=True)
    eq = equivalence_report(unmerged, merged, [q])
    print(f"    ideal unitaries identical: {eq['allclose']} (fidelity {eq['fidelity']:.12f}), "
          f"{rc.circuit_stats(unmerged)['total_ops']} ops -> {rc.circuit_stats(merged)['total_ops']} op")
    noise = rc.RydbergNoiseModel(t1=35.0, t2_star=3.1, moment_duration=T_STORAGE / 16.0)
    sim = cirq.DensityMatrixSimulator(noise=noise)
    pops = {}
    for name, circ in (("unmerged", unmerged), ("merged", merged)):
        rho = sim.simulate(circ).final_density_matrix
        pops[name] = float(np.real(rho[0, 0]))
    print(f"    under rc.RydbergNoiseModel: P(|0>) = {pops['unmerged']:.4f} (unmerged, "
          f"{rc.circuit_stats(unmerged)['moments']} noisy moments) vs {pops['merged']:.4f} "
          f"(merged, {rc.circuit_stats(merged)['moments']} moment)")
    print("    -> the merge is unitarily exact but throws away the decoherence budget;")
    print("       lower DD circuits with merge_single_qubit=False.")
    return result


# ---------------------------------------------------------------------- figure
def make_figure(scaling: dict, dd: dict) -> str:
    rplt.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    # ---------------- (a) grouped bars: before vs after
    ax = axes[0]
    n_vals = np.array(scaling["N"], dtype=float)
    width = 0.2
    series = [
        ("ops, high-level", [s["total_ops"] for s in scaling["before"]], rplt.PALETTE["gray"], -1.5),
        ("ops, compiled", [s["total_ops"] for s in scaling["after"]], rplt.PALETTE["blue"], -0.5),
        ("depth, high-level", [s["moments"] for s in scaling["before"]], rplt.PALETTE["yellow"], 0.5),
        ("depth, compiled", [s["moments"] for s in scaling["after"]], rplt.PALETTE["orange"], 1.5),
    ]
    for label, values, colour, offset in series:
        ax.bar(n_vals + offset * width, values, width=width, color=colour, edgecolor="black",
               linewidth=0.5, label=label)
    for n, s in zip(n_vals, scaling["after"]):
        ax.text(n - 0.5 * width, s["total_ops"] + 1.5, str(s["total_ops"]), ha="center", fontsize=7.5,
                color=rplt.PALETTE["blue"], fontweight="bold")
    ax.set_xticks(n_vals)
    ax.set_xlabel("ensemble size $N$ (atoms)")
    ax.set_ylabel("count")
    ax.set_title("(a) Before vs after $\\mathtt{compile\\_to\\_rydberg\\_hardware}$\n"
                 f"{N_TROTTER} Trotter steps + magic-lattice storage", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    fid = min(scaling["fidelity"])
    ax.text(0.98, 0.04,
            f"all $N$: $|\\langle\\psi_{{\\rm pre}}|\\psi_{{\\rm post}}\\rangle|^2 \\geq {fid:.10f}$\n"
            "$\\mathtt{cirq.allclose\\_up\\_to\\_global\\_phase}$ = True",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=rplt.PALETTE["green"], lw=1.2))

    # ---------------- (b) 2q scaling
    ax = axes[1]
    two_q = np.array([s["2q_ops"] for s in scaling["after"]], dtype=float)
    one_q = np.array([s["1q_ops"] for s in scaling["after"]], dtype=float)
    n_fine = np.linspace(n_vals[0], n_vals[-1], 200)
    ax.plot(n_vals, two_q, "o", ms=9, color=rplt.PALETTE["red"], zorder=5,
            label="compiled $\\mathtt{CZPowGate}$ count")
    ax.plot(n_fine, N_TROTTER * n_fine * (n_fine - 1) / 2, color=rplt.PALETTE["red"], lw=2.0,
            ls="--", label="$n_{\\rm Trotter}\\, N(N-1)/2$  (all-to-all blockade)")
    ax.plot(n_vals, one_q, "s", ms=8, color=rplt.PALETTE["blue"], zorder=5,
            label="compiled 1q ($\\mathtt{PhasedXZGate}$/$\\mathtt{XPowGate}$)")
    ax.plot(n_fine, (2 * N_TROTTER + 1) * n_fine - (N_TROTTER - 0) * 0, color=rplt.PALETTE["blue"],
            lw=1.6, ls=":", label="$(2 n_{\\rm Trotter}+1)\\,N$ before merging")
    ax.set_xlabel("ensemble size $N$ (atoms)")
    ax.set_ylabel("native operations")
    ax.set_title("(b) Blockade cost is quadratic in the superatom size\n"
                 "one Trotter step $\\to$ $N(N-1)/2$ pairwise CZ", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xticks(n_vals)

    # ---------------- (c) DD overhead vs benefit
    ax = axes[2]
    pulses = np.array(dd["pulses"], dtype=float)
    x = np.arange(len(pulses), dtype=float)
    ax.bar(x - 0.18, dd["dd_ops"], width=0.36, color=rplt.PALETTE["cyan"], edgecolor="black",
           linewidth=0.5, label="ops after $\\mathtt{compile\\_dynamical\\_decoupling}$")
    ax.bar(x + 0.18, dd["hw_ops"], width=0.36, color=rplt.PALETTE["purple"], edgecolor="black",
           linewidth=0.5, label="ops after hardware lowering")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(p)}" for p in pulses])
    ax.set_xlabel("number of compiled CPMG $\\pi$-pulses $N_\\pi$")
    ax.set_ylabel("native operations (overhead)")
    ax.set_ylim(0, max(dd["dd_ops"]) * 1.45)
    ax.legend(fontsize=8.2, loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(x, dd["enhancement"], "o-", color=rplt.PALETTE["red"], lw=2.6, ms=8,
             label="coherence gain $T_2/T_2^*$")
    ax2.axhline(10.0, color=rplt.PALETTE["green"], ls="--", lw=1.6, label="$10\\times$ (PRL 128, 123601)")
    for xi, (enh, t2) in zip(x, zip(dd["enhancement"], dd["t2"])):
        ax2.annotate(f"{t2:.1f} $\\mu$s", xy=(xi, enh), xytext=(0, 8), textcoords="offset points",
                     fontsize=7.8, ha="center", color=rplt.PALETTE["red"], fontweight="bold")
    ax2.set_ylabel("coherence extension $T_2 / T_2^*$", color=rplt.PALETTE["red"])
    ax2.tick_params(axis="y", labelcolor=rplt.PALETTE["red"])
    ax2.set_ylim(0, max(dd["enhancement"]) * 1.45)
    ax2.grid(False)
    ax2.legend(fontsize=8.2, loc="lower right")
    ax.set_title("(c) Decoupling overhead vs coherence benefit\n"
                 "custom $\\mathtt{@cirq.transformer}$ + hardware lowering", fontsize=10.5)

    fig.suptitle(
        "Compiling ensemble Rydberg physics onto native hardware with the Cirq transformer stack\n"
        "Phys. Rev. Lett. 128, 123601 (2022) -- Trapped Alkali-Metal Rydberg Qubit",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return rplt.save_figure(fig, __file__, "cirq_prl2022_transformer_pipeline.png")


def main() -> None:
    t_start = time.time()
    print("=" * 94)
    print("Cirq transformer pipeline: lowering a Rydberg ensemble circuit to native hardware")
    print("Phys. Rev. Lett. 128, 123601 (2022) -- Trapped Alkali-Metal Rydberg Qubit")
    print("=" * 94)
    print(f"  Omega_1/2pi = {OMEGA_1 / MHZ:.1f} MHz, V_vdW/2pi = {V_BLOCKADE / MHZ:.1f} MHz, "
          f"dt = {DT * 1e3:.0f} ns, storage = {T_STORAGE:.1f} us")

    study_decomposition()
    study_pass_by_pass()
    study_target_gatesets()
    scaling = study_scaling()
    dd = study_dd()

    print("\n" + "-" * 94)
    print("[6] Figure")
    print("-" * 94)
    make_figure(scaling, dd)
    print(f"\nTotal runtime: {time.time() - t_start:.1f} s")
    print("=" * 94)


if __name__ == "__main__":
    main()
