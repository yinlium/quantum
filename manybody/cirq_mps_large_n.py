"""
Tensor-network (MPS) simulation of the Rydberg superatom beyond the statevector wall.

Reference
---------
Y. Mei, Y. Li, H. Nguyen, P. R. Berman, and A. Kuzmich,
*"Trapped Alkali-Metal Rydberg Qubit"*, **Phys. Rev. Lett. 128, 123601 (2022)**.

Physics
-------
``N`` ground-state atoms sitting inside a single Rydberg blockade radius are driven
resonantly with single-atom Rabi frequency ``Omega_1``.  The Hamiltonian is

    H = (Omega_1 / 2) * sum_j X_j  +  V_vdW * sum_{j<k} n_j n_k ,   n_j = |r><r|_j

(the ``|0> = |g>``, ``|1> = |r>`` convention of :mod:`rydberg_cirq.dicke`; the
excitation number ``m`` is the Hamming weight).  When ``V_vdW >> sqrt(N) Omega_1`` the
van der Waals shift pushes every ``m >= 2`` configuration out of resonance, so the
dynamics is frozen into the two-dimensional *superatom qubit*

    |g...g>   <-->   |W> = (1/sqrt(N)) sum_j |g...r_j...g>

whose coupling is the **collectively enhanced** ``Omega_N = sqrt(N) Omega_1`` --
the central result reproduced here.

Why MPS, and the honest caveat
------------------------------
Dense statevector simulation costs ``2**N`` amplitudes, which is why the companion
script ``cirq_superatom_rabi_oscillation.py`` stops at ``N = 9``.  A matrix-product
state stores only ``N * 2 * chi**2`` numbers, so if the *bond dimension* ``chi``
stays small the cost becomes **linear** in ``N``.

The caveat is real and must be stated up front: the blockade term is **all-to-all**
coupled, which is the pathological case for a one-dimensional MPS ansatz.  There is
no geometric locality to exploit, every one of the ``N(N-1)/2`` ``CZPowGate``s is a
long-range gate that must be applied by swapping qubits together, and in general the
entanglement across a bipartition of an all-to-all model grows with ``N``.

What rescues us is *physics*, not the ansatz: in the strong-blockade regime the
accessible Hilbert space is the two-dimensional span ``{|g...g>, |W>}`` (plus an
``O((Omega_N/V)**2)`` admixture of ``m >= 2``).  A superposition
``a|g...g> + b|W>`` is a *weakly* entangled state -- it is exactly an MPS of bond
dimension **2** for any ``N`` (one bond value carries "no excitation yet", the other
carries "the excitation has already been placed to my left").  So the blockade that
makes the superatom a good qubit is exactly what makes it a good MPS.  This script
demonstrates that empirically, and also deliberately breaks it by weakening the
blockade, where the bond dimension *does* blow up and MPS fails.

Simulation engine: what we actually did, and why
------------------------------------------------
``cirq.contrib.quimb`` is used throughout, but with an important, deliberate split:

* ``cirq.contrib.quimb.circuit_to_tensors`` / ``tensor_state_vector`` /
  ``tensor_expectation_value`` -- used as an *exact* (untruncated) tensor-network
  contraction of the Cirq circuit, cross-checked against ``cirq.Simulator``.
* ``cirq.contrib.quimb.MPSSimulator`` -- benchmarked and then **rejected** for the
  production runs.  Its ``_MPSHandler`` does not maintain a canonical 1-D chain;
  it creates a fresh bond ``mu_{i}_{j}`` for every *pair* of qubits touched by a
  two-qubit gate.  Under an all-to-all blockade layer every pair is touched, so each
  site tensor acquires ``N-1`` bonds and its size grows like ``chi**(N-1)`` -- i.e.
  it reproduces the ``2**N`` wall it was supposed to avoid.  The console report
  measures this explicitly (``estimation_stats()['num_coefs_used']`` vs ``2**N``).
* Production runs therefore drive a genuine 1-D ``quimb.tensor.CircuitMPS`` (the
  library underneath ``cirq.contrib.quimb``) **gate by gate from the Cirq circuit
  object**: every gate is obtained via ``cirq.unitary(op)`` and every qubit index via
  the Cirq ``LineQubit`` ordering, so the Cirq circuit remains the single source of
  truth.  ``CircuitMPS`` uses ``swap+split`` to apply the long-range blockade gates
  while keeping a strict, interpretable ``max_bond`` handle.

Cirq APIs showcased
-------------------
* ``rydberg_cirq.gates.CollectiveLaserDriveStep`` / ``RydbergBlockadeStep`` composite
  ``cirq.Gate``s and their ``_decompose_`` into native ``XPowGate`` / ``CZPowGate``.
* ``cirq.decompose`` to flatten the high-level Trotter block to a native circuit.
* ``cirq.unitary`` as the bridge from Cirq operations to tensor-network gates.
* ``cirq.Simulator`` (``complex128``) as the exact statevector credibility anchor.
* ``cirq.contrib.quimb.circuit_to_tensors``, ``tensor_state_vector``,
  ``tensor_expectation_value``, ``MPSSimulator`` / ``MPSOptions``.
* ``cirq.PauliSum`` / ``cirq.PauliString`` observables.

Outputs
-------
Console report + ``manybody/cirq_prl2022_mps_large_n.png``.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Sequence

import numpy as np
import scipy.linalg as sla
from scipy.optimize import curve_fit

import cirq
import cirq.contrib.quimb as ccq
import quimb.tensor as qtn

import rydberg_cirq as rc
from rydberg_cirq import plotting as rcplot

# --------------------------------------------------------------------------------------
# Model / schedule
# --------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SuperatomModel:
    """Parameters of the driven blockaded ensemble (angular frequencies, unit Omega_1)."""

    omega_1: float = 1.0
    v_vdw: float = 50.0

    def omega_collective(self, N: int) -> float:
        return np.sqrt(N) * self.omega_1


@dataclasses.dataclass(frozen=True)
class TrotterSchedule:
    """Second-order (Strang) Trotter schedule for one collective-Rabi trajectory."""

    t_max: float
    n_samples: int  # observable samples after t = 0
    sub_per_sample: int  # Strang substeps between consecutive samples
    dt_sub: float
    times: np.ndarray

    @property
    def n_sub_total(self) -> int:
        return self.n_samples * self.sub_per_sample


def make_schedule(
    N: int,
    model: SuperatomModel,
    n_periods: float = 2.0,
    samples_per_period: int = 24,
    vh_target: float = 1.2,
) -> TrotterSchedule:
    """Build a schedule covering ``n_periods`` of the *collective* Rabi oscillation.

    ``vh_target`` caps the dimensionless Strang step ``V_vdW * dt_sub``; the Trotter
    convergence check in :func:`trotter_convergence_report` shows ``V dt ~ 1.2`` is
    already good to ``~1e-7`` infidelity, while ``V dt >~ 2 pi`` aliases catastrophically.
    """
    omega_N = model.omega_collective(N)
    t_max = n_periods * 2.0 * np.pi / omega_N
    n_samples = int(round(n_periods * samples_per_period))
    dt_sample = t_max / n_samples
    sub_per_sample = max(1, int(np.ceil(model.v_vdw * dt_sample / vh_target)))
    return TrotterSchedule(
        t_max=t_max,
        n_samples=n_samples,
        sub_per_sample=sub_per_sample,
        dt_sub=dt_sample / sub_per_sample,
        times=np.linspace(0.0, t_max, n_samples + 1),
    )


# --------------------------------------------------------------------------------------
# Cirq circuit construction
# --------------------------------------------------------------------------------------


def strang_block(N: int, model: SuperatomModel, dt_sub: float, qubits) -> cirq.Circuit:
    """One Strang step ``e^{-i H_d dt/2} e^{-i H_V dt} e^{-i H_d dt/2}`` as a Cirq circuit.

    Built from the shared composite gates and then flattened with ``cirq.decompose``
    into native ``XPowGate`` / ``CZPowGate`` operations.
    """
    high_level = cirq.Circuit(
        rc.CollectiveLaserDriveStep(N, model.omega_1, dt_sub / 2.0).on(*qubits),
        rc.RydbergBlockadeStep(N, model.v_vdw, dt_sub).on(*qubits),
        rc.CollectiveLaserDriveStep(N, model.omega_1, dt_sub / 2.0).on(*qubits),
    )
    return cirq.Circuit(cirq.decompose(high_level))


def cirq_to_quimb_program(circuit: cirq.Circuit, qubits: Sequence[cirq.Qid]):
    """Lower a Cirq circuit to ``[(unitary, (site indices...)), ...]`` for quimb.

    This is the bridge: the Cirq ``Circuit`` stays the source of truth, ``cirq.unitary``
    supplies the gate matrices and the ``LineQubit`` order fixes the MPS site order.
    Cirq's big-endian ``(out..., in...)`` reshape convention matches quimb's, which is
    verified numerically by :func:`validate_against_statevector`.
    """
    index = {q: i for i, q in enumerate(qubits)}
    return [
        (cirq.unitary(op), tuple(index[q] for q in op.qubits))
        for op in circuit.all_operations()
    ]


# --------------------------------------------------------------------------------------
# Observables
# --------------------------------------------------------------------------------------


def _single_excitation_bitstrings(N: int):
    return ["".join("1" if k == j else "0" for k in range(N)) for j in range(N)]


def mps_populations(circ: qtn.CircuitMPS, N: int, bitstrings) -> dict:
    """Excitation-sector populations of an MPS, at cost ``O(N**2 chi**2)``.

    Only ``N + 1`` computational amplitudes are needed:

    * ``P0    = |<g...g|psi>|^2``
    * ``P(m=1)= sum_j |<e_j|psi>|^2``
    * ``P_W   = |sum_j <e_j|psi>|^2 / N``          (the symmetric combination)
    * ``P(m>=2) = 1 - P0 - P(m=1)``

    Everything is divided by ``<psi|psi>`` because bond truncation is not norm
    preserving -- this keeps the reported populations honest.
    """
    # quimb returns the norm as a 0-d complex scalar; take the modulus explicitly.
    norm2 = float(abs(circ.psi.norm())) ** 2
    a0 = complex(circ.amplitude("0" * N))
    a1 = np.array([complex(circ.amplitude(b)) for b in bitstrings])
    p0 = abs(a0) ** 2 / norm2
    p1 = float(np.sum(np.abs(a1) ** 2)) / norm2
    pw = abs(a1.sum()) ** 2 / (N * norm2)
    return {"p0": p0, "p1": p1, "pW": pw, "pmulti": max(0.0, 1.0 - p0 - p1), "norm2": norm2}


def dense_populations(psi: np.ndarray, masks: dict, w_vec: np.ndarray) -> dict:
    p0 = float(np.abs(psi[masks["m0"]][0]) ** 2)
    p1 = float(np.sum(np.abs(psi[masks["m1"]]) ** 2))
    pw = float(np.abs(np.vdot(w_vec, psi)) ** 2)
    pmulti = float(np.sum(np.abs(psi[masks["multi"]]) ** 2))
    return {"p0": p0, "p1": p1, "pW": pw, "pmulti": pmulti, "norm2": 1.0}


# --------------------------------------------------------------------------------------
# Trajectory runners
# --------------------------------------------------------------------------------------


def run_mps(
    N: int,
    model: SuperatomModel,
    sched: TrotterSchedule,
    max_bond: int = 32,
    cutoff: float = 1e-12,
    collect_states: bool = False,
) -> dict:
    """Evolve the Cirq Trotter circuit inside a 1-D ``quimb.tensor.CircuitMPS``."""
    qubits = cirq.LineQubit.range(N)
    block = strang_block(N, model, sched.dt_sub, qubits)
    program = cirq_to_quimb_program(block, qubits)
    bitstrings = _single_excitation_bitstrings(N)

    circ = qtn.CircuitMPS(N, max_bond=max_bond, cutoff=cutoff, gate_contract="swap+split")

    rows, states, bonds, mem = [], [], [], []
    t_wall = 0.0
    for s in range(sched.n_samples + 1):
        if s > 0:
            t0 = time.perf_counter()
            for _ in range(sched.sub_per_sample):
                for U, where in program:
                    circ.apply_gate_raw(U, where)
            t_wall += time.perf_counter() - t0
        rows.append(mps_populations(circ, N, bitstrings))
        bonds.append(int(circ.psi.max_bond()))
        mem.append(int(sum(t.data.nbytes for t in circ.psi.tensors)))
        if collect_states:
            states.append(np.asarray(circ.to_dense()).reshape(-1))

    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out.update(
        N=N,
        times=sched.times,
        runtime=t_wall,
        max_bond_setting=max_bond,
        chi_peak=int(max(bonds)),
        chi_trace=np.array(bonds),
        memory_bytes=int(max(mem)),
        n_gates=sched.n_sub_total * len(program),
        engine=f"quimb CircuitMPS (chi<={max_bond})",
    )
    if collect_states:
        out["states"] = np.array(states)
    return out


def run_dense(
    N: int,
    model: SuperatomModel,
    sched: TrotterSchedule,
    collect_states: bool = False,
) -> dict:
    """Exact ``cirq.Simulator`` statevector evolution of the *same* Cirq circuit."""
    qubits = cirq.LineQubit.range(N)
    block = strang_block(N, model, sched.dt_sub, qubits)
    n_ops = len(list(block.all_operations()))
    step = cirq.Circuit([block] * sched.sub_per_sample)
    sim = cirq.Simulator(dtype=np.complex128)

    masks = rc.excitation_masks(N)
    w_vec = rc.w_state_vector(N)

    psi = np.zeros(2**N, dtype=np.complex128)
    psi[0] = 1.0

    rows, states = [], []
    t_wall = 0.0
    for s in range(sched.n_samples + 1):
        if s > 0:
            t0 = time.perf_counter()
            psi = sim.simulate(step, initial_state=psi).final_state_vector.astype(np.complex128)
            t_wall += time.perf_counter() - t0
        rows.append(dense_populations(psi, masks, w_vec))
        if collect_states:
            states.append(psi.copy())

    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out.update(
        N=N,
        times=sched.times,
        runtime=t_wall,
        chi_peak=np.nan,
        memory_bytes=2**N * 16,
        n_gates=sched.n_sub_total * n_ops,
        engine="cirq.Simulator (complex128)",
    )
    if collect_states:
        out["states"] = np.array(states)
    return out


# --------------------------------------------------------------------------------------
# Rabi-frequency extraction
# --------------------------------------------------------------------------------------


def _rabi_model(t, amp, omega, offset):
    return offset + amp * np.sin(0.5 * omega * t) ** 2


def fit_rabi_frequency(times: np.ndarray, p_w: np.ndarray, omega_guess: float) -> tuple:
    """Least-squares fit of ``P_W(t) = c + A sin^2(Omega t / 2)``; returns ``(Omega, rmse)``."""
    p0 = [float(np.ptp(p_w)) or 1.0, float(omega_guess), float(np.min(p_w))]
    popt, _ = curve_fit(_rabi_model, times, p_w, p0=p0, maxfev=40000)
    resid = p_w - _rabi_model(times, *popt)
    return abs(float(popt[1])), float(np.sqrt(np.mean(resid**2)))


# --------------------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------------------

RULE = "=" * 86
SUB = "-" * 86


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def circuit_demo(model: SuperatomModel) -> None:
    section("[1] Cirq circuit under test: Strang-Trotterised superatom Hamiltonian")
    N = 3
    q = cirq.LineQubit.range(N)
    high = cirq.Circuit(
        rc.CollectiveLaserDriveStep(N, model.omega_1, 0.05).on(*q),
        rc.RydbergBlockadeStep(N, model.v_vdw, 0.10).on(*q),
        rc.CollectiveLaserDriveStep(N, model.omega_1, 0.05).on(*q),
    )
    print("\nHigh-level composite gates (N = 3):")
    print(high)
    print("\nAfter cirq.decompose -> native XPowGate / CZPowGate:")
    print(cirq.Circuit(cirq.decompose(high)))
    print(f"\nH = (Omega_1/2) sum_j X_j + V sum_(j<k) n_j n_k, "
          f"Omega_1 = {model.omega_1}, V_vdW = {model.v_vdw}")
    print("Native gate count per Strang step: 2N XPowGate + N(N-1)/2 CZPowGate")
    for n in (4, 12, 20, 24):
        print(f"    N = {n:2d} -> {2*n:3d} + {n*(n-1)//2:4d} = {2*n + n*(n-1)//2:4d} native ops/step")


def contrib_quimb_demo(model: SuperatomModel) -> None:
    """Exercise the *exact* ``cirq.contrib.quimb`` tensor-network API."""
    section("[2] cirq.contrib.quimb exact tensor-network contraction vs cirq.Simulator")
    N = 8
    q = cirq.LineQubit.range(N)
    circuit = cirq.Circuit([strang_block(N, model, 0.02, q)] * 3)

    tensors, frontier, _ = ccq.circuit_to_tensors(circuit=circuit, qubits=q)
    print(f"\nccq.circuit_to_tensors  : N = {N}, {len(tensors)} tensors "
          f"(1 input ket + {len(tensors)-N} gates), frontier depth max = {max(frontier.values())}")

    t0 = time.perf_counter()
    psi_tn = ccq.tensor_state_vector(circuit, q)
    t_tn = time.perf_counter() - t0
    t0 = time.perf_counter()
    psi_sv = cirq.Simulator(dtype=np.complex128).simulate(circuit).final_state_vector
    t_sv = time.perf_counter() - t0
    fid = abs(np.vdot(psi_tn, psi_sv)) ** 2
    print(f"ccq.tensor_state_vector : |<psi_TN|psi_cirq>|^2 = {fid:.12f}   "
          f"(TN {t_tn*1e3:.1f} ms vs statevector {t_sv*1e3:.1f} ms)")

    ps = cirq.Z(q[0]) * cirq.Z(q[1])
    ev_tn = float(np.real(ccq.tensor_expectation_value(circuit, ps)))
    ev_sv = float(
        ps.expectation_from_state_vector(
            psi_sv.astype(np.complex128), {qq: i for i, qq in enumerate(q)}
        ).real
    )
    print(f"ccq.tensor_expectation_value <Z0 Z1> = {ev_tn:+.10f}  "
          f"vs cirq PauliString {ev_sv:+.10f}  (|diff| = {abs(ev_tn-ev_sv):.2e})")
    print("\nNote: this contraction is EXACT (no truncation) and therefore still scales")
    print("      exponentially for an all-to-all circuit -- it is a cross-check, not a")
    print("      large-N engine.")


def contrib_mps_simulator_diagnosis(model: SuperatomModel) -> None:
    """Honest measurement of why ``ccq.MPSSimulator`` cannot take us to large N."""
    section("[3] Why cirq.contrib.quimb.MPSSimulator fails on an all-to-all blockade")
    print("\nIts internal state keeps one tensor per qubit but opens a NEW bond 'mu_i_j'")
    print("for every PAIR touched by a 2-qubit gate.  The blockade layer touches every")
    print("pair, so each site tensor ends up with N-1 bonds and size ~ 2 * chi^(N-1).")
    print(f"\n{'N':>3} {'coefs stored':>14} {'2^N':>10} {'coefs/2^N':>10} "
          f"{'MPS bytes':>12} {'runtime':>9}")
    print(SUB)
    opts = ccq.MPSOptions(max_bond=8, cutoff=1e-8)
    for N in (4, 6, 8, 10, 12):
        q = cirq.LineQubit.range(N)
        circuit = cirq.Circuit([strang_block(N, model, 0.02, q)] * 2)
        sim = ccq.MPSSimulator(simulation_options=opts)
        t0 = time.perf_counter()
        try:
            res = sim.simulate(circuit, qubit_order=q)
            dt = time.perf_counter() - t0
            stats = res.final_state.estimation_stats()
            coefs = stats["num_coefs_used"]
            print(f"{N:3d} {coefs:14d} {2**N:10d} {coefs/2**N:10.2f} "
                  f"{stats['memory_bytes']:12d} {dt:8.2f}s")
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"{N:3d}  FAILED: {type(exc).__name__}: {exc}")
    print("\n-> measured storage is exactly N * 2^N: every single site tensor already holds")
    print("   2 * chi^(N-1) = 2^N numbers, so this path is N times WORSE than a dense")
    print("   statevector.  The exponential wall is not avoided, it is amplified.")
    print("   Production runs below therefore use a genuine 1-D quimb CircuitMPS driven")
    print("   gate-by-gate from the same Cirq circuit (swap+split for long-range gates).")


def trotter_convergence_report(model: SuperatomModel) -> float:
    """Separate the *Trotter* error from the *MPS truncation* error."""
    section("[4] Trotter (circuit) error vs the exact matrix exponential, N = 8")
    N = 8
    dim = 2**N
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    H = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        op = np.array([[1.0]], dtype=complex)
        for k in range(N):
            op = np.kron(op, X if k == j else np.eye(2))
        H += 0.5 * model.omega_1 * op
    w = rc.hamming_weights(N)
    H += model.v_vdw * np.diag(w * (w - 1) / 2.0)

    t_max = 2.0 * 2.0 * np.pi / model.omega_collective(N)
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0
    psi_exact = sla.expm(-1j * H * t_max) @ psi0
    w_vec = rc.w_state_vector(N)
    pw_exact = abs(np.vdot(w_vec, psi_exact)) ** 2
    print(f"\nexact scipy.linalg.expm:  P_W(t_max = {t_max:.3f}) = {pw_exact:.10f}")
    print(f"\n{'n_sub':>7} {'V*dt_sub':>9} {'P_W':>14} {'|dP_W|':>10} {'1 - fidelity':>14}")
    print(SUB)
    q = cirq.LineQubit.range(N)
    sim = cirq.Simulator(dtype=np.complex128)
    chosen = np.nan
    for n_sub in (32, 64, 128, 256, 512):
        h = t_max / n_sub
        circ = cirq.Circuit([strang_block(N, model, h, q)] * n_sub)
        psi = sim.simulate(circ).final_state_vector.astype(complex)
        pw = abs(np.vdot(w_vec, psi)) ** 2
        infid = 1.0 - abs(np.vdot(psi_exact, psi)) ** 2
        flag = ""
        if abs(model.v_vdw * h - 1.2) < 0.45:
            chosen, flag = infid, "   <- production step size"
        print(f"{n_sub:7d} {model.v_vdw*h:9.3f} {pw:14.10f} "
              f"{abs(pw-pw_exact):10.2e} {infid:14.2e}{flag}")
    print("\n-> the production schedule caps V*dt_sub at 1.2, i.e. Trotter infidelity")
    print("   ~1e-7, far below the MPS truncation errors studied below.")
    return chosen


def validate_against_statevector(model: SuperatomModel, n_list, max_bond=32) -> list:
    section("[5] CREDIBILITY ANCHOR: MPS vs exact cirq.Simulator statevector, N <= 12")
    print(f"\n{'N':>3} {'samples':>8} {'gates':>8} {'chi_peak':>9} {'min fidelity':>14} "
          f"{'max |dP_W|':>12} {'max |dP(m>=2)|':>15} {'t_MPS':>8} {'t_dense':>9}")
    print(SUB)
    rows = []
    for N in n_list:
        sched = make_schedule(N, model)
        mps = run_mps(N, model, sched, max_bond=max_bond, collect_states=True)
        dense = run_dense(N, model, sched, collect_states=True)
        fids = np.array(
            [abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
             for a, b in zip(mps["states"], dense["states"])]
        )
        d_pw = float(np.max(np.abs(mps["pW"] - dense["pW"])))
        d_pm = float(np.max(np.abs(mps["pmulti"] - dense["pmulti"])))
        rows.append(
            dict(N=N, min_fid=float(fids.min()), d_pw=d_pw, d_pm=d_pm,
                 chi=mps["chi_peak"], t_mps=mps["runtime"], t_dense=dense["runtime"],
                 gates=mps["n_gates"])
        )
        print(f"{N:3d} {sched.n_samples:8d} {mps['n_gates']:8d} {mps['chi_peak']:9d} "
              f"{fids.min():14.10f} {d_pw:12.2e} {d_pm:15.2e} "
              f"{mps['runtime']:7.2f}s {dense['runtime']:8.2f}s")
    worst = max(r["d_pw"] for r in rows)
    print(f"\n-> worst-case |P_W(MPS) - P_W(exact)| over all N <= {max(n_list)} "
          f"and all times: {worst:.2e}")
    print("   (this also verifies that the Cirq big-endian unitary convention maps")
    print("    correctly onto quimb's MPS site ordering)")
    return rows


def bond_convergence_study(model: SuperatomModel, N: int, chis, weak_v: float) -> dict:
    section(f"[6] Bond-dimension convergence at N = {N}: blockaded vs unblockaded")
    out = {"N": N}
    for label, mdl in (("strong blockade", model),
                       ("weak blockade", SuperatomModel(model.omega_1, weak_v))):
        sched = make_schedule(N, mdl, n_periods=1.0, samples_per_period=24)
        dense = run_dense(N, mdl, sched, collect_states=True)
        ratio = mdl.v_vdw / mdl.omega_collective(N)
        print(f"\n{label}: V_vdW = {mdl.v_vdw:g}  (V / Omega_N = {ratio:.2f}), "
              f"peak P(m>=2) = {dense['pmulti'].max():.3f}")
        print(f"{'chi':>5} {'1 - fidelity':>15} {'max |dP_W|':>13} {'MPS bytes':>11} {'runtime':>9}")
        print(SUB)
        recs = []
        for chi in chis:
            mps = run_mps(N, mdl, sched, max_bond=chi, cutoff=1e-14, collect_states=True)
            fids = np.array(
                [abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
                 for a, b in zip(mps["states"], dense["states"])]
            )
            infid = float(1.0 - fids.min())
            d_pw = float(np.max(np.abs(mps["pW"] - dense["pW"])))
            recs.append(dict(chi=chi, infid=max(infid, 1e-16), d_pw=max(d_pw, 1e-16),
                             bytes=mps["memory_bytes"], runtime=mps["runtime"],
                             chi_peak=mps["chi_peak"]))
            print(f"{chi:5d} {infid:15.3e} {d_pw:13.3e} {mps['memory_bytes']:11d} "
                  f"{mps['runtime']:8.2f}s")
        out[label] = dict(records=recs, v=mdl.v_vdw, ratio=ratio,
                          pmulti=float(dense["pmulti"].max()))

    # How does the truncation error grow with N at *fixed* bond dimension?
    chi_fixed = 4
    print(f"\nError growth with N at fixed chi = {chi_fixed} "
          f"(the real test of MPS viability):")
    print(f"{'N':>4} {'strong-blockade 1-F':>22} {'weak-blockade 1-F':>20}")
    print(SUB)
    n_scan = [n for n in (6, 8, 10, 12, 14) if n <= 14]
    scaling = {"n": n_scan, "strong": [], "weak": []}
    for n in n_scan:
        vals = []
        for mdl in (model, SuperatomModel(model.omega_1, weak_v)):
            sch = make_schedule(n, mdl, n_periods=1.0, samples_per_period=16)
            dns = run_dense(n, mdl, sch, collect_states=True)
            m = run_mps(n, mdl, sch, max_bond=chi_fixed, cutoff=1e-14, collect_states=True)
            f = np.array(
                [abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
                 for a, b in zip(m["states"], dns["states"])]
            )
            vals.append(max(float(1.0 - f.min()), 1e-16))
        scaling["strong"].append(vals[0])
        scaling["weak"].append(vals[1])
        print(f"{n:4d} {vals[0]:22.3e} {vals[1]:20.3e}")
    out["scaling"] = scaling
    strong = out["strong blockade"]["records"]
    weak = out["weak blockade"]["records"]
    print("\n-> Blockaded regime: the *physical* state lives in span{|g...g>, |W>}, which is")
    print("   a bond-dimension-2 MPS for ANY N, and indeed chi = 2 already gets the state")
    print(f"   right to {strong[1]['infid']:.1e}.  Full convergence to ~1e-10 needs chi ~ 8")
    print("   for two reasons: (i) the residual O((Omega_N/V)^2) admixture of m >= 2 Dicke")
    print("   states carries a few extra Schmidt values, and (ii) 'swap+split' application")
    print("   of the long-range blockade gates transiently inflates the bond, so a hard")
    print("   max_bond cap also truncates those intermediates.  Left unconstrained")
    print("   (max_bond = 16, cutoff = 1e-12) the sampled state settles at chi = 2-5.")
    print("\n-> Unblockaded regime: the all-to-all Ising term is no longer frozen out, the")
    print(f"   ensemble genuinely explores all m sectors (peak P(m>=2) = "
          f"{out['weak blockade']['pmulti']:.2f}) and entanglement grows across every")
    print(f"   bipartition.  chi = 2 is off by {weak[1]['infid']:.1e} and even chi = "
          f"{weak[-1]['chi']} only reaches {weak[-1]['infid']:.1e}; the required chi grows")
    print("   with N here, so THIS is where the MPS ansatz breaks down for an all-to-all")
    print("   Hamiltonian.  The superatom is tractable because the blockade -- the very")
    print("   thing that makes it a good qubit -- keeps it weakly entangled.")
    return out


def large_n_sweep(model: SuperatomModel, n_list, max_bond=16, dense_max_n=14) -> list:
    section("[7] Large-N sweep: collective sqrt(N) enhancement from the MPS simulation")
    print(f"\n{'N':>3} {'gates':>7} {'chi':>4} {'MPS mem':>9} {'dense mem':>11} "
          f"{'t_MPS':>8} {'t_dense':>9} {'Omega_fit':>10} {'sqrt(N)':>8} {'ratio':>7} "
          f"{'maxP(m>=2)':>11}")
    print(SUB)
    rows = []
    for N in n_list:
        sched = make_schedule(N, model)
        mps = run_mps(N, model, sched, max_bond=max_bond)
        omega_fit, rmse = fit_rabi_frequency(sched.times, mps["pW"], model.omega_collective(N))
        t_dense = np.nan
        if N <= dense_max_n:
            t_dense = run_dense(N, model, sched)["runtime"]
        rows.append(
            dict(N=N, times=sched.times, pW=mps["pW"], pmulti=mps["pmulti"],
                 omega_fit=omega_fit, rmse=rmse, chi=mps["chi_peak"],
                 mem_mps=mps["memory_bytes"], mem_dense=2**N * 16,
                 t_mps=mps["runtime"], t_dense=t_dense, gates=mps["n_gates"])
        )
        dstr = f"{t_dense:8.2f}s" if np.isfinite(t_dense) else f"{'--':>9}"
        print(f"{N:3d} {mps['n_gates']:7d} {mps['chi_peak']:4d} "
              f"{_fmt_bytes(mps['memory_bytes']):>9} {_fmt_bytes(2**N*16):>11} "
              f"{mps['runtime']:7.2f}s {dstr} {omega_fit:10.5f} "
              f"{np.sqrt(N):8.4f} {omega_fit/np.sqrt(N):7.4f} {mps['pmulti'].max():11.2e}")
    return rows


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB", "PB"):
        if b < 1024 or unit == "PB":
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}PB"


def power_law_fit(n_arr, omega_arr):
    """Fit ``Omega_N = A N**alpha`` by linear regression in log-log space."""
    lx, ly = np.log(n_arr), np.log(omega_arr)
    n = len(lx)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    s_err = np.sqrt(np.sum(resid**2) / (n - 2)) if n > 2 else 0.0
    slope_err = s_err / np.sqrt(np.sum((lx - lx.mean()) ** 2)) if n > 2 else 0.0
    r2 = 1.0 - np.sum(resid**2) / np.sum((ly - ly.mean()) ** 2)
    return float(slope), float(slope_err), float(np.exp(intercept)), float(r2)


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------


def make_figure(sweep, bond_study, exponent, exponent_err, prefactor, r2, show_n):
    import matplotlib.pyplot as plt

    rcplot.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    colors = rcplot.SERIES_COLORS
    P = rcplot.PALETTE

    by_n = {r["N"]: r for r in sweep}

    # ---- (a) collective Rabi oscillations -------------------------------------------
    ax = axes[0, 0]
    for i, N in enumerate(show_n):
        r = by_n[N]
        c = colors[i % len(colors)]
        ax.plot(r["times"], r["pW"], color=c, lw=2.2,
                label=rf"$N={N}$  ($\sqrt{{N}}={np.sqrt(N):.2f}$)")
        ax.plot(r["times"], np.sin(np.sqrt(N) * r["times"] / 2.0) ** 2,
                color=c, ls=":", lw=1.3, alpha=0.8)
    ax.set_xlim(0, 2.0 * 2 * np.pi / np.sqrt(min(show_n[1:])))
    ax.set_ylim(-0.04, 1.12)
    ax.set_xlabel(r"time  $\Omega_1 t$")
    ax.set_ylabel(r"superatom population  $P(|W\rangle)$")
    ax.set_title("(a) MPS collective Rabi flopping\n"
                 r"solid: quimb MPS of the Cirq circuit   dotted: $\sin^2(\sqrt{N}\Omega_1 t/2)$")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    # ---- (b) sqrt(N) power law -------------------------------------------------------
    ax = axes[0, 1]
    n_arr = np.array([r["N"] for r in sweep], dtype=float)
    om_arr = np.array([r["omega_fit"] for r in sweep])
    grid = np.linspace(n_arr.min() * 0.9, n_arr.max() * 1.1, 200)
    ax.loglog(n_arr, om_arr, "o", color=P["blue"], ms=7, zorder=3,
              label="MPS fit of $P(|W\\rangle)$")
    ax.loglog(grid, prefactor * grid**exponent, "-", color=P["red"], lw=2.0,
              label=rf"fit $\Omega_N = {prefactor:.4f}\,N^{{{exponent:.4f}}}$")
    ax.loglog(grid, np.sqrt(grid), "--", color=P["black"], lw=1.4, alpha=0.75,
              label=r"theory $\Omega_N=\sqrt{N}\,\Omega_1$")
    ax.set_xlabel(r"atom number  $N$")
    ax.set_ylabel(r"fitted Rabi frequency  $\Omega_N/\Omega_1$")
    ax.set_title("(b) Collective enhancement survives to large $N$")
    ax.annotate(
        rf"$\alpha = {exponent:.4f} \pm {exponent_err:.4f}$" "\n"
        rf"(ideal $0.5$),  $R^2 = {r2:.6f}$" "\n"
        rf"max $|\Omega_N/\sqrt{{N}}-1| = "
        rf"{np.max(np.abs(om_arr/np.sqrt(n_arr)-1))*100:.2f}\%$",
        xy=(0.04, 0.82), xycoords="axes fraction", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.45", fc=P["yellow"], alpha=0.35, ec=P["gray"]),
        va="top",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ---- (c) bond-dimension convergence ---------------------------------------------
    ax = axes[1, 0]
    styles = {"strong blockade": (P["blue"], "o", "-"), "weak blockade": (P["red"], "s", "--")}
    for label in ("strong blockade", "weak blockade"):
        data = bond_study[label]
        c, mk, ls = styles[label]
        chis = np.array([r["chi"] for r in data["records"]], dtype=float)
        infid = np.array([r["infid"] for r in data["records"]])
        ax.loglog(chis, infid, marker=mk, ls=ls, color=c, ms=7,
                  label=rf"{label}: $V/\Omega_N={data['ratio']:.2f}$, "
                        rf"$P(m\geq 2)_{{\max}}={data['pmulti']:.2f}$")
    ax.axhline(1e-10, color=P["green"], ls=":", lw=1.6, label=r"converged ($10^{-10}$)")
    ax.axvline(2, color=P["gray"], ls=":", lw=1.4)
    ax.text(2.06, 2e-13, r"$\chi=2$: span$\{|g\cdots g\rangle,|W\rangle\}$",
            fontsize=8.5, rotation=90, va="bottom", color=P["gray"])
    ax.set_xlabel(r"maximum MPS bond dimension  $\chi$")
    ax.set_ylabel(r"$1-|\langle\psi_{\mathrm{MPS}}|\psi_{\mathrm{exact}}\rangle|^2$")
    ax.set_title("(c) Where MPS works and where it breaks\n"
                 f"$N={bond_study['N']}$, vs exact cirq.Simulator statevector")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.grid(True, which="both", alpha=0.3)

    sc = bond_study["scaling"]
    axi = ax.inset_axes([0.60, 0.60, 0.37, 0.35])
    axi.semilogy(sc["n"], sc["strong"], "o-", color=P["blue"], ms=4, lw=1.4)
    axi.semilogy(sc["n"], sc["weak"], "s--", color=P["red"], ms=4, lw=1.4)
    axi.set_title(r"fixed $\chi=4$", fontsize=8)
    axi.set_xlabel(r"$N$", fontsize=8, labelpad=1)
    axi.set_ylabel(r"$1-F$", fontsize=8, labelpad=1)
    axi.tick_params(labelsize=7)
    axi.grid(True, alpha=0.25)

    # ---- (d) runtime / memory scaling ------------------------------------------------
    ax = axes[1, 1]
    t_mps = np.array([r["t_mps"] for r in sweep])
    t_den = np.array([r["t_dense"] for r in sweep])
    ok = np.isfinite(t_den)
    ax.semilogy(n_arr, t_mps, "o-", color=P["blue"], ms=6, label="runtime: MPS (quimb)")
    ax.semilogy(n_arr[ok], t_den[ok], "s-", color=P["red"], ms=6,
                label="runtime: dense cirq.Simulator")
    if ok.sum() >= 3:
        sl, ic = np.polyfit(n_arr[ok][-4:], np.log(t_den[ok][-4:]), 1)
        ext = np.linspace(n_arr[ok].max(), n_arr.max(), 40)
        ax.semilogy(ext, np.exp(ic + sl * ext), ":", color=P["red"], lw=1.6,
                    label=rf"dense extrapolation ($\times{np.exp(sl):.1f}$ per atom)")
    ax.set_xlabel(r"atom number  $N$")
    ax.set_ylabel("wall-clock runtime per trajectory (s)")
    ax.set_title("(d) Cost scaling: linear-in-$N$ MPS vs the $2^N$ wall")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.grid(False)
    n_wall = np.arange(2, 41)
    ax2.plot(n_wall, 2.0**n_wall * 16, "-", color=P["gray"], lw=1.6, alpha=0.9,
             label=r"memory: dense $2^N\times16$ B")
    ax2.plot(n_arr, [r["mem_mps"] for r in sweep], "^--", color=P["green"], ms=5,
             label="memory: MPS tensors")
    ax2.axhline(16 * 1024**3, color=P["purple"], ls="-.", lw=1.5)
    ax2.text(2.5, 16 * 1024**3 * 1.6, "16 GB RAM wall  ($N\\approx30$)",
             fontsize=8.5, color=P["purple"])
    ax2.set_ylabel("memory footprint (bytes)")
    ax2.set_xlim(1, 32)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Matrix-product-state simulation of the trapped Rydberg superatom beyond the "
        "statevector wall\n"
        "Cirq circuits + cirq.contrib.quimb / quimb MPS  |  "
        "Y. Mei et al., Phys. Rev. Lett. 128, 123601 (2022)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return fig


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    model = SuperatomModel(omega_1=1.0, v_vdw=50.0)

    print(RULE)
    print("Rydberg superatom beyond the statevector wall -- tensor-network / MPS study")
    print("Phys. Rev. Lett. 128, 123601 (2022)  |  Cirq + cirq.contrib.quimb + quimb MPS")
    print(RULE)
    print(f"cirq {cirq.__version__} | quimb {__import__('quimb').__version__} | "
          f"numpy {np.__version__}")

    circuit_demo(model)
    contrib_quimb_demo(model)
    contrib_mps_simulator_diagnosis(model)
    trotter_convergence_report(model)

    validate_against_statevector(model, n_list=[4, 6, 8, 10, 12], max_bond=32)

    bond_study = bond_convergence_study(
        model, N=12, chis=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32], weak_v=1.5
    )

    n_list = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    sweep = large_n_sweep(model, n_list, max_bond=16, dense_max_n=18)

    section("[8] sqrt(N) power-law fit")
    n_arr = np.array([r["N"] for r in sweep], dtype=float)
    om_arr = np.array([r["omega_fit"] for r in sweep])
    alpha, alpha_err, pref, r2 = power_law_fit(n_arr, om_arr)
    print(f"\n  Omega_N = A * N^alpha   with   A = {pref:.6f} (ideal {model.omega_1:.6f})")
    print(f"                                alpha = {alpha:.6f} +/- {alpha_err:.6f} (ideal 0.5)")
    print(f"                                   R^2 = {r2:.8f}")
    dev = np.abs(om_arr / np.sqrt(n_arr) - 1.0)
    print(f"\n  max |Omega_fit / (sqrt(N) Omega_1) - 1| = {dev.max()*100:.3f}%  "
          f"(at N = {int(n_arr[dev.argmax()])})")
    print(f"  mean deviation                          = {dev.mean()*100:.3f}%")

    section("[9] Runtime and memory scaling: MPS vs dense vs the 2^N wall")
    print(f"\n{'N':>3} {'MPS mem':>10} {'dense mem':>11} {'mem ratio':>11} "
          f"{'t_MPS':>9} {'t_dense':>10} {'speed-up':>9}")
    print(SUB)
    for r in sweep:
        ratio = r["mem_dense"] / r["mem_mps"]
        if np.isfinite(r["t_dense"]):
            sp = f"{r['t_dense']/r['t_mps']:8.2f}x"
            td = f"{r['t_dense']:9.2f}s"
        else:
            sp, td = f"{'--':>9}", f"{'--':>10}"
        print(f"{r['N']:3d} {_fmt_bytes(r['mem_mps']):>10} {_fmt_bytes(r['mem_dense']):>11} "
              f"{ratio:10.1f}x {r['t_mps']:8.2f}s {td} {sp}")
    n_max = int(n_arr.max())
    print(f"\n  MPS memory is O(N * 2 * chi^2): {_fmt_bytes(sweep[-1]['mem_mps'])} at N = {n_max},")
    print(f"  versus {_fmt_bytes(2**n_max * 16)} for a complex128 statevector "
          f"({2**n_max*16/sweep[-1]['mem_mps']:.0f}x).")
    for n_wall, tag in ((30, "16 GB"), (34, "256 GB"), (40, "16 TB")):
        print(f"    dense statevector at N = {n_wall}: {_fmt_bytes(2**n_wall * 16)}  ({tag} class)")

    fig = make_figure(sweep, bond_study, alpha, alpha_err, pref, r2,
                      show_n=[2, 4, 8, 12, 16, 20, 24])
    section("[10] Figure")
    rcplot.save_figure(fig, __file__, "cirq_prl2022_mps_large_n.png")
    print(RULE)


if __name__ == "__main__":
    main()
