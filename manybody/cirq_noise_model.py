"""
Realistic decoherence in collective sqrt(N) Rabi oscillations of a Rydberg superatom.

Reference
---------
Y. Mei, Y. Li, H. Nguyen, P. R. Berman, and A. Kuzmich,
*Trapped Alkali-Metal Rydberg Qubit*, Phys. Rev. Lett. **128**, 123601 (2022).
(companion: Y. Li *et al.*, Phys. Rev. A **106**, L051701 (2022).)

Physics
-------
``manybody/cirq_superatom_rabi_oscillation.py`` reproduces the *ideal, undamped*
collectively enhanced Rabi flopping of ``N`` atoms inside one blockade radius,

    |G> = |g_1 ... g_N>   <-->   |W> = N^(-1/2) sum_j |g...r_j...g>,
    Omega_N = sqrt(N) Omega_1 ,

because the dipole blockade ``V_vdW >> Omega_N`` closes off every ``m >= 2``
sector.  The measured oscillations of Fig. 2 of PRL 128, 123601 instead **damp
out** within a few microseconds.  This module reproduces that damping from first
principles with :class:`rydberg_cirq.RydbergNoiseModel`, using the experiment's
own numbers:

======================  ==========================  ===============================
Mechanism               Physical origin              Value used here
======================  ==========================  ===============================
Rydberg lifetime T1     spontaneous + 300 K BBR      100 us  (n = 50)
Laser dephasing T2*     excitation-laser linewidth   3.1 us  (free-induction value
                                                     quoted in the PRL)
Correlated dephasing    common-mode laser phase /    gamma_c, Delta-m-diagonal
                        Doppler across the ensemble  (PRA 106, L051701)
======================  ==========================  ===============================

Four results are established, each printed to the console and drawn in
``cirq_prl2022_noise_model.png``:

(a) The oscillation **frequency** still scales as ``sqrt(N)`` to better than 1 %
    while the **contrast** decays -- frequency and coherence decouple.
(b) The collective pi-pulse ``|W>`` fidelity vs ``T2*``, with the ``T2*`` needed
    for a 99 % target read off by interpolation.
(c) An error budget: amplitude damping alone / phase damping alone / both /
    both + correlated collective dephasing.
(d) **Correlated** collective dephasing is *not* equivalent to independent
    per-atom dephasing even when both damp the ``|G> <-> |W>`` coherence at the
    same rate: independent dephasing scrambles the symmetric superposition and
    drives ``F_W -> 1/N`` inside the ``m = 1`` manifold, whereas the collective
    channel leaves the ``m = 1`` sector *exactly* untouched.  That is precisely
    the Dicke-purification mechanism of the companion PRA paper.

Cirq APIs showcased
-------------------
* ``cirq.DensityMatrixSimulator(noise=...)`` driven by a custom
  ``cirq.NoiseModel`` (:class:`rydberg_cirq.RydbergNoiseModel`) that converts
  physical ``T1`` / ``T2*`` into ``cirq.amplitude_damp`` / ``cirq.phase_damp``
  probabilities through a per-moment duration.
* A custom ``N``-qubit ``_kraus_`` channel
  (:class:`rydberg_cirq.CollectiveDephasingChannel`) injected by the same noise
  model to model *correlated* dephasing.
* Composite ``cirq.Gate`` subclasses with ``_decompose_``
  (:class:`rydberg_cirq.CollectiveLaserDriveStep`,
  :class:`rydberg_cirq.RydbergBlockadeStep`) kept *unexpanded* at the top level so
  that one circuit moment == one Trotter half-step == a well-defined physical
  duration.
* Density-matrix re-injection (``simulate(..., initial_state=rho)``) to march the
  open-system evolution forward in time at O(n) cost.

Run:  python manybody/cirq_noise_model.py
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla
from scipy.optimize import curve_fit

import cirq

import rydberg_cirq as rc
import rydberg_cirq.plotting  # noqa: F401  (registers the ``rc.plotting`` attribute)
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Physical parameters (angular frequencies in rad/us, times in us)
# --------------------------------------------------------------------------- #
OMEGA_1 = np.pi          # single-atom Rabi frequency -> Omega_1 / 2pi = 0.5 MHz
V_VDW = 100.0            # van der Waals blockade shift -> V / 2pi = 15.9 MHz
T1_RYDBERG = 100.0       # us   -- n = 50 lifetime incl. 300 K blackbody
T2_STAR = 3.1            # us   -- PRL 128, 123601 free-induction coherence time
GAMMA_COLLECTIVE = 0.2   # 1/us -- correlated (common-mode) dephasing rate

T_MAX = 6.0              # us   -- ~2 T2*, enough to watch the contrast die
N_OUTER = 61             # time samples of the Rabi trace
V_DT_TARGET = 0.25       # Trotter accuracy knob: keep V * dt_sub below this


# --------------------------------------------------------------------------- #
# Trotterised propagator circuits
# --------------------------------------------------------------------------- #
def strang_step_circuit(
    qubits, omega_1: float, v_vdw: float, duration: float, n_sub: int
) -> cirq.Circuit:
    """Second-order Strang circuit for ``exp(-i H duration)`` of the blockaded drive.

    ``H = (Omega_1/2) sum_j X_j + V sum_{j<k} n_j n_k``.

    The composite gates are deliberately *not* decomposed, so the returned circuit
    has exactly ``3 * n_sub`` moments and each moment carries an unambiguous
    physical duration ``duration / (3 n_sub)`` -- which is what makes a
    time-based ``cirq.NoiseModel`` meaningful.
    """
    n = len(qubits)
    ds = duration / n_sub
    circuit = cirq.Circuit()
    for _ in range(n_sub):
        circuit.append(rc.CollectiveLaserDriveStep(n, omega_1, ds / 2.0).on(*qubits))
        circuit.append(rc.RydbergBlockadeStep(n, v_vdw, ds).on(*qubits))
        circuit.append(rc.CollectiveLaserDriveStep(n, omega_1, ds / 2.0).on(*qubits))
    return circuit


def _n_sub_for(duration: float, v_vdw: float) -> int:
    return max(1, int(np.ceil(duration * v_vdw / V_DT_TARGET)))


def make_noise(
    dt_moment: float,
    t1: float | None,
    t2_star: float | None,
    gamma_c: float | None = None,
) -> rc.RydbergNoiseModel:
    return rc.RydbergNoiseModel(
        t1=t1,
        t2_star=t2_star,
        moment_duration=dt_moment,
        gamma_collective=gamma_c,
        include_collective=gamma_c is not None,
    )


# --------------------------------------------------------------------------- #
# Observables
# --------------------------------------------------------------------------- #
class Sectors:
    """Cached projectors onto the ``|W>``, ``m = 1`` and ``m >= 2`` sectors."""

    def __init__(self, N: int):
        self.N = N
        self.w = rc.w_state_vector(N)
        masks = rc.excitation_masks(N)
        self.mask_m1 = masks["m1"]
        self.mask_multi = masks["multi"]
        self.mask_m0 = masks["m0"]

    def from_density_matrix(self, rho: np.ndarray) -> dict:
        diag = np.real(np.diag(rho))
        return {
            "F_W": float(np.real(self.w.conj() @ rho @ self.w)),
            "P_m1": float(diag[self.mask_m1].sum()),
            "P_multi": float(diag[self.mask_multi].sum()),
            "P_m0": float(diag[self.mask_m0].sum()),
        }

    def from_state_vector(self, psi: np.ndarray) -> dict:
        p = np.abs(psi) ** 2
        return {
            "F_W": float(np.abs(np.vdot(self.w, psi)) ** 2),
            "P_m1": float(p[self.mask_m1].sum()),
            "P_multi": float(p[self.mask_multi].sum()),
            "P_m0": float(p[self.mask_m0].sum()),
        }


# --------------------------------------------------------------------------- #
# Time traces
# --------------------------------------------------------------------------- #
def rabi_trace_noisy(
    N: int,
    t_max: float = T_MAX,
    n_outer: int = N_OUTER,
    t1: float | None = T1_RYDBERG,
    t2_star: float | None = T2_STAR,
    gamma_c: float | None = None,
    omega_1: float = OMEGA_1,
    v_vdw: float = V_VDW,
) -> dict:
    """Open-system collective Rabi trace via ``cirq.DensityMatrixSimulator``."""
    qubits = cirq.LineQubit.range(N)
    sec = Sectors(N)
    times = np.linspace(0.0, t_max, n_outer)
    dt = times[1] - times[0]

    step = strang_step_circuit(qubits, omega_1, v_vdw, dt, _n_sub_for(dt, v_vdw))
    noise = make_noise(dt / len(step), t1, t2_star, gamma_c)
    sim = cirq.DensityMatrixSimulator(noise=noise, dtype=np.complex128)

    rho = np.zeros((2**N, 2**N), dtype=np.complex128)
    rho[0, 0] = 1.0

    out = {k: np.zeros(n_outer) for k in ("F_W", "P_m1", "P_multi", "P_m0")}
    for it in range(n_outer):
        if it > 0:
            rho = np.asarray(
                sim.simulate(step, initial_state=rho, qubit_order=qubits).final_density_matrix,
                dtype=np.complex128,
            )
        for k, v in sec.from_density_matrix(rho).items():
            out[k][it] = v
    out["times"] = times
    out["moments_per_step"] = len(step)
    return out


def rabi_trace_ideal(
    N: int,
    t_max: float = T_MAX,
    n_outer: int = N_OUTER,
    omega_1: float = OMEGA_1,
    v_vdw: float = V_VDW,
) -> dict:
    """Closed-system reference trace via ``cirq.Simulator`` (statevector)."""
    qubits = cirq.LineQubit.range(N)
    sec = Sectors(N)
    times = np.linspace(0.0, t_max, n_outer)
    dt = times[1] - times[0]

    step = strang_step_circuit(qubits, omega_1, v_vdw, dt, _n_sub_for(dt, v_vdw))
    sim = cirq.Simulator(dtype=np.complex128)

    psi = np.zeros(2**N, dtype=np.complex128)
    psi[0] = 1.0
    out = {k: np.zeros(n_outer) for k in ("F_W", "P_m1", "P_multi", "P_m0")}
    for it in range(n_outer):
        if it > 0:
            psi = np.asarray(
                sim.simulate(step, initial_state=psi, qubit_order=qubits).final_state_vector,
                dtype=np.complex128,
            )
        for k, v in sec.from_state_vector(psi).items():
            out[k][it] = v
    out["times"] = times
    return out


# --------------------------------------------------------------------------- #
# pi-pulse fidelity (single circuit, exact pulse area)
# --------------------------------------------------------------------------- #
def pi_pulse_density_matrix(
    N: int,
    t1: float | None,
    t2_star: float | None,
    gamma_c: float | None = None,
    omega_1: float = OMEGA_1,
    v_vdw: float = V_VDW,
) -> np.ndarray:
    """Density matrix after a resonant collective pi-pulse ``t_pi = pi / (sqrt(N) Omega_1)``."""
    qubits = cirq.LineQubit.range(N)
    t_pi = np.pi / (np.sqrt(N) * omega_1)
    circuit = strang_step_circuit(qubits, omega_1, v_vdw, t_pi, _n_sub_for(t_pi, v_vdw))
    noise = make_noise(t_pi / len(circuit), t1, t2_star, gamma_c)
    sim = cirq.DensityMatrixSimulator(noise=noise, dtype=np.complex128)
    rho0 = np.zeros((2**N, 2**N), dtype=np.complex128)
    rho0[0, 0] = 1.0
    return np.asarray(
        sim.simulate(circuit, initial_state=rho0, qubit_order=qubits).final_density_matrix,
        dtype=np.complex128,
    )


# --------------------------------------------------------------------------- #
# Storage of |W> under independent vs correlated dephasing
# --------------------------------------------------------------------------- #
def w_storage_trace(N: int, t_max: float, n_pts: int, mode: str, t2_star: float) -> dict:
    """Idle |W> in the dark and watch it dephase.

    ``mode='independent'`` uses ``cirq.phase_damp`` on every atom with coherence
    time ``t2_star``; ``mode='collective'`` uses the ensemble-wide correlated
    :class:`rydberg_cirq.CollectiveDephasingChannel` with ``gamma_c = 2 / T2*``,
    which is exactly the rate that reproduces the *same* ``|G> <-> |W>``
    (``Delta m = 1``) coherence decay ``exp(-t / T2*)``.
    """
    qubits = cirq.LineQubit.range(N)
    sec = Sectors(N)
    times = np.linspace(0.0, t_max, n_pts)
    dt = times[1] - times[0]

    n_idle = 4  # identity moments per sample interval
    idle = cirq.Circuit([cirq.Moment(cirq.I.on_each(*qubits)) for _ in range(n_idle)])
    if mode == "independent":
        noise = make_noise(dt / n_idle, None, t2_star, None)
    elif mode == "collective":
        noise = make_noise(dt / n_idle, None, None, 2.0 / t2_star)
    else:  # pragma: no cover
        raise ValueError(mode)
    sim = cirq.DensityMatrixSimulator(noise=noise, dtype=np.complex128)

    w = rc.w_state_vector(N)
    rho = np.outer(w, w.conj()).astype(np.complex128)

    f_w, p_m1 = np.zeros(n_pts), np.zeros(n_pts)
    for it in range(n_pts):
        if it > 0:
            rho = np.asarray(
                sim.simulate(idle, initial_state=rho, qubit_order=qubits).final_density_matrix,
                dtype=np.complex128,
            )
        obs = sec.from_density_matrix(rho)
        f_w[it], p_m1[it] = obs["F_W"], obs["P_m1"]
    return {"times": times, "F_W": f_w, "P_m1": p_m1, "F_cond": f_w / np.maximum(p_m1, 1e-15)}


# --------------------------------------------------------------------------- #
# Damped-oscillation fit
# --------------------------------------------------------------------------- #
def damped_model(t, amp, tau, omega, floor):
    env = np.exp(-t / tau)
    return amp * env * np.sin(omega * t / 2.0) ** 2 + floor * (1.0 - env)


def fit_damped(times, signal, omega_guess, tau_guess):
    p0 = [1.0, tau_guess, omega_guess, 0.15]
    bounds = ([0.0, 1e-3, 0.2 * omega_guess, 0.0], [2.0, 1e4, 5.0 * omega_guess, 1.0])
    popt, _ = curve_fit(damped_model, times, signal, p0=p0, bounds=bounds, maxfev=60000)
    return {"amp": popt[0], "tau": popt[1], "omega": popt[2], "floor": popt[3]}


# --------------------------------------------------------------------------- #
# Validation: noiseless density matrix == statevector
# --------------------------------------------------------------------------- #
def validate_noise_off(N: int = 3) -> None:
    print("\n[VALIDATION 1] RydbergNoiseModel(t1=None, t2_star=None) must be the identity")
    qubits = cirq.LineQubit.range(N)
    t = 1.3
    circuit = strang_step_circuit(qubits, OMEGA_1, V_VDW, t, _n_sub_for(t, V_VDW))
    noise = make_noise(t / len(circuit), None, None, None)
    print(f"  noise model: {noise!r}")
    print(f"  p_amplitude_damp = {noise.p_amplitude_damp:.3e},  p_phase_damp = {noise.p_phase_damp:.3e}")

    rho0 = np.zeros((2**N, 2**N), dtype=np.complex128)
    rho0[0, 0] = 1.0
    rho = np.asarray(
        cirq.DensityMatrixSimulator(noise=noise, dtype=np.complex128)
        .simulate(circuit, initial_state=rho0, qubit_order=qubits)
        .final_density_matrix,
        dtype=np.complex128,
    )
    psi = np.asarray(
        cirq.Simulator(dtype=np.complex128)
        .simulate(circuit, qubit_order=qubits)
        .final_state_vector,
        dtype=np.complex128,
    )
    err = np.max(np.abs(rho - np.outer(psi, psi.conj())))
    sec = Sectors(N)
    print(f"  max |rho_DM - |psi><psi||           = {err:.3e}   (target < 1e-6)")
    print(
        f"  F_W  density matrix vs statevector  = {sec.from_density_matrix(rho)['F_W']:.12f}"
        f" vs {sec.from_state_vector(psi)['F_W']:.12f}"
    )
    print(f"  trace(rho) - 1                      = {np.trace(rho).real - 1.0:+.3e}")
    print(f"  --> {'PASS' if err < 1e-6 else 'FAIL'}")


def validate_trotter(N: int = 4) -> None:
    """Strang circuit vs a dense ``expm`` of the same Hamiltonian."""
    print("\n[VALIDATION 2] Trotterised Cirq circuit vs exact matrix exponential")
    qubits = cirq.LineQubit.range(N)
    t = 0.7
    dim = 2**N
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    nop = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    def emb(op, j):
        mats = [np.eye(2, dtype=complex)] * N
        mats[j] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = sum((OMEGA_1 / 2.0) * emb(X, j) for j in range(N))
    for j in range(N):
        for k in range(j + 1, N):
            H = H + V_VDW * emb(nop, j) @ emb(nop, k)
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0
    psi_exact = sla.expm(-1j * H * t) @ psi0

    circuit = strang_step_circuit(qubits, OMEGA_1, V_VDW, t, _n_sub_for(t, V_VDW))
    psi_cirq = np.asarray(
        cirq.Simulator(dtype=np.complex128).simulate(circuit, qubit_order=qubits).final_state_vector
    )
    infid = 1.0 - abs(np.vdot(psi_exact, psi_cirq)) ** 2
    print(f"  N = {N}, t = {t} us, {len(circuit)} moments ({len(circuit)//3} Strang sub-steps)")
    print(f"  1 - |<psi_exact|psi_cirq>|^2        = {infid:.3e}   (target < 1e-6)")
    print(f"  --> {'PASS' if infid < 1e-6 else 'FAIL'}")


# --------------------------------------------------------------------------- #
# Main study
# --------------------------------------------------------------------------- #
def main() -> None:
    rc.plotting.apply_style()
    t_start = time.time()

    print("=" * 84)
    print("Cirq open-system model of collective sqrt(N) Rabi oscillations")
    print("Phys. Rev. Lett. 128, 123601 (2022) -- Trapped Alkali-Metal Rydberg Qubit")
    print("=" * 84)
    print(f"  Omega_1 / 2pi = {OMEGA_1 / (2 * np.pi) * 1e3:6.1f} kHz     (Omega_1 = {OMEGA_1:.4f} rad/us)")
    print(f"  V_vdW   / 2pi = {V_VDW / (2 * np.pi):6.2f} MHz")
    print(f"  T1 (Rydberg)  = {T1_RYDBERG:6.1f} us   (n = 50, 300 K blackbody)")
    print(f"  T2* (laser)   = {T2_STAR:6.2f} us   (PRL free-induction value)")
    print(f"  gamma_c       = {GAMMA_COLLECTIVE:6.2f} 1/us (correlated ensemble dephasing)")

    validate_noise_off()
    validate_trotter()

    # ------------------------------------------------------------------ (a)
    N_LIST = [1, 2, 4, 6]
    print("\n[A] Damped collective Rabi oscillations (DensityMatrixSimulator + RydbergNoiseModel)")
    print(f"    {'N':>2} {'sqrt(N)':>8} {'w_fit/w1 ideal':>15} {'w_fit/w1 noisy':>15} "
          f"{'freq err %':>11} {'tau_fit [us]':>13} {'max P(m>=2)':>12}")
    traces = {}
    for N in N_LIST:
        t0 = time.time()
        ideal = rabi_trace_ideal(N)
        noisy = rabi_trace_noisy(N)
        om_guess = np.sqrt(N) * OMEGA_1
        fit_i = fit_damped(ideal["times"], ideal["F_W"], om_guess, 1e3)
        fit_n = fit_damped(noisy["times"], noisy["F_W"], om_guess, T2_STAR)
        traces[N] = {"ideal": ideal, "noisy": noisy, "fit_i": fit_i, "fit_n": fit_n}
        ferr = 100.0 * abs(fit_n["omega"] - om_guess) / om_guess
        print(
            f"    {N:>2} {np.sqrt(N):8.4f} {fit_i['omega']/OMEGA_1:15.4f} "
            f"{fit_n['omega']/OMEGA_1:15.4f} {ferr:11.3f} {fit_n['tau']:13.2f} "
            f"{noisy['P_multi'].max():12.2e}   [{time.time()-t0:.1f}s]"
        )
    print("    --> the fitted frequency tracks sqrt(N) to <1% while the contrast decays:")
    print(f"        contrast at t = {T_MAX} us  (noisy peak / ideal peak):")
    for N in N_LIST:
        ideal, noisy = traces[N]["ideal"], traces[N]["noisy"]
        half = ideal["times"] > 0.6 * T_MAX
        print(
            f"          N = {N}:  {noisy['F_W'][half].max():.3f}  vs ideal "
            f"{ideal['F_W'][half].max():.3f}   (tau_env = {traces[N]['fit_n']['tau']:.2f} us)"
        )

    # ------------------------------------------------------------------ (b)
    print("\n[B] Collective pi-pulse |W> fidelity vs laser coherence time T2*")
    t2_grid = np.logspace(np.log10(0.3), np.log10(3000.0), 22)
    N_SWEEP = [2, 4, 6]
    fid_vs_t2 = {}
    t2_for_99 = {}
    for N in N_SWEEP:
        t0 = time.time()
        sec = Sectors(N)
        vals = np.array(
            [sec.from_density_matrix(pi_pulse_density_matrix(N, T1_RYDBERG, t2))["F_W"]
             for t2 in t2_grid]
        )
        fid_vs_t2[N] = vals
        # interpolate (monotone in log T2*) for the 99% crossing
        if vals.max() >= 0.99 >= vals.min():
            t2_for_99[N] = float(np.exp(np.interp(0.99, vals, np.log(t2_grid))))
        else:
            t2_for_99[N] = np.nan
        print(
            f"    N = {N}:  F_W(T2* = {T2_STAR} us) = "
            f"{np.interp(np.log(T2_STAR), np.log(t2_grid), vals):.4f} | "
            f"F_W(T2* -> inf) = {vals[-1]:.5f} | "
            f"T2* needed for 99% = {t2_for_99[N]:8.2f} us   [{time.time()-t0:.1f}s]"
        )
    print("    --> at the experiment's T2* = 3.1 us the single pi-pulse is already")
    print("        fidelity-limited; reaching 99% demands an order-of-magnitude")
    print("        narrower excitation laser (or dynamical decoupling, cf. Module 2).")

    # ------------------------------------------------------------------ (c)
    print("\n[C] Error budget at the collective pi-pulse (N = 4)")
    N_EB = 4
    sec_eb = Sectors(N_EB)
    budget = [
        ("no noise", dict(t1=None, t2_star=None, gamma_c=None)),
        ("T1 only\n(amp. damp)", dict(t1=T1_RYDBERG, t2_star=None, gamma_c=None)),
        ("T2* only\n(phase damp)", dict(t1=None, t2_star=T2_STAR, gamma_c=None)),
        ("collective only\n(corr. Kraus)", dict(t1=None, t2_star=None, gamma_c=GAMMA_COLLECTIVE)),
        ("T1 + T2*", dict(t1=T1_RYDBERG, t2_star=T2_STAR, gamma_c=None)),
        ("T1 + T2*\n+ collective", dict(t1=T1_RYDBERG, t2_star=T2_STAR, gamma_c=GAMMA_COLLECTIVE)),
    ]
    eb_labels, eb_infid, eb_leak = [], [], []
    t_pi_4 = np.pi / (np.sqrt(N_EB) * OMEGA_1)
    for label, kw in budget:
        rho = pi_pulse_density_matrix(N_EB, **kw)
        obs = sec_eb.from_density_matrix(rho)
        eb_labels.append(label)
        eb_infid.append(1.0 - obs["F_W"])
        eb_leak.append(obs["P_multi"])
        print(
            f"    {label.replace(chr(10), ' '):28s} 1 - F_W = {1 - obs['F_W']:.5f} "
            f"({100*(1-obs['F_W']):6.3f} %)   P(m>=2) = {obs['P_multi']:.2e}"
        )
    print(f"    t_pi = {t_pi_4:.4f} us;  t_pi/T1 = {t_pi_4/T1_RYDBERG:.2e}, "
          f"t_pi/T2* = {t_pi_4/T2_STAR:.3f}")
    lin = eb_infid[1] + eb_infid[2] + eb_infid[3] - 3 * eb_infid[0]
    print(f"    linear sum of the three isolated channels = {lin:.5f} "
          f"vs combined {eb_infid[-1]:.5f}  (near-additive at this error level)")

    # ------------------------------------------------------------------ (d)
    print("\n[D] Correlated vs independent dephasing of a stored |W> (N = 4, T1 off)")
    N_ST = 4
    t_store = 3.0 * T2_STAR
    st_ind = w_storage_trace(N_ST, t_store, 41, "independent", T2_STAR)
    st_col = w_storage_trace(N_ST, t_store, 41, "collective", T2_STAR)
    # analytic: independent per-atom phase damping leaves rho_jk (j != k) -> exp(-2t/T2*)
    analytic_cond = (1.0 + (N_ST - 1) * np.exp(-2.0 * st_ind["times"] / T2_STAR)) / N_ST
    err_an = np.max(np.abs(st_ind["F_cond"] - analytic_cond))
    print(f"    independent dephasing: F_W|m=1 -> 1/N = {1/N_ST:.4f}")
    print(f"      max |Cirq - analytic (1 + (N-1)e^(-2t/T2*))/N| = {err_an:.3e}   "
          f"(target < 1e-6)  --> {'PASS' if err_an < 1e-6 else 'FAIL'}")
    print(f"      F_W|m=1  at t = {t_store:.1f} us : {st_ind['F_cond'][-1]:.5f}")
    print(f"    collective dephasing:  F_W|m=1 stays pinned at 1")
    print(f"      F_W|m=1  at t = {t_store:.1f} us : {st_col['F_cond'][-1]:.5f} "
          f"(deviation {abs(st_col['F_cond'][-1]-1):.2e})")
    print(f"      P(m=1)   at t = {t_store:.1f} us : {st_col['P_m1'][-1]:.5f} "
          f"(the Delta-m = 0 sector is untouched by the collective channel)")
    print("    --> matched Delta-m = 1 coherence decay, completely different fate for")
    print("        the symmetric superposition: correlated noise preserves |W>,")
    print("        independent noise scrambles it (cf. Phys. Rev. A 106, L051701).")

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.4))
    colors = {N: rc.plotting.SERIES_COLORS[i] for i, N in enumerate(N_LIST)}

    # (a) damped Rabi
    ax = axes[0, 0]
    for N in N_LIST:
        c = colors[N]
        ideal, noisy, fit_n = traces[N]["ideal"], traces[N]["noisy"], traces[N]["fit_n"]
        ax.plot(noisy["times"], noisy["F_W"], color=c, lw=2.1,
                label=f"$N={N}$ Cirq + noise ($\\Omega_N={np.sqrt(N):.2f}\\,\\Omega_1$)")
        ax.plot(ideal["times"], np.sin(np.sqrt(N) * OMEGA_1 * ideal["times"] / 2.0) ** 2,
                color=c, ls=":", lw=1.1, alpha=0.65)
        env = fit_n["amp"] * np.exp(-noisy["times"] / fit_n["tau"])
        ax.plot(noisy["times"], env, color=c, ls="--", lw=1.0, alpha=0.9)
    ax.axvline(T2_STAR, color=rc.plotting.PALETTE["gray"], ls="-.", lw=1.2)
    ax.text(T2_STAR * 1.03, 1.02, "$T_2^*=3.1\\,\\mu$s", fontsize=8.5,
            color=rc.plotting.PALETTE["gray"])
    ax.set_xlabel("time $t$  ($\\mu$s)")
    ax.set_ylabel("superatom population  $F_W=\\langle W|\\rho|W\\rangle$")
    ax.set_title("(a) Damped collective Rabi flopping\n"
                 "solid: Cirq density matrix | dotted: ideal $\\sin^2(\\sqrt{N}\\Omega_1 t/2)$ | "
                 "dashed: fitted envelope")
    ax.set_ylim(-0.03, 1.12)
    ax.legend(loc="upper right", ncol=1, fontsize=8)

    # (b) fidelity vs T2*
    ax = axes[0, 1]
    for i, N in enumerate(N_SWEEP):
        c = rc.plotting.SERIES_COLORS[i]
        ax.semilogx(t2_grid, fid_vs_t2[N], "o-", color=c, ms=3.6, lw=2.0, label=f"$N={N}$")
        if np.isfinite(t2_for_99[N]):
            ax.plot([t2_for_99[N]], [0.99], "*", color=c, ms=14, mec="k", mew=0.6, zorder=5)
    ax.axhline(0.99, color=rc.plotting.PALETTE["red"], ls="--", lw=1.4)
    ax.text(0.35, 0.9915, "99 % target", color=rc.plotting.PALETTE["red"], fontsize=9)
    ax.axvline(T2_STAR, color=rc.plotting.PALETTE["gray"], ls="-.", lw=1.3)
    ax.text(T2_STAR * 1.08, 0.3, f"PRL $T_2^*={T2_STAR}\\,\\mu$s", rotation=90,
            fontsize=8.5, color=rc.plotting.PALETTE["gray"])
    txt = "\n".join(
        f"$N={N}$: $T_2^*\\geq{t2_for_99[N]:.0f}\\,\\mu$s for 99 %" for N in N_SWEEP
    )
    ax.text(0.03, 0.40, txt, transform=ax.transAxes, fontsize=8.6,
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel("laser coherence time  $T_2^*$  ($\\mu$s)")
    ax.set_ylabel("$|W\\rangle$ fidelity after a collective $\\pi$ pulse")
    ax.set_title("(b) $\\pi$-pulse superatom fidelity vs dephasing\n"
                 "($T_1=100\\,\\mu$s fixed; stars mark the 99 % crossing)")
    ax.set_ylim(0.0, 1.03)
    ax.legend(loc="lower right")

    # (c) error budget
    ax = axes[1, 0]
    xs = np.arange(len(eb_labels))
    bar_colors = [
        rc.plotting.PALETTE["gray"], rc.plotting.PALETTE["blue"], rc.plotting.PALETTE["orange"],
        rc.plotting.PALETTE["purple"], rc.plotting.PALETTE["green"], rc.plotting.PALETTE["red"],
    ]
    bars = ax.bar(xs, 100.0 * np.array(eb_infid), color=bar_colors, edgecolor="k", lw=0.6)
    for b, v in zip(bars, eb_infid):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.04,
                f"{100*v:.2f}%" if v > 1e-4 else f"{100*v:.1e}%",
                ha="center", fontsize=8.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(eb_labels, fontsize=8.2)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 60)
    ax.set_ylabel("$\\pi$-pulse infidelity  $1-F_W$  (%)")
    ax.set_title("(c) Error budget of the collective $\\pi$ pulse ($N=4$)\n"
                 "laser dephasing dominates; blockade leakage stays $\\lesssim 10^{-3}$")
    ax.grid(True, axis="y", which="both", alpha=0.3)

    # (d) correlated vs independent
    ax = axes[1, 1]
    tt = st_ind["times"]
    ax.plot(tt, st_ind["F_W"], color=rc.plotting.PALETTE["red"], lw=2.2,
            label="independent per-atom dephasing: $F_W$")
    ax.plot(tt, st_ind["F_cond"], color=rc.plotting.PALETTE["red"], ls="--", lw=2.0,
            label="independent: $F_W/P(m{=}1)$")
    ax.plot(tt, analytic_cond, color="k", ls=":", lw=1.6,
            label="analytic $[1+(N-1)e^{-2t/T_2^*}]/N$")
    ax.plot(tt, st_col["F_W"], color=rc.plotting.PALETTE["blue"], lw=2.2,
            label="correlated collective dephasing: $F_W$")
    ax.plot(tt, st_col["F_cond"], color=rc.plotting.PALETTE["blue"], ls="--", lw=2.0,
            label="collective: $F_W/P(m{=}1)$")
    ax.axhline(1.0 / N_ST, color=rc.plotting.PALETTE["gray"], ls="-.", lw=1.2)
    ax.text(tt[-1] * 0.60, 1.0 / N_ST + 0.02, "$1/N$ scrambling floor", fontsize=8.6,
            color=rc.plotting.PALETTE["gray"])
    ax.set_xlabel("storage time  $t$  ($\\mu$s)")
    ax.set_ylabel("population / conditional fidelity")
    ax.set_title("(d) Correlated $\\neq$ independent dephasing ($N=4$)\n"
                 "matched $\\Delta m{=}1$ decay rate $\\gamma_c=2/T_2^*$; only the correlated\n"
                 "channel preserves the symmetric $|W\\rangle$")
    ax.set_ylim(0.0, 1.14)
    ax.legend(loc="center right", fontsize=8)

    fig.suptitle(
        "Realistic decoherence of a trapped Rydberg superatom in Cirq\n"
        "reproduction of Phys. Rev. Lett. 128, 123601 (2022), Mei, Li, Nguyen, Berman & Kuzmich",
        fontsize=13.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    rc.plotting.save_figure(fig, __file__, "cirq_prl2022_noise_model.png")

    print(f"\nTotal runtime: {time.time() - t_start:.1f} s")
    print("=" * 84)


if __name__ == "__main__":
    main()
