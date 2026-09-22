"""
Adiabatic (chirped) versus resonant sqrt(N) pi-pulse preparation of a Rydberg |W> state.

Reference
---------
Y. Mei, Y. Li, H. Nguyen, P. R. Berman, and A. Kuzmich,
*Trapped Alkali-Metal Rydberg Qubit*, Phys. Rev. Lett. **128**, 123601 (2022).

Physics
-------
Inside a single blockade volume the driven ensemble is, to an excellent
approximation, a **two-level system**

    |G> = |g_1 ... g_N>                       (energy 0)
    |W> = N^(-1/2) sum_j |g...r_j...g>        (energy -Delta)

coupled with the collectively enhanced matrix element

    <W| (Omega/2) sum_j X_j |G> = sqrt(N) Omega / 2 = Omega_N / 2 .

The PRL prepares the superatom with a **resonant** ``pi`` pulse of area
``Omega_N t = pi``.  That is the fastest route, but the transferred population
``sin^2(Omega_N t / 2)`` is quadratically sensitive to any calibration error in
the pulse area -- a +/-20 % error in ``Omega_1`` (atom-number fluctuations,
intensity drift, position in the Gaussian beam) costs ~10 % of the fidelity.

The alternative is **adiabatic rapid passage (ARP)**: chirp the laser detuning
from far below to far above resonance,

    H(t) = (Omega(t)/2) sum_j X_j  -  Delta(t) sum_j n_j  +  V sum_{j<k} n_j n_k ,
    Delta: -Delta_max -> +Delta_max ,   Omega: 0 -> Omega_peak -> 0 ,

so that the system never leaves the instantaneous *lower* dressed state, which
is adiabatically connected to ``|G>`` at ``Delta = -Delta_max`` and to ``|W>`` at
``Delta = +Delta_max``.  The transfer then depends only on the sweep being slow
compared with the avoided-crossing gap ``Omega_N``, not on any pulse area -- the
fidelity develops a broad plateau in ``Omega``.  The price is duration.

What this module establishes
----------------------------
(a) the ``Omega(t)``, ``Delta(t)`` profiles and the population trajectory
    ``P_G -> P_W`` with the ``m >= 2`` leakage staying at the ``10^-3`` level,
(b) ``|W>`` fidelity versus sweep duration: the ARP converges to 1, the resonant
    ``pi`` pulse is duration-independent (until it gets so short that it breaks
    the blockade),
(c) ``|W>`` fidelity versus a Rabi-frequency calibration error: the resonant
    pulse is a narrow ``sin^2`` peak (99 % window ~ +/-6 %), the ARP is flat over
    tens of percent.

Validation performed and printed
--------------------------------
* Trotterised Cirq circuit vs a dense ``scipy.linalg.expm`` of the same
  Hamiltonian (fixes the ``cirq.ZPowGate`` sign of the detuning term).
* The fast ``cirq.unitary`` + ``matrix_power`` propagator vs an explicit
  end-to-end ``cirq.Simulator`` run of the fully expanded circuit.
* The full ``2^N`` Cirq result vs a numerically integrated ``2 x 2``
  ``{|G>, |W>}`` model with coupling ``sqrt(N) Omega``.
* The constant-``Omega`` linear chirp vs the closed-form **Landau-Zener**
  transfer probability ``1 - exp(-pi Omega_N^2 / (2 |dDelta/dt|))``.

Cirq APIs showcased
-------------------
* custom ``cirq.Gate`` with ``_decompose_`` for the detuning term
  (:class:`GlobalDetuningStep`, local to this module) alongside the shared
  :class:`rydberg_cirq.CollectiveLaserDriveStep` /
  :class:`rydberg_cirq.RydbergBlockadeStep`,
* ``cirq.unitary`` of a Trotter sub-step circuit + ``numpy.linalg.matrix_power``
  for time-dependent Hamiltonians whose coefficients are frozen inside a step,
* ``cirq.Simulator`` statevector cross-check of the fully expanded circuit,
* ``cirq.decompose`` / ``cirq.Circuit`` diagrams for the native gate lowering.

Units: ``Omega_1 = 1``, all times are in units of ``1 / Omega_1`` and all
frequencies in units of ``Omega_1``.

Run:  python manybody/cirq_adiabatic_w_state.py
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla

import cirq

import rydberg_cirq as rc
import rydberg_cirq.plotting  # noqa: F401  (registers the ``rc.plotting`` attribute)
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Parameters (Omega_1 = 1)
# --------------------------------------------------------------------------- #
N_ATOMS = 4
OMEGA_1 = 1.0            # peak single-atom Rabi frequency
V_VDW = 200.0            # blockade shift;  V / Omega_N = 100 for N = 4
DELTA_MAX = 20.0         # chirp half-range;  Delta_max << V so |m=2> never crosses
V_DT_TARGET = 0.25       # Trotter knob: keep V * dt_sub below this
SLICES_PER_UNIT_TIME = 10.0   # time slices per 1/Omega_1 (midpoint freezing of Omega, Delta)
N_OUTER_MIN, N_OUTER_MAX = 150, 900


def n_outer_for(T: float) -> int:
    """Number of midpoint slices: enough that ``Omega dt`` and ``Delta' dt^2`` stay small."""
    return int(np.clip(round(SLICES_PER_UNIT_TIME * T), N_OUTER_MIN, N_OUTER_MAX))


# --------------------------------------------------------------------------- #
# Local gate: the detuning term  -Delta * sum_j n_j
# --------------------------------------------------------------------------- #
class GlobalDetuningStep(cirq.Gate):
    """Trotter step of the global laser detuning, ``exp(-i H dt)`` with

        H_det = -Delta * sum_j n_j ,     n_j = |r><r|_j = (1 - Z_j)/2 .

    The propagator is ``prod_j diag(1, exp(+i Delta dt))``.  Since
    ``cirq.ZPowGate(exponent=e) = diag(1, exp(i pi e))``, the native lowering is a
    single ``ZPowGate`` per atom with ``e = Delta * dt / pi``.

    (Defined locally -- ``rydberg_cirq`` is shared read-only across agents.)
    """

    def __init__(self, num_qubits: int, delta: float, dt: float):
        super().__init__()
        self._n = int(num_qubits)
        self.delta = float(delta)
        self.dt = float(dt)

    def _num_qubits_(self) -> int:
        return self._n

    def _decompose_(self, qubits):
        exponent = (self.delta * self.dt) / np.pi
        for q in qubits:
            yield cirq.ZPowGate(exponent=exponent).on(q)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return ["Detune(D*dt)"] * self._n

    def __repr__(self) -> str:
        return f"GlobalDetuningStep({self._n}, {self.delta!r}, {self.dt!r})"


# --------------------------------------------------------------------------- #
# Pulse shapes
# --------------------------------------------------------------------------- #
def pulse_profile(protocol: str, t: np.ndarray, T: float, omega_scale: float = 1.0,
                  N: int = N_ATOMS) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Omega(t), Delta(t))`` for one of the three protocols.

    ``'arp'``        smooth ``sin^2`` envelope, linear chirp  (the practical ARP)
    ``'lz'``         constant ``Omega``, linear chirp         (textbook Landau-Zener)
    ``'resonant'``   ``Delta = 0`` and ``Omega`` set so the pulse area is exactly
                     ``pi`` over the *same* total duration ``T`` -- a fair
                     apples-to-apples comparison at fixed protocol time.
    """
    s = np.asarray(t, dtype=float) / T
    if protocol == "arp":
        omega = omega_scale * OMEGA_1 * np.sin(np.pi * s) ** 2
        delta = DELTA_MAX * (2.0 * s - 1.0)
    elif protocol == "lz":
        omega = omega_scale * OMEGA_1 * np.ones_like(s)
        delta = DELTA_MAX * (2.0 * s - 1.0)
    elif protocol == "resonant":
        omega = omega_scale * (np.pi / (np.sqrt(N) * T)) * np.ones_like(s)
        delta = np.zeros_like(s)
    else:  # pragma: no cover
        raise ValueError(protocol)
    return omega, delta


# --------------------------------------------------------------------------- #
# Cirq propagation
# --------------------------------------------------------------------------- #
def strang_substep_circuit(qubits, omega: float, delta: float, v_vdw: float,
                           ds: float) -> cirq.Circuit:
    """One second-order Strang sub-step ``exp(-i H ds)`` at frozen ``Omega, Delta``.

    ``H_diag = -Delta sum_j n_j + V sum_{j<k} n_j n_k`` is diagonal, so the
    detuning and blockade factors commute exactly and the only splitting error is
    between the drive and the diagonal part: ``O(ds^3)`` per sub-step.
    """
    n = len(qubits)
    return cirq.Circuit(
        rc.CollectiveLaserDriveStep(n, omega, ds / 2.0).on(*qubits),
        GlobalDetuningStep(n, delta, ds).on(*qubits),
        rc.RydbergBlockadeStep(n, v_vdw, ds).on(*qubits),
        rc.CollectiveLaserDriveStep(n, omega, ds / 2.0).on(*qubits),
    )


def run_sweep(protocol: str, T: float, N: int = N_ATOMS, omega_scale: float = 1.0,
              v_vdw: float = V_VDW, n_outer: int | None = None,
              delta_max: float = DELTA_MAX, record: bool = False) -> dict:
    """Propagate ``|g...g>`` through the chirped sweep with Cirq circuits.

    Within each of the ``n_outer`` slices ``Omega`` and ``Delta`` are frozen at
    their midpoint values, so the slice propagator is the ``n_sub``-th power of a
    single Strang sub-step unitary obtained from ``cirq.unitary``.  Using
    ``numpy.linalg.matrix_power`` instead of simulating ``n_sub`` copies of the
    circuit is *numerically identical* and ~3 orders of magnitude faster; the
    equivalence is verified in :func:`validate_fast_propagator`.
    """
    qubits = cirq.LineQubit.range(N)
    n_outer = n_outer_for(T) if n_outer is None else n_outer
    dt = T / n_outer
    n_sub = max(1, int(np.ceil(dt * v_vdw / V_DT_TARGET)))
    ds = dt / n_sub

    t_mid = (np.arange(n_outer) + 0.5) * dt
    omega_mid, delta_mid = pulse_profile(protocol, t_mid, T, omega_scale, N, delta_max)

    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0

    w = rc.w_state_vector(N)
    masks = rc.excitation_masks(N)
    hist = {"t": [0.0], "P_G": [1.0], "P_W": [0.0], "P_m1": [0.0], "P_multi": [0.0]}

    for k in range(n_outer):
        sub = strang_substep_circuit(qubits, omega_mid[k], delta_mid[k], v_vdw, ds)
        u_slice = np.linalg.matrix_power(cirq.unitary(sub), n_sub)
        psi = u_slice @ psi
        if record:
            p = np.abs(psi) ** 2
            hist["t"].append((k + 1) * dt)
            hist["P_G"].append(float(p[masks["m0"]].sum()))
            hist["P_W"].append(float(abs(np.vdot(w, psi)) ** 2))
            hist["P_m1"].append(float(p[masks["m1"]].sum()))
            hist["P_multi"].append(float(p[masks["multi"]].sum()))

    p = np.abs(psi) ** 2
    out = {
        "psi": psi,
        "F_W": float(abs(np.vdot(w, psi)) ** 2),
        "P_multi": float(p[masks["multi"]].sum()),
        "P_m1": float(p[masks["m1"]].sum()),
        "P_G": float(p[masks["m0"]].sum()),
        "n_sub": n_sub,
        "n_outer": n_outer,
    }
    if record:
        out["hist"] = {k: np.asarray(v) for k, v in hist.items()}
        out["t_grid"] = np.linspace(0.0, T, 400)
        out["omega_t"], out["delta_t"] = pulse_profile(protocol, out["t_grid"], T,
                                                       omega_scale, N, delta_max)
    return out


# --------------------------------------------------------------------------- #
# Reduced two-level (Landau-Zener) reference model
# --------------------------------------------------------------------------- #
def _two_level_propagator(g: float, delta: float, dt: float) -> np.ndarray:
    """Exact ``exp(-i H dt)`` for ``H = [[0, g], [g, -delta]]`` (closed form, no expm).

    Writing ``H = -(delta/2) I + (g sigma_x + (delta/2) sigma_z)`` and using
    ``exp(-i theta n.sigma) = cos(theta) I - i sin(theta) n.sigma``.
    """
    r = np.hypot(g, delta / 2.0)
    c, s = np.cos(r * dt), (np.sinc(r * dt / np.pi) * dt)  # s = sin(r dt)/r, safe at r -> 0
    u = np.array(
        [[c - 1j * s * (delta / 2.0), -1j * s * g],
         [-1j * s * g, c + 1j * s * (delta / 2.0)]],
        dtype=complex,
    )
    return np.exp(1j * delta * dt / 2.0) * u


def run_two_level(protocol: str, T: float, N: int = N_ATOMS, omega_scale: float = 1.0,
                  n_outer: int = 20000, delta_max: float = DELTA_MAX) -> float:
    """Numerically integrate the ``{|G>, |W>}`` model with coupling ``sqrt(N) Omega``.

    ``H_2(t) = [[0, Omega_N(t)/2], [Omega_N(t)/2, -Delta(t)]]``, propagated with an
    exact 2x2 matrix exponential per midpoint slice (no Trotter error, no blockade
    leakage, and a 100x finer time grid than the many-body run).
    """
    dt = T / n_outer
    t_mid = (np.arange(n_outer) + 0.5) * dt
    omega, delta = pulse_profile(protocol, t_mid, T, omega_scale, N, delta_max)
    g = np.sqrt(N) * omega / 2.0

    psi = np.array([1.0 + 0j, 0.0 + 0j])
    for k in range(n_outer):
        psi = _two_level_propagator(g[k], delta[k], dt) @ psi
    return float(abs(psi[1]) ** 2)


def landau_zener_probability(T: float, N: int = N_ATOMS, omega_scale: float = 1.0,
                             delta_max: float = DELTA_MAX) -> float:
    """Closed-form LZ transfer for a linear chirp at constant ``Omega``.

    Gap at the avoided crossing ``= Omega_N = sqrt(N) Omega``; sweep rate
    ``alpha = dDelta/dt = 2 Delta_max / T``.  The diabatic (failure) probability is
    ``exp(-2 pi (Omega_N/2)^2 / alpha)``.
    """
    omega_n = np.sqrt(N) * omega_scale * OMEGA_1
    alpha = 2.0 * delta_max / T
    return 1.0 - np.exp(-2.0 * np.pi * (omega_n / 2.0) ** 2 / alpha)
    """Propagate ``|g...g>`` through the chirped sweep with Cirq circuits.

    Within each of the ``n_outer`` slices ``Omega`` and ``Delta`` are frozen at
    their midpoint values, so the slice propagator is the ``n_sub``-th power of a
    single Strang sub-step unitary obtained from ``cirq.unitary``.  Using
    ``numpy.linalg.matrix_power`` instead of simulating ``n_sub`` copies of the
    circuit is *numerically identical* and ~3 orders of magnitude faster; the
    equivalence is verified in :func:`validate_fast_propagator`.
    """
    qubits = cirq.LineQubit.range(N)
    dt = T / n_outer
    n_sub = max(1, int(np.ceil(dt * v_vdw / V_DT_TARGET)))
    ds = dt / n_sub

    t_mid = (np.arange(n_outer) + 0.5) * dt
    omega_mid, delta_mid = pulse_profile(protocol, t_mid, T, omega_scale, N)

    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0

    w = rc.w_state_vector(N)
    masks = rc.excitation_masks(N)
    hist = {"t": [0.0], "P_G": [1.0], "P_W": [0.0], "P_m1": [0.0], "P_multi": [0.0]}

    for k in range(n_outer):
        sub = strang_substep_circuit(qubits, omega_mid[k], delta_mid[k], v_vdw, ds)
        u_slice = np.linalg.matrix_power(cirq.unitary(sub), n_sub)
        psi = u_slice @ psi
        if record:
            p = np.abs(psi) ** 2
            hist["t"].append((k + 1) * dt)
            hist["P_G"].append(float(p[masks["m0"]].sum()))
            hist["P_W"].append(float(abs(np.vdot(w, psi)) ** 2))
            hist["P_m1"].append(float(p[masks["m1"]].sum()))
            hist["P_multi"].append(float(p[masks["multi"]].sum()))

    p = np.abs(psi) ** 2
    out = {
        "psi": psi,
        "F_W": float(abs(np.vdot(w, psi)) ** 2),
        "P_multi": float(p[masks["multi"]].sum()),
        "P_m1": float(p[masks["m1"]].sum()),
        "P_G": float(p[masks["m0"]].sum()),
        "n_sub": n_sub,
        "n_outer": n_outer,
    }
    if record:
        out["hist"] = {k: np.asarray(v) for k, v in hist.items()}
        out["t_grid"] = np.linspace(0.0, T, 400)
        out["omega_t"], out["delta_t"] = pulse_profile(protocol, out["t_grid"], T,
                                                       omega_scale, N)
    return out


# --------------------------------------------------------------------------- #
# Reduced two-level (Landau-Zener) reference model
# --------------------------------------------------------------------------- #
def run_two_level(protocol: str, T: float, N: int = N_ATOMS, omega_scale: float = 1.0,
                  n_outer: int = 4000) -> float:
    """Numerically integrate the ``{|G>, |W>}`` model with coupling ``sqrt(N) Omega``.

    ``H_2(t) = [[0, Omega_N(t)/2], [Omega_N(t)/2, -Delta(t)]]``, propagated with an
    exact 2x2 ``expm`` per midpoint slice (no Trotter error, no blockade leakage).
    """
    dt = T / n_outer
    t_mid = (np.arange(n_outer) + 0.5) * dt
    omega, delta = pulse_profile(protocol, t_mid, T, omega_scale, N)
    omega_n = np.sqrt(N) * omega

    psi = np.array([1.0 + 0j, 0.0 + 0j])
    for k in range(n_outer):
        h = np.array([[0.0, omega_n[k] / 2.0], [omega_n[k] / 2.0, -delta[k]]], dtype=complex)
        psi = sla.expm(-1j * h * dt) @ psi
    return float(abs(psi[1]) ** 2)


def landau_zener_probability(T: float, N: int = N_ATOMS, omega_scale: float = 1.0) -> float:
    """Closed-form LZ transfer for a linear chirp at constant ``Omega``.

    Gap at the avoided crossing ``= Omega_N = sqrt(N) Omega``; sweep rate
    ``alpha = dDelta/dt = 2 Delta_max / T``.  The diabatic (failure) probability is
    ``exp(-2 pi (Omega_N/2)^2 / alpha)``.
    """
    omega_n = np.sqrt(N) * omega_scale * OMEGA_1
    alpha = 2.0 * DELTA_MAX / T
    return 1.0 - np.exp(-2.0 * np.pi * (omega_n / 2.0) ** 2 / alpha)


# --------------------------------------------------------------------------- #
# Validations
# --------------------------------------------------------------------------- #
def validate_trotter_vs_expm(N: int = 4) -> None:
    print("\n[VALIDATION 1] Strang sub-step circuit vs dense expm (fixes the ZPowGate sign)")
    qubits = cirq.LineQubit.range(N)
    omega, delta, v, t = 0.83, -3.7, V_VDW, 0.4

    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    nop = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)

    def emb(op, j):
        mats = [np.eye(2, dtype=complex)] * N
        mats[j] = op
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = sum((omega / 2.0) * emb(X, j) for j in range(N))
    H = H - delta * sum(emb(nop, j) for j in range(N))
    for j in range(N):
        for k in range(j + 1, N):
            H = H + v * emb(nop, j) @ emb(nop, k)

    n_sub = int(np.ceil(t * v / 0.02))
    ds = t / n_sub
    u_cirq = np.linalg.matrix_power(
        cirq.unitary(strang_substep_circuit(qubits, omega, delta, v, ds)), n_sub
    )
    u_exact = sla.expm(-1j * H * t)
    # remove the (physically irrelevant) global phase before comparing
    phase = np.vdot(u_exact.ravel(), u_cirq.ravel())
    phase /= abs(phase)
    err = np.max(np.abs(u_cirq - phase * u_exact))
    print(f"  N = {N}, Omega = {omega}, Delta = {delta}, V = {v}, t = {t}, {n_sub} sub-steps")
    print(f"  max |U_cirq - e^(i phi) U_expm|     = {err:.3e}   (target < 1e-6)")
    print(f"  --> {'PASS' if err < 1e-6 else 'FAIL'}")


def validate_fast_propagator() -> None:
    print("\n[VALIDATION 2] cirq.unitary + matrix_power vs an explicit cirq.Simulator run")
    N, T, n_outer = N_ATOMS, 4.0, 20
    qubits = cirq.LineQubit.range(N)
    dt = T / n_outer
    n_sub = max(1, int(np.ceil(dt * V_VDW / V_DT_TARGET)))
    ds = dt / n_sub
    t_mid = (np.arange(n_outer) + 0.5) * dt
    omega_mid, delta_mid = pulse_profile("arp", t_mid, T, 1.0, N)

    full = cirq.Circuit()
    for k in range(n_outer):
        for _ in range(n_sub):
            full += strang_substep_circuit(qubits, omega_mid[k], delta_mid[k], V_VDW, ds)
    psi_sim = np.asarray(
        cirq.Simulator(dtype=np.complex128).simulate(full, qubit_order=qubits).final_state_vector
    )
    psi_fast = run_sweep("arp", T, N, n_outer=n_outer)["psi"]
    fid = abs(np.vdot(psi_sim, psi_fast)) ** 2
    print(f"  explicit circuit: {len(full)} moments, "
          f"{len(list(cirq.decompose(full)))} native ops after cirq.decompose")
    print(f"  1 - |<psi_Simulator|psi_fast>|^2    = {1 - fid:.3e}   (target < 1e-9)")
    print(f"  --> {'PASS' if 1 - fid < 1e-9 else 'FAIL'}")


def validate_two_level(durations) -> None:
    print("\n[VALIDATION 3] full 2^N Cirq sweep vs the reduced {|G>,|W>} two-level model")
    print(f"    {'T [1/Om1]':>10} {'F_W (Cirq)':>12} {'F_W (2-level)':>14} "
          f"{'|diff|':>10} {'P(m>=2)':>10}")
    worst = 0.0
    for T in durations:
        res = run_sweep("arp", T)
        f2 = run_two_level("arp", T)
        d = abs(res["F_W"] - f2)
        worst = max(worst, d)
        print(f"    {T:10.1f} {res['F_W']:12.6f} {f2:14.6f} {d:10.2e} {res['P_multi']:10.2e}")
    print(f"  worst deviation = {worst:.3e}  (limited by the finite blockade,")
    print(f"  expected scale (Omega_N/2V)^2 = {(np.sqrt(N_ATOMS)/(2*V_VDW))**2:.2e})")
    print(f"  --> {'PASS' if worst < 5e-3 else 'FAIL'}")


def validate_landau_zener(durations) -> None:
    print("\n[VALIDATION 4] constant-Omega linear chirp vs the closed-form Landau-Zener law")
    print(f"    Omega_N = sqrt(N) Omega_1 = {np.sqrt(N_ATOMS)*OMEGA_1:.3f}, "
          f"alpha = 2 Delta_max / T")
    print(f"    {'T [1/Om1]':>10} {'F_W (Cirq)':>12} {'1-exp(-pi Om_N^2/2alpha)':>26} "
          f"{'|diff|':>10}")
    for T in durations:
        res = run_sweep("lz", T)
        lz = landau_zener_probability(T)
        print(f"    {T:10.1f} {res['F_W']:12.6f} {lz:26.6f} {abs(res['F_W']-lz):10.2e}")
    print("  (the residual is the finite-Delta_max start/stop transient, which decays")
    print("   as (Omega_N / 2 Delta_max)^2 = "
          f"{(np.sqrt(N_ATOMS)*OMEGA_1/(2*DELTA_MAX))**2:.2e}; the asymptotic trend is reproduced)")


# --------------------------------------------------------------------------- #
# Main study
# --------------------------------------------------------------------------- #
def main() -> None:
    rc.plotting.apply_style()
    t_start = time.time()
    N = N_ATOMS
    omega_n = np.sqrt(N) * OMEGA_1

    print("=" * 88)
    print("Adiabatic chirped sweep vs resonant sqrt(N) pi-pulse preparation of |W>")
    print("Phys. Rev. Lett. 128, 123601 (2022) -- Trapped Alkali-Metal Rydberg Qubit")
    print("=" * 88)
    print(f"  N = {N} atoms,  Omega_1 = {OMEGA_1},  Omega_N = sqrt(N) Omega_1 = {omega_n:.4f}")
    print(f"  V_vdW = {V_VDW}  (V / Omega_N = {V_VDW/omega_n:.1f}),  "
          f"Delta_max = {DELTA_MAX} (= {DELTA_MAX/omega_n:.1f} Omega_N)")
    print(f"  resonant pi-pulse time t_pi = pi / Omega_N = {np.pi/omega_n:.4f} / Omega_1")

    # Show the Cirq circuit for one Trotter sub-step.
    q = cirq.LineQubit.range(3)
    demo = strang_substep_circuit(q, 0.8, -2.0, 20.0, 0.05)
    print("\n[Composite Cirq sub-step circuit, N = 3]")
    print(demo)
    print("\n[Lowered to the native gateset via cirq.decompose]")
    print(cirq.Circuit(cirq.decompose(demo)))

    validate_trotter_vs_expm()
    validate_fast_propagator()
    validate_two_level([6.0, 20.0, 45.0])
    validate_landau_zener([6.0, 12.0, 24.0, 48.0])

    # ------------------------------------------------------------------ (a)
    T_SHOW = 45.0
    print(f"\n[A] Population trajectory through the chirped sweep (T = {T_SHOW}/Omega_1)")
    traj = run_sweep("arp", T_SHOW, record=True)
    h = traj["hist"]
    print(f"    Trotter: {traj['n_outer']} slices x {traj['n_sub']} Strang sub-steps "
          f"= {traj['n_outer']*traj['n_sub']} sub-steps, V*ds = "
          f"{V_VDW*T_SHOW/traj['n_outer']/traj['n_sub']:.3f}")
    print(f"    final  P_G = {traj['P_G']:.6f}   P_W = {traj['F_W']:.6f}   "
          f"P(m=1) = {traj['P_m1']:.6f}   P(m>=2) = {traj['P_multi']:.3e}")
    print(f"    max leakage P(m>=2) along the whole sweep = {h['P_multi'].max():.3e}")
    print(f"    W purity inside the m=1 sector at the end = "
          f"{traj['F_W']/max(traj['P_m1'],1e-15):.6f}  (blockade holds throughout)")

    # ------------------------------------------------------------------ (b)
    print("\n[B] |W> fidelity vs protocol duration: adiabatic vs resonant pi pulse")
    durations = np.array([1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0, 16.0, 22.0,
                          30.0, 40.0, 55.0, 75.0, 100.0])
    f_arp, f_lz, f_res, leak_res, leak_arp = [], [], [], [], []
    for T in durations:
        a = run_sweep("arp", T)
        l = run_sweep("lz", T)
        r = run_sweep("resonant", T)
        f_arp.append(a["F_W"]); leak_arp.append(a["P_multi"])
        f_lz.append(l["F_W"])
        f_res.append(r["F_W"]); leak_res.append(r["P_multi"])
    f_arp, f_lz, f_res = map(np.asarray, (f_arp, f_lz, f_res))
    leak_res, leak_arp = np.asarray(leak_res), np.asarray(leak_arp)
    print(f"    {'T':>7} {'F_W ARP':>10} {'F_W LZ':>10} {'F_W pi-pulse':>13} "
          f"{'leak ARP':>10} {'leak pi':>10}")
    for i, T in enumerate(durations):
        print(f"    {T:7.1f} {f_arp[i]:10.6f} {f_lz[i]:10.6f} {f_res[i]:13.6f} "
              f"{leak_arp[i]:10.2e} {leak_res[i]:10.2e}")
    ok99 = durations[f_arp >= 0.99]
    ok999 = durations[f_arp >= 0.999]
    T99 = float(ok99[0]) if ok99.size else np.nan
    T999 = float(ok999[0]) if ok999.size else np.nan
    print(f"    --> ARP reaches 99%  fidelity for T >~ {T99:.0f}/Omega_1 "
          f"({T99*omega_n/np.pi:.0f} x t_pi)")
    print(f"    --> ARP reaches 99.9% fidelity for T >~ {T999:.0f}/Omega_1 "
          f"({T999*omega_n/np.pi:.0f} x t_pi)")
    print(f"    --> the resonant pi pulse is essentially duration-independent "
          f"(F_W = {f_res[-1]:.6f} at T = {durations[-1]:.0f}) but collapses at short T")
    print(f"        because Omega = pi/(sqrt(N) T) then rivals V: "
          f"F_W = {f_res[0]:.4f}, leakage = {leak_res[0]:.2e} at T = {durations[0]:.1f}")

    # ------------------------------------------------------------------ (c)
    print("\n[C] Robustness to a Rabi-frequency / pulse-area calibration error")
    eps = np.linspace(-0.35, 0.35, 29)
    T_ROB = [20.0, 45.0]
    rob_arp = {T: np.array([run_sweep("arp", T, omega_scale=1 + e)["F_W"] for e in eps])
               for T in T_ROB}
    T_RES = 45.0
    rob_res = np.array([run_sweep("resonant", T_RES, omega_scale=1 + e)["F_W"] for e in eps])
    analytic_res = np.sin((1.0 + eps) * np.pi / 2.0) ** 2

    def width99(curve):
        good = np.abs(eps[curve >= 0.99])
        return 2.0 * good.max() if good.size else 0.0

    print(f"    {'eps':>7} " + " ".join(f"{'ARP T='+str(int(T)):>12}" for T in T_ROB)
          + f" {'pi pulse':>10} {'sin^2 theory':>13}")
    for i, e in enumerate(eps):
        if i % 3:
            continue
        row = " ".join(f"{rob_arp[T][i]:12.6f}" for T in T_ROB)
        print(f"    {e:+7.2f} {row} {rob_res[i]:10.6f} {analytic_res[i]:13.6f}")
    print(f"    max |pi-pulse Cirq - sin^2((1+eps) pi/2)| = "
          f"{np.max(np.abs(rob_res - analytic_res)):.3e}  (validates the pulse-area model)")
    for T in T_ROB:
        print(f"    99% robustness window, ARP (T = {T:.0f}/Omega_1): "
              f"Delta eps = +/-{width99(rob_arp[T])/2*100:5.1f} %")
    print(f"    99% robustness window, resonant pi pulse:    "
          f"Delta eps = +/-{width99(rob_res)/2*100:5.1f} %")
    best = max(T_ROB, key=lambda T: width99(rob_arp[T]))
    ratio = width99(rob_arp[best]) / max(width99(rob_res), 1e-9)
    print(f"    --> the adiabatic protocol tolerates {ratio:.1f}x more miscalibration")
    print("        than the resonant pi pulse at the 99% level -- the entire point of ARP.")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.26)
    P = rc.plotting.PALETTE

    # (a) profiles + trajectory
    ax = fig.add_subplot(gs[0, :])
    ax2 = ax.twinx()
    ax2.plot(traj["t_grid"], traj["omega_t"] / OMEGA_1, color=P["gray"], lw=1.8, ls="-")
    ax2.fill_between(traj["t_grid"], 0, traj["omega_t"] / OMEGA_1, color=P["gray"], alpha=0.13)
    ax2.plot(traj["t_grid"], traj["delta_t"] / DELTA_MAX, color=P["yellow"], lw=1.8, ls="-.")
    ax2.set_ylabel("$\\Omega(t)/\\Omega_1$   and   $\\Delta(t)/\\Delta_{\\max}$",
                   color="0.35")
    ax2.set_ylim(-1.15, 1.35)
    ax2.grid(False)
    ax2.tick_params(axis="y", colors="0.35")

    ax.plot(h["t"], h["P_G"], color=P["blue"], lw=2.4, label="$P(|g\\ldots g\\rangle)$")
    ax.plot(h["t"], h["P_W"], color=P["red"], lw=2.6, label="$F_W=|\\langle W|\\psi\\rangle|^2$")
    ax.plot(h["t"], h["P_m1"], color=P["green"], lw=1.4, ls="--", label="$P(m=1)$")
    ax.plot(h["t"], 1e3 * h["P_multi"], color=P["purple"], lw=1.6, ls=":",
            label="$10^3\\times P(m\\geq 2)$ (blockade leakage)")
    ax.axvline(T_SHOW / 2, color="0.5", lw=1.0, ls=":")
    ax.text(T_SHOW / 2 + 0.6, 0.55, "$\\Delta=0$\navoided crossing", fontsize=8.5, color="0.35")
    ax.set_xlabel("time  $t$  (units of $1/\\Omega_1$)")
    ax.set_ylabel("population")
    ax.set_ylim(-0.04, 1.14)
    ax.set_title(f"(a) Chirped adiabatic rapid passage $|g\\ldots g\\rangle\\to|W\\rangle$, "
                 f"$N={N}$, $T={T_SHOW:.0f}/\\Omega_1$\n"
                 "grey: $\\Omega(t)$ envelope,  yellow dash-dot: linear detuning chirp "
                 f"$\\Delta:-{DELTA_MAX:.0f}\\to+{DELTA_MAX:.0f}\\,\\Omega_1$")
    ax.legend(loc="center left", fontsize=9)

    # (b) fidelity vs duration
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(durations, f_arp, "o-", color=P["red"], ms=5,
                label="adiabatic chirp (ARP, $\\sin^2$ envelope)")
    ax.semilogx(durations, f_lz, "s--", color=P["orange"], ms=4, lw=1.6,
                label="linear chirp, constant $\\Omega$ (Landau-Zener)")
    tt = np.logspace(np.log10(durations[0]), np.log10(durations[-1]), 200)
    ax.semilogx(tt, [landau_zener_probability(T) for T in tt], ":", color="k", lw=1.5,
                label="LZ law $1-e^{-\\pi\\Omega_N^2/2\\alpha}$")
    ax.semilogx(durations, f_res, "^-", color=P["blue"], ms=5,
                label="resonant $\\sqrt{N}$ $\\pi$ pulse (area $\\pi$ in time $T$)")
    ax.axhline(0.99, color=P["gray"], ls="--", lw=1.2)
    ax.text(durations[0] * 1.05, 0.955, "99 %", fontsize=8.6, color=P["gray"])
    if np.isfinite(T99):
        ax.plot([T99], [f_arp[durations == T99][0]], "*", color=P["red"], ms=15,
                mec="k", mew=0.6, zorder=6)
    ax.set_xlabel("protocol duration  $T$  (units of $1/\\Omega_1$)")
    ax.set_ylabel("$|W\\rangle$ fidelity")
    ax.set_ylim(0.0, 1.06)
    ax.set_title("(b) Fidelity vs duration: adiabaticity has a speed cost\n"
                 "(the $\\pi$ pulse is fast but exact only if perfectly calibrated)")
    ax.legend(loc="lower right", fontsize=8.2)

    # (c) robustness
    ax = fig.add_subplot(gs[1, 1])
    for i, T in enumerate(T_ROB):
        ax.plot(100 * eps, rob_arp[T], "-o", ms=3.6,
                color=[P["red"], P["purple"]][i],
                label=f"adiabatic chirp, $T={T:.0f}/\\Omega_1$")
    ax.plot(100 * eps, rob_res, "-^", color=P["blue"], ms=3.6,
            label=f"resonant $\\pi$ pulse, $T={T_RES:.0f}/\\Omega_1$")
    ax.plot(100 * eps, analytic_res, ":", color="k", lw=1.5,
            label="$\\sin^2[(1+\\epsilon)\\pi/2]$ (pulse-area theory)")
    ax.axhline(0.99, color=P["gray"], ls="--", lw=1.2)
    ax.axvspan(-100 * width99(rob_res) / 2, 100 * width99(rob_res) / 2,
               color=P["blue"], alpha=0.10)
    ax.axvspan(-100 * width99(rob_arp[best]) / 2, 100 * width99(rob_arp[best]) / 2,
               color=P["red"], alpha=0.07)
    ax.text(0.03, 0.06,
            f"99 % window:\n  $\\pi$ pulse  $\\pm${100*width99(rob_res)/2:.1f} %\n"
            f"  ARP ($T={best:.0f}$)  $\\pm${100*width99(rob_arp[best])/2:.1f} %",
            transform=ax.transAxes, fontsize=8.8,
            bbox=dict(fc="white", ec="0.7", alpha=0.92))
    ax.set_xlabel("Rabi-frequency calibration error  $\\epsilon$  (%)   "
                  "[$\\Omega\\to(1+\\epsilon)\\Omega$]")
    ax.set_ylabel("$|W\\rangle$ fidelity")
    ax.set_ylim(0.0, 1.06)
    ax.set_title("(c) Robustness: the adiabatic plateau\n"
                 "resonant transfer is a narrow $\\sin^2$ peak, ARP is flat")
    ax.legend(loc="lower center", fontsize=8.2)

    fig.suptitle(
        "Adiabatic vs resonant preparation of a Rydberg superatom $|W\\rangle$ state in Cirq\n"
        "reproduction of Phys. Rev. Lett. 128, 123601 (2022), Mei, Li, Nguyen, Berman & Kuzmich",
        fontsize=13.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    rc.plotting.save_figure(fig, __file__, "cirq_prl2022_adiabatic_w_state.png")

    print(f"\nTotal runtime: {time.time() - t_start:.1f} s")
    print("=" * 88)


if __name__ == "__main__":
    main()
