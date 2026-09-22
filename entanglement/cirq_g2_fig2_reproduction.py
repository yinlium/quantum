r"""
Cirq reproduction of Fig. 2 of Phys. Rev. A 106, L051701 (2022)
===============================================================

    Y. Li, Y. Mei, H. Nguyen, P. R. Berman, and A. Kuzmich,
    "Dynamics of collective-dephasing-induced multi-atom entanglement",
    Phys. Rev. A **106**, L051701 (2022) [+ Supplemental Material].

Physics
-------
A short two-photon pulse leaves the ensemble in an *unentangled* product state,
which in the collective spin-wave basis reads ``|Psi_0> = sum_m c_m |m>`` with
Poissonian ``|c_m|^2`` (mean ``m_bar``).  During the storage interval ``T_s`` the
Rydberg pairs dephase under

    H_c = sum_{mu<nu} hbar kappa_{mu nu} n_mu n_nu ,   kappa_{mu nu} = C6 / R_{mu nu}^6,
    U(T_s) = prod_{mu<nu} exp(-i Phi_{mu nu} n_mu n_nu),   Phi_{mu nu} = kappa_{mu nu} T_s.

The ``m = 0`` and ``m = 1`` sectors are *exactly* untouched (a lone Rydberg atom has
no partner), while every ``m >= 2`` sector acquires position-dependent random phases.
The phase-matched retrieval therefore loses its multi-excitation content and the
retrieved field becomes antibunched:

    g2(T_s) = sum_{m>=2} |c_m|^2 m(m-1) X_m(T_s) / ( sum_{m>=1} |c_m|^2 m Y_m(T_s) )^2

with the overlap factors (Supplemental Material)

    X_m = <m| U^dag S^dag S^dag S S U |m> / [m(m-1)],
    Y_m = <m| U^dag S^dag S U |m> / m,          S = (1/sqrt(N)) sum_j sigma_-^(j).

What this script does
---------------------
1.  **Exact Cirq circuits** on a statevector-simulable register (``N_MICRO = 12``
    qubits sampled from the *same* Gaussian cloud as the real ensemble, so the
    pair-separation statistics -- and hence the two-atom coherence ``eta`` -- are
    identical) give ``X_2(T_s)`` and ``Y_2(T_s)`` with no approximation.
2.  The elementary coherence is inverted from the ``m = 2`` sector,

        eta_X = (X_2 - 2/N^2) / (1 - 1/N - 2/N^2),   eta_Y = (Y_2 - 1/N) / (1 - 2/N),

    and Eqs. (S.7)/(S.8) extrapolate to the macroscopic cloud (``N = 270``):

        X_m = [(N-m)^2 + 3(N-m)]/N^2 * eta_X^(2m-3) + 2/N^2,
        Y_m = (N-m)/N * eta_Y^(m-1) + 1/N.
3.  ``g2(T_s)`` is assembled with Poissonian ``|c_m|^2`` up to ``m = 15`` and
    averaged over >= 150 independent cloud realizations; the shaded band is the
    quoted +-20% experimental uncertainty on the longitudinal cloud size ``sigma_z``.

Cirq APIs showcased
-------------------
* ``sympy.Symbol('T_s')``-**parametrized** ``cirq.CZPowGate`` exponents -- *one*
  circuit per cloud realization, swept over storage time with ``cirq.Points`` /
  ``cirq.Linspace`` / ``cirq.ParamResolver`` via ``Simulator.simulate_sweep``
  (a ~4x speed-up over rebuilding a circuit per time point, printed at run time).
* ``rydberg_cirq.gates.PairwisePhaseGate`` -- the exact (Trotter-error-free)
  storage propagator, used as the reference the symbolic circuit is checked against.
* ``rydberg_cirq.cloud`` (``sample_cloud``, ``phase_matrix``, ``coherence_eta``),
  ``rydberg_cirq.dicke`` (``dicke_state_vector``, ``poisson_excitation_amplitudes``),
  ``rydberg_cirq.metrics.g2_from_populations``, ``rydberg_cirq.plotting``.

Run:  python cirq_g2_fig2_reproduction.py      (figure: cirq_g2_fig2_reproduction.png)
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np
import sympy
import cirq

import rydberg_cirq as rc
from rydberg_cirq.plotting import PALETTE, apply_style, save_figure

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Experimental configuration (Phys. Rev. A 106, L051701, main text + Supplemental)
# --------------------------------------------------------------------------------------
N_MACRO = 270          # atoms in the excitation volume
N_MICRO = 12           # qubits in the exactly simulated Cirq register
M_MAX = 15             # highest excitation number kept in the g2 sums
SIGMA_Z_UNCERTAINTY = 0.20   # +-20% experimental uncertainty on sigma_z

TS_SYMBOL = sympy.Symbol("T_s")

# The four experimentally relevant (cloud length, principal quantum number) combinations.
SHORT_N50 = rc.SHORT_CLOUD                                   # n=50, C6=15.44, sigma_z=5.25
SHORT_N40 = dataclasses.replace(                             # n=40, C6=1.00,  sigma_z=5.25
    rc.SHORT_CLOUD, c6=1.00, m_bar=1.63, label="short cloud (n=40)"
)
LONG_N40 = rc.LONG_CLOUD                                     # n=40, C6=1.00,  sigma_z=115
LONG_N50 = dataclasses.replace(                              # n=50, C6=15.44, sigma_z=115
    rc.LONG_CLOUD, c6=15.44, m_bar=0.79, label="long cloud (n=50)"
)

# Experimental data digitised from Fig. 2 (see reproduce_paper_exact.py in this folder).
EXP = {
    "short_n40": dict(
        ts=np.array([0.4, 0.8, 1.2, 3.1, 5.1, 8.2, 10.1, 15.1, 20.2, 25.0]),
        g2=np.array([0.89, 0.85, 0.82, 0.79, 0.76, 0.69, 0.63, 0.67, 0.63, 0.59]),
        err=np.array([0.06, 0.05, 0.05, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06]),
    ),
    "short_n50": dict(
        ts=np.array([0.6, 1.1, 3.1, 5.1, 8.2, 10.1, 15.1, 20.2, 25.0]),
        g2=np.array([0.53, 0.56, 0.34, 0.23, 0.20, 0.18, 0.15, 0.06, 0.11]),
        err=np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.04, 0.07]),
    ),
    "long_n50": dict(
        ts=np.array([0.5, 1.0, 5.1, 10.1, 15.1, 20.2]),
        g2=np.array([1.08, 0.98, 1.01, 1.01, 1.07, 1.06]),
        err=np.array([0.09, 0.08, 0.12, 0.11, 0.13, 0.21]),
    ),
}


# --------------------------------------------------------------------------------------
# Local helpers (defined here rather than in rydberg_cirq/ so the shared package
# is left untouched).
# --------------------------------------------------------------------------------------
def spinwave_lowering_tables(num_qubits: int):
    """Index tables implementing ``S = (1/sqrt N) sum_j sigma_-^(j)`` on a statevector.

    ``sigma_-^(j) = |g_j><r_j|`` flips ``|1> -> |0>`` on qubit ``j``.  Cirq orders the
    statevector with qubit 0 as the most significant bit.
    """
    dim = 1 << num_qubits
    idx = np.arange(dim)
    tables = []
    for j in range(num_qubits):
        bit = 1 << (num_qubits - 1 - j)
        src = idx[(idx & bit) != 0]
        tables.append((src, src ^ bit))
    return tables


def apply_spinwave_lowering(state: np.ndarray, tables, num_qubits: int) -> np.ndarray:
    """Return ``S |state>`` (phase-matched collective annihilation, ``k_0 . r`` gauged out)."""
    out = np.zeros_like(state)
    for src, dst in tables:
        out[dst] += state[src]
    return out / np.sqrt(num_qubits)


def build_parametrized_storage_circuit(rate_matrix: np.ndarray, qubits, symbol) -> cirq.Circuit:
    """One ``T_s``-parametrized circuit for ``U(T_s) = prod exp(-i kappa_{jk} T_s n_j n_k)``.

    ``rate_matrix[j, k] = kappa_{jk}`` is the *phase rate* (rad/us); the symbolic
    ``CZPowGate`` exponent ``-kappa_{jk} T_s / pi`` is what makes a single circuit
    sweepable over the whole storage-time axis.
    """
    n = rate_matrix.shape[0]
    circuit = cirq.Circuit()
    for j in range(n):
        for k in range(j + 1, n):
            rate = float(rate_matrix[j, k])
            if rate != 0.0:
                circuit.append(
                    cirq.CZPowGate(exponent=-rate * symbol / np.pi).on(qubits[j], qubits[k])
                )
    return circuit


def x2_y2_from_states(states, tables, num_qubits: int):
    """Exact ``X_2`` and ``Y_2`` from a list of evolved ``|m=2>`` statevectors."""
    x2 = np.empty(len(states))
    y2 = np.empty(len(states))
    for i, psi in enumerate(states):
        s1 = apply_spinwave_lowering(np.asarray(psi, dtype=complex), tables, num_qubits)
        s2 = apply_spinwave_lowering(s1, tables, num_qubits)
        y2[i] = float(np.vdot(s1, s1).real) / 2.0          # / m
        x2[i] = float(np.vdot(s2, s2).real) / 2.0          # / [m(m-1)]
    return x2, y2


def x2_y2_numpy(rate_matrix: np.ndarray, storage_times: np.ndarray):
    """Closed-form reference for the ``m = 2`` overlaps (no quantum circuit involved).

    Starting from ``|2> = binom(N,2)^(-1/2) sum_{mu<nu} |mu nu>`` one finds

        X_2 = |sum_{mu != nu} e^{-i Phi_{mu nu}}|^2 / [N^3 (N-1)],
        Y_2 = sum_nu |sum_{mu != nu} e^{-i Phi_{mu nu}}|^2 / [N^2 (N-1)].
    """
    n = rate_matrix.shape[0]
    x2 = np.empty(len(storage_times))
    y2 = np.empty(len(storage_times))
    off = ~np.eye(n, dtype=bool)
    for i, ts in enumerate(storage_times):
        e = np.exp(-1j * rate_matrix * ts) * off
        a = e.sum(axis=1)
        x2[i] = np.abs(a.sum()) ** 2 / (n**3 * (n - 1))
        y2[i] = float(np.sum(np.abs(a) ** 2)) / (n**2 * (n - 1))
    return x2, y2


def eta_from_x2(x2: np.ndarray, n: int) -> np.ndarray:
    """Invert Eq. (S.7) at ``m = 2``:  ``eta = (X_2 - 2/N^2) / (1 - 1/N - 2/N^2)``."""
    return np.clip((x2 - 2.0 / n**2) / (1.0 - 1.0 / n - 2.0 / n**2), 0.0, 1.0)


def eta_from_y2(y2: np.ndarray, n: int) -> np.ndarray:
    """Invert Eq. (S.8) at ``m = 2``:  ``eta = (Y_2 - 1/N) / (1 - 2/N)``."""
    return np.clip((y2 - 1.0 / n) / (1.0 - 2.0 / n), 0.0, 1.0)


def scaling_ansatz_xy(eta_x: float, eta_y: float, n: int, m_max: int):
    """Eqs. (S.7)/(S.8): overlap factors of the macroscopic ensemble.

    ``X_m`` is irrelevant for ``m < 2`` (the ``m(m-1)`` weight vanishes) and is set to
    zero to keep the ``eta^(2m-3)`` power from diverging as ``eta -> 0``.
    """
    m = np.arange(m_max + 1)
    x_m = np.zeros(m_max + 1)
    m2 = m[2:]
    x_m[2:] = ((n - m2) ** 2 + 3.0 * (n - m2)) / n**2 * eta_x ** (2 * m2 - 3) + 2.0 / n**2
    y_m = (n - m) / n * eta_y ** np.maximum(m - 1, 0) + 1.0 / n
    return x_m, y_m


def g2_curve_from_eta(eta_x: np.ndarray, eta_y: np.ndarray, m_bar: float) -> np.ndarray:
    """Assemble ``g2(T_s)`` for the ``N = 270`` cloud from the extrapolated overlaps."""
    p_m = rc.poisson_excitation_amplitudes(N_MACRO, m_bar, m_max=M_MAX)
    out = np.empty(len(eta_x))
    for i in range(len(eta_x)):
        x_m, y_m = scaling_ansatz_xy(eta_x[i], eta_y[i], N_MACRO, M_MAX)
        out[i] = rc.g2_from_populations(p_m, x_m, y_m)
    return out


# --------------------------------------------------------------------------------------
# The Cirq engine: one parametrized circuit per cloud realization, one sweep in T_s.
# --------------------------------------------------------------------------------------
def cirq_x2_y2_ensemble(params, storage_times, n_realizations, rng, simulator, qubits,
                        psi2, tables, progress_label=""):
    """Average the exact ``X_2(T_s)``, ``Y_2(T_s)`` over independent cloud realizations.

    Every realization draws ``N_MICRO`` atoms from the Gaussian excitation volume, builds
    *one* ``T_s``-parametrized circuit, and sweeps it with ``cirq.Points``.
    """
    n_t = len(storage_times)
    x2_all = np.empty((n_realizations, n_t))
    y2_all = np.empty((n_realizations, n_t))
    sweep = cirq.Points(TS_SYMBOL.name, list(map(float, storage_times)))
    iu = np.triu_indices(N_MICRO, k=1)
    c_pool = np.zeros(n_t, dtype=complex)   # pooled <exp(-i Phi_{mu nu})> over all pairs

    t0 = time.time()
    for r in range(n_realizations):
        positions = rc.sample_cloud(params, N_MICRO, rng)
        # phase_matrix at T_s = 1 us is exactly the phase *rate* kappa_{jk}.
        rate = rc.phase_matrix(positions, 1.0, params.c6)
        circuit = build_parametrized_storage_circuit(rate, qubits, TS_SYMBOL)
        results = simulator.simulate_sweep(circuit, params=sweep, initial_state=psi2)
        x2_all[r], y2_all[r] = x2_y2_from_states(
            [res.final_state_vector for res in results], tables, N_MICRO
        )
        c_pool += np.exp(-1j * np.outer(storage_times, rate[iu])).mean(axis=1)
        if progress_label and (r + 1) % 50 == 0:
            print(f"      {progress_label}: {r + 1}/{n_realizations} realizations "
                  f"({time.time() - t0:.1f} s)")
    return x2_all, y2_all, np.abs(c_pool / n_realizations) ** 2


def run_cloud(params, storage_times, n_realizations, rng, simulator, qubits, psi2, tables,
              label=""):
    """Full pipeline for one cloud: Cirq -> eta -> Eqs. (S.7)/(S.8) -> g2(T_s)."""
    x2_all, y2_all, eta_pool = cirq_x2_y2_ensemble(
        params, storage_times, n_realizations, rng, simulator, qubits, psi2, tables, label
    )
    x2 = x2_all.mean(axis=0)
    y2 = y2_all.mean(axis=0)
    eta_x = eta_from_x2(x2, N_MICRO)
    eta_y = eta_from_y2(y2, N_MICRO)
    g2 = g2_curve_from_eta(eta_x, eta_y, params.m_bar)

    # Bootstrap the realization spread to quantify the statistical (not systematic) error.
    boot = np.empty((40, len(storage_times)))
    for b in range(40):
        pick = rng.integers(0, n_realizations, n_realizations)
        boot[b] = g2_curve_from_eta(
            eta_from_x2(x2_all[pick].mean(axis=0), N_MICRO),
            eta_from_y2(y2_all[pick].mean(axis=0), N_MICRO),
            params.m_bar,
        )
    return dict(params=params, x2=x2, y2=y2, eta_x=eta_x, eta_y=eta_y, eta_pool=eta_pool,
                g2=g2, g2_sem=boot.std(axis=0), n_realizations=n_realizations)


def sigma_z_band(params, storage_times, n_realizations, rng, simulator, qubits, psi2, tables,
                 label=""):
    """``g2`` envelope for ``sigma_z`` varied by the quoted +-20%."""
    curves = []
    for scale in (1.0 - SIGMA_Z_UNCERTAINTY, 1.0 + SIGMA_Z_UNCERTAINTY):
        p = dataclasses.replace(params, sigma_z=params.sigma_z * scale)
        res = cirq_x2_y2_ensemble(p, storage_times, n_realizations, rng, simulator, qubits,
                                  psi2, tables, f"{label} sigma_z x{scale:.1f}")
        curves.append(
            g2_curve_from_eta(
                eta_from_x2(res[0].mean(axis=0), N_MICRO),
                eta_from_y2(res[1].mean(axis=0), N_MICRO),
                params.m_bar,
            )
        )
    lo = np.minimum(curves[0], curves[1])
    hi = np.maximum(curves[0], curves[1])
    return lo, hi


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate(simulator, qubits, psi2, tables, rng):
    """Print the three correctness checks demanded of a quantum-circuit reproduction."""
    print("-" * 86)
    print("VALIDATION 1 -- symbolic CZPowGate circuit == rydberg_cirq.PairwisePhaseGate")
    print("-" * 86)
    positions = rc.sample_cloud(SHORT_N50, N_MICRO, rng)
    rate = rc.phase_matrix(positions, 1.0, SHORT_N50.c6)
    circuit = build_parametrized_storage_circuit(rate, qubits, TS_SYMBOL)
    print(f"  parametrized circuit: {N_MICRO} qubits, "
          f"{len(list(circuit.all_operations()))} symbolic CZPowGate ops, "
          f"free symbols = {sorted(map(str, cirq.parameter_names(circuit)))}")

    ts_probe = 0.37
    resolver = cirq.ParamResolver({TS_SYMBOL.name: ts_probe})
    psi_sym = simulator.simulate(circuit, param_resolver=resolver,
                                 initial_state=psi2).final_state_vector

    exact_gate = rc.PairwisePhaseGate(rc.phase_matrix(positions, ts_probe, SHORT_N50.c6))
    psi_gate = simulator.simulate(cirq.Circuit(exact_gate.on(*qubits)),
                                  initial_state=psi2).final_state_vector
    overlap = abs(np.vdot(psi_gate, psi_sym))
    print(f"  cirq.ParamResolver(T_s={ts_probe}) vs PairwisePhaseGate(Phi(T_s)):")
    print(f"    max |dpsi| = {np.max(np.abs(psi_sym - psi_gate)):.3e}, "
          f"|<psi_gate|psi_sym>| = {overlap:.12f}")
    eta_single = rc.coherence_eta(rc.phase_matrix(positions, ts_probe, SHORT_N50.c6))
    print(f"  rydberg_cirq.coherence_eta for this realization: |<exp(-i Phi)>| = "
          f"{eta_single:.6f}  ->  eta = |<exp(-i Phi)>|^2 = {eta_single**2:.6f}")

    demo_qubits = cirq.LineQubit.range(4)
    demo = build_parametrized_storage_circuit(rate[:4, :4], demo_qubits, TS_SYMBOL)
    print("  4-qubit excerpt of the parametrized storage circuit:")
    for line in str(demo).splitlines():
        print("    " + line)

    print()
    print("-" * 86)
    print("VALIDATION 2 -- Cirq X_2, Y_2 vs direct NumPy evaluation (cirq.Linspace sweep)")
    print("-" * 86)
    lin = cirq.Linspace(TS_SYMBOL.name, start=0.0, stop=2.0, length=9)
    ts_lin = np.array([float(r.value_of(TS_SYMBOL.name)) for r in lin])
    results = simulator.simulate_sweep(circuit, params=lin, initial_state=psi2)
    x2_c, y2_c = x2_y2_from_states([r.final_state_vector for r in results], tables, N_MICRO)
    x2_n, y2_n = x2_y2_numpy(rate, ts_lin)
    print(f"    {'T_s [us]':>9} {'X2 (Cirq)':>13} {'X2 (NumPy)':>13} {'|dX2|':>10}"
          f" {'Y2 (Cirq)':>13} {'Y2 (NumPy)':>13} {'|dY2|':>10}")
    for i, ts in enumerate(ts_lin):
        print(f"    {ts:9.4f} {x2_c[i]:13.9f} {x2_n[i]:13.9f} {abs(x2_c[i]-x2_n[i]):10.2e}"
              f" {y2_c[i]:13.9f} {y2_n[i]:13.9f} {abs(y2_c[i]-y2_n[i]):10.2e}")
    res_x = float(np.max(np.abs(x2_c - x2_n)))
    res_y = float(np.max(np.abs(y2_c - y2_n)))
    print(f"  max residual: |dX_2| = {res_x:.3e}, |dY_2| = {res_y:.3e}  "
          f"({'PASS' if max(res_x, res_y) < 1e-9 else 'CHECK'} at the 1e-9 level)")

    n = N_MICRO
    print(f"  T_s = 0 sanity: X_2 = Y_2 = (N-1)/N = {(n - 1) / n:.9f} "
          f"(Cirq gives {x2_c[0]:.9f}, {y2_c[0]:.9f})")

    print()
    print("-" * 86)
    print("VALIDATION 3 -- parametrized sweep vs rebuilding one circuit per time point")
    print("-" * 86)
    ts_bench = np.linspace(0.0, 5.0, 40)
    t0 = time.time()
    sweep = cirq.Points(TS_SYMBOL.name, list(ts_bench))
    out_sweep = simulator.simulate_sweep(circuit, params=sweep, initial_state=psi2)
    t_sweep = time.time() - t0
    x2_sweep, _ = x2_y2_from_states([r.final_state_vector for r in out_sweep], tables, N_MICRO)

    t0 = time.time()
    states = []
    for ts in ts_bench:
        gate = rc.PairwisePhaseGate(rc.phase_matrix(positions, float(ts), SHORT_N50.c6))
        states.append(
            simulator.simulate(cirq.Circuit(gate.on(*qubits)),
                               initial_state=psi2).final_state_vector
        )
    t_rebuild = time.time() - t0
    x2_rebuild, _ = x2_y2_from_states(states, tables, N_MICRO)

    print(f"  1 symbolic circuit + cirq.Points sweep over {len(ts_bench)} T_s : "
          f"{t_sweep:.3f} s")
    print(f"  {len(ts_bench)} freshly built concrete circuits                 : "
          f"{t_rebuild:.3f} s")
    print(f"  speed-up = {t_rebuild / max(t_sweep, 1e-9):.2f}x, "
          f"max |dX_2| between the two routes = {np.max(np.abs(x2_sweep - x2_rebuild)):.3e}")
    print()
    return res_x, res_y


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--realizations", type=int, default=150,
                        help="cloud realizations per central curve (default 150)")
    parser.add_argument("--band-realizations", type=int, default=60,
                        help="cloud realizations per +-20%% sigma_z band edge (default 60)")
    parser.add_argument("--time-points", type=int, default=55,
                        help="logarithmic storage-time points (default 55)")
    args = parser.parse_args()

    apply_style()
    rng = np.random.default_rng(20221101)

    print("=" * 86)
    print("g2(T_s) of the retrieved field -- Cirq reproduction of Fig. 2")
    print("Phys. Rev. A 106, L051701 (2022), Y. Li, Y. Mei, H. Nguyen, P. R. Berman, A. Kuzmich")
    print("=" * 86)
    print(f"  macroscopic ensemble      N        = {N_MACRO} atoms")
    print(f"  exact Cirq register       N_micro  = {N_MICRO} qubits "
          f"({2**N_MICRO} amplitudes, {N_MICRO * (N_MICRO - 1) // 2} CZPowGates)")
    print(f"  excitation sum truncated  m_max    = {M_MAX}")
    for p in (SHORT_N50, SHORT_N40, LONG_N40, LONG_N50):
        print(f"  {p.label:22s} sigma = ({p.sigma_x:.3f}, {p.sigma_y:.3f}, {p.sigma_z:7.2f}) um, "
              f"C6/h = {p.c6:6.2f} GHz um^6, m_bar = {p.m_bar:.2f}")
    print()

    qubits = cirq.LineQubit.range(N_MICRO)
    simulator = cirq.Simulator(dtype=np.complex128, seed=7)
    tables = spinwave_lowering_tables(N_MICRO)
    psi2 = rc.dicke_state_vector(N_MICRO, 2)

    res_x, res_y = validate(simulator, qubits, psi2, tables, rng)

    # Storage-time grid: logarithmic so the sub-microsecond collapse is resolved while
    # still reaching the 30 us of the experiment.
    storage_times = np.concatenate(
        [[0.0], np.logspace(-3.0, np.log10(30.0), args.time_points)]
    )

    n_main = args.realizations
    n_band = args.band_realizations

    print("-" * 86)
    print(f"CIRQ ENSEMBLES -- {n_main} realizations per curve, {len(storage_times)} T_s points "
          f"per parametrized circuit")
    print("-" * 86)

    runs = {}
    bands = {}
    t_start = time.time()
    for key, params, want_band in (
        ("short_n50", SHORT_N50, True),
        ("short_n40", SHORT_N40, True),
        ("long_n40", LONG_N40, True),
        ("long_n50", LONG_N50, False),
    ):
        print(f"  [{key}] {params.label} ...")
        runs[key] = run_cloud(params, storage_times, n_main, rng, simulator, qubits, psi2,
                              tables, label=key)
        if want_band:
            bands[key] = sigma_z_band(params, storage_times, n_band, rng, simulator, qubits,
                                      psi2, tables, label=key)
    print(f"  total Cirq wall time: {time.time() - t_start:.1f} s")
    print()

    # ---------------------------------------------------------------- numbers / claims
    print("-" * 86)
    print("KEY NUMBERS")
    print("-" * 86)
    p_ref = rc.poisson_excitation_amplitudes(N_MACRO, SHORT_N50.m_bar, m_max=M_MAX)
    m_arr = np.arange(M_MAX + 1)
    poisson_g2 = (np.sum(p_ref * m_arr * (m_arr - 1)) / np.sum(p_ref * m_arr) ** 2)
    print(f"  Poissonian (no interaction) reference for m_bar = {SHORT_N50.m_bar}: "
          f"g2 = <m(m-1)>/<m>^2 = {poisson_g2:.6f}  (= 1 - 1/N for a binomial spin wave)")
    print()
    print(f"  {'cloud':<22}{'eta(0)':>9}{'g2(0)':>9}{'g2(1us)':>10}{'g2(5us)':>10}"
          f"{'g2(15us)':>10}{'g2(30us)':>10}{'eta(30us)':>11}")
    probes = [0.0, 1.0, 5.0, 15.0, 30.0]
    for key in ("short_n50", "short_n40", "long_n40", "long_n50"):
        r = runs[key]
        g = np.interp(probes, storage_times, r["g2"])
        e0 = r["eta_x"][0]
        e_end = np.interp(30.0, storage_times, r["eta_x"])
        print(f"  {r['params'].label:<22}{e0:9.5f}{g[0]:9.4f}{g[1]:10.4f}{g[2]:10.4f}"
              f"{g[3]:10.4f}{g[4]:10.4f}{e_end:11.2e}")
    print()
    print("  Central claim of the paper -- interaction-induced dephasing during storage")
    print("  converts the unentangled spin wave into a single-excitation Dicke (W) state:")
    for key in ("short_n50", "short_n40"):
        r = runs[key]
        g0 = r["g2"][0]
        g_end = r["g2"][-1]
        print(f"    {r['params'].label:<22} g2(0) = {g0:.4f} (Poissonian ~ 1)  ->  "
              f"g2(30 us) = {g_end:.4f}   suppression x{g0 / max(g_end, 1e-12):.1f}")
    r = runs["long_n40"]
    print(f"    {r['params'].label:<22} g2(0) = {r['g2'][0]:.4f}  ->  "
          f"g2(30 us) = {r['g2'][-1]:.4f}   (dilute cloud: no dephasing, stays ~ 1)")
    print()
    print(f"  bootstrap statistical error on the mean curve (max over T_s): "
          + ", ".join(f"{k} {runs[k]['g2_sem'].max():.4f}" for k in runs))
    print()

    # ------------------------------------------------- validation 4: eta is N-independent
    print("-" * 86)
    print("VALIDATION 4 -- eta from the Cirq circuits == |<exp(-i Phi)>|^2 of the pair ensemble")
    print("-" * 86)
    print("  Treating the pair phases as i.i.d. with c = <exp(-i Phi_{mu nu})>, the exact")
    print("  m = 2 expectation values give X_2 = (1 - 1/N - 2/N^2)|c|^2 + 2/N^2 and")
    print("  Y_2 = (N-2)/N |c|^2 + 1/N, i.e. eta = |c|^2 carries no N dependence at all.")
    print("  This is what licenses reading eta off a 12-qubit register and inserting it")
    print("  into Eqs. (S.7)/(S.8) at N = 270.")
    dev_max = 0.0
    for key in ("short_n50", "long_n40"):
        r = runs[key]
        print(f"  {r['params'].label}:")
        print(f"    {'T_s [us]':>9}{'eta_X (Cirq)':>15}{'eta_Y (Cirq)':>15}"
              f"{'|<e^-iPhi>|^2':>16}{'|dev|':>10}")
        for ts_probe in (0.0, 0.01, 0.1, 1.0, 5.0, 30.0):
            i = int(np.argmin(np.abs(storage_times - ts_probe)))
            dev = max(abs(r["eta_x"][i] - r["eta_pool"][i]),
                      abs(r["eta_y"][i] - r["eta_pool"][i]))
            dev_max = max(dev_max, dev)
            print(f"    {storage_times[i]:9.3f}{r['eta_x'][i]:15.6f}{r['eta_y'][i]:15.6f}"
                  f"{r['eta_pool'][i]:16.6f}{dev:10.4f}")
    print(f"  max |eta_circuit - |<exp(-i Phi)>|^2| over the probes = {dev_max:.4f}")
    print("  (residual scatter is the finite-realization Monte-Carlo noise plus the")
    print("   correlations between pair phases that share an atom -- exactly the")
    print("   approximation the paper's ansatz makes.)")
    print()

    # ------------------------------------------------------- comparison to experiment
    print("-" * 86)
    print("COMPARISON WITH THE EXPERIMENTAL POINTS OF FIG. 2")
    print("-" * 86)
    chi2_report = {}
    for key, exp_key in (("short_n50", "short_n50"), ("short_n40", "short_n40"),
                         ("long_n50", "long_n50")):
        d = EXP[exp_key]
        theory = np.interp(d["ts"], storage_times, runs[key]["g2"])
        resid = theory - d["g2"]
        chi2 = float(np.sum((resid / d["err"]) ** 2) / len(resid))
        chi2_report[key] = chi2
        print(f"  {runs[key]['params'].label}   (chi^2/dof = {chi2:.2f}, "
              f"RMS residual = {np.sqrt(np.mean(resid**2)):.3f})")
        print(f"    {'T_s [us]':>9}{'g2 exp':>10}{'+-':>7}{'g2 Cirq':>10}{'resid':>9}"
              f"{'n sigma':>9}")
        for i, ts in enumerate(d["ts"]):
            print(f"    {ts:9.2f}{d['g2'][i]:10.3f}{d['err'][i]:7.3f}{theory[i]:10.3f}"
                  f"{resid[i]:9.3f}{resid[i] / d['err'][i]:9.2f}")
        print()

    # ------------------------------------------------------------------------- figure
    print("-" * 86)
    print("FIGURE")
    print("-" * 86)
    fig = make_figure(storage_times, runs, bands, res_x, res_y)
    save_figure(fig, __file__, "cirq_g2_fig2_reproduction.png")

    print()
    print("=" * 86)
    print("SUMMARY")
    print("=" * 86)
    print(f"  X_2/Y_2 Cirq-vs-NumPy residuals: {res_x:.2e} / {res_y:.2e}  (exact circuits)")
    print(f"  g2(T_s -> 0)  ~ {runs['short_n50']['g2'][0]:.3f} (short n=50), "
          f"{runs['short_n40']['g2'][0]:.3f} (short n=40) -- unentangled Poissonian spin wave")
    print(f"  g2(T_s = 30 us) = {runs['short_n50']['g2'][-1]:.3f} (short n=50) -- "
          f"antibunched single photon from an entangled W state")
    print(f"  long cloud stays classical: g2(30 us) = {runs['long_n40']['g2'][-1]:.3f} "
          f"(n=40), {runs['long_n50']['g2'][-1]:.3f} (n=50)")
    print("  chi^2/dof vs experiment: "
          + ", ".join(f"{k} = {v:.2f}" for k, v in chi2_report.items()))
    print("=" * 86)


def make_figure(ts, runs, bands, res_x, res_y):
    """Four-panel publication figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6))
    c50 = PALETTE["green"]
    c40 = PALETTE["orange"]
    cblue = PALETTE["blue"]
    cred = PALETTE["red"]
    cpurple = PALETTE["purple"]

    # ---------------------------------------------------------------- (a) short cloud
    ax = axes[0, 0]
    for key, color, name in (("short_n40", c40, r"$n=40$"), ("short_n50", c50, r"$n=50$")):
        if key in bands:
            ax.fill_between(ts, bands[key][0], bands[key][1], color=color, alpha=0.18, lw=0)
        ax.plot(ts, runs[key]["g2"], color=color, lw=2.4,
                label=f"Cirq {name}, " r"$C_6/h=" +
                      (f"{runs[key]['params'].c6:g}" ) + r"$ GHz$\,\mu$m$^6$")
    for key, color, marker, name in (("short_n40", c40, "o", r"Expt. $n=40$"),
                                     ("short_n50", c50, "s", r"Expt. $n=50$")):
        d = EXP[key]
        ax.errorbar(d["ts"], d["g2"], yerr=d["err"], fmt=marker, color=color, ecolor=color,
                    ms=6.0, elinewidth=1.4, capsize=2.5, ls="none", label=name, zorder=5)
    ax.axhline(1.0, color="0.45", ls=":", lw=1.4)
    ax.text(0.6, 1.02, "Poissonian (unentangled spin wave)", color="0.35", fontsize=8.5,
            ha="left", va="bottom")
    ax.set_xlim(-0.6, 30.5)
    ax.set_ylim(0.0, 1.45)
    ax.set_xlabel(r"storage time  $T_s$  ($\mu$s)")
    ax.set_ylabel(r"$g^{(2)}(T_s)$")
    ax.set_title(r"(a) short cloud, $\sigma_z = 5.25\ \mu$m  ($D_z = 10.5\ \mu$m)"
                 "\n" r"shaded: $\pm 20\%$ experimental uncertainty on $\sigma_z$")
    ax.legend(loc="upper right", ncol=1)

    # ----------------------------------------------------------------- (b) long cloud
    ax = axes[0, 1]
    for key, color, name in (("long_n40", c40, r"$n=40$"), ("long_n50", c50, r"$n=50$")):
        if key in bands:
            ax.fill_between(ts, bands[key][0], bands[key][1], color=color, alpha=0.18, lw=0)
        ax.plot(ts, runs[key]["g2"], color=color, lw=2.4,
                label=f"Cirq {name}, " r"$C_6/h=" + f"{runs[key]['params'].c6:g}" +
                      r"$ GHz$\,\mu$m$^6$")
    d = EXP["long_n50"]
    ax.errorbar(d["ts"], d["g2"], yerr=d["err"], fmt="s", mfc="white", mec=c50, mew=1.8,
                ecolor=c50, ms=7.0, elinewidth=1.4, capsize=2.5, ls="none",
                label=r"Expt. $n=50$, $D_z=230\ \mu$m", zorder=5)
    ax.axhline(1.0, color="0.45", ls=":", lw=1.4)
    ax.set_xlim(-0.6, 30.5)
    ax.set_ylim(0.0, 1.32)
    ax.set_xlabel(r"storage time  $T_s$  ($\mu$s)")
    ax.set_ylabel(r"$g^{(2)}(T_s)$")
    ax.set_title(r"(b) long cloud, $\sigma_z = 115\ \mu$m  ($D_z = 230\ \mu$m)")
    ax.legend(loc="lower left")

    # ------------------------------------------------- (c) coherence / overlap factors
    ax = axes[1, 0]
    styles = {"short_n50": (c50, "-"), "short_n40": (c40, "-"),
              "long_n40": (c40, "--"), "long_n50": (c50, "--")}
    tpos = ts.copy()
    tpos[0] = ts[1] * 0.5
    for key, (color, ls) in styles.items():
        ax.plot(tpos, runs[key]["eta_x"], color=color, ls=ls, lw=2.4,
                label=r"$\eta_X$ " + runs[key]["params"].label)
        ax.plot(tpos, runs[key]["eta_y"], color=color, ls=ls, lw=1.0, alpha=0.6)
    # classical cross-check: eta = |<exp(-i Phi)>|^2 of the pair ensemble (N-independent)
    ax.plot(tpos[::3], runs["short_n50"]["eta_pool"][::3], "k.", ms=5.0, zorder=6,
            label=r"$|\langle e^{-i\Phi}\rangle|^2$ (classical, $n=50$ short)")
    ax.plot(tpos[::3], runs["long_n40"]["eta_pool"][::3], "k.", ms=5.0, zorder=6)
    ax.set_xscale("log")
    ax.set_xlim(tpos[0], 31.0)
    ax.set_ylim(-0.03, 1.32)
    ax.set_xlabel(r"storage time  $T_s$  ($\mu$s)")
    ax.set_ylabel(r"two-atom coherence  $\eta(T_s)$")
    ax.set_title(r"(c) $\eta$ inverted from the exact Cirq $X_2,\,Y_2$"
                 "\n" r"thick: $\eta_X$ [Eq. (S.7)],  thin: $\eta_Y$ [Eq. (S.8)]")
    ax.legend(loc="lower left", fontsize=7.6)
    ax.text(0.975, 0.955,
            "Cirq vs NumPy reference\n" r"$\max|\Delta X_2| = $" f"{res_x:.1e}" "\n"
            r"$\max|\Delta Y_2| = $" f"{res_y:.1e}",
            transform=ax.transAxes, fontsize=8.0, va="top", ha="right",
            bbox=dict(fc="white", ec="0.7", alpha=0.9, boxstyle="round,pad=0.35"))

    # ------------------------------------------- (d) macroscopic X_m, Y_m from Eq. S.7/8
    ax = axes[1, 1]
    r = runs["short_n50"]
    colors_m = [cblue, c50, c40, cred, cpurple]
    for i, m in enumerate([2, 3, 4, 5, 6]):
        x_m = np.array([scaling_ansatz_xy(ex, ey, N_MACRO, M_MAX)[0][m]
                        for ex, ey in zip(r["eta_x"], r["eta_y"])])
        y_m = np.array([scaling_ansatz_xy(ex, ey, N_MACRO, M_MAX)[1][m]
                        for ex, ey in zip(r["eta_x"], r["eta_y"])])
        col = colors_m[i % len(colors_m)]
        ax.plot(tpos, x_m, color=col, lw=2.2, label=rf"$X_{{{m}}}$")
        ax.plot(tpos, y_m, color=col, lw=1.1, ls="--", alpha=0.7)
    ax.axhline(2.0 / N_MACRO**2, color="0.4", ls=":", lw=1.3)
    ax.text(0.012, 2.4 / N_MACRO**2, r"$2/N^2$", color="0.3", fontsize=9)
    ax.axhline(1.0 / N_MACRO, color="0.4", ls="-.", lw=1.3)
    ax.text(0.012, 1.2 / N_MACRO, r"$1/N$", color="0.3", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(tpos[0], 31.0)
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlabel(r"storage time  $T_s$  ($\mu$s)")
    ax.set_ylabel(r"overlap factors  $X_m,\ Y_m$")
    ax.set_title(r"(d) $N=270$ extrapolation, Eqs. (S.7)/(S.8), short cloud $n=50$"
                 "\n" r"(solid: $X_m\sim\eta^{2m-3}$;  dashed: $Y_m\sim\eta^{m-1}$)")
    ax.legend(loc="lower left", ncol=5, fontsize=8.5)

    fig.suptitle(
        r"Interaction-induced dephasing $\Rightarrow$ antibunched retrieval:  "
        r"Cirq reproduction of Fig. 2 of Phys. Rev. A $\bf{106}$, L051701 (2022)"
        "\n"
        rf"exact {N_MICRO}-qubit $T_s$-parametrized $C\!Z^{{\,\theta}}$ circuits "
        rf"$\rightarrow$ $\eta(T_s)$ $\rightarrow$ Eqs. (S.7)/(S.8) at $N={N_MACRO}$",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    return fig


if __name__ == "__main__":
    main()
