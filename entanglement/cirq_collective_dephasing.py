r"""
Genuine Cirq density-matrix simulation of dephasing-induced Dicke-state
purification in a trapped Rydberg ensemble.

Reference
---------
Y. Li, Y. Mei, H. Nguyen, P. R. Berman and A. Kuzmich,
*Dynamics of collective-dephasing-induced multiatom entanglement*,
**Phys. Rev. A 106, L051701 (2022)**.

What this script replaces
-------------------------
``entanglement/simulation_dicke.py`` integrates

    d(rho)/dt = -i [H, rho] + (gamma_c / 2) (2 Jz rho Jz - Jz^2 rho - rho Jz^2)

by hand, in the (N+1)-dimensional Dicke ladder, with a Strang (Trotter-Suzuki)
operator splitting.  Here the *same* physics is obtained from a real
``cirq.DensityMatrixSimulator`` running on the full ``2^N``-dimensional Hilbert
space, driven by the exact CPTP Kraus channel
:class:`rydberg_cirq.channels.CollectiveDephasingChannel`.  The hand-rolled
solver is retained only as a validation oracle
(:func:`_numpy_master_equation_oracle`), and both are benchmarked against an
*exact* Liouvillian matrix exponential (:func:`_exact_liouvillian_solution`).

Physics
-------
**Initial state.**  A short two-photon pulse leaves the ensemble in a strictly
*unentangled product* spin wave, Eq. (2) of the paper,

    |Psi_0> = prod_j ( a |g_j> + b |r_j> ),     |b|^2 = m_bar / N,

whose excitation statistics ``|c_m|^2 = binom(N, m) |a|^{2(N-m)} |b|^{2m}`` are
binomial (Poissonian for ``N >> m_bar``).  In Cirq this is one single-qubit
layer, ``cirq.ry(theta).on_each(*qubits)`` with ``sin^2(theta/2) = m_bar / N``.

**Two competing dephasing mechanisms during storage.**  This is the central
point of the script, and it is a genuinely falsifiable statement:

1. *Collective (global-phase) dephasing* -- laser phase noise, common-mode
   Doppler/Stark shifts.  Generator ``Jz = -(1/2) sum_j Z_j`` is a **sum of
   single-atom operators**, so the channel is a mixture of *local* unitaries
   ``exp(-i phi Jz) = prod_j exp(-i phi Z_j / 2)``.  A mixture of local
   unitaries is a separable (LOCC) map, therefore, by convexity of the
   concurrence, it **can never create entanglement** from a product state.
   It also commutes with the excitation number, so every ``p_m`` is frozen.
   This is ``rydberg_cirq.CollectiveDephasingChannel``, and the simulation
   below confirms ``C(t) = 0`` to machine precision at all times.

2. *Interaction-induced (pairwise) dephasing* -- the actual mechanism of
   PRA 106, L051701.  The van der Waals Hamiltonian
   ``H_c = sum_{mu<nu} kappa_{mu nu} n_mu n_nu`` is **quadratic** in the atomic
   excitation operators and the couplings ``kappa_{mu nu} = C6 / R_{mu nu}^6``
   are random from shot to shot because the atoms sit at random positions in
   the cloud.  Crucially ``H_c |m=0> = H_c |m=1> = 0``: a lone Rydberg atom has
   no partner.  Disorder-averaging ``exp(-i H_c T_s)`` therefore leaves the
   ``m <= 1`` sectors untouched while scrambling every ``m >= 2`` sector out of
   the phase-matched spin-wave mode.  That is what converts the product spin
   wave into ``|W>`` and drives ``g^(2) -> 0``.

Because ``rydberg_cirq`` does not (yet) ship a disorder-averaged version of the
pairwise channel, one is defined **locally** in this module as
:class:`InteractionDephasingChannel`, built with exactly the same exact-finite-
Kraus construction as ``rydberg_cirq.channels.collective_dephasing_kraus``.
Nothing inside ``rydberg_cirq/`` is modified.

**Observable state.**  Retrieval is mode selective: only the phase-matched
symmetric spin wave radiates into the detected mode.  The operational state is
therefore the post-selected

    rho_eff(t) = Pi rho(t) Pi / Tr[Pi rho(t) Pi],
    Pi = sum_m |D_m><D_m|   (``rydberg_cirq.dicke_projector``),

which at ``t = 0`` is *exactly* the initial product state (so the "concurrence
starts at zero" prediction is not built in by hand) and whose Dicke populations
``p_m(t) = <D_m|rho_eff(t)|D_m>`` are the paper's ``p_m``.

Cirq APIs showcased
-------------------
* ``cirq.DensityMatrixSimulator`` + ``simulate_moment_steps`` (complex128)
* ``cirq.ry(...).on_each(...)`` to build the unentangled spin wave as a circuit
* ``cirq.Gate`` subclasses exposing ``_kraus_`` (``rc.CollectiveDephasingChannel``
  and the local ``InteractionDephasingChannel``); ``cirq.kraus`` /
  ``cirq.has_kraus`` for CPTP introspection
* ``cirq.MatrixGate`` for the exact single-atom drive propagator, cross-checked
  against ``rc.CollectiveLaserDriveStep``
* ``cirq.PauliSum`` observables via ``rc.collective_spin_paulisums`` and
  ``expectation_from_density_matrix`` (inside ``rc.spin_observables``)
* ``rc.transformers.circuit_stats`` for circuit bookkeeping

Run
---
    /Users/yinliyl/quantum/.venv/bin/python entanglement/cirq_collective_dephasing.py
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg as la

import cirq
import rydberg_cirq as rc
from rydberg_cirq import plotting as rcplot

# --------------------------------------------------------------------------
# Global model parameters
# --------------------------------------------------------------------------
N_ATOMS = 6              # 2^N x 2^N density matrix; 6-8 is cheap
M_BAR = 1.6              # mean excitation number, n = 40 long-cloud value (paper: 1.63)
GAMMA = 1.0              # dephasing rate; time is measured in units of 1 / GAMMA
T_MAX = 6.0
N_STEPS = 40

# Validation (driven master equation) parameters
VAL_OMEGA = 1.3          # Rabi frequency of the collective drive, Omega
VAL_DELTA = 0.7          # detuning, Delta
VAL_GAMMA = 0.9          # collective dephasing rate
VAL_TMAX = 1.5
VAL_THETA0 = 0.7         # initial coherent-spin-state tipping angle

RNG_SEED = 20221101
DTYPE = np.complex128


# ==========================================================================
# 1.  Local helper: disorder-averaged pairwise interaction dephasing channel
# ==========================================================================
def pair_dephasing_exponent(num_qubits: int) -> np.ndarray:
    r"""Number of *pair bonds* by which two computational basis states differ.

    For a basis state ``s`` let ``P^s_{mu nu} = n^s_mu n^s_nu in {0, 1}`` flag an
    excited pair.  Then

        E[s, s'] = sum_{mu<nu} (P^s_{mu nu} - P^{s'}_{mu nu})^2
                 = binom(m_s, 2) + binom(m_s', 2) - 2 binom(|s AND s'|, 2),

    because ``(a - b)^2 = a XOR b`` for binary ``a, b`` and the pairs common to
    both states are exactly the pairs inside the overlap ``s AND s'``.

    Note ``E[s, s'] = 0`` whenever both states have ``m <= 1``: the ``m = 0`` and
    ``m = 1`` sectors are *immune* to interaction dephasing, which is the whole
    reason the spin wave purifies into ``|W>``.
    """
    idx = np.arange(2**num_qubits, dtype=np.int64)
    m = rc.hamming_weights(num_qubits).astype(np.int64)
    overlap = np.bitwise_count(np.bitwise_and(idx[:, None], idx[None, :])).astype(np.int64)

    def binom2(x):
        return x * (x - 1) // 2

    return (binom2(m)[:, None] + binom2(m)[None, :] - 2 * binom2(overlap)).astype(float)


def interaction_dephasing_damping(num_qubits: int, gamma_p: float, tau: float) -> np.ndarray:
    r"""Exact disorder-averaged damping matrix of the pairwise interaction channel.

    With i.i.d. random couplings ``kappa_{mu nu}`` the shot-averaged propagator
    ``< U rho U^dag >``, ``U = exp(-i tau sum_{mu<nu} kappa_{mu nu} n_mu n_nu)``,
    acts elementwise on the computational-basis density matrix,

        rho[s, s'] -> rho[s, s'] * exp(-gamma_p tau E[s, s'] / 2),

    with ``E`` from :func:`pair_dephasing_exponent` and
    ``gamma_p tau = <(kappa tau)^2>`` the elementary single-bond phase variance.
    The elementary two-atom coherence factor of the paper is therefore
    ``eta = exp(-gamma_p tau / 2)``.
    """
    if gamma_p * tau <= 0.0:
        return np.ones((2**num_qubits, 2**num_qubits))
    return np.exp(-0.5 * gamma_p * tau * pair_dephasing_exponent(num_qubits))


def _exact_kraus_from_damping(damping: np.ndarray, tol: float = 1e-12) -> list[np.ndarray]:
    """Exact finite Kraus set for an elementwise (Schur) damping channel.

    Identical construction to ``rydberg_cirq.channels.collective_dephasing_kraus``:
    the damping matrix is a positive-semidefinite Gram matrix, so its
    eigendecomposition ``D = sum_k lambda_k v_k v_k^T`` yields diagonal Kraus
    operators ``K_k = sqrt(lambda_k) diag(v_k)``, and ``D[s, s] = 1`` guarantees
    ``sum_k K_k^dag K_k = I``.
    """
    eigvals, eigvecs = np.linalg.eigh(damping)
    keep = eigvals > tol
    eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]
    diags = np.sqrt(eigvals)[:, None] * eigvecs.T.astype(complex)
    total = np.sum(np.abs(diags) ** 2, axis=0)
    diags *= 1.0 / np.sqrt(np.maximum(total, 1e-300))[None, :]
    return [np.diag(d) for d in diags]


class InteractionDephasingChannel(cirq.Gate):
    r"""``N``-qubit CPTP channel: disorder-averaged Rydberg interaction dephasing.

    Local sibling of :class:`rydberg_cirq.channels.CollectiveDephasingChannel`.
    Where that channel is generated by the *linear* collective operator ``Jz``,
    this one is generated by the *quadratic* van der Waals Hamiltonian
    ``sum_{mu<nu} kappa_{mu nu} n_mu n_nu`` of Phys. Rev. A 106, L051701 Eq. (3),
    averaged over the random atomic positions of the cloud.

    Consequences of being quadratic:

    * ``m = 0`` and ``m = 1`` are dark to it (no pair -> no phase), so the
      single-excitation Dicke state ``|W>`` is a *fixed point*;
    * it damps coherences *within* an ``m >= 2`` sector, hence the phase-matched
      (symmetric) spin-wave population leaks into non-radiating modes;
    * it is **not** a mixture of local unitaries, so unlike collective dephasing
      it can and does generate entanglement.

    This gate is deliberately defined here rather than in ``rydberg_cirq/`` (the
    shared package must not be edited); promoting it would be a natural
    follow-up.
    """

    def __init__(self, num_qubits: int, gamma_p: float, tau: float, tol: float = 1e-12):
        super().__init__()
        self._n = int(num_qubits)
        self.gamma_p = float(gamma_p)
        self.tau = float(tau)
        self.tol = float(tol)
        self._cached: tuple | None = None

    def _num_qubits_(self) -> int:
        return self._n

    def _kraus_(self):
        if self._cached is None:
            damping = interaction_dephasing_damping(self._n, self.gamma_p, self.tau)
            self._cached = tuple(_exact_kraus_from_damping(damping, self.tol))
        return self._cached

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return [f"PairDeph(g={self.gamma_p:.3g},t={self.tau:.3g})"] + ["#"] * (self._n - 1)

    def __repr__(self) -> str:
        return f"InteractionDephasingChannel({self._n}, {self.gamma_p!r}, {self.tau!r})"


# ==========================================================================
# 2.  Circuit construction and Cirq density-matrix propagation
# ==========================================================================
def spin_wave_angle(num_qubits: int, m_bar: float) -> float:
    """``theta`` with ``sin^2(theta/2) = m_bar / N`` (paper Eq. (2))."""
    return 2.0 * np.arcsin(np.sqrt(m_bar / num_qubits))


def spin_wave_layer(qubits, m_bar: float) -> cirq.Moment:
    """The unentangled product spin wave as a single ``cirq.ry`` layer."""
    theta = spin_wave_angle(len(qubits), m_bar)
    return cirq.Moment(cirq.ry(theta).on_each(*qubits))


def storage_circuit(qubits, channel: cirq.Gate, num_steps: int, m_bar: float) -> cirq.Circuit:
    """``ry`` layer followed by ``num_steps`` applications of ``channel``."""
    return cirq.Circuit(
        [spin_wave_layer(qubits, m_bar)] + [channel.on(*qubits) for _ in range(num_steps)]
    )


def propagate(circuit: cirq.Circuit, qubits) -> list[np.ndarray]:
    """Density matrix after every moment, via ``simulate_moment_steps``."""
    dim = 2 ** len(qubits)
    sim = cirq.DensityMatrixSimulator(dtype=DTYPE, seed=RNG_SEED)
    out = []
    for step in sim.simulate_moment_steps(circuit, qubit_order=qubits):
        out.append(np.asarray(step.density_matrix(copy=True)).reshape(dim, dim))
    return out


# ==========================================================================
# 3.  Phase-matched (retrievable) state and its entanglement metrics
# ==========================================================================
def phase_matched_state(rho: np.ndarray, projector: np.ndarray):
    r"""Post-select on the excitation still living in the phase-matched mode.

    Returns ``(p_m, rho_eff, retrieval_weight)`` where

        rho_D  = P rho P^dag                    ((N+1) x (N+1), Dicke ladder)
        weight = Tr(rho_D)                      (symmetric-subspace population)
        p_m    = diag(rho_D) / weight
        rho_eff= P^dag rho_D P / weight         (embedded back into 2^N)

    At ``t = 0`` the spin wave is entirely symmetric so ``rho_eff = rho``; no
    information is discarded and the "``C(0) = 0``" test is a genuine prediction.
    """
    rho_d = projector @ rho @ projector.conj().T
    weight = float(np.real(np.trace(rho_d)))
    rho_d = rho_d / weight
    p_m = np.real(np.diag(rho_d)).copy()
    p_m = np.clip(p_m, 0.0, None)
    rho_eff = projector.conj().T @ rho_d @ projector
    rho_eff = 0.5 * (rho_eff + rho_eff.conj().T)
    return p_m, rho_eff, weight


def qfi_matrix(rho: np.ndarray, j_mats) -> np.ndarray:
    r"""``3 x 3`` quantum Fisher information matrix for ``(Jx, Jy, Jz)``.

        F[a, b] = 2 sum_{jk} (p_j - p_k)^2 / (p_j + p_k) <j|J_a|k><k|J_b|j>

    Its largest eigenvalue is the QFI optimised over the collective-spin
    direction; for any product (coherent spin) state it equals exactly ``N``,
    i.e. the standard quantum limit.
    """
    eigvals, eigvecs = la.eigh(rho)
    rotated = [eigvecs.conj().T @ J @ eigvecs for J in j_mats]
    p = eigvals[:, None]
    q = eigvals[None, :]
    den = p + q
    with np.errstate(divide="ignore", invalid="ignore"):
        wgt = np.where(den > 1e-13, (p - q) ** 2 / np.where(den > 1e-13, den, 1.0), 0.0)
    F = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            F[a, b] = 2.0 * float(np.real(np.sum(wgt * rotated[a] * rotated[b].T)))
    return 0.5 * (F + F.T)


def analyse(rho: np.ndarray, qubits, projector: np.ndarray, j_mats) -> dict:
    """All tracked observables of the phase-matched state ``rho_eff``."""
    N = len(qubits)
    p_m, rho_eff, weight = phase_matched_state(rho, projector)

    obs = rc.spin_observables(rho_eff, qubits)
    sq = rc.squeezing_parameters(obs, N)
    conc = rc.concurrence(rc.two_atom_reduced_dm(obs, N))

    ones = np.ones_like(p_m)
    g2 = rc.g2_from_populations(p_m, ones, ones)

    F = qfi_matrix(rho_eff, j_mats)
    qfi_opt = float(np.max(np.linalg.eigvalsh(F)))
    qfi_jx = rc.quantum_fisher_information(rho_eff, j_mats[0])

    # concurrence of the *full* atomic state, for comparison
    obs_full = rc.spin_observables(rho, qubits)
    conc_full = rc.concurrence(rc.two_atom_reduced_dm(obs_full, N))

    return {
        "p_m": p_m,
        "retrieval_weight": weight,
        "concurrence": conc,
        "concurrence_full": conc_full,
        "dicke_purity": rc.dicke_purity(p_m),
        "xi_R2": sq["xi_R2"],
        "xi_S2": sq["xi_S2"],
        "qfi_opt": qfi_opt,
        "qfi_jx": qfi_jx,
        "g2": g2,
        "J_len": obs["J_len"],
    }


def run_storage_model(qubits, channel: cirq.Gate, label: str) -> dict:
    """Full Cirq density-matrix storage run + observable extraction."""
    N = len(qubits)
    projector = rc.dicke_projector(N)
    J = rc.collective_spin_paulisums(qubits)
    j_mats = [J["Jx"].matrix(qubits), J["Jy"].matrix(qubits), J["Jz"].matrix(qubits)]

    circuit = storage_circuit(qubits, channel, N_STEPS, M_BAR)
    t0 = time.time()
    rhos = propagate(circuit, qubits)
    wall = time.time() - t0

    dt = T_MAX / N_STEPS
    times = np.arange(len(rhos)) * dt
    recs = [analyse(rho, qubits, projector, j_mats) for rho in rhos]

    out = {k: np.array([r[k] for r in recs]) for k in recs[0] if k != "p_m"}
    out["p_m"] = np.array([r["p_m"] for r in recs])
    out["times"] = times
    out["label"] = label
    out["wall"] = wall
    out["circuit"] = circuit
    out["rhos"] = rhos
    out["traces"] = np.array([float(np.real(np.trace(r))) for r in rhos])
    return out


# ==========================================================================
# 4.  Validation oracles
# ==========================================================================
def _dicke_observables(rho_d: np.ndarray, Jx, Jy, Jz) -> dict:
    """``rc.spin_observables``-compatible dict, computed in the Dicke ladder."""

    def ex(op):
        return float(np.real(np.trace(rho_d @ op)))

    def sym(a, b):
        return ex(0.5 * (a @ b + b @ a))

    out = {
        "Jx": ex(Jx), "Jy": ex(Jy), "Jz": ex(Jz),
        "Jx2": sym(Jx, Jx), "Jy2": sym(Jy, Jy), "Jz2": sym(Jz, Jz),
        "JxJy": sym(Jx, Jy), "JxJz": sym(Jx, Jz), "JyJz": sym(Jy, Jz),
    }
    out["J_mean"] = np.array([out["Jx"], out["Jy"], out["Jz"]])
    out["J_len"] = float(np.linalg.norm(out["J_mean"]))
    return out


def _numpy_master_equation_oracle(
    N: int, omega: float, delta: float, gamma_c: float,
    t_max: float, num_steps: int, theta0: float,
):
    r"""Faithful port of ``entanglement/simulation_dicke.py`` (Strang splitting).

    Integrates, in the ``(N+1)``-dimensional Dicke basis,

        d(rho)/dt = -i [H, rho] + (gamma_c/2) (2 Jz rho Jz - Jz^2 rho - rho Jz^2),
        H = Omega Jx + Delta Jz,

    with the second-order Strang step

        rho -> S(dt/2) . exp(-i H dt) . S(dt/2),
        S(tau): rho_{mm'} -> rho_{mm'} exp(-gamma_c (M_m - M_m')^2 tau / 2).

    The splitting is unconditionally stable and CPTP but carries an ``O(dt^2)``
    local error; :func:`_exact_liouvillian_solution` provides the exact answer.
    """
    Jx, Jy, Jz, M_vals = rc.get_dicke_operators(N)
    H = omega * Jx + delta * Jz
    dt = t_max / num_steps

    dM = M_vals[:, None] - M_vals[None, :]
    damping_half = np.exp(-0.25 * gamma_c * (dM**2) * dt)
    U = la.expm(-1j * H * dt)
    U_dag = U.conj().T

    psi0 = rc.coherent_spin_state(N, theta0)
    rho = np.outer(psi0, psi0.conj())

    obs_hist, pm_hist = [], []
    for step in range(num_steps + 1):
        rho = 0.5 * (rho + rho.conj().T)
        tr = float(np.real(np.trace(rho)))
        if tr > 0:
            rho = rho / tr
        obs_hist.append(_dicke_observables(rho, Jx, Jy, Jz))
        pm_hist.append(np.real(np.diag(rho)).copy())
        if step < num_steps:
            rho = rho * damping_half
            rho = U @ rho @ U_dag
            rho = rho * damping_half

    return {
        "times": np.linspace(0.0, t_max, num_steps + 1),
        "obs": obs_hist,
        "p_m": np.array(pm_hist),
        "rho_final": rho,
    }


def _exact_liouvillian_solution(N: int, omega: float, delta: float, gamma_c: float,
                                times, theta0: float):
    r"""Exact ``exp(L t)`` reference in the Dicke ladder (no splitting error).

    Row-major (C-order) vectorisation, for which ``vec(A rho B) = (A kron B^T) vec(rho)``.
    """
    Jx, Jy, Jz, _ = rc.get_dicke_operators(N)
    H = omega * Jx + delta * Jz
    D = N + 1
    I = np.eye(D)
    Jz2 = Jz @ Jz

    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    L += 0.5 * gamma_c * (2.0 * np.kron(Jz, Jz.T) - np.kron(Jz2, I) - np.kron(I, Jz2.T))

    psi0 = rc.coherent_spin_state(N, theta0)
    v0 = np.outer(psi0, psi0.conj()).reshape(-1)

    obs_hist, pm_hist = [], []
    for t in times:
        rho = (la.expm(L * t) @ v0).reshape(D, D)
        rho = 0.5 * (rho + rho.conj().T)
        obs_hist.append(_dicke_observables(rho, Jx, Jy, Jz))
        pm_hist.append(np.real(np.diag(rho)).copy())
    return {"obs": obs_hist, "p_m": np.array(pm_hist)}


def _drive_gate(omega: float, delta: float, dt: float) -> cirq.MatrixGate:
    r"""Exact single-atom propagator of ``H = Omega Jx + Delta Jz``.

    ``Jx = +(1/2) sum X_j`` and ``Jz = -(1/2) sum Z_j`` are sums of *commuting*
    single-atom terms, so ``exp(-i H dt)`` factorises exactly -- there is no
    Trotter error between ``Jx`` and ``Jz``.
    """
    h = 0.5 * omega * cirq.unitary(cirq.X) - 0.5 * delta * cirq.unitary(cirq.Z)
    return cirq.MatrixGate(la.expm(-1j * h * dt), name="U_drive")


def cirq_driven_run(qubits, omega, delta, gamma_c, t_max, num_steps, theta0, strang=True):
    """Cirq density-matrix integration of the driven + collectively dephased ensemble."""
    N = len(qubits)
    dt = t_max / num_steps
    drive = _drive_gate(omega, delta, dt)

    if strang:
        half = rc.CollectiveDephasingChannel(N, gamma_c, dt / 2.0)
        step_ops = [cirq.Moment(half.on(*qubits)),
                    cirq.Moment(drive.on_each(*qubits)),
                    cirq.Moment(half.on(*qubits))]
    else:  # first-order Lie-Trotter
        full = rc.CollectiveDephasingChannel(N, gamma_c, dt)
        step_ops = [cirq.Moment(drive.on_each(*qubits)),
                    cirq.Moment(full.on(*qubits))]

    # initial coherent spin state: ry(theta0) on each qubit
    circuit = cirq.Circuit([cirq.Moment(cirq.ry(theta0).on_each(*qubits))]
                           + step_ops * num_steps)

    dim = 2**N
    sim = cirq.DensityMatrixSimulator(dtype=DTYPE, seed=RNG_SEED)
    per_step = len(step_ops)
    rhos = []
    for i, st in enumerate(sim.simulate_moment_steps(circuit, qubit_order=qubits)):
        if (i - 1) % per_step == per_step - 1 or i == 0:
            rhos.append(np.asarray(st.density_matrix(copy=True)).reshape(dim, dim))

    projector = rc.dicke_projector(N)
    Jx, Jy, Jz, _ = rc.get_dicke_operators(N)
    obs_hist, pm_hist = [], []
    for rho in rhos:
        rho_d = projector @ rho @ projector.conj().T
        obs_hist.append(_dicke_observables(rho_d, Jx, Jy, Jz))
        pm_hist.append(np.real(np.diag(rho_d)).copy())
    return {"obs": obs_hist, "p_m": np.array(pm_hist), "circuit": circuit,
            "rho_final": rhos[-1]}


_OBS_KEYS = ("Jx", "Jy", "Jz", "Jx2", "Jy2", "Jz2", "JxJy", "JxJz", "JyJz")


def _obs_deviation(a_obs, b_obs, a_pm, b_pm) -> float:
    dev = 0.0
    for oa, ob in zip(a_obs, b_obs):
        dev = max(dev, max(abs(oa[k] - ob[k]) for k in _OBS_KEYS))
    dev = max(dev, float(np.max(np.abs(np.asarray(a_pm) - np.asarray(b_pm)))))
    return dev


# ==========================================================================
# 5.  Console report sections
# ==========================================================================
def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report_setup(qubits) -> None:
    N = len(qubits)
    _rule("1. UNENTANGLED PRODUCT SPIN WAVE  (PRA 106, L051701 Eq. 2)")
    theta = spin_wave_angle(N, M_BAR)
    circuit = cirq.Circuit(spin_wave_layer(qubits, M_BAR))
    print(f"  N = {N} atoms, m_bar = {M_BAR}  ->  |b|^2 = m_bar/N = {M_BAR/N:.6f}")
    print(f"  single-qubit layer: cirq.ry(theta).on_each(...),  theta = {theta:.6f} rad")
    print(f"  sin^2(theta/2) = {np.sin(theta/2)**2:.6f}")
    print(textwrap_indent(str(circuit), "    "))

    psi = cirq.final_state_vector(circuit, qubit_order=qubits, dtype=DTYPE)
    w = rc.hamming_weights(N)
    p_circuit = np.array([float(np.sum(np.abs(psi[w == m]) ** 2)) for m in range(N + 1)])
    p_exact = rc.poisson_excitation_amplitudes(N, M_BAR)
    print("\n  excitation statistics |c_m|^2  (Cirq circuit vs analytic binomial):")
    print("    m      Cirq          analytic      |diff|")
    for m in range(N + 1):
        print(f"    {m}   {p_circuit[m]:.10f}  {p_exact[m]:.10f}  {abs(p_circuit[m]-p_exact[m]):.2e}")
    print(f"  max |Cirq - analytic| = {np.max(np.abs(p_circuit - p_exact)):.3e}")
    print(f"  sum |c_m|^2 = {p_circuit.sum():.12f},  <m> = {np.sum(p_circuit*np.arange(N+1)):.10f}"
          f"  (target {M_BAR})")


def textwrap_indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + ln for ln in text.splitlines())


def report_conventions(qubits) -> None:
    """Check that the Pauli-sum spin operators and the Dicke ladder agree."""
    N = len(qubits)
    _rule("2. CONVENTION CHECK: cirq.PauliSum spin operators vs Dicke ladder")
    J = rc.collective_spin_paulisums(qubits)
    P = rc.dicke_projector(N)
    Jx_d, Jy_d, Jz_d, _ = rc.get_dicke_operators(N)
    worst = 0.0
    for name, dicke_op in (("Jx", Jx_d), ("Jy", Jy_d), ("Jz", Jz_d)):
        full = J[name].matrix(qubits)
        proj = P @ full @ P.conj().T
        dev = float(np.max(np.abs(proj - dicke_op)))
        worst = max(worst, dev)
        print(f"  || P {name}_pauli P^dag - {name}_dicke ||_max = {dev:.3e}")
    jx, jy, jz = (J["Jx"].matrix(qubits), J["Jy"].matrix(qubits), J["Jz"].matrix(qubits))
    comm = float(np.max(np.abs(jx @ jy - jy @ jx - 1j * jz)))
    print(f"  || [Jx, Jy] - i Jz ||_max = {comm:.3e}   (angular-momentum algebra)")
    assert worst < 1e-10 and comm < 1e-10


def report_channels(qubits) -> None:
    N = len(qubits)
    _rule("3. THE TWO STORAGE CHANNELS (CPTP verification)")
    tau = T_MAX / N_STEPS
    for label, ch in (("rc.CollectiveDephasingChannel (linear, Jz)",
                       rc.CollectiveDephasingChannel(N, GAMMA, tau)),
                      ("InteractionDephasingChannel (quadratic, n_mu n_nu)",
                       InteractionDephasingChannel(N, GAMMA, tau))):
        ks = cirq.kraus(ch)
        closure = sum(k.conj().T @ k for k in ks)
        dev = float(np.max(np.abs(closure - np.eye(2**N))))
        print(f"  {label}")
        print(f"     cirq.has_kraus = {cirq.has_kraus(ch)},  #Kraus = {len(ks)},"
              f"  || sum K^dag K - I ||_max = {dev:.3e}")
        assert dev < 1e-10

    print("\n  Sector immunity of the interaction channel (paper: H_c|m<=1> = 0):")
    E = pair_dephasing_exponent(N)
    w = rc.hamming_weights(N)
    low = (w <= 1)[:, None] & (w <= 1)[None, :]
    print(f"     max E[s, s'] over all m_s, m_s' <= 1  = {E[low].max():.0f}  (must be 0)")
    print(f"     max E[s, s'] overall                  = {E.max():.0f}"
          f"  (= binom({N},2) = {N*(N-1)//2})")
    assert E[low].max() == 0

    print("\n  Semigroup / exactness check (pure dephasing has no Trotter error):")
    one_shot = interaction_dephasing_damping(N, GAMMA, 8 * tau)
    stepped = np.ones_like(one_shot)
    for _ in range(8):
        stepped = stepped * interaction_dephasing_damping(N, GAMMA, tau)
    print(f"     || D(8 dt) - D(dt)^8 ||_max = {np.max(np.abs(one_shot - stepped)):.3e}")


def report_cirq_vs_analytic(model: dict, qubits, channel_damping) -> None:
    """Cirq density matrices vs the exact elementwise-damping analytic solution."""
    N = len(qubits)
    psi0 = cirq.final_state_vector(cirq.Circuit(spin_wave_layer(qubits, M_BAR)),
                                   qubit_order=qubits, dtype=DTYPE)
    rho0 = np.outer(psi0, psi0.conj())
    dt = T_MAX / N_STEPS
    worst = 0.0
    for k, rho in enumerate(model["rhos"]):
        ref = rho0 * channel_damping(N, GAMMA, k * dt)
        worst = max(worst, float(np.max(np.abs(rho - ref))))
    print(f"     max_t || rho_Cirq(t) - rho_analytic(t) ||_max = {worst:.3e}")
    print(f"     max_t |Tr rho_Cirq(t) - 1|                    = "
          f"{np.max(np.abs(model['traces'] - 1.0)):.3e}")
    return worst


def report_validation(qubits) -> dict:
    """Cirq vs ported Strang oracle vs exact Liouvillian, with dt refinement."""
    N = len(qubits)
    _rule("5. VALIDATION: Cirq DM sim  vs  ported Strang oracle  vs  exact exp(L t)")
    print(f"  Driven collective master equation: Omega = {VAL_OMEGA}, Delta = {VAL_DELTA}, "
          f"gamma_c = {VAL_GAMMA}")
    print(f"  Initial coherent spin state theta0 = {VAL_THETA0} rad, t_max = {VAL_TMAX}")
    print("\n  Sanity: rc.CollectiveLaserDriveStep vs the exact drive gate (Delta = 0)")
    dt_chk = 0.1
    u_rc = cirq.unitary(rc.CollectiveLaserDriveStep(1, VAL_OMEGA, dt_chk))
    u_ex = cirq.unitary(_drive_gate(VAL_OMEGA, 0.0, dt_chk))
    phase = np.trace(u_ex.conj().T @ u_rc) / 2.0
    print(f"     || U_rc - e^(i phi) U_exact ||_max = "
          f"{np.max(np.abs(u_rc - phase * u_ex)):.3e}  (equal up to global phase)")

    print("\n  dt refinement (all deviations are max over t and over "
          "{Jx,Jy,Jz,Jx2,Jy2,Jz2,JxJy,JxJz,JyJz} and p_m):\n")
    header = (f"   {'steps':>6} {'dt':>8} | {'Strang oracle':>14} {'Cirq Strang':>14} "
              f"{'Cirq Lie-Trot':>14} | {'Cirq vs oracle':>15}")
    print(header)
    print("   " + "-" * (len(header) - 3))

    rows = []
    for num_steps in (6, 12, 24, 48, 96):
        dt = VAL_TMAX / num_steps
        times = np.linspace(0.0, VAL_TMAX, num_steps + 1)
        exact = _exact_liouvillian_solution(N, VAL_OMEGA, VAL_DELTA, VAL_GAMMA,
                                            times, VAL_THETA0)
        oracle = _numpy_master_equation_oracle(N, VAL_OMEGA, VAL_DELTA, VAL_GAMMA,
                                               VAL_TMAX, num_steps, VAL_THETA0)
        cq_s = cirq_driven_run(qubits, VAL_OMEGA, VAL_DELTA, VAL_GAMMA,
                               VAL_TMAX, num_steps, VAL_THETA0, strang=True)
        cq_l = cirq_driven_run(qubits, VAL_OMEGA, VAL_DELTA, VAL_GAMMA,
                               VAL_TMAX, num_steps, VAL_THETA0, strang=False)

        e_or = _obs_deviation(oracle["obs"], exact["obs"], oracle["p_m"], exact["p_m"])
        e_cs = _obs_deviation(cq_s["obs"], exact["obs"], cq_s["p_m"], exact["p_m"])
        e_cl = _obs_deviation(cq_l["obs"], exact["obs"], cq_l["p_m"], exact["p_m"])
        e_co = _obs_deviation(cq_s["obs"], oracle["obs"], cq_s["p_m"], oracle["p_m"])
        rows.append((num_steps, dt, e_or, e_cs, e_cl, e_co))
        print(f"   {num_steps:6d} {dt:8.4f} | {e_or:14.3e} {e_cs:14.3e} "
              f"{e_cl:14.3e} | {e_co:15.3e}")

    def order(col):
        r = np.array([row[col] for row in rows])
        d = np.array([row[1] for row in rows])
        return float(np.polyfit(np.log(d), np.log(np.maximum(r, 1e-16)), 1)[0])

    print(f"\n   fitted convergence order:  Strang oracle  p = {order(2):.2f}  (expect 2)")
    print(f"                              Cirq Strang    p = {order(3):.2f}  (expect 2)")
    print(f"                              Cirq Lie-Trot  p = {order(4):.2f}  (expect 1)")
    print(f"   Cirq-Strang vs NumPy-Strang agree to {max(r[5] for r in rows):.3e} "
          f"(machine precision: identical splitting, independent codes)")
    return {"rows": rows}


def report_cloud_calibration() -> float:
    """Map the dimensionless rate onto the paper's n = 40 long cloud."""
    _rule("6. PHYSICAL TIME SCALE (n = 40 long cloud, PRA 106, L051701 Fig. 2)")
    cloud = rc.LONG_CLOUD
    rng = np.random.default_rng(RNG_SEED)
    pos = rc.sample_cloud(cloud, n=300, rng=rng)
    target = np.exp(-0.5)  # eta = exp(-gamma_p T_s / 2) = e^{-1/2} at gamma_p T_s = 1
    ts_grid = np.logspace(-2, 1.5, 120)
    etas = np.array([rc.coherence_eta(rc.phase_matrix(pos, ts, cloud.c6)) for ts in ts_grid])
    idx = int(np.argmin(np.abs(etas - target)))
    t_unit = float(ts_grid[idx])
    print(f"  cloud: {cloud.label}, sigma = {cloud.sigmas} um, C6/h = {cloud.c6} GHz um^6")
    print(f"  two-atom coherence eta(T_s) = |<exp(-i dPhi)>| crosses e^(-1/2) = {target:.4f}"
          f" at T_s ~ {t_unit:.3f} us")
    print(f"  => one unit of the dimensionless storage time (gamma_p t = 1) ~ {t_unit:.3f} us")
    print(f"  => the full window plotted, gamma_p t = {T_MAX}, is T_s ~ {T_MAX*t_unit:.2f} us")
    return t_unit


# ==========================================================================
# 6.  Figure
# ==========================================================================
def make_figure(coll: dict, pair: dict, t_unit: float, val: dict):
    import matplotlib.pyplot as plt

    rcplot.apply_style()
    N = N_ATOMS
    t = pair["times"]
    colors = rcplot.SERIES_COLORS
    C_PAIR = rcplot.PALETTE["red"]
    C_COLL = rcplot.PALETTE["blue"]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # ---- (a) populations -------------------------------------------------
    m_show = min(4, N)
    for m in range(m_show + 1):
        ax_a.plot(t, pair["p_m"][:, m], color=colors[m % len(colors)], lw=2.2,
                  label=rf"$p_{m}$  (pairwise)")
        ax_a.plot(t, coll["p_m"][:, m], color=colors[m % len(colors)], lw=1.2, ls=":")
    ax_a.set_xlabel(r"storage time  $\gamma_p t$")
    ax_a.set_ylabel(r"phase-matched population  $p_m(t)$")
    ax_a.set_title("(a) excitation-sector populations", loc="left")
    ax_a.set_xlim(t[0], t[-1])
    ax_a.set_ylim(-0.02, 0.72)
    ax_a.legend(ncol=2, loc="center right", fontsize=8.5)
    ax_a.text(0.03, 0.96,
              "solid: interaction (pairwise) dephasing\n"
              "dotted: collective dephasing (frozen)",
              transform=ax_a.transAxes, va="top", fontsize=8.5,
              bbox=dict(fc="white", ec="0.7", alpha=0.9))

    # ---- (b) concurrence -------------------------------------------------
    nc_pair = N * pair["concurrence"]
    nc_coll = N * coll["concurrence"]
    ax_b.plot(t, nc_pair, color=C_PAIR, lw=2.6, label=r"pairwise (interaction) dephasing")
    ax_b.plot(t, nc_coll, color=C_COLL, lw=2.2, ls="--",
              label=r"collective dephasing  $\equiv 0$")
    ax_b.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    k_pk = int(np.argmax(nc_pair))
    ax_b.plot([t[k_pk]], [nc_pair[k_pk]], "o", color=C_PAIR, ms=7, zorder=5)
    ax_b.annotate(rf"peak $N\,C = {nc_pair[k_pk]:.3f}$" "\n" rf"at $\gamma_p t = {t[k_pk]:.2f}$",
                  xy=(t[k_pk], nc_pair[k_pk]), xytext=(0.45, 0.45),
                  textcoords="axes fraction", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color=C_PAIR, lw=1.4))
    ax_b.annotate(rf"$N\,C(0) = {nc_pair[0]:.1e}$" "\n" r"product state: $\bf{no}$ entanglement",
                  xy=(t[0], nc_pair[0]), xytext=(0.06, 0.70),
                  textcoords="axes fraction", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2))
    ax_b.set_xlabel(r"storage time  $\gamma_p t$")
    ax_b.set_ylabel(r"scaled concurrence  $N \times C(\rho_{12})$")
    ax_b.set_title("(b) entanglement is GENERATED by dephasing", loc="left")
    ax_b.set_xlim(t[0], t[-1])
    ax_b.legend(loc="lower right", fontsize=9)

    # ---- (c) Dicke purity ------------------------------------------------
    ax_c.plot(t, pair["dicke_purity"], color=C_PAIR, lw=2.6, label="pairwise dephasing")
    ax_c.plot(t, coll["dicke_purity"], color=C_COLL, lw=2.2, ls="--",
              label="collective dephasing")
    ax_c.axhline(1.0, color=rcplot.PALETTE["gray"], ls=":", lw=1.8,
                 label=r"pure $|W\rangle$ (Dicke $m=1$)")
    ax_c.set_xlabel(r"storage time  $\gamma_p t$")
    ax_c.set_ylabel(r"Dicke purity  $p_1 / (1 - p_0)$")
    ax_c.set_title("(c) purification into the single-excitation Dicke state", loc="left")
    ax_c.set_xlim(t[0], t[-1])
    ax_c.set_ylim(0.3, 1.06)
    ax_c.legend(loc="lower right", fontsize=9)
    ax_c.text(0.03, 0.95,
              rf"$p_1/(1-p_0):\ {pair['dicke_purity'][0]:.3f} \rightarrow "
              rf"{pair['dicke_purity'][-1]:.3f}$",
              transform=ax_c.transAxes, va="top", fontsize=9.5,
              bbox=dict(fc="white", ec="0.7", alpha=0.9))

    # ---- (d) QFI ---------------------------------------------------------
    ax_d.plot(t, pair["qfi_opt"] / N, color=C_PAIR, lw=2.6,
              label=r"$F_Q^{\max}/N$  pairwise")
    ax_d.plot(t, coll["qfi_opt"] / N, color=C_COLL, lw=2.2, ls="--",
              label=r"$F_Q^{\max}/N$  collective")
    ax_d.axhline(1.0, color=rcplot.PALETTE["gray"], ls=":", lw=1.8,
                 label="standard quantum limit")
    ax_d.set_xlabel(r"storage time  $\gamma_p t$")
    ax_d.set_ylabel(r"$F_Q^{\max} / N$")
    ax_d.set_title("(d) metrologically useful entanglement", loc="left")
    ax_d.set_xlim(t[0], t[-1])
    ax_d.legend(loc="center right", fontsize=9)

    ax_d2 = ax_d.twinx()
    ax_d2.plot(t, pair["xi_R2"], color=rcplot.PALETTE["green"], lw=1.6, ls="-.",
               label=r"$\xi_R^2$ pairwise")
    ax_d2.set_ylabel(r"Wineland  $\xi_R^2$", color=rcplot.PALETTE["green"])
    ax_d2.tick_params(axis="y", labelcolor=rcplot.PALETTE["green"])
    ax_d2.grid(False)
    ax_d2.legend(loc="upper right", fontsize=8.5)

    # secondary physical time axis on panel (a)
    ax_a2 = ax_a.secondary_xaxis("top", functions=(lambda x: x * t_unit,
                                                   lambda x: x / t_unit))
    ax_a2.set_xlabel(r"storage time $T_s$ ($\mu$s), $n=40$ long cloud", fontsize=9)

    fig.suptitle(
        "Interaction-induced dephasing converts an unentangled Rydberg spin wave into "
        r"an entangled Dicke state" "\n"
        rf"Cirq {N}-qubit density-matrix simulation, $\bar m = {M_BAR}$   |   "
        "Y. Li, Y. Mei, H. Nguyen, P. R. Berman, A. Kuzmich, "
        "Phys. Rev. A 106, L051701 (2022)",
        fontsize=12.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# ==========================================================================
# 7.  Main
# ==========================================================================
def main() -> None:
    np.set_printoptions(linewidth=140, suppress=False)
    t_start = time.time()
    N = N_ATOMS
    qubits = cirq.LineQubit.range(N)

    print("#" * 78)
    print("#  Cirq density-matrix study of collective vs interaction-induced dephasing")
    print("#  Phys. Rev. A 106, L051701 (2022) -- Li, Mei, Nguyen, Berman, Kuzmich")
    print("#" * 78)

    report_setup(qubits)
    report_conventions(qubits)
    report_channels(qubits)

    # ---- storage runs ----------------------------------------------------
    _rule("4. STORAGE DYNAMICS (cirq.DensityMatrixSimulator, simulate_moment_steps)")
    dt = T_MAX / N_STEPS
    print(f"  N_steps = {N_STEPS}, dt = {dt:.4f} / gamma, t_max = {T_MAX}")

    coll_ch = rc.CollectiveDephasingChannel(N, GAMMA, dt)
    pair_ch = InteractionDephasingChannel(N, GAMMA, dt)

    coll = run_storage_model(qubits, coll_ch, "collective dephasing")
    pair = run_storage_model(qubits, pair_ch, "interaction (pairwise) dephasing")

    for model, damp in ((coll, rc.dicke_dephasing_factors), (pair, None)):
        stats = rc.circuit_stats(model["circuit"])
        print(f"\n  [{model['label']}]  wall time {model['wall']:.2f} s")
        print(f"     circuit stats: {stats}")

    print("\n  Cirq vs exact analytic elementwise-damping solution:")
    print("   [collective]")
    dev_coll = report_cirq_vs_analytic(
        coll, qubits,
        lambda n, g, tau: rc.dicke_dephasing_factors(n, g, tau)[
            np.ix_(rc.hamming_weights(n), rc.hamming_weights(n))])
    print("   [pairwise]")
    dev_pair = report_cirq_vs_analytic(pair, qubits, interaction_dephasing_damping)
    assert dev_coll < 1e-9 and dev_pair < 1e-9

    # ---- headline numbers ------------------------------------------------
    _rule("4b. HEADLINE RESULT")
    nc_p = N * pair["concurrence"]
    nc_c = N * coll["concurrence"]
    k_pk = int(np.argmax(nc_p))
    print(f"  Initial (t=0) state is the product spin wave:")
    print(f"     N x C(0)                        = {nc_p[0]:.3e}   (must be ~0)")
    print(f"     N x C(0), full atomic rho       = {N*pair['concurrence_full'][0]:.3e}")
    print(f"     F_Q^max(0)/N                    = {pair['qfi_opt'][0]/N:.8f}   "
          f"(product state => exactly the SQL, 1)")
    print(f"     retrieval weight Tr(Pi rho Pi)  = {pair['retrieval_weight'][0]:.10f}")
    print("\n  Interaction (pairwise) dephasing:")
    print(f"     peak  N x C = {nc_p[k_pk]:.5f}  at gamma_p t = {pair['times'][k_pk]:.3f}")
    print(f"     final N x C = {nc_p[-1]:.5f}")
    print(f"     Dicke purity p1/(1-p0):  {pair['dicke_purity'][0]:.5f} -> "
          f"{pair['dicke_purity'][-1]:.5f}")
    print(f"     g^(2)                  :  {pair['g2'][0]:.5f} -> {pair['g2'][-1]:.5f}")
    print(f"     F_Q^max/N              :  {pair['qfi_opt'][0]/N:.4f} -> "
          f"{pair['qfi_opt'][-1]/N:.4f}   (SQL = 1)")
    print(f"     xi_R^2                 :  {pair['xi_R2'][0]:.4f} -> {pair['xi_R2'][-1]:.4f}")
    print(f"     p_2                    :  {pair['p_m'][0,2]:.5f} -> {pair['p_m'][-1,2]:.5f}")
    print(f"     symmetric-subspace weight: {pair['retrieval_weight'][0]:.4f} -> "
          f"{pair['retrieval_weight'][-1]:.4f}")
    print("\n  Collective dephasing (null result - a separable, local-unitary mixture):")
    print(f"     max_t  N x C            = {np.max(np.abs(nc_c)):.3e}  (machine zero)")
    print(f"     max_t |p_m(t) - p_m(0)| = "
          f"{np.max(np.abs(coll['p_m'] - coll['p_m'][0])):.3e}  (populations frozen)")
    print(f"     Dicke purity            : {coll['dicke_purity'][0]:.5f} -> "
          f"{coll['dicke_purity'][-1]:.5f}")
    print(f"     F_Q^max/N               : {coll['qfi_opt'][0]/N:.4f} -> "
          f"{coll['qfi_opt'][-1]/N:.4f}")
    print("\n  Analytic cross-check of the paper's finite-N asymptote")
    print("     p_m(inf) -> |c_m|^2 / binom(N, m)  (residual phase-matched background):")
    c2 = rc.poisson_excitation_amplitudes(N, M_BAR)
    from math import comb
    q_inf = np.array([c2[m] / comb(N, m) if m >= 2 else c2[m] for m in range(N + 1)])
    q_inf = q_inf / q_inf.sum()
    print(f"     analytic p_m(inf) = {np.array2string(q_inf, precision=5)}")
    print(f"     Cirq     p_m(T)   = {np.array2string(pair['p_m'][-1], precision=5)}")
    print(f"     max |diff|        = {np.max(np.abs(q_inf - pair['p_m'][-1])):.3e}"
          "   (finite T_max, residual exp(-gamma_p T))")

    val = report_validation(qubits)
    t_unit = report_cloud_calibration()

    # ---- figure ----------------------------------------------------------
    _rule("7. FIGURE")
    fig = make_figure(coll, pair, t_unit, val)
    rcplot.save_figure(fig, __file__, "cirq_collective_dephasing.png")

    print(f"\n  total wall time: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
