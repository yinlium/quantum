"""
Photon-counting (shot-noise-limited) measurement of ``g2`` for a dephased Rydberg
spin wave -- the way the experiment actually does it.

Reference
---------
Y. Li, Y. Mei, H. Nguyen, P. R. Berman and A. Kuzmich,
*Dynamics of collective-dephasing-induced multiatom entanglement*,
Phys. Rev. A **106**, L051701 (2022)  [and its Supplemental Material].

Motivation
----------
Everywhere else in ``entanglement/`` the second-order autocorrelation

    g2(T_s) = <S+ S+ S S> / <S+ S>^2 ,
    S = S_{k0} = (1 / sqrt(N)) sum_mu exp(i k0 . r_mu) sigma^{gr}_mu

is obtained from *exact expectation values*.  A real experiment never has access
to an expectation value: it retrieves the spin wave into the phase-matched
optical mode, splits the light onto an array of single-photon counters and
accumulates **finite photon-counting statistics**.  This module reproduces that
procedure inside Cirq and quantifies the resulting shot noise.

Mapping: phase-matched retrieved field  ->  detector clicks
----------------------------------------------------------
After the storage interval the ensemble is read out with a phase-matched control
field.  In the standard DLCZ treatment the positive-frequency part of the
retrieved field operator in the phase-matched direction ``k0`` is proportional to
the collective spin-wave destruction operator ``S_{k0}`` (the commutator
``[S, S+] = 1 - O(m/N)`` is bosonic to ``O(1/N)``), so the photon-number operator
of the retrieved mode is ``n_ph = eta_r S+ S`` and the **two-fold coincidence
operator** is ``n_ph (n_ph - 1) = eta_r^2 S+ S+ S S``.  Light emitted into any
other spin-wave mode is not phase matched, misses the fibre, and is simply lost --
that loss is exactly what the factors ``Y_m`` and ``X_m`` of Eqs. (S.7)-(S.8)
describe.

We implement the detection stage literally, as a **multiplexed Hanbury Brown -
Twiss detector array** made of ``K`` ancilla "detector" qubits.  Each detector
``d`` is weakly coupled to the collective mode by a beam-splitter interaction

    H_d = (theta / sqrt(N)) sum_mu ( sigma^+_mu sigma^-_d + h.c. )
        -> prod_mu  exp( -i (theta/sqrt(N)) (X_mu X_d + Y_mu Y_d) / 2 )
        -> prod_mu  cirq.ISwapPowGate(exponent = -2 theta / (pi sqrt(N)))

so that, to leading order in ``theta``, the amplitude for detector ``d`` to click
is ``-i theta S |psi>``.  Consequently, with ``n`` the total number of clicks in a
shot,

    <n>        = K       theta^2 <S+ S>            (singles rate)
    <n(n-1)>   = K(K-1)  theta^4 <S+ S+ S S>       (two-fold coincidence rate)

and the detector-efficiency-independent estimator is the normalised second
factorial moment of the sampled click distribution

    g2_hat = K/(K-1) * <n(n-1)> / <n>^2  ->  <S+S+SS> / <S+S>^2 .

The prefactor ``K/(K-1)`` removes the finite-pixel-number bias of an array of
on/off (non-number-resolving) detectors; every other instrumental factor
(``theta``, ``eta_r``, ``K``) cancels in the ratio, exactly as detection
efficiency cancels in a real HBT measurement.  The residual systematic is
``O(theta^2)`` from detector saturation/back-action; it is measured explicitly
below and kept below the quoted statistical error bars.

Physics being sampled
---------------------
* Spin-wave preparation: :class:`rydberg_cirq.CollectiveLaserDriveStep` with
  ``Omega_1 dt = 2 arcsin(sqrt(m_bar / N))``, giving the binomial (-> Poissonian)
  amplitudes ``c_m = sqrt(C(N,m)) a^{N-m} b^m`` of Eq. (2).
* Storage: :class:`rydberg_cirq.PairwisePhaseGate` built from a Gaussian cloud
  sampled with :func:`rydberg_cirq.sample_cloud` / :func:`rydberg_cirq.phase_matrix`
  (``N(N-1)/2`` exact ``CZPowGate``s, no Trotter error).  Every shot batch uses a
  fresh spatial configuration, reproducing the ensemble average over atomic
  positions.
* A control experiment with :class:`rydberg_cirq.CollectiveDephasingChannel`
  demonstrates the central claim of the paper: *collective* (m-diagonal)
  dephasing leaves ``g2`` **exactly** invariant, whereas *interaction-induced*
  (position-dependent) dephasing -- which scrambles population among collective
  modes at fixed ``m`` -- is what drives ``g2 -> 0``.

Cirq APIs showcased
-------------------
``cirq.Simulator.run`` with ``repetitions`` (true sampling), ``cirq.measure``,
``cirq.ISwapPowGate``, ``cirq.DensityMatrixSimulator`` (for the CPTP control),
custom gates/channels from ``rydberg_cirq``, and ``cirq.Simulator.simulate`` for
the exact reference values.

Outputs
-------
``cirq_g2_shot_noise.png`` with four panels and a console report containing the
shots-vs-estimate-vs-error-vs-deviation convergence table.
"""

from __future__ import annotations

import time

import numpy as np
import cirq

import rydberg_cirq as rc
from rydberg_cirq.plotting import PALETTE, apply_style, save_figure

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
N_ATOMS = 8            # statevector-simulable ensemble
N_DETECTORS = 10       # multiplexed HBT detector pixels
THETA = 0.10           # weak detector coupling (systematic bias ~ theta^2)
M_BAR = 1.0            # mean number of Rydberg excitations
CLOUD = rc.SHORT_CLOUD  # n = 50, C6/h = 15.44 GHz um^6, sigma_z = 5.25 um

STORAGE_TIMES = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
N_CONFIGS_CURVE = 12
SHOTS_PER_CONFIG_CURVE = 30_000

TS_REF = 1.0                    # storage time used for the convergence study
N_CONFIGS_POOL = 25
SHOTS_PER_CONFIG_POOL = 200_000
SHOT_GRID = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 2_000_000])

N_BOOTSTRAP = 600
SEED = 20221101


# --------------------------------------------------------------------------- #
# Local helpers (nothing in rydberg_cirq/ is modified)
# --------------------------------------------------------------------------- #
def spin_wave_operator(n_atoms: int) -> np.ndarray:
    """Dense ``2^N x 2^N`` matrix of ``S_{k0} = (1/sqrt(N)) sum_mu sigma^-_mu``.

    Working in the co-moving frame of the write beam absorbs ``exp(i k0 . r_mu)``,
    so the phase-matched mode is the *uniform* superposition.  ``S`` is
    permutation invariant, hence independent of the qubit ordering convention.
    """
    dim = 2**n_atoms
    op = np.zeros((dim, dim), dtype=complex)
    for s in range(dim):
        for j in range(n_atoms):
            bit = 1 << (n_atoms - 1 - j)
            if s & bit:                       # atom j excited -> de-excite it
                op[s ^ bit, s] += 1.0
    return op / np.sqrt(n_atoms)


def retrieved_moments(state: np.ndarray, s_op: np.ndarray) -> tuple[float, float]:
    """``(<S+S>, <S+S+SS>)`` -- singles and two-fold coincidence rates."""
    if state.ndim == 1:
        s1 = s_op @ state
        s2 = s_op @ s1
        return float(np.vdot(s1, s1).real), float(np.vdot(s2, s2).real)
    # density matrix
    sd = s_op.conj().T
    return (
        float(np.trace(sd @ s_op @ state).real),
        float(np.trace(sd @ sd @ s_op @ s_op @ state).real),
    )


def spin_wave_circuit(atoms, m_bar: float) -> cirq.Circuit:
    """Short excitation pulse producing the unentangled spin wave ``sum_m c_m |m>``."""
    n = len(atoms)
    pulse_area = 2.0 * np.arcsin(np.sqrt(m_bar / n))   # Omega_1 Omega_2 T_p / (2 Delta)
    return cirq.Circuit(rc.CollectiveLaserDriveStep(n, omega_1=pulse_area, dt=1.0).on(*atoms))


def storage_circuit(atoms, positions: np.ndarray, storage_time: float) -> cirq.Circuit:
    """Exact ``U(T_s) = prod_{mu<nu} exp(-i Phi_{mu nu} n_mu n_nu)``."""
    phi = rc.phase_matrix(positions, storage_time, CLOUD.c6)
    return cirq.Circuit(rc.PairwisePhaseGate(phi).on(*atoms))


def detector_array_circuit(atoms, detectors, theta: float) -> cirq.Circuit:
    """Weak collective beam-splitter coupling of the spin wave to ``K`` detectors."""
    exponent = -2.0 * (theta / np.sqrt(len(atoms))) / np.pi
    circuit = cirq.Circuit()
    for det in detectors:
        for atom in atoms:
            circuit.append(cirq.ISwapPowGate(exponent=exponent).on(atom, det))
    return circuit


def g2_from_click_histogram(counts: np.ndarray, n_det: int) -> float:
    """``g2_hat = K/(K-1) <n(n-1)>/<n>^2`` from a histogram of clicks-per-shot."""
    n_vals = np.arange(len(counts))
    total = counts.sum()
    if total == 0:
        return np.nan
    mean_n = float(counts @ n_vals) / total
    mean_nn = float(counts @ (n_vals * (n_vals - 1))) / total
    if mean_n <= 0.0:
        return np.nan
    return (n_det / (n_det - 1.0)) * mean_nn / mean_n**2


def bootstrap_g2_error(counts: np.ndarray, n_det: int, rng, n_boot: int = N_BOOTSTRAP) -> float:
    """Non-parametric bootstrap std of ``g2_hat``.

    The estimator depends on the shots only through the click histogram, so an
    exact bootstrap is obtained by resampling the multinomial counts -- ``O(K)``
    per replica instead of ``O(M)``.
    """
    total = int(counts.sum())
    if total == 0:
        return np.nan
    probs = counts / total
    draws = rng.multinomial(total, probs, size=n_boot).astype(float)
    n_vals = np.arange(len(counts), dtype=float)
    mean_n = draws @ n_vals / total
    mean_nn = draws @ (n_vals * (n_vals - 1.0)) / total
    good = mean_n > 0
    if good.sum() < 2:
        return np.nan
    vals = (n_det / (n_det - 1.0)) * mean_nn[good] / mean_n[good] ** 2
    return float(np.std(vals, ddof=1))


def delta_method_g2_error(counts: np.ndarray, n_det: int) -> float:
    """Analytic (multinomial / delta-method) error propagation, for cross-check.

    With ``a_i = n_i(n_i-1)``, ``b_i = n_i``, ``A = <a>``, ``B = <b>`` and
    ``kappa = K/(K-1)``,

        Var(g2) = kappa^2 / M * [ Var(a)/B^4 - 4 A Cov(a,b)/B^5 + 4 A^2 Var(b)/B^6 ].
    """
    total = int(counts.sum())
    if total == 0:
        return np.nan
    p = counts / total
    n_vals = np.arange(len(counts), dtype=float)
    a = n_vals * (n_vals - 1.0)
    b = n_vals
    A, B = float(p @ a), float(p @ b)
    if B <= 0.0:
        return np.nan
    var_a = float(p @ a**2) - A**2
    var_b = float(p @ b**2) - B**2
    cov_ab = float(p @ (a * b)) - A * B
    kappa = n_det / (n_det - 1.0)
    var = kappa**2 / total * (
        var_a / B**4 - 4.0 * A * cov_ab / B**5 + 4.0 * A**2 * var_b / B**6
    )
    return float(np.sqrt(max(var, 0.0)))


def scaling_ansatz_g2(x2: float, y2: float, n_atoms: int, p_m: np.ndarray) -> float:
    """``g2`` from the paper's ``N^2`` scaling ansatz, Eqs. (S.7)-(S.8).

    ``X_m = [(N-m)^2 + 3(N-m)]/N^2 * eta_x^(2m-3) + 2/N^2``
    ``Y_m = (N-m)/N * eta_y^(m-1) + 1/N``
    with ``eta`` back-extracted from the exactly simulated ``X_2``, ``Y_2``.
    """
    n = n_atoms
    eta_x = max(0.0, (x2 - 2.0 / n**2) / (1.0 - 1.0 / n - 2.0 / n**2))
    eta_y = max(0.0, (y2 - 1.0 / n) / (1.0 - 2.0 / n))
    m = np.arange(len(p_m))
    x_m = np.where(
        m >= 2,
        ((n - m) ** 2 + 3.0 * (n - m)) / n**2 * eta_x ** np.clip(2 * m - 3, 0, None) + 2.0 / n**2,
        0.0,
    )
    y_m = (n - m) / n * eta_y ** np.clip(m - 1, 0, None) + 1.0 / n
    return rc.g2_from_populations(p_m, x_m, y_m)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    t_start = time.time()
    apply_style()
    rng = np.random.default_rng(SEED)
    sim = cirq.Simulator(seed=SEED)

    atoms = cirq.LineQubit.range(N_ATOMS)
    detectors = cirq.LineQubit.range(N_ATOMS, N_ATOMS + N_DETECTORS)
    s_op = spin_wave_operator(N_ATOMS)
    p_m_exact = rc.poisson_excitation_amplitudes(N_ATOMS, M_BAR)
    dicke2 = rc.dicke_state_vector(N_ATOMS, 2).astype(np.complex64)
    n_vals = np.arange(N_DETECTORS + 1)

    print("=" * 78)
    print("  SHOT-NOISE-LIMITED g2 OF A DEPHASED RYDBERG SPIN WAVE")
    print("  Phys. Rev. A 106, L051701 (2022) -- sampled photon coincidences in Cirq")
    print("=" * 78)
    print(f"  atoms N                 : {N_ATOMS}   (statevector dim 2^{N_ATOMS})")
    print(f"  detector pixels K       : {N_DETECTORS}  ->  {N_ATOMS + N_DETECTORS}-qubit circuits")
    print(f"  detector coupling theta : {THETA}")
    print(f"  mean excitation m_bar   : {M_BAR}")
    print(f"  cloud                   : {CLOUD.label}, C6/h = {CLOUD.c6} GHz um^6")
    print(f"  |c_m|^2 (binomial)      : " + " ".join(f"{v:.4f}" for v in p_m_exact[:5]) + " ...")

    # ------------------------------------------------------------------ #
    # 0.  Circuit anatomy
    # ------------------------------------------------------------------ #
    demo_pos = rc.sample_cloud(CLOUD, n=N_ATOMS, rng=np.random.default_rng(0))
    demo_circuit = (
        spin_wave_circuit(atoms, M_BAR)
        + storage_circuit(atoms, demo_pos, TS_REF)
        + detector_array_circuit(atoms, detectors, THETA)
        + cirq.Circuit(cirq.measure(*detectors, key="clicks"))
    )
    stats = rc.circuit_stats(cirq.expand_composite(demo_circuit))
    print("\n[0] Circuit anatomy (after cirq.expand_composite)")
    print(f"    qubits={stats['qubits']}  moments={stats['moments']}  "
          f"1q ops={stats['1q_ops']}  2q ops={stats['2q_ops']}")
    print("    stages: CollectiveLaserDriveStep -> PairwisePhaseGate -> "
          "K x (N x ISwapPowGate) -> cirq.measure")

    # ------------------------------------------------------------------ #
    # 1.  Control: collective vs interaction-induced dephasing
    # ------------------------------------------------------------------ #
    print("\n[1] Control -- which kind of dephasing actually suppresses g2?")
    dm_sim = cirq.DensityMatrixSimulator()
    base = spin_wave_circuit(atoms, M_BAR)
    rho0 = dm_sim.simulate(base).final_density_matrix.astype(complex)
    s1_0, s2_0 = retrieved_moments(rho0, s_op)
    print(f"    no dephasing                       : g2 = {s2_0 / s1_0**2:.6f}")
    for gamma_tau in (0.5, 5.0, 50.0):
        circ = base + cirq.Circuit(
            rc.CollectiveDephasingChannel(N_ATOMS, gamma_c=gamma_tau, tau=1.0).on(*atoms)
        )
        rho = dm_sim.simulate(circ).final_density_matrix.astype(complex)
        s1, s2 = retrieved_moments(rho, s_op)
        print(f"    CollectiveDephasingChannel g*t={gamma_tau:5.1f} : g2 = {s2 / s1**2:.6f}   "
              f"(purity {np.trace(rho @ rho).real:.4f})")
    ctrl_pos = rc.sample_cloud(CLOUD, n=N_ATOMS, rng=np.random.default_rng(1))
    circ = base + storage_circuit(atoms, ctrl_pos, 10.0)
    psi = sim.simulate(circ).final_state_vector.astype(complex)
    s1, s2 = retrieved_moments(psi, s_op)
    print(f"    PairwisePhaseGate  T_s = 10 us     : g2 = {s2 / s1**2:.6f}")
    print("    => m-diagonal collective dephasing leaves g2 EXACTLY invariant;")
    print("       only position-dependent pairwise phases (which move population")
    print("       out of the phase-matched collective mode) give g2 -> 0.")

    # ------------------------------------------------------------------ #
    # 2.  Detector-model systematic vs theta
    # ------------------------------------------------------------------ #
    print("\n[2] Detector back-action systematic (expected O(theta^2))")
    sys_pos = rc.sample_cloud(CLOUD, n=N_ATOMS, rng=np.random.default_rng(2))
    atom_only = spin_wave_circuit(atoms, M_BAR) + storage_circuit(atoms, sys_pos, TS_REF)
    psi_ref = sim.simulate(atom_only).final_state_vector.astype(complex)
    s1_ref, s2_ref = retrieved_moments(psi_ref, s_op)
    g2_ref_single = s2_ref / s1_ref**2
    print(f"    {'theta':>7} {'<n>':>9} {'<n(n-1)>':>11} {'g2_model':>10} {'bias':>9}")
    for theta in (0.05, 0.10, 0.20, 0.30):
        full = atom_only + detector_array_circuit(atoms, detectors, theta)
        sv = sim.simulate(full).final_state_vector
        probs = (np.abs(sv) ** 2).reshape(2**N_ATOMS, 2**N_DETECTORS).sum(axis=0)
        clicks = np.array([int(i).bit_count() for i in range(2**N_DETECTORS)])
        hist = np.bincount(clicks, weights=probs, minlength=N_DETECTORS + 1)
        mean_n = float(hist @ n_vals)
        mean_nn = float(hist @ (n_vals * (n_vals - 1)))
        g2_model = (N_DETECTORS / (N_DETECTORS - 1.0)) * mean_nn / mean_n**2
        print(f"    {theta:7.2f} {mean_n:9.5f} {mean_nn:11.3e} {g2_model:10.6f} "
              f"{100 * (g2_model / g2_ref_single - 1):+8.2f}%")
    print(f"    exact operator value g2 = <S+S+SS>/<S+S>^2 = {g2_ref_single:.6f}")

    # ------------------------------------------------------------------ #
    # 3.  Sampled g2 vs storage time
    # ------------------------------------------------------------------ #
    print(f"\n[3] Sampling g2(T_s):  {len(STORAGE_TIMES)} storage times x "
          f"{N_CONFIGS_CURVE} cloud configurations x {SHOTS_PER_CONFIG_CURVE} shots")
    clouds_curve = [rc.sample_cloud(CLOUD, n=N_ATOMS, rng=rng) for _ in range(N_CONFIGS_CURVE)]

    g2_sampled, g2_err, g2_exact_curve, g2_ansatz_curve = [], [], [], []
    for t_s in STORAGE_TIMES:
        hist = np.zeros(N_DETECTORS + 1, dtype=np.int64)
        num = den = 0.0
        x2_acc = y2_acc = 0.0
        for positions in clouds_curve:
            store = storage_circuit(atoms, positions, t_s)
            prep = spin_wave_circuit(atoms, M_BAR) + store

            # exact reference from the atoms-only statevector
            psi = sim.simulate(prep).final_state_vector.astype(complex)
            a, b = retrieved_moments(psi, s_op)
            den += a
            num += b

            # X_2, Y_2 straight out of the Cirq |m=2> circuit (paper's prescription)
            ev = sim.simulate(store, initial_state=dicke2).final_state_vector.astype(complex)
            s1v = s_op @ ev
            s2v = s_op @ s1v
            y2_acc += float(np.vdot(s1v, s1v).real) / 2.0
            x2_acc += float(np.vdot(s2v, s2v).real) / 2.0

            # sampled photon counting
            circuit = (
                prep
                + detector_array_circuit(atoms, detectors, THETA)
                + cirq.Circuit(cirq.measure(*detectors, key="clicks"))
            )
            result = sim.run(circuit, repetitions=SHOTS_PER_CONFIG_CURVE)
            clicks = result.measurements["clicks"].sum(axis=1)
            hist += np.bincount(clicks, minlength=N_DETECTORS + 1).astype(np.int64)

        num /= N_CONFIGS_CURVE
        den /= N_CONFIGS_CURVE
        g2_exact_curve.append(num / den**2)
        g2_ansatz_curve.append(
            scaling_ansatz_g2(x2_acc / N_CONFIGS_CURVE, y2_acc / N_CONFIGS_CURVE,
                              N_ATOMS, p_m_exact)
        )
        g2_sampled.append(g2_from_click_histogram(hist, N_DETECTORS))
        g2_err.append(bootstrap_g2_error(hist, N_DETECTORS, rng))

    g2_sampled = np.array(g2_sampled)
    g2_err = np.array(g2_err)
    g2_exact_curve = np.array(g2_exact_curve)
    g2_ansatz_curve = np.array(g2_ansatz_curve)
    pulls = (g2_sampled - g2_exact_curve) / g2_err

    print(f"    {'T_s (us)':>9} {'g2 sampled':>12} {'+- stat':>9} {'g2 exact':>10} "
          f"{'ansatz':>9} {'pull':>7}")
    for i, t_s in enumerate(STORAGE_TIMES):
        print(f"    {t_s:9.2f} {g2_sampled[i]:12.4f} {g2_err[i]:9.4f} "
              f"{g2_exact_curve[i]:10.4f} {g2_ansatz_curve[i]:9.4f} {pulls[i]:+7.2f}")
    print(f"    RMS pull = {np.sqrt(np.mean(pulls**2)):.2f} (expect ~1 for correct error bars)")
    print(f"    max |ansatz - exact| = {np.max(np.abs(g2_ansatz_curve - g2_exact_curve)):.4f}")

    # ------------------------------------------------------------------ #
    # 4.  Convergence study at T_s = TS_REF
    # ------------------------------------------------------------------ #
    print(f"\n[4] Shot-noise convergence at T_s = {TS_REF} us "
          f"({N_CONFIGS_POOL} configs x {SHOTS_PER_CONFIG_POOL} shots "
          f"= {N_CONFIGS_POOL * SHOTS_PER_CONFIG_POOL:,} shots)")
    clouds_pool = [rc.sample_cloud(CLOUD, n=N_ATOMS, rng=rng) for _ in range(N_CONFIGS_POOL)]
    pool = np.empty(N_CONFIGS_POOL * SHOTS_PER_CONFIG_POOL, dtype=np.int8)
    num = den = 0.0
    model_num = model_den = 0.0
    for i, positions in enumerate(clouds_pool):
        prep = spin_wave_circuit(atoms, M_BAR) + storage_circuit(atoms, positions, TS_REF)
        psi = sim.simulate(prep).final_state_vector.astype(complex)
        a, b = retrieved_moments(psi, s_op)
        den += a
        num += b

        detected = prep + detector_array_circuit(atoms, detectors, THETA)
        sv = sim.simulate(detected).final_state_vector
        probs = (np.abs(sv) ** 2).reshape(2**N_ATOMS, 2**N_DETECTORS).sum(axis=0)
        bits = np.array([int(k).bit_count() for k in range(2**N_DETECTORS)])
        hist_exact = np.bincount(bits, weights=probs, minlength=N_DETECTORS + 1)
        model_den += float(hist_exact @ n_vals)
        model_num += float(hist_exact @ (n_vals * (n_vals - 1)))

        result = sim.run(
            detected + cirq.Circuit(cirq.measure(*detectors, key="clicks")),
            repetitions=SHOTS_PER_CONFIG_POOL,
        )
        lo = i * SHOTS_PER_CONFIG_POOL
        pool[lo:lo + SHOTS_PER_CONFIG_POOL] = result.measurements["clicks"].sum(axis=1)

    rng.shuffle(pool)   # mix configurations so sub-samples are representative
    g2_exact_ref = (num / N_CONFIGS_POOL) / (den / N_CONFIGS_POOL) ** 2
    g2_model_ref = (N_DETECTORS / (N_DETECTORS - 1.0)) * (model_num / N_CONFIGS_POOL) / (
        model_den / N_CONFIGS_POOL
    ) ** 2
    systematic = abs(g2_model_ref - g2_exact_ref)
    print(f"    exact operator g2      = {g2_exact_ref:.6f}")
    print(f"    detector-model g2      = {g2_model_ref:.6f}   "
          f"(theta^2 systematic {100 * systematic / g2_exact_ref:+.2f}%)")

    rows = []
    for m_shots in SHOT_GRID:
        n_blocks = int(min(40, max(3, pool.size // m_shots)))
        ests = []
        for k in range(n_blocks):
            block = pool[k * m_shots:(k + 1) * m_shots]
            ests.append(g2_from_click_histogram(
                np.bincount(block, minlength=N_DETECTORS + 1), N_DETECTORS))
        ests = np.array(ests, dtype=float)
        first_hist = np.bincount(pool[:m_shots], minlength=N_DETECTORS + 1)
        rows.append(
            dict(
                M=int(m_shots),
                est=float(ests[0]),
                boot=bootstrap_g2_error(first_hist, N_DETECTORS, rng),
                delta=delta_method_g2_error(first_hist, N_DETECTORS),
                rms=float(np.sqrt(np.mean((ests - g2_model_ref) ** 2))),
                blocks=n_blocks,
                coinc=int((pool[:m_shots] >= 2).sum()),
            )
        )

    print(f"\n    {'shots M':>10} {'g2 est':>9} {'boot err':>9} {'delta err':>10} "
          f"{'RMS err':>9} {'dev/exact':>10} {'pull':>7} {'n>=2':>7}")
    for r in rows:
        dev = r["est"] - g2_exact_ref
        pull = dev / r["boot"] if r["boot"] and np.isfinite(r["boot"]) and r["boot"] > 0 else np.nan
        print(f"    {r['M']:10,} {r['est']:9.4f} {r['boot']:9.4f} {r['delta']:10.4f} "
              f"{r['rms']:9.4f} {dev:+10.4f} {pull:+7.2f} {r['coinc']:7d}")

    finite = [r for r in rows if np.isfinite(r["rms"]) and r["rms"] > 0]
    slope = np.polyfit(np.log10([r["M"] for r in finite[2:]]),
                       np.log10([r["rms"] for r in finite[2:]]), 1)[0]
    print(f"    fitted log-log slope of RMS error vs M : {slope:+.3f}  (ideal -0.500)")

    # ------------------------------------------------------------------ #
    # 5.  Sampled excitation-number histogram
    # ------------------------------------------------------------------ #
    print("\n[5] Sampled atomic excitation-number distribution (source characterisation)")
    src = spin_wave_circuit(atoms, M_BAR) + cirq.Circuit(cirq.measure(*atoms, key="m"))
    src_shots = 400_000
    m_samples = sim.run(src, repetitions=src_shots).measurements["m"].sum(axis=1)
    p_m_sampled = np.bincount(m_samples, minlength=N_ATOMS + 1) / src_shots
    p_m_poisson = np.exp(-M_BAR) * M_BAR ** np.arange(N_ATOMS + 1) / np.array(
        [np.math.factorial(k) if hasattr(np, "math") else float(np.prod(range(1, k + 1)) or 1)
         for k in range(N_ATOMS + 1)]
    )
    print(f"    {'m':>3} {'sampled':>10} {'binomial':>10} {'Poisson':>10}")
    for m in range(min(5, N_ATOMS + 1)):
        print(f"    {m:3d} {p_m_sampled[m]:10.5f} {p_m_exact[m]:10.5f} {p_m_poisson[m]:10.5f}")
    print(f"    sampled <m> = {m_samples.mean():.5f}  (target m_bar = {M_BAR})")
    print(f"    max |sampled - binomial| = {np.max(np.abs(p_m_sampled - p_m_exact)):.5f} "
          f"(1-sigma shot noise ~ {np.sqrt(0.25 / src_shots):.5f})")

    # ------------------------------------------------------------------ #
    # 6.  Figure
    # ------------------------------------------------------------------ #
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    t_plot = np.where(STORAGE_TIMES > 0, STORAGE_TIMES, 3e-3)

    # (a) g2 vs storage time
    ax_a.plot(t_plot, g2_exact_curve, "-", color=PALETTE["blue"],
              label=r"exact $\langle S^\dagger S^\dagger SS\rangle/\langle S^\dagger S\rangle^2$")
    ax_a.plot(t_plot, g2_ansatz_curve, "--", color=PALETTE["green"], lw=1.8,
              label=r"scaling ansatz Eqs. (S.7)-(S.8)")
    ax_a.errorbar(t_plot, g2_sampled, yerr=g2_err, fmt="o", color=PALETTE["red"],
                  ecolor=PALETTE["red"], ms=6, capsize=3, lw=0, elinewidth=1.6,
                  label=f"sampled, {N_CONFIGS_CURVE * SHOTS_PER_CONFIG_CURVE // 1000}k shots/point")
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"storage time $T_s$ ($\mu$s)")
    ax_a.set_ylabel(r"$g^{(2)}(T_s)$")
    ax_a.set_title("(a) photon-counting $g^{(2)}$ vs exact theory", loc="left")
    ax_a.set_ylim(0.0, 1.05)
    ax_a.axhline(1.0, color=PALETTE["gray"], ls=":", lw=1.2)
    ax_a.legend(loc="lower left")
    ax_a.text(0.97, 0.93, f"$N={N_ATOMS}$, $\\bar m={M_BAR}$\n{CLOUD.label}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=8.5,
              bbox=dict(fc="white", ec="0.8", alpha=0.9))

    # (b) error vs shots
    m_arr = np.array([r["M"] for r in rows], dtype=float)
    rms_arr = np.array([r["rms"] for r in rows], dtype=float)
    boot_arr = np.array([r["boot"] for r in rows], dtype=float)
    ax_b.loglog(m_arr, rms_arr, "o-", color=PALETTE["blue"], ms=6,
                label="RMS error over disjoint blocks")
    ax_b.loglog(m_arr, boot_arr, "s--", color=PALETTE["orange"], ms=5,
                label="bootstrap $1\\sigma$ estimate")
    guide = rms_arr[-1] * np.sqrt(m_arr[-1] / m_arr)
    ax_b.loglog(m_arr, guide, ":", color=PALETTE["black"], lw=1.8,
                label=r"$\propto M^{-1/2}$ guide")
    ax_b.axhline(systematic, color=PALETTE["purple"], ls="-.", lw=1.6,
                 label=rf"$\theta^2$ detector systematic ({systematic:.4f})")
    ax_b.set_xlabel("number of shots $M$")
    ax_b.set_ylabel(r"$|\Delta g^{(2)}|$")
    ax_b.set_title("(b) shot-noise scaling of the estimator", loc="left")
    ax_b.legend(loc="lower left", fontsize=8.5)

    # (c) estimate vs shots
    est_arr = np.array([r["est"] for r in rows], dtype=float)
    ax_c.errorbar(m_arr, est_arr, yerr=boot_arr, fmt="o", color=PALETTE["red"],
                  ms=6, capsize=3, lw=0, elinewidth=1.6, label="sampled estimate")
    ax_c.axhline(g2_model_ref, color=PALETTE["blue"], lw=2.0,
                 label=f"detector-model exact ({g2_model_ref:.4f})")
    ax_c.axhline(g2_exact_ref, color=PALETTE["green"], ls="--", lw=1.8,
                 label=f"operator exact ({g2_exact_ref:.4f})")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("number of shots $M$")
    ax_c.set_ylabel(r"$\hat g^{(2)}$")
    ax_c.set_title(rf"(c) convergence at $T_s={TS_REF}\,\mu$s", loc="left")
    ax_c.legend(loc="upper right", fontsize=8.5)

    # (d) excitation-number histogram
    m_axis = np.arange(N_ATOMS + 1)
    width = 0.38
    ax_d.bar(m_axis - width / 2, p_m_sampled, width, color=PALETTE["red"], alpha=0.85,
             yerr=np.sqrt(np.maximum(p_m_sampled, 0) * (1 - p_m_sampled) / src_shots),
             capsize=2, label=f"sampled ({src_shots // 1000}k shots)")
    ax_d.bar(m_axis + width / 2, p_m_exact, width, color=PALETTE["blue"], alpha=0.85,
             label=r"binomial $|c_m|^2=\binom{N}{m}a^{2(N-m)}b^{2m}$")
    ax_d.plot(m_axis, p_m_poisson, "k^--", ms=6, lw=1.6,
              label=rf"Poisson($\bar m={M_BAR}$)")
    ax_d.set_yscale("log")
    ax_d.set_ylim(1e-6, 1.3)
    ax_d.set_xlim(-0.6, 5.6)
    ax_d.set_xlabel("excitation number $m$")
    ax_d.set_ylabel(r"$P(m)$")
    ax_d.set_title("(d) sampled spin-wave excitation statistics", loc="left")
    ax_d.legend(loc="lower left", fontsize=8.5)

    fig.suptitle(
        "Shot-noise-limited $g^{(2)}$ from sampled photon coincidences in Cirq\n"
        "interaction-induced dephasing of a Rydberg spin wave "
        "[Y. Li $et~al.$, Phys. Rev. A $\\bf{106}$, L051701 (2022)]",
        fontsize=12.5, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save_figure(fig, __file__, "cirq_g2_shot_noise.png")

    print(f"\n  total runtime {time.time() - t_start:.1f} s")
    print("=" * 78)


if __name__ == "__main__":
    main()
