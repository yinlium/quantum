"""Interaction-induced dephasing of g2 for an extended atomic cloud (Sec. IV).

Reproduces the interaction-induced-dephasing part of P.R. Berman, H. Nguyen,
Y. Mei, Y. Li, A. Kuzmich, "Interference bunching and antibunching of coherent
atomic radiation fields," Phys. Rev. A 108, 043713 (2023), Sec. IV. Atoms
excited to a Rydberg state interact via the van der Waals C6/R^6 potential, so
each atom pair accumulates a random relative phase during a retrieval pulse of
duration Tp; this dephases the ensemble's collective spin-wave phase Q(t) and
its magnitude-squared correlator G(t) = <|Q(t)|^2>, and hence g2(t). Q, G are
evaluated by Monte Carlo averaging over random atomic positions within a
sphere of radius R (uniform cloud); the point-pulse limit (Tp -> tau) analogs
are labeled Q_pt, G_pt. The n=75 Rb Rydberg C3/C6 coefficients feeding the
pair interaction are obtained with the ``arc`` (Alkali Rydberg Calculator)
package, cross-checked against the Kaulakys radial-matrix-element formula used
in the original analysis.

Monte Carlo sample counts and atom numbers are reduced from the original
notebook (100 seeds, N=400) to keep runtime short for a demo script; the
formulas are unchanged. Source: Dephasing_230317.ipynb.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# ---------------------------------------------------------------------------
# Rydberg (n S_1/2 <-> n P_3/2, (n-1) P_3/2) atomic structure for Rb87
# (quantum-defect formulas from Li, Han et al.; only the S and P3/2 branches
#  that Rydberg_para() below actually needs are kept).
# ---------------------------------------------------------------------------
_HCONST = 6.62606896e-34  # Planck constant
_CLIGHT = 2.99792458e8  # speed of light
_RY = 1.0973731568527e7  # Rydberg constant
_A0 = 0.52917720859e-10  # Bohr radius
_ALPHA = 7.2973525376e-3  # fine-structure constant


@jit(nopython=True, nogil=True)
def rydberg_energy(n, l, j):
    """Energy of an alkali Rydberg level via quantum defects (S_1/2 or P_3/2)."""
    if l == 0:
        delta, delta2 = 3.1311804, 0.1784
    else:  # l == 1, j == 3/2
        delta, delta2 = 2.6416737, 0.2950
    nu = n - delta - delta2 / (n - delta) ** 2
    return -(_HCONST * _CLIGHT * _RY) / nu**2


@jit(nopython=True, nogil=True)
def radial_matrix_element(n, l, j, na, la, ja):
    """<na la ja| r |n l j> via the Kaulakys (1995) quasiclassical formula."""
    Z = 1
    if l == 0:
        delta, delta2 = 3.1311804, 0.1784
    else:
        delta, delta2 = 2.6416737, 0.2950
    nu = n - delta - delta2 / (n - delta) ** 2

    if la == 0:
        deltaa, delta2a = 3.1311804, 0.1784
    else:
        deltaa, delta2a = 2.6416737, 0.2950
    nua = na - deltaa - delta2a / (na - deltaa) ** 2

    s_a = nua - nu
    nu_c_a = (2 * (nu * nua) ** 2 / (nu + nua)) ** (1 / 3)
    e_a = np.sqrt(1 - ((l + la + 1) / (2 * nu_c_a)) ** 2)

    csi = np.arange(0, np.pi, 1e-3)
    j_a = (1 / np.pi) * np.sum(np.cos(s_a * csi + s_a * e_a * np.sin(csi))) * 1e-3
    dj_a = (1 / np.pi) * np.sum(-np.sin(s_a * csi + s_a * e_a * np.sin(csi)) * np.sin(csi)) * 1e-3

    if (la - l) == 1:
        D_pa = (1 / s_a) * (dj_a + np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))
    else:
        D_pa = (1 / s_a) * (dj_a - np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))

    return ((-1) ** (n - na)) * nu_c_a**5 / (Z * (nu * nua) ** (3 / 2)) * D_pa


def c6_over_r6_coefficient(n, omega, dist):
    """C6/R^6-equivalent dephasing coefficient (MHz * um^6) for the nS Rydberg pair.

    Follows the "Rydberg_para" analysis: the nS_1/2 Rydberg state is nearly
    resonantly Forster-coupled to nP_3/2 + (n-1)P_3/2 with a small defect and
    dipole-dipole coefficient C3; C6 = C3^2 / defect is the effective van der
    Waals coefficient used for the interaction-induced dephasing below.
    """
    l, j = 0, 0.5
    n1, l1, j1 = n, 1, 1.5
    n2, l2, j2 = n - 1, 1, 1.5

    defect = (rydberg_energy(n1, l1, j1) + rydberg_energy(n2, l2, j2) - 2 * rydberg_energy(n, l, j)) / _HCONST * 1e-6
    C3 = (
        1e18
        * 1e-6
        * (_ALPHA * _CLIGHT / (2 * np.pi) * _A0**2)
        * radial_matrix_element(n, l, j, n1, l1, j1)
        * radial_matrix_element(n, l, j, n2, l2, j2)
    )
    return C3**2 / defect / 1000


def _arc_c6_cross_check(n):
    """Independent C6 (GHz um^6) from arc's perturbative pair-state calculation,
    as a sanity check on the Kaulakys-formula C3/defect estimate above."""
    from arc import PairStateInteractions, Rubidium

    calc = PairStateInteractions(Rubidium(), n, 0, 0.5, n, 0, 0.5, 0.5, 0.5)
    return calc.getC6perturbatively(0, 0, 10, 25.0e9)


# ---------------------------------------------------------------------------
# Monte Carlo dephasing: Q(t), G(t) for atoms randomly placed in a cloud
# ---------------------------------------------------------------------------
@jit(nopython=True, nogil=True)
def _positions_uniform_sphere(N, R, seed):
    np.random.seed(seed)
    phi = np.random.uniform(0, 2 * np.pi, N)
    costheta = np.random.uniform(-1, 1, N)
    u = np.random.uniform(0, 1, N)
    theta = np.arccos(costheta)
    r = R * u ** (1.0 / 3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


@jit(nopython=True, nogil=True)
def _positions_gaussian_cloud(N, R, seed):
    np.random.seed(seed)
    w = R / np.sqrt(2)
    x = np.random.normal(0, w, N)
    y = np.random.normal(0, w, N)
    z = np.random.normal(0, w, N)
    return x, y, z


@jit(nopython=True, nogil=True)
def _dephasing_qg(x, y, z, time, Tp, f, ph, C6_c3):
    """Single-shot (fixed atom positions) Q(t), G(t) and the resulting g2(t).

    Q is the ensemble-averaged interaction-induced phase factor of the
    collective spin wave; G = <|sum_j exp(i*phase_j)|^2> is its magnitude-
    squared correlator. For t < Tp, the retrieval pulse is still being
    applied ("in-pulse"); for t >= Tp it has ended and the phase is frozen at
    its value accumulated during Tp ("point-pulse" limit).
    """
    N = len(x)
    dist3D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                dist3D[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)

    Q = np.zeros(len(time), dtype=np.complex64)
    G = np.zeros(len(time))
    for tt in range(len(time)):
        t = time[tt]
        phase_num = 0.0 + 0.0j
        phase_num_G = 0.0
        if t < Tp:
            for i in range(N):
                inner = 0.0 + 0.0j
                for j in range(N):
                    if i != j:
                        detuning_ij = C6_c3 * 1000 / (dist3D[i, j] ** 6)
                        exp_t = np.exp(-2j * np.pi * detuning_ij * t)
                        temp = (-exp_t + (1 - 1j * detuning_ij * 2 * np.pi * t)) / (detuning_ij * 2 * np.pi) ** 2
                        phase_num += temp
                        inner += temp
                phase_num_G += np.abs(inner) ** 2
            Q[tt] = phase_num / t**2 * 2 / N**2
            G[tt] = phase_num_G / t**4 * 4 / N**3
        else:
            for i in range(N):
                inner = 0.0 + 0.0j
                for j in range(N):
                    if i != j:
                        detuning_ij = C6_c3 * 1000 / (dist3D[i, j] ** 6)
                        expt = np.exp(-2j * np.pi * detuning_ij * t)
                        expTp = np.exp(2j * np.pi * detuning_ij * Tp)
                        temp = expt * (-1 + expTp * (1 - 1j * detuning_ij * 2 * np.pi * Tp)) / (detuning_ij * 2 * np.pi) ** 2
                        phase_num += temp
                        inner += temp
                phase_num_G += np.abs(inner) ** 2
            Q[tt] = phase_num / Tp**2 * 2 / N**2
            G[tt] = phase_num_G / Tp**4 * 4 / N**3

    epsi = 0.1  # small-dephasing expansion parameter (N * chi^2 * Tp^2 truncated to leading order)
    intti = 1 + 2 * f * np.cos(ph) + f**2 + epsi * (
        -f * np.cos(ph) + f * (np.exp(1j * ph) * Q + np.conjugate(np.exp(1j * ph) * Q)) + f**2 * G
    )
    ati = (
        1
        + 4 * f * (1 - epsi / 2) * np.cos(ph)
        + 4 * f**2
        + 2 * epsi * (f * (np.exp(1j * ph) * Q + np.conjugate(np.exp(1j * ph) * Q)) + f**2 * G)
        + f**2 * (np.exp(2j * ph) * Q + np.conjugate(np.exp(2j * ph) * Q))
        + 2 * f**3 * (np.exp(1j * ph) * Q + np.conjugate(np.exp(1j * ph) * Q))
        + f**4 * np.abs(Q) ** 2
    )
    g2ti = np.abs(ati) / np.abs(intti) ** 2
    return g2ti, Q, G


def dephasing_qg(N, R, time, seed, Tp, f, ph, C6_c3, cloud="uniform"):
    positions = _positions_uniform_sphere(N, R, seed) if cloud == "uniform" else _positions_gaussian_cloud(N, R, seed)
    return _dephasing_qg(*positions, time, Tp, f, ph, C6_c3)


def g2_approx(f, ph, epsi=0.1):
    """Analytic g2 in the tau_1 >> 1 (fully dephased) limit."""
    return (1 + 4 * f * (1 - epsi / 2) * np.cos(ph) + 4 * f**2) / (1 + 2 * f * (1 - epsi / 2) * np.cos(ph) + f**2) ** 2


def mc_average_g2(N, R, time, Tp, f, ph, C6_c3, n_seeds, cloud="uniform"):
    g2_avg = np.zeros(len(time))
    for seed in range(n_seeds):
        g2ti, _, _ = dephasing_qg(N, R, time, seed, Tp, f, ph, C6_c3, cloud=cloud)
        g2_avg += g2ti / n_seeds
    return g2_avg


def main():
    n = 75
    R = 10.0  # cloud radius, um
    omega = max(1 / 0.100 / (2 * np.pi), 40 * 40 / 2 / 480)
    C6_c3 = c6_over_r6_coefficient(n, omega, 15.0)
    C6_arc = _arc_c6_cross_check(n)
    print(f"C6/defect (Kaulakys formula) = {C6_c3:.3f} MHz um^6")
    print(f"C6 (arc, perturbative pair-state calc) = {C6_arc:.3f} GHz um^6 (independent cross-check)")

    time_scale = -1000 * 2 * np.pi * C6_c3 / 1e6  # converts raw time (us) to dimensionless C6/R^6 * t

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) g2 vs beam-splitter ratio f, uniform sphere, N=100
    N = 100
    seed = 1
    chi = 0.0057 * 2 * np.pi
    t_pair = np.array([1.0, 2.0]) / time_scale  # Tp=0 (t~0+) and Tp finite short pulse
    Tp_short = 1.0 / time_scale
    f_list = np.linspace(0, 2, 60)
    g2_vs_f = np.array([dephasing_qg(N, R, t_pair, seed, Tp_short, ff, np.pi, C6_c3)[0] for ff in f_list])
    axes[0, 0].plot(f_list, g2_vs_f[:, 0], label="Tp=1, tau=1 (no free evolution)")
    axes[0, 0].plot(f_list, g2_vs_f[:, 1], label="Tp=1, tau=2 (0.1 free evolution)")
    axes[0, 0].set_xlabel(r"$f$")
    axes[0, 0].set_ylabel(r"$g^{(2)}$")
    axes[0, 0].set_title("g2 vs f, uniform cloud")
    axes[0, 0].legend(fontsize=8)

    # (b) g2 vs tau, MC average over seeds, uniform sphere vs Gaussian cloud
    n_seeds = 20
    n_time = 400
    Tp = 200.0 / time_scale
    f = 1.0
    t_list = np.linspace(10, 200, n_time) / time_scale
    t_list_scaled = time_scale * t_list

    g2_uniform = mc_average_g2(N, R, t_list, Tp, f, np.pi, C6_c3, n_seeds, cloud="uniform")
    g2_gaussian = mc_average_g2(N, R, t_list, Tp, f, np.pi, C6_c3, n_seeds, cloud="gaussian")
    g2_approx_curve = g2_approx(f, np.pi) * np.ones(n_time)

    axes[0, 1].plot(t_list_scaled, g2_uniform, "-", color="tab:red", label="uniform sphere (MC)")
    axes[0, 1].plot(t_list_scaled, g2_gaussian, "--", color="tab:purple", label="Gaussian cloud (MC)")
    axes[0, 1].plot(t_list_scaled, g2_approx_curve, "-", color="tab:orange", label=r"approx, $\tau_1 \gg 1$")
    axes[0, 1].set_title("h=1")
    axes[0, 1].set_ylabel(r"$g^{(2)}$")
    axes[0, 1].set_xlabel(r"$C_6/R^6 \cdot T_P$")
    axes[0, 1].legend(fontsize=8)

    # (c)/(d) point-pulse limit: G_pt(tau,tau) and |Q_pt(tau,tau)|^2 vs tau,
    # for a short (Tp=2) and a longer (Tp=30) retrieval pulse, uniform sphere.
    N_pt = 200
    n_seeds_pt = 20
    n_time_pt = 150
    Tp_a = 30.0 / time_scale
    Tp_b = 2.0 / time_scale
    t_list_pt = np.linspace(2, 30, n_time_pt) / time_scale
    t_list_pt_scaled = time_scale * t_list_pt

    G_a = np.zeros(n_time_pt)
    Q2_a = np.zeros(n_time_pt)
    G_b = np.zeros(n_time_pt)
    Q2_b = np.zeros(n_time_pt)
    for seed in range(n_seeds_pt):
        _, Q_a_s, G_a_s = dephasing_qg(N_pt, R, t_list_pt, seed, Tp_a, 1.0, np.pi, C6_c3, cloud="uniform")
        _, Q_b_s, G_b_s = dephasing_qg(N_pt, R, t_list_pt, seed, Tp_b, 1.0, np.pi, C6_c3, cloud="uniform")
        G_a += G_a_s / n_seeds_pt
        Q2_a += np.abs(Q_a_s) ** 2 / n_seeds_pt
        G_b += G_b_s / n_seeds_pt
        Q2_b += np.abs(Q_b_s) ** 2 / n_seeds_pt

    axes[1, 0].plot(t_list_pt_scaled, G_a, "-", color="tab:green", label=r"$G_{pt}(30,\tau)$")
    axes[1, 0].plot(t_list_pt_scaled, Q2_a, "--", color="tab:blue", label=r"$|Q_{pt}(30,\tau)|^2$")
    axes[1, 0].plot(t_list_pt_scaled, G_b, "-", color="tab:red", label=r"$G_{pt}(2,\tau)$")
    axes[1, 0].plot(t_list_pt_scaled, Q2_b, "--", color="tab:orange", label=r"$|Q_{pt}(2,\tau)|^2$")
    axes[1, 0].set_title("Spherical uniform distribution")
    axes[1, 0].set_ylabel(r"$G_{pt},|Q_{pt}|^2$")
    axes[1, 0].set_xlabel(r"$\tau$")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
