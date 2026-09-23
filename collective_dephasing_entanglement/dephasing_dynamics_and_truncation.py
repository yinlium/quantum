"""Full g^(2)(T_s) dynamics, Rydberg-blockade effect, and Fock-truncation
sensitivity, for Y. Li, Y. Mei, H. Nguyen, P. R. Berman, A. Kuzmich,
"Dynamics of Collective-Dephasing-Induced Multiatom Entanglement,"
Phys. Rev. A 106, L051701 (2022), Supplemental Material figures.

Three related checks on the van-der-Waals dephasing model used to fit the
storage-time data in g2_storage_time_fit.py:

(a) Fock-state truncation (Fig. S1): the multi-excitation amplitude c_m and
    the truncated g^(2)(T_s=0) estimate, as a function of the truncation
    order m, for n=40 and n=50 -- showing the truncation used elsewhere
    (m ~ 15-50) is well converged.
(b) Full g^(2)(T_s) dynamics (Fig. S2) from ns to ms storage time for
    n=40, 50, 75, with and without a Rydberg-blockade cutoff on the
    pairwise dephasing (atom pairs closer than the blockade radius do not
    contribute a dephasing phase). Blockade barely changes the curve, since
    the relevant interatomic separations are set by the (much larger) cloud
    size, not the blockade radius.
(c) g^(2)(T_s=1 us) vs. principal quantum number n (Fig. S3): dephasing
    grows with n (larger C3), so g^(2) decreases monotonically with n at
    fixed storage time; overlaid are the measured g^(2)(T_s=1 us) values at
    n=40, 50 (data also used in g2_storage_time_fit.py).

Monte Carlo seed counts and the n-scan resolution are reduced from the
original notebook to keep runtime short; the model is unchanged. Source:
Supplement figure.ipynb ("g2 revival" folder), Figures S1-S3.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import jit


# ---------------------------------------------------------------------------
# Rb87 Rydberg (nS_1/2 <-> nP_3/2 / (n-1)P_3/2) structure, quantum-defect
# formulas (only the S and P3/2 branches needed here are kept).
# ---------------------------------------------------------------------------
@jit(nopython=True, nogil=True)
def rydberg_energy(n, l, j):
    if l == 0:
        delta, delta2 = 3.1311804, 0.1784
    else:  # l == 1, j == 3/2
        delta, delta2 = 2.6416737, 0.2950
    nu = n - delta - delta2 / (n - delta) ** 2
    hconst, clight, Ry = 6.62606896e-34, 2.99792458e8, 1.0973731568527e7
    return -(hconst * clight * Ry) / nu**2


@jit(nopython=True, nogil=True)
def radial_matrix_element(n, l, j, na, la, ja):
    """<na la ja| r |n l j> via the Kaulakys (1995) quasiclassical formula."""
    Z = 1
    delta, delta2 = (3.1311804, 0.1784) if l == 0 else (2.6416737, 0.2950)
    nu = n - delta - delta2 / (n - delta) ** 2
    deltaa, delta2a = (3.1311804, 0.1784) if la == 0 else (2.6416737, 0.2950)
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


def omega_2_rabi_mhz(n):
    """Coupling-laser two-photon Rabi frequency (MHz) to Rydberg state n,
    from the 480 nm dipole matrix element (arc's Rubidium87 HFS data)."""
    import arc
    from scipy.constants import c as C_c
    from scipy.constants import e as C_e
    from scipy.constants import epsilon_0, hbar, pi
    from scipy.constants import physical_constants
    from math import sqrt

    bohr_radius = physical_constants["Bohr radius"][0]
    atom = arc.Rubidium87()
    laser_power, laser_waist = 30e-3, 15e-6
    max_intensity = 2 * laser_power / (pi * laser_waist**2)
    electric_field = sqrt(2.0 * max_intensity / (C_c * epsilon_0))
    dipole = atom.getDipoleMatrixElementHFS(5, 1, 3 / 2, 3, -3, n, 0, 1 / 2, 2, -2, +1, s=0.5) * C_e * bohr_radius
    freq = electric_field * abs(dipole) / hbar
    return freq / 1e6 / (2 * pi) / 1.46


@jit(nopython=True, nogil=True)
def fock_amplitude_ci(omega_1, omega_2, delta, N, t, i):
    """Binomial multi-excitation Fock amplitude c_i for i excitations out of N
    atoms (mean-field beam-splitter-like weighting; see g2_storage_time_fit.py)."""
    a = np.cos(2 * np.pi * omega_1 * omega_2 / 4 / delta * t)
    b = np.sin(2 * np.pi * omega_1 * omega_2 / 4 / delta * t)
    binomial = 1.0
    for m in range(i):
        binomial = binomial * (N - m) / (m + 1)
    return a ** (N - i) * b**i * np.sqrt(binomial)


@jit(nopython=True, nogil=True)
def dephasing_g2_blockade(N, wx, wy, wz, n, omega_2, time, seed, trunc, blockade):
    """One Monte Carlo draw of g^(2)(T_s), as in g2_storage_time_fit.py, with
    an optional Rydberg-blockade cutoff: pairs closer than the blockade
    radius Rb (set by the two-photon Rabi frequency and C3) are excluded
    from the dephasing sum instead of contributing their van-der-Waals phase."""
    a_0 = 0.52917720859e-10
    alpha = 7.2973525376e-3
    hconst = 6.62606896e-34
    clight = 2.99792458e8

    np.random.seed(seed)
    wx2, wy2, wz2 = wx / 2.0, wy / 2.0, wz / 2.0
    xvec = np.random.normal(0, wx2, N)
    yvec = np.random.normal(0, wy2, N)
    zvec = np.random.normal(0, wz2, N)

    l, j = 0, 0.5
    n1, l1, j1 = n, 1, 1.5
    n2, l2, j2 = n - 1, 1, 1.5
    omega = 1 / 0.1
    defect = (rydberg_energy(n1, l1, j1) + rydberg_energy(n2, l2, j2) - 2 * rydberg_energy(n, l, j)) / hconst * 1e-6
    C3 = (1e18) * (1e-6) * ((alpha * clight / (2 * np.pi)) * a_0**2) \
        * radial_matrix_element(n, l, j, n1, l1, j1) * radial_matrix_element(n, l, j, n2, l2, j2)
    Rb = (-C3 / np.sqrt((omega + np.abs(defect / 2)) ** 2 - (defect / 2) ** 2)) ** (1 / 3)

    c_i_list = np.zeros(trunc + 1)
    for i in range(trunc + 1):
        c_i_list[i] = fock_amplitude_ci(9.2, omega_2, 480.0, N, 0.1, i)
    c_i_list = c_i_list / np.sqrt(np.sum(c_i_list**2))

    dist3D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                dist3D[i, k] = np.sqrt((xvec[i] - xvec[k]) ** 2 + (yvec[i] - yvec[k]) ** 2 + (zvec[i] - zvec[k]) ** 2)

    g2 = np.zeros(len(time))
    for tt in range(len(time)):
        phase_num = 0.0 + 0.0j
        phase_denom = 0.0
        for i in range(N):
            inner = 0.0 + 0.0j
            for k in range(N):
                if i != k:
                    detuning_ik = np.abs(defect / 2) - np.sqrt((defect / 2) ** 2 + (C3 / dist3D[i, k] ** 3) ** 2)
                    term = 0.0 + 0.0j
                    if blockade == 1:
                        if dist3D[i, k] > Rb:
                            term = np.exp(2j * np.pi * detuning_ik * time[tt])
                    else:
                        term = np.exp(2j * np.pi * detuning_ik * time[tt])
                    phase_num += term
                    inner += term
            phase_denom += np.abs(inner) ** 2

        x_test = np.abs(phase_num) ** 2 / N**2 / (N - 1) ** 2
        y_test = phase_denom / N / (N - 1) ** 2
        x_test = (x_test - 2 / N**2) / (1 - 2 / N**2)
        y_test = (y_test - 1 / N) / (1 - 1 / N)

        g2_denom = c_i_list[1] ** 2
        g2_num = 0.0
        for m in range(2, trunc + 1):
            g2_denom += c_i_list[m] ** 2 * m * ((1 - 1 / N) * y_test ** (m - 1) + 1 / N)
            g2_num += c_i_list[m] ** 2 * m * (m - 1) * ((1 - 2 / N**2) * x_test ** (2 * m - 3) + 2 / N**2)
        g2[tt] = g2_num / g2_denom**2

    return g2


def mc_average_g2(N, wx, wy, wz, n, omega_2, time, trunc, blockade, n_seeds):
    g2_avg = np.zeros(len(time))
    for seed in range(n_seeds):
        g2_avg += dephasing_g2_blockade(N, wx, wy, wz, n, omega_2, time, seed, trunc, blockade) / n_seeds
    return g2_avg


def truncated_fock_amplitudes(omega_1, omega_2, delta, N, t, n_trunc):
    """Normalized multi-excitation amplitudes c_0..c_{n_trunc} -- Fig. S1's c_m bars."""
    c_i_list = np.array([fock_amplitude_ci(omega_1, omega_2, delta, N, t, i) for i in range(n_trunc + 1)])
    return c_i_list / np.sqrt(np.sum(c_i_list**2))


def truncated_g2_at_t(omega_1, omega_2, delta, N, t, n_trunc):
    """g^(2)(t) from the c_i Fock-state expansion alone (no dephasing),
    truncated at n_trunc >= 2 excitations -- Fig. S1's g2(0) curve."""
    c_i_list = truncated_fock_amplitudes(omega_1, omega_2, delta, N, t, n_trunc)
    nom = sum(i * (i - 1) * c_i_list[i] ** 2 for i in range(2, n_trunc + 1))
    denom = sum(i * c_i_list[i] ** 2 for i in range(1, n_trunc + 1)) ** 2
    return nom / denom, c_i_list


def main():
    N = 273
    wx = wy = 5.85  # microns
    wz = 10.5  # microns, longitudinal cloud size for this (unfit) demo
    trunc = 12  # reduced from 50 for runtime; truncation-convergence is shown in panel (a)
    n_seeds = 15  # reduced from 100

    omega_2_40 = omega_2_rabi_mhz(40)
    omega_2_50 = omega_2_rabi_mhz(50)
    omega_2_75 = omega_2_rabi_mhz(75)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Fig. S1: truncation-order dependence of c_m and g2(T_s=0)
    trunc_list = np.arange(0, 11)
    g2_trunc_40 = np.full(11, np.nan)
    g2_trunc_50 = np.full(11, np.nan)
    c_list_40 = np.zeros(11)
    c_list_50 = np.zeros(11)
    for x in trunc_list:
        c_i_40 = truncated_fock_amplitudes(9.2, omega_2_40, 480.0, 273, 0.1, x)
        c_list_40[x] = c_i_40[x]
        c_i_50 = truncated_fock_amplitudes(9.2, omega_2_50, 480.0, 273, 0.1, x)
        c_list_50[x] = c_i_50[x]
        if x >= 2:
            g2_trunc_40[x], _ = truncated_g2_at_t(9.2, omega_2_40, 480.0, 273, 0.1, x)
            g2_trunc_50[x], _ = truncated_g2_at_t(9.2, omega_2_50, 480.0, 273, 0.1, x)

    ax_a2 = axes[0].twinx()
    axes[0].bar(trunc_list - 0.2, c_list_40, 0.35, color="darkgray", label="n=40")
    axes[0].bar(trunc_list + 0.2, c_list_50, 0.35, color="lightgray", label="n=50")
    ax_a2.plot(trunc_list, g2_trunc_40, color="purple", label="n=40")
    ax_a2.plot(trunc_list, g2_trunc_50, color="purple", alpha=0.5, label="n=50")
    axes[0].set_xlabel("truncation order m")
    axes[0].set_ylabel(r"$c_m$")
    ax_a2.set_ylabel(r"$g^{(2)}(T_s=0)$")
    axes[0].set_ylim(0, 1.05)
    ax_a2.set_ylim(0, 1.05)
    axes[0].set_title("Fig. S1: Fock-truncation convergence")
    axes[0].legend(fontsize=8, loc="upper right")

    # (b) Fig. S2: full g2(Ts) dynamics, n=40/50/75, blockade on/off
    t_list = 10 ** np.linspace(-5, 3, 40)
    colors = {40: "tab:blue", 50: "tab:orange", 75: "tab:green"}
    for n, omega_2 in [(40, omega_2_40), (50, omega_2_50), (75, omega_2_75)]:
        g2_no_blockade = mc_average_g2(N, wx, wy, wz, n, omega_2, t_list, trunc, blockade=0, n_seeds=n_seeds)
        g2_blockade = mc_average_g2(N, wx, wy, wz, n, omega_2, t_list, trunc, blockade=1, n_seeds=n_seeds)
        axes[1].plot(t_list, g2_no_blockade, "-", color=colors[n], label=f"n={n}, no blockade")
        axes[1].plot(t_list, g2_blockade, "--", dashes=(4, 4), color=colors[n], alpha=0.7, label=f"n={n}, blockade")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"storage time $T_s$ ($\mu$s)")
    axes[1].set_ylabel(r"$g^{(2)}(T_s)$")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Fig. S2: full dynamics & blockade effect")
    axes[1].legend(fontsize=6, loc="lower left")

    # (c) Fig. S3: g2(Ts=1 us) vs. principal quantum number n
    n_list = np.arange(30, 81, 5)
    g2_vs_n = np.zeros(len(n_list))
    for idx, n in enumerate(n_list):
        omega_2_n = omega_2_rabi_mhz(int(n))
        g2_vs_n[idx] = mc_average_g2(N, wx, wy, wz, int(n), omega_2_n, np.array([1.0]), trunc, blockade=0, n_seeds=n_seeds)[0]
    axes[2].plot(n_list, g2_vs_n, "o-", color="tab:purple", label="model (no blockade)")
    # measured g2(Ts=1 us) at n=40, 50 (same data as g2_storage_time_fit.py)
    axes[2].errorbar([40, 50], [0.810455317, 0.560277426], yerr=[0.05233634, 0.047491352],
                      fmt="s", color="black", label="data")
    axes[2].set_xlabel("Rydberg principal quantum number n")
    axes[2].set_ylabel(r"$g^{(2)}(T_s=1~\mu s)$")
    axes[2].set_title("Fig. S3: dephasing grows with n")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
