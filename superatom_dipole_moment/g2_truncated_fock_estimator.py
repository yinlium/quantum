"""
Sensitivity of the estimated second-order correlation g^(2) to the assumed
mean excitation number mbar, for the superatom dipole-moment experiment of

    B. Yang et al., "Dipole Moment of a Superatom", PRL 133, 213601 (2024).

The collective coherence (Re, Im) and its second moment y(t) are generated
by the same van-der-Waals dephasing Monte Carlo as
interaction_phase_and_g2_dynamics.py (n = 75 Rydberg state, experimental trap
waists). They are then combined with a truncated Fock-state expansion of the
retrieved light, with amplitudes c_i(mbar, N, i) for i = 0..trunc excitations
out of N atoms (a beam-splitter-like binomial weighting parameterized by the
mean excitation number mbar), to give an alternative g^(2) estimate. Sweeping
mbar shows how sensitive this "source" g^(2) is to the assumed mean excitation.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def rydberg_energy(n, l, j):
    """Quantum-defect energy of Rb 87 |n l j> (fine-structure levels only)."""
    hconst = 6.62606896e-34
    clight = 2.99792458e8
    Ry = 1.0973731568527e7

    if l == 0:
        delta_s, delta_s2 = 3.1311804, 0.1784
        nu = n - delta_s - delta_s2 / (n - delta_s) ** 2
    if (l == 1) and (j == 1 / 2):
        delta_p, delta_p2 = 2.6548849, 0.2900
        nu = n - delta_p - delta_p2 / (n - delta_p) ** 2
    if (l == 1) and (j == 3 / 2):
        delta_p, delta_p2 = 2.6416737, 0.2950
        nu = n - delta_p - delta_p2 / (n - delta_p) ** 2
    if (l == 2) and (j == 3 / 2):
        delta_d, delta_d2 = 1.34809171, -0.60286
        nu = n - delta_d - delta_d2 / (n - delta_d) ** 2
    if (l == 2) and (j == 5 / 2):
        delta_d, delta_d2 = 1.34646572, -0.59600
        nu = n - delta_d - delta_d2 / (n - delta_d) ** 2
    if (l == 3) and (j == 5 / 2):
        delta_f, delta_f2 = 0.0165192, -0.085
        nu = n - delta_f - delta_f2 / (n - delta_f) ** 2
    if (l == 3) and (j == 7 / 2):
        delta_f, delta_f2 = 0.0165437, -0.086
        nu = n - delta_f - delta_f2 / (n - delta_f) ** 2

    return -(hconst * clight * Ry) / nu**2


@jit(nopython=True, nogil=True)
def radial_matrix_element(n, l, j, na, la, ja):
    """<na la ja| r |n l j> via the Kaulakys (1995) quasiclassical formula."""
    Z = 1
    if l == 0:
        delta_s, delta_s2 = 3.1311804, 0.1784
        nu = n - delta_s - delta_s2 / (n - delta_s) ** 2
    if (l == 1) and (j == 1 / 2):
        delta_p, delta_p2 = 2.6548849, 0.2900
        nu = n - delta_p - delta_p2 / (n - delta_p) ** 2
    if (l == 1) and (j == 3 / 2):
        delta_p, delta_p2 = 2.6416737, 0.2950
        nu = n - delta_p - delta_p2 / (n - delta_p) ** 2
    if (l == 2) and (j == 3 / 2):
        delta_d, delta_d2 = 1.34809171, -0.60286
        nu = n - delta_d - delta_d2 / (n - delta_d) ** 2
    if (l == 2) and (j == 5 / 2):
        delta_d, delta_d2 = 1.34646572, -0.59600
        nu = n - delta_d - delta_d2 / (n - delta_d) ** 2
    if (l == 3) and (j == 5 / 2):
        delta_f, delta_f2 = 0.0165192, -0.085
        nu = n - delta_f - delta_f2 / (n - delta_f) ** 2
    if (l == 3) and (j == 7 / 2):
        delta_f, delta_f2 = 0.0165437, -0.086
        nu = n - delta_f - delta_f2 / (n - delta_f) ** 2

    if la == 0:
        delta_s, delta_s2 = 3.1311804, 0.1784
        nua = na - delta_s - delta_s2 / (na - delta_s) ** 2
    if (la == 1) and (ja == 1 / 2):
        delta_p, delta_p2 = 2.6548849, 0.2900
        nua = na - delta_p - delta_p2 / (na - delta_p) ** 2
    if (la == 1) and (ja == 3 / 2):
        delta_p, delta_p2 = 2.6416737, 0.2950
        nua = na - delta_p - delta_p2 / (na - delta_p) ** 2
    if (la == 2) and (ja == 3 / 2):
        delta_d, delta_d2 = 1.34809171, -0.60286
        nua = na - delta_d - delta_d2 / (na - delta_d) ** 2
    if (la == 2) and (ja == 5 / 2):
        delta_d, delta_d2 = 1.34646572, -0.59600
        nua = na - delta_d - delta_d2 / (na - delta_d) ** 2
    if (la == 3) and (ja == 5 / 2):
        delta_f, delta_f2 = 0.0165192, -0.085
        nua = na - delta_f - delta_f2 / (na - delta_f) ** 2
    if (la == 3) and (ja == 7 / 2):
        delta_f, delta_f2 = 0.0165437, -0.086
        nua = na - delta_f - delta_f2 / (na - delta_f) ** 2

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


@jit(nopython=True, nogil=True)
def collective_coherence(N, wx, wy, wz, n, time, seed, blockade):
    """One Monte Carlo draw of the pairwise-averaged collective coherence and
    its second moment y(t), dephased by the van-der-Waals interaction."""
    a_0 = 0.52917720859e-10
    alpha = 7.2973525376e-3
    hconst = 6.62606896e-34
    clight = 2.99792458e8

    wx, wy, wz = wx / 2.0, wy / 2.0, wz / 2.0
    xvec = np.random.normal(0, wx, N)
    yvec = np.random.normal(0, wy, N)
    zvec = np.random.normal(0, wz, N)

    l, j = 0, 1 / 2
    n1, l1, j1 = n, 1, 3 / 2
    n2, l2, j2 = n - 1, 1, 3 / 2
    omega = 1
    defect = (rydberg_energy(n1, l1, j1) + rydberg_energy(n2, l2, j2) - 2 * rydberg_energy(n, l, j)) / hconst * 1e-6
    C3 = (1e18) * (1e-6) * ((alpha * clight / (2 * np.pi)) * a_0**2) \
        * radial_matrix_element(n, l, j, n1, l1, j1) * radial_matrix_element(n, l, j, n2, l2, j2)

    dist3D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i != k:
                dist3D[i, k] = np.sqrt((xvec[i] - xvec[k]) ** 2 + (yvec[i] - yvec[k]) ** 2 + (zvec[i] - zvec[k]) ** 2)

    x_real = np.zeros(len(time))
    x_imag = np.zeros(len(time))
    y_list = np.zeros(len(time))
    for tt in range(len(time)):
        phase_num = 0.0 + 0.0j
        phase_denom = 0.0
        for i in range(N):
            inner = 0.0 + 0.0j
            for k in range(N):
                if i != k:
                    detuning_ij = np.abs(defect / 2) - np.sqrt((defect / 2) ** 2 + (C3 / dist3D[i, k] ** 3) ** 2)
                    if blockade == 1:
                        term = omega / np.sqrt(omega**2 + detuning_ij**2) * np.exp(2j * np.pi * detuning_ij * time[tt])
                    else:
                        term = np.exp(2j * np.pi * detuning_ij * time[tt])
                    phase_num += term
                    inner += term
            phase_denom += np.abs(inner) ** 2

        coherence = phase_num / N / (N - 1)
        x_real[tt] = coherence.real - 2 / N**2
        x_imag[tt] = coherence.imag
        y_list[tt] = phase_denom / N / (N - 1) ** 2

    return x_real, x_imag, y_list


@jit(nopython=True, nogil=True)
def fock_amplitude(mbar, N, i):
    """Binomial Fock-state amplitude c_i for i excitations out of N atoms,
    parameterized by the mean excitation number mbar = N * b^2."""
    b = np.sqrt(mbar / N)
    a = np.sqrt(1 - b**2)
    binomial = 1.0
    for m in range(i):
        binomial = binomial * (N - m) / (m + 1)
    return a ** (N - i) * b**i * np.sqrt(binomial)


@jit(nopython=True, nogil=True)
def g2_estimate(x_real, x_imag, y, mbar, N, trunc):
    """g^(2)(t) estimated from a truncated (0..trunc excitations) Fock-state
    expansion of the retrieved field, weighted by fock_amplitude(mbar, N, i)."""
    c_i_list = np.zeros(trunc + 1)
    for i in range(trunc + 1):
        c_i_list[i] = fock_amplitude(mbar, N, i)
    norm = np.sqrt(np.sum(c_i_list**2))
    c_i_list = c_i_list / norm

    x_test = np.abs(x_real) ** 2 + np.abs(x_imag) ** 2
    y_test = np.abs(y)

    g2 = np.zeros(len(x_real))
    for tt in range(len(x_real)):
        g2_denom = c_i_list[1] ** 2
        g2_num = 0.0
        for m in range(2, trunc + 1):
            g2_denom += c_i_list[m] ** 2 * m * (y_test[tt] ** (m - 1))
            g2_num += c_i_list[m] ** 2 * m * (m - 1) * (x_test[tt] ** (2 * m - 3))
        g2[tt] = g2_num / g2_denom**2
    return g2


def main():
    N = 273  # sample size used for the n = 75 Rydberg-state dephasing run
    waist_xy, waist_z = 5.85, 10.5  # microns
    n_rydberg = 75
    n_seeds = 100
    trunc = 20
    t_list = np.linspace(0, 2, 100)  # storage time, microseconds

    x_real = np.zeros(len(t_list))
    x_imag = np.zeros(len(t_list))
    y_list = np.zeros(len(t_list))
    for seed in range(n_seeds):
        xr, xi, y = collective_coherence(N, waist_xy, waist_xy, waist_z, n_rydberg, t_list, seed, blockade=0)
        x_real += xr / n_seeds
        x_imag += xi / n_seeds
        y_list += y / n_seeds

    mbar_list = np.arange(0.1, 2.0, 0.3)
    fig, ax = plt.subplots(figsize=(7, 5))
    for mbar in mbar_list:
        g2 = g2_estimate(x_real, x_imag, y_list, mbar, N=1000, trunc=trunc)
        ax.plot(t_list, g2, label=f"mbar = {mbar:.2f}")
    ax.set_xlabel(r"storage time $T_s$ ($\mu$s)")
    ax.set_ylabel(r"estimated $g^{(2)}(T_s)$")
    ax.set_title("Truncated Fock-state g$^{(2)}$ estimate vs mean excitation mbar")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
