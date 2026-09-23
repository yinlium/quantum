"""
Monte Carlo estimate of the interaction-induced phase shift phi' and the
"source" g^(2) value s = 2|beta_2|^2/|beta_1|^4 used as fixed parameters in

    B. Yang et al., "Dipole Moment of a Superatom", PRL 133, 213601 (2024),

where phi' = -0.3*pi and s = 0.12 at n = 75 (quoted from the companion paper,
Y. Li et al., PRA 106, L051701 (2022), and this Letter's Supplemental Material).

An ensemble of N atoms is placed at random positions in a Gaussian cloud
(the trap waists of the experiment). Every atom pair dephases the collective
coherence via its van-der-Waals shift C3/r_ij^3 (C3 obtained from a
quantum-defect radial matrix element). Averaging exp(i * 2*pi * detuning_ij * t)
over pairs and Monte Carlo samples gives the (real, imag) collective coherence
and its second moment, from which phi'(t) = arg(coherence) and the dynamical
g^(2)(t) = 4|coherence(t)|^2 / (1 + y(t))^2 follow.
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
    hconst = 6.62606896e-34  # Planck constant
    clight = 2.99792458e8  # speed of light
    Ry = 1.0973731568527e7  # Rydberg constant

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
    e_a = np.sqrt(1 - ((l + la + 1) / (2 * nu_c_a)) ** 2)  # eccentricity

    csi = np.arange(0, np.pi, 1e-3)
    j_a = (1 / np.pi) * np.sum(np.cos(s_a * csi + s_a * e_a * np.sin(csi))) * 1e-3
    dj_a = (1 / np.pi) * np.sum(-np.sin(s_a * csi + s_a * e_a * np.sin(csi)) * np.sin(csi)) * 1e-3

    if (la - l) == 1:
        D_pa = (1 / s_a) * (dj_a + np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))
    else:  # la - l == -1
        D_pa = (1 / s_a) * (dj_a - np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))

    return ((-1) ** (n - na)) * nu_c_a**5 / (Z * (nu * nua) ** (3 / 2)) * D_pa


@jit(nopython=True, nogil=True)
def collective_coherence(N, wx, wy, wz, n, time, seed, blockade):
    """One Monte Carlo draw of N atom positions -> (Re, Im, |.|^2 moment) of
    the pairwise-averaged collective coherence exp(i*2*pi*detuning_ij*t),
    dephased by the van-der-Waals shift between np|n-1,p> pair states."""
    a_0 = 0.52917720859e-10  # Bohr radius
    alpha = 7.2973525376e-3  # fine-structure constant
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


def main():
    # Experimental parameters (Fig. 1 caption / "Experimental protocol"):
    # ~10 um longitudinal x ~6 um transverse cloud, n = 75 Rydberg state.
    N = 700
    waist_xy, waist_z = 5.85, 10.5  # microns
    n_rydberg = 75
    n_seeds = 10  # Monte Carlo draws (notebook used 20; reduced for runtime)
    t_list = np.linspace(0, 2.5, 300)  # storage time, microseconds

    x_real = np.zeros(len(t_list))
    x_imag = np.zeros(len(t_list))
    y_list = np.zeros(len(t_list))
    for seed in range(n_seeds):
        xr, xi, y = collective_coherence(N, waist_xy, waist_xy, waist_z, n_rydberg, t_list, seed, blockade=0)
        x_real += xr / n_seeds
        x_imag += xi / n_seeds
        y_list += y / n_seeds

    phi_prime = np.arctan2(x_imag, x_real)
    g2_dynamic = 4 * (x_real**2 + x_imag**2) / (1 + y_list) ** 2

    # Value quoted in the Letter for the experimental storage time Ts ~ 0.3 us.
    Ts = 0.3
    idx = np.argmin(np.abs(t_list - Ts))
    print(f"At n={n_rydberg}, Ts={t_list[idx]:.3f} us: phi' = {phi_prime[idx]:.3f} rad "
          f"({phi_prime[idx] / np.pi:.3f} pi), s = g2 = {g2_dynamic[idx]:.3f}")
    print("(Letter quotes phi' = -0.3*pi, s = 0.12 at n = 75, Ts ~ 0.3 us.)")

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(t_list, phi_prime / np.pi)
    axes[0].axvline(Ts, color="gray", ls="--", lw=1)
    axes[0].set_ylabel(r"$\phi' / \pi$")
    axes[1].plot(t_list, g2_dynamic)
    axes[1].axvline(Ts, color="gray", ls="--", lw=1)
    axes[1].set_ylabel(r"$g^{(2)}_{\rm source}(t) = s(t)$")
    axes[1].set_xlabel(r"storage time $T_s$ ($\mu$s)")
    fig.suptitle(f"Interaction-induced dephasing, n = {n_rydberg}")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
