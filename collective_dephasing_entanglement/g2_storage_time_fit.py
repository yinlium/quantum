"""Second-order correlation g^(2)(T_s) of the retrieved spin wave vs. storage time.

Reproduces the central experimental result of Y. Li, Y. Mei, H. Nguyen, P. R.
Berman, A. Kuzmich, "Dynamics of Collective-Dephasing-Induced Multiatom
Entanglement," Phys. Rev. A 106, L051701 (2022): interaction-induced (van der
Waals) dephasing among Rydberg-excited atoms during a storage time T_s
converts an initially unentangled, singly-excited collective spin wave into
an entangled multiatom (Dicke-like) state, whose signature is the decay of
g^(2)(T_s) below the uncorrelated value as T_s increases.

For each pair of interacting atoms i,j at Rydberg principal quantum number n,
the van der Waals shift C3/R_ij^3 between the nS_1/2 Rydberg pair state and
its near-resonant (n,n-1)P_3/2 Forster partner dephases the relative phase of
the collective spin wave. Averaging exp(i*2*pi*detuning_ij*T_s) over atom
pairs (positions drawn from a Gaussian cloud of transverse size wx=wy and
longitudinal size Lz) and Monte Carlo samples gives the coherence moments
x(T_s), y(T_s), which are combined with a truncated multi-excitation
Fock-state expansion (amplitudes c_i, set by the two-photon excitation Rabi
frequencies and pulse duration) to estimate g^(2)(T_s).

The single free parameter, the cloud's longitudinal size Lz, is fit by
least-squares to the measured g^(2)(T_s) at Rydberg states n=40 and n=50
(100 ns retrieval pulse). A third dataset, n=50 with the atoms pinned in an
optical lattice (suppressing atomic motion but not the interaction), is
plotted for comparison but not fit.

Monte Carlo seed counts, the Lz scan resolution, and the Fock-state
truncation are reduced from the original notebook to keep runtime short; the
model and fit procedure are unchanged. Source: Supplement figure.ipynb
("g2 revival" folder), cells computing g2_n40/g2_n50 vs. the dephasing model
and optimizing Lz (`optimal_param`); the pairwise dephasing model itself
("final"/"test0526_blockade") also appears (unfit) in
g2_revival-09-07-2022.ipynb and g2_revival-08-22-2022.ipynb. Data:
g2_n40_pw100_final.csv, g2_n50_pw100_final.csv, g2_n50_pw100_latt_final.csv,
originally under "Many body rabi oscillation/simulation/dephasing/" (a
sibling folder to both source folders, referenced by relative path from
Supplement figure.ipynb).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit

DATA_DIR = Path(__file__).parent / "data"


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
    atoms, driven by two-photon Rabi frequencies omega_1, omega_2 at detuning
    delta over pulse duration t (mean-field beam-splitter-like weighting)."""
    a = np.cos(2 * np.pi * omega_1 * omega_2 / 4 / delta * t)
    b = np.sin(2 * np.pi * omega_1 * omega_2 / 4 / delta * t)
    binomial = 1.0
    for m in range(i):
        binomial = binomial * (N - m) / (m + 1)
    return a ** (N - i) * b**i * np.sqrt(binomial)


@jit(nopython=True, nogil=True)
def dephasing_g2(N, wx, wy, Lz, n, omega_2, time, seed, trunc):
    """One Monte Carlo draw (fixed atom positions) of g^(2)(T_s) for storage
    times `time`, from van-der-Waals dephasing of the collective coherence
    combined with the truncated Fock-state expansion c_i."""
    a_0 = 0.52917720859e-10
    alpha = 7.2973525376e-3
    hconst = 6.62606896e-34
    clight = 2.99792458e8

    np.random.seed(seed)
    wx2, wy2, wz2 = wx / 2.0, wy / 2.0, Lz / 2.0
    xvec = np.random.normal(0, wx2, N)
    yvec = np.random.normal(0, wy2, N)
    zvec = np.random.normal(0, wz2, N)

    l, j = 0, 0.5
    n1, l1, j1 = n, 1, 1.5
    n2, l2, j2 = n - 1, 1, 1.5
    defect = (rydberg_energy(n1, l1, j1) + rydberg_energy(n2, l2, j2) - 2 * rydberg_energy(n, l, j)) / hconst * 1e-6
    C3 = (1e18) * (1e-6) * ((alpha * clight / (2 * np.pi)) * a_0**2) \
        * radial_matrix_element(n, l, j, n1, l1, j1) * radial_matrix_element(n, l, j, n2, l2, j2)

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


def mc_average_g2(N, wx, wy, Lz, n, omega_2, time, trunc, n_seeds):
    g2_avg = np.zeros(len(time))
    for seed in range(n_seeds):
        g2_avg += dephasing_g2(N, wx, wy, Lz, n, omega_2, time, seed, trunc) / n_seeds
    return g2_avg


def main():
    N = 273  # atom number calibrated from the Rabi-flopping fit (calibration_fits.py)
    wx = wy = 5.85 * np.sqrt(2)  # microns, transverse cloud size
    trunc = 15  # Fock-state truncation (reduced from 50; c_i decays fast well before this)
    n_seeds_scan = 8  # Monte Carlo seeds while scanning Lz (reduced from 200)
    n_seeds_final = 20  # Monte Carlo seeds for the final smooth curves

    omega_2_40 = omega_2_rabi_mhz(40)
    omega_2_50 = omega_2_rabi_mhz(50)
    print(f"omega_2(n=40) = {omega_2_40:.2f} MHz, omega_2(n=50) = {omega_2_50:.2f} MHz")

    df_n40 = pd.read_csv(DATA_DIR / "g2_n40_pw100_final.csv")
    df_n50 = pd.read_csv(DATA_DIR / "g2_n50_pw100_final.csv")
    df_n50_latt = pd.read_csv(DATA_DIR / "g2_n50_pw100_latt_final.csv")
    x_n40, y_n40, yerr_n40 = df_n40["time"].to_numpy(), df_n40["g2"].to_numpy(), df_n40["g2 err"].to_numpy()
    x_n50, y_n50, yerr_n50 = df_n50["time"].to_numpy(), df_n50["g2"].to_numpy(), df_n50["g2 err"].to_numpy()
    x_latt, y_latt, yerr_latt = (
        df_n50_latt["time"].to_numpy(),
        df_n50_latt["g2"].to_numpy(),
        df_n50_latt["g2 err"].to_numpy(),
    )

    # --- fit the single free parameter (cloud longitudinal size Lz) to the
    # n=40 and n=50 (no lattice) data by least-squares (reduced scan grid) ---
    Lz_list = np.arange(2.0, 16.0, 2.0)
    sq_err = np.zeros(len(Lz_list))
    for idx, Lz in enumerate(Lz_list):
        g2_40 = mc_average_g2(N, wx, wy, Lz, 40, omega_2_40, x_n40, trunc, n_seeds_scan)
        g2_50 = mc_average_g2(N, wx, wy, Lz, 50, omega_2_50, x_n50, trunc, n_seeds_scan)
        sq_err[idx] = np.sum((y_n40 - g2_40) ** 2) + np.sum((y_n50 - g2_50) ** 2)
        print(f"Lz = {Lz:5.1f} um  ->  sum squared error = {sq_err[idx]:.4f}")
    Lz_best = Lz_list[np.argmin(sq_err)]
    print(f"Best-fit cloud longitudinal size: Lz = {Lz_best:.1f} um")

    # --- smooth theory curves at the best-fit Lz ---
    t_smooth = np.linspace(0.05, 25, 30)
    g2_40_smooth = mc_average_g2(N, wx, wy, Lz_best, 40, omega_2_40, t_smooth, trunc, n_seeds_final)
    g2_50_smooth = mc_average_g2(N, wx, wy, Lz_best, 50, omega_2_50, t_smooth, trunc, n_seeds_final)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(Lz_list, sq_err, "o-")
    axes[0].axvline(Lz_best, color="gray", ls="--", lw=1)
    axes[0].set_xlabel(r"cloud longitudinal size $L_z$ ($\mu$m)")
    axes[0].set_ylabel("summed squared error (n=40 & n=50)")
    axes[0].set_title("Least-squares fit of cloud size")

    axes[1].errorbar(x_n40, y_n40, yerr=yerr_n40, fmt="o", color="tab:blue", label="n=40 data")
    axes[1].plot(t_smooth, g2_40_smooth, "-", color="tab:blue", alpha=0.7, label="n=40 model")
    axes[1].errorbar(x_n50, y_n50, yerr=yerr_n50, fmt="s", color="tab:red", label="n=50 data")
    axes[1].plot(t_smooth, g2_50_smooth, "-", color="tab:red", alpha=0.7, label="n=50 model")
    axes[1].errorbar(x_latt, y_latt, yerr=yerr_latt, fmt="^", color="tab:green", label="n=50, lattice (not fit)")
    axes[1].set_xlabel(r"storage time $T_s$ ($\mu$s)")
    axes[1].set_ylabel(r"$g^{(2)}(T_s)$")
    axes[1].set_title(f"Dephasing-induced entanglement ($L_z$ = {Lz_best:.1f} $\\mu$m)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
