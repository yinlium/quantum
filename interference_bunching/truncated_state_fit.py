"""Factorized- and truncated-state g2(phi)/g2(f), fit to real phase-scan data.

Reproduces the theory and data fits of P.R. Berman, H. Nguyen, Y. Mei, Y. Li,
A. Kuzmich, "Interference bunching and antibunching of coherent atomic
radiation fields," Phys. Rev. A 108, 043713 (2023), Secs. III.A-III.C. Besides
the single-excitation factorized state, a state truncated at 4 collective
excitations (amplitudes beta_0..beta_4) is used to fit the measured efficiency
eta(phi) and g2(phi)/g2(f) of light retrieved from an atomic ensemble
interfering with a reference field of relative strength f and phase phi. Real
photon-counting data (data/best_g2.csv, phase-resolved coincidence counts) are
fit with scipy.optimize.curve_fit; additional (phi, f)-summary points measured
across several beam-splitter ratios f are fit separately (hardcoded arrays,
digitized from the same experiment).

Source: intbun_new.ipynb.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special
from scipy.optimize import curve_fit

DATA_CSV = Path(__file__).parent / "data" / "best_g2.csv"


# ---------------------------------------------------------------------------
# Factorized state (single collective excitation)
# ---------------------------------------------------------------------------
def inf(f, N, b, phi, c, bg):
    p = b**2
    eff_per_exc = c * b**2 * N
    inf0 = 1 + 2 * f * np.cos(phi) + f**2 + f**2 * p / N / (1 - p)
    return inf0 * (eff_per_exc / f**2) + bg / f**2


def af(f, N, b, phi, c, bg):
    p = b**2
    eff_per_exc = c * b**2 * N
    R1 = (4 * f**2 * p - 2 * f**4 * (1 - 3 * p)) / (1 - p) - 2 * f**2 * np.cos(
        2 * phi
    ) - 4 * f**3 * (1 - 3 * p) / (1 - p) * np.cos(phi)
    R2 = -8 * f**3 * p / (1 - p) * np.cos(phi) + f**4 * (1 - 10 * p + 11 * p**2) / (
        1 - p
    ) ** 2
    R3 = 2 * f**4 * p * (2 - 3 * p) / (1 - p) ** 2
    af0 = (1 + 2 * f * np.cos(phi) + f**2) ** 2 + R1 / N + R2 / N**2 + R3 / N**3
    return af0 * (eff_per_exc / f**2) ** 2 + 2 * bg / f**2 * inf(f, N, b, phi, c, bg)


def g2f(f, N, b, phi, c, bg):
    return af(f, N, b, phi, c, bg) / inf(f, N, b, phi, c, bg) ** 2


# ---------------------------------------------------------------------------
# Truncated state (up to 4 collective excitations)
# ---------------------------------------------------------------------------
def beta(n, N, b):
    """Binomial (coherent-spin-state) amplitude of the n-excitation Dicke state."""
    return np.sqrt(scipy.special.binom(N, n)) * (1 - b**2) ** ((N - n) / 2) * b**n


def intt(f, N, b, b0, b1, b2, b3, phi, c, bg):
    """Intensity for the state truncated at 3 excitations (beta_0..beta_3)."""
    g = np.sqrt(N) * b * np.sqrt(1 - b**2)
    eff_per_exc = c * b**2 * N
    intt0 = 1 + 2 * f * np.cos(phi) * (
        b0 * b1 + np.sqrt(2 * (N - 1) / N) * b2 * b1 + np.sqrt(3 * (N - 2) / N) * b3 * b2
    ) / g + f**2 * (b1**2 + 2 * (N - 1) / N * b2**2 + 3 * (N - 2) / N * b3**2) / g**2
    return intt0 * (eff_per_exc / f**2) + bg / f**2


def at(f, N, b, b0, b1, b2, b3, b4, phi, c, bg):
    """Coincidence rate for the state truncated at 4 excitations (beta_0..beta_4)."""
    g = np.sqrt(N) * b * np.sqrt(1 - b**2)
    eff_per_exc = c * b**2 * N
    at0 = (
        1
        + 4
        * f
        * np.cos(phi)
        * (b0 * b1 + np.sqrt(2 * (N - 1) / N) * b2 * b1 + np.sqrt(3 * (N - 2) / N) * b3 * b2)
        / g
        + 4 * f**2 * (b1**2 + 2 * (N - 1) / N * b2**2 + 3 * (N - 2) / N * b3**2) / g**2
        + 2
        * f**2
        * np.cos(2 * phi)
        * (
            np.sqrt(2 * N * (N - 1) / N**2) * b0 * b2
            + np.sqrt(6 * (N - 1) * (N - 2) / N**2) * b1 * b3
            + np.sqrt(12 * (N - 2) * (N - 3) / N**2) * b2 * b4
        )
        / g**2
        + 4
        * f**3
        * np.cos(phi)
        * (
            np.sqrt(2 * (N - 1) / N) * b1 * b2
            + 2 * (N - 1) / N * np.sqrt(3 * (N - 2) / N) * b2 * b3
            + 3 * (N - 2) / N * np.sqrt(4 * (N - 3) / N) * b3 * b4
        )
        / g**3
        + f**4
        * (2 * N * (N - 1) / N**2 * b2**2 + 6 * (N - 1) * (N - 2) / N**2 * b3**2 + 12 * (N - 2) * (N - 3) / N**2 * b4**2)
        / g**4
    )
    return at0 * (eff_per_exc / f**2) ** 2 + 2 * bg / f**2 * intt(f, N, b, b0, b1, b2, b3, phi, c, bg)


def g2t(f, N, s, b1, phi, c, bg, b, b0, b2, b3, b4):
    """g2 for the truncated state, with beta_3 = beta_4 = 0 (parametrized by s, beta_1)."""
    return at(f, N, b, b0, b1, b2, b3, b4, phi, c, bg) / intt(f, N, b, b0, b1, b2, b3, phi, c, bg) ** 2


def _truncated_amplitudes(N, s, b1):
    """beta_0, beta_2 from the pair-correlation parameter s (beta_3 = beta_4 = 0)."""
    b = b1 / np.sqrt(N)
    b2 = np.sqrt(((1 - 2 * s * b1**2) - np.sqrt(1 - 4 * s * b1**2)) / (4 * s))
    b3 = 0.0
    b4 = 0.0
    b0 = np.sqrt(1 - b1**2 - b2**2 - b3**2 - b4**2)
    return b, b0, b2, b3, b4


def main():
    N = 100
    s = 0.8
    b1 = 0.2
    b, b0, b2, b3, b4 = _truncated_amplitudes(N, s, b1)
    phi0 = np.pi
    c = 0.03
    bg = 0.0001
    cycle_num = 2000 * 60 / 3 * 60

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))

    # --- theory only: g2 vs phase -------------------------------------------------
    phi_list = np.linspace(0, 2 * np.pi, 100)
    g2t_list = g2t(1.0, N, s, b1, phi_list, c, bg, b, b0, b2, b3, b4)
    axes[0, 0].plot(phi_list, g2t_list)
    axes[0, 0].set_xlabel("phase")
    axes[0, 0].set_ylabel("g2")
    axes[0, 0].set_title("truncated state, g2(phi)")

    # --- theory only: g2 vs f (wide range + zoom) -----------------------------
    f_list = np.linspace(0.01, 100, 10000)
    g2t_list_f = g2t(f_list, N, s, b1, phi0, c, bg, b, b0, b2, b3, b4)
    axes[0, 1].plot(f_list[:99], g2t_list_f[:99])
    axes[0, 1].set_ylim(0, 2.5)
    axes[0, 1].set_xlabel("f")
    axes[0, 1].set_ylabel("g2")
    axes[0, 1].set_title("truncated state, g2(f) [zoom]")

    axes[0, 2].plot(f_list[:399], g2t_list_f[:399])
    axes[0, 2].set_xlabel("f")
    axes[0, 2].set_ylabel("g2")
    axes[0, 2].set_title("truncated state, g2(f) [wide]")

    # --- theoretical V(f) and g2(f) with synthetic counting-statistics errors -----
    f_data = np.arange(0.5, 2, 0.1)
    f_theo = np.arange(0.01, 2.5, 0.01)
    phi_theo = np.linspace(0, 2 * np.pi, 100)

    v_theo = [
        (
            intt(fi, N, b, b0, b1, b2, b3, phi_theo, c, bg).max()
            - intt(fi, N, b, b0, b1, b2, b3, phi_theo, c, bg).min()
        )
        / (
            intt(fi, N, b, b0, b1, b2, b3, phi_theo, c, bg).max()
            + intt(fi, N, b, b0, b1, b2, b3, phi_theo, c, bg).min()
        )
        for fi in f_theo
    ]
    axes[1, 0].plot(f_theo, v_theo, color="tab:green")
    axes[1, 0].set_xlabel("f")
    axes[1, 0].set_ylabel("V")
    axes[1, 0].set_title("visibility vs f (theory)")

    counts_data = np.array([intt(fi, N, b, b0, b1, b2, b3, phi0, c, bg) * cycle_num for fi in f_data])
    double_data = np.array([at(fi, N, b, b0, b1, b2, b3, b4, phi0, c, bg) * cycle_num for fi in f_data])
    g2_data = double_data / counts_data**2 * cycle_num
    g2_data_err = np.sqrt(double_data) / counts_data**2 * cycle_num
    g2_theo = g2t(f_theo, N, s, b1, phi0, c, bg, b, b0, b2, b3, b4)
    axes[1, 1].plot(f_theo, g2_theo, color="tab:blue")
    axes[1, 1].errorbar(f_data, g2_data, yerr=g2_data_err, fmt="o", color="black")
    axes[1, 1].set_ylim(0, 2)
    axes[1, 1].set_xlim(0, 0.9)
    axes[1, 1].set_xlabel("f")
    axes[1, 1].set_ylabel("g2")
    axes[1, 1].set_title("g2 vs f (theory + synthetic counting stats)")

    # --- theoretical efficiency(phi) and g2(phi) with synthetic errors, f=1 -------
    f = 1
    phi_theo = np.linspace(0, 2 * np.pi, 100)
    phi_data = np.arange(0, 2, 0.1) * np.pi

    counts_data = intt(f, N, b, b0, b1, b2, b3, phi_data, c, bg) * cycle_num
    eff_data = 2 * counts_data / cycle_num
    eff_data_err = 2 * np.sqrt(counts_data) / cycle_num
    eff_theo = 2 * intt(f, N, b, b0, b1, b2, b3, phi_theo, c, bg)

    double_data = at(f, N, b, b0, b1, b2, b3, b4, phi_data, c, bg) * cycle_num
    g2_data = double_data / counts_data**2 * cycle_num
    g2_data_err = np.sqrt(double_data) / counts_data**2 * cycle_num
    g2_theo = g2t(f, N, s, b1, phi_theo, c, bg, b, b0, b2, b3, b4)

    axes[1, 2].plot(phi_theo / np.pi, eff_theo, color="tab:orange")
    axes[1, 2].errorbar(phi_data / np.pi, eff_data, yerr=eff_data_err, fmt="o", color="black")
    axes[1, 2].set_xlabel("phi (pi)")
    axes[1, 2].set_ylabel("eta")
    axes[1, 2].set_title("efficiency vs phase (theory + synthetic stats)")

    axes[2, 0].plot(phi_theo / np.pi, g2_theo, color="tab:blue")
    axes[2, 0].errorbar(phi_data / np.pi, g2_data, yerr=g2_data_err, fmt="o", color="black")
    axes[2, 0].set_xlabel("phi (pi)")
    axes[2, 0].set_ylabel("g2")
    axes[2, 0].set_title("g2 vs phase (theory + synthetic stats)")

    # --- fit to real phase-scan data (data/best_g2.csv) ---------------------------
    df = pd.read_csv(DATA_CSV, sep=",")
    df["phase"] = df["phase"] * np.pi / 180
    # "eff err" is not stored in best_g2.csv; propagate Poisson counting error from
    # the raw coincidence channels the same way the companion Figure2.ipynb does.
    df["eff err"] = (np.sqrt(df["A3"]) + np.sqrt(df["A1"])) / df["total"]

    s_fit = 1.01  # pair-correlation parameter held fixed for this fit, as in the notebook

    def efficiency_fitting(phi, b1_, phi_0, N_, c_, bg_):
        phi = phi - phi_0
        b_ = b1_ / np.sqrt(N_)
        b2_ = np.sqrt(s_fit) * b1_**2 / np.sqrt(2)
        b3_ = 0.0
        b4_ = 0.0
        b0_ = np.sqrt(1 - b1_**2 - b2_**2 - b3_**2 - b4_**2)
        eff_per_exc = c_ * b_**2 * N_
        f = 1
        return intt(f, N_, b_, b0_, b1_, b2_, b3_, phi, c_, bg_) * (eff_per_exc / f**2) + bg_ / f**2

    def g2_fitting(phi, b1_, phi_0, N_, c_, bg_):
        f = 1
        phi = phi - phi_0
        b_, b0_, b2_, b3_, b4_ = _truncated_amplitudes(N_, s_fit, b1_)
        return g2t(f, N_, s_fit, b1_, phi, c_, bg_, b_, b0_, b2_, b3_, b4_)

    popt_eff, _ = curve_fit(
        efficiency_fitting,
        df["phase"],
        df["eff"],
        p0=[0.5, 1.2, 100, 0.03, 0.0001],
        bounds=([0, -2 * np.pi, 100, 0, 0], [0.5, 2 * np.pi, np.inf, 1, 1]),
        sigma=df["eff err"],
    )
    phi_theo = np.linspace(0, 2 * np.pi, 100)
    eff_fitted = efficiency_fitting(phi_theo + popt_eff[1], *popt_eff)
    axes[2, 1].errorbar(
        np.mod(df["phase"] - popt_eff[1], 2 * np.pi), df["eff"], yerr=df["eff err"], fmt="o"
    )
    axes[2, 1].plot(phi_theo, eff_fitted)
    axes[2, 1].set_xlabel("phase")
    axes[2, 1].set_ylabel("eff")
    axes[2, 1].set_title("efficiency fit to best_g2.csv")
    vis_fitted = (eff_fitted.max() - eff_fitted.min()) / (eff_fitted.max() + eff_fitted.min())
    print("efficiency fit: beta_1, phi_0, N, c, bg =", popt_eff)
    print("  fitted visibility =", vis_fitted)

    popt_g2, _ = curve_fit(
        g2_fitting,
        df["phase"],
        df["g2"],
        p0=[0.3, 2.55, 100, 0.03, 0.0001],
        bounds=([0, 0, 100, 0, 0], [0.5, 2 * np.pi, np.inf, 1, 1]),
        sigma=df["g2 err"],
    )
    axes[2, 2].errorbar(
        np.mod(df["phase"] - popt_g2[1], 2 * np.pi), df["g2"], yerr=df["g2 err"], fmt="o"
    )
    axes[2, 2].plot(phi_theo, g2_fitting(phi_theo + popt_g2[1], *popt_g2))
    axes[2, 2].set_xlabel("phi (pi)")
    axes[2, 2].set_ylabel("g2")
    axes[2, 2].set_title("g2 fit to best_g2.csv")
    print("g2 fit: beta_1, phi_0, N, c, bg =", popt_g2)

    # --- fit V(f), g2(f) and efficiency contrast to real (f, V, g2) summary data --
    # Digitized from repeated phase scans at several beam-splitter ratios f
    # (the notebook's own hardcoded summary of that experiment).
    f_list = np.array([0.49, 0.91, 1.03, 1.23, 1.47, 1.96])
    eff_max = np.array([0.065762376, 0.035433333, 0.028338983, 0.018393939, 0.019015152, 0.017909091])
    eff_min = np.array([0.015212871, 0.002758333, 0.00209322, 0.00340404, 0.003919192, 0.003545918])
    eff_diff_list = eff_max - eff_min
    v_list = np.array([0.624258732, 0.855553131, 0.862433862, 0.687673772, 0.658225061, 0.669455434])
    v_err_list = np.array([0.026549388, 0.027992619, 0.031637224, 0.035292761, 0.035733962, 0.036391687])
    g2_list = np.array([0.867, 2.46, 2.86, 1.34, 0.353, 0.81])
    g2_err_list = np.array([0.12, 0.48, 0.62, 0.27, 0.25, 0.29])

    s_ff = 0.9

    def eff_diff_fit(f, b1_, N_, c_):
        b_ = b1_ / np.sqrt(N_)
        b2_ = np.sqrt(s_ff) * b1_**2 / np.sqrt(2)
        b3_, b4_ = 0.0, 0.0
        b0_ = np.sqrt(1 - b1_**2 - b2_**2 - b3_**2 - b4_**2)
        eff_per_exc = c_ * b_**2 * N_
        g_ = np.sqrt(N_) * b_ * np.sqrt(1 - b_**2)
        Itmax = 1 + 2 * f * (b0_ * b1_ + np.sqrt(2 * (N_ - 1) / N_) * b2_ * b1_ + np.sqrt(3 * (N_ - 2) / N_) * b3_ * b2_) / g_ + f**2 * (b1_**2 + 2 * (N_ - 1) / N_ * b2_**2 + 3 * (N_ - 2) / N_ * b3_**2) / g_**2
        Itmin = 1 - 2 * f * (b0_ * b1_ + np.sqrt(2 * (N_ - 1) / N_) * b2_ * b1_ + np.sqrt(3 * (N_ - 2) / N_) * b3_ * b2_) / g_ + f**2 * (b1_**2 + 2 * (N_ - 1) / N_ * b2_**2 + 3 * (N_ - 2) / N_ * b3_**2) / g_**2
        return (Itmax - Itmin) * eff_per_exc / f**2

    def v_f_fit(f, b1_, N_, c_, bg_):
        b_ = b1_ / np.sqrt(N_)
        b2_ = np.sqrt(s_ff) * b1_**2 / np.sqrt(2)
        b3_, b4_ = 0.0, 0.0
        b0_ = np.sqrt(1 - b1_**2 - b2_**2 - b3_**2 - b4_**2)
        c_fixed = popt_eff_diff[2]
        eff_per_exc = c_fixed * b_**2 * N_
        Itmax = intt(f, N_, b_, b0_, b1_, b2_, b3_, 0, c_fixed, bg_)
        Itmin = intt(f, N_, b_, b0_, b1_, b2_, b3_, np.pi, c_fixed, bg_)
        return ((Itmax - Itmin) * eff_per_exc / f**2) / ((Itmax + Itmin) * eff_per_exc / f**2 + 2 * bg_ / f**2)

    def g2_f_fit(f, b1_, N_, c_, bg_):
        c_fixed = popt_eff_diff[2]
        b_, b0_, b2_, b3_, b4_ = _truncated_amplitudes(N_, s_ff, b1_)
        return g2t(f, N_, s_ff, b1_, np.pi, c_fixed, bg_, b_, b0_, b2_, b3_, b4_)

    f_theo2 = np.linspace(0.01, 5, 100)
    popt_eff_diff, _ = curve_fit(
        eff_diff_fit, f_list, eff_diff_list, p0=[0.3, 200, 0.03], bounds=([0, 100, 0], [0.5, np.inf, 1])
    )
    popt_v_f, _ = curve_fit(
        v_f_fit, f_list, v_list, p0=[0.3, 200, 0.06, 0.0001],
        bounds=([0, 100, 0, 0], [0.5, np.inf, 1, 1]), sigma=v_err_list,
    )
    popt_g2_f, _ = curve_fit(
        g2_f_fit, f_list, g2_list, p0=[0.3, 200, 0.06, 0.0001],
        bounds=([0, 100, 0, 0], [0.5, np.inf, 1, 1]), sigma=g2_err_list,
    )
    print("eff_diff fit: beta_1, N, c =", popt_eff_diff)
    print("V(f) fit: beta_1, N, c, bg =", popt_v_f)
    print("g2(f) fit: beta_1, N, c, bg =", popt_g2_f)

    axes[3, 0].errorbar(f_list, eff_diff_list, fmt="o")
    axes[3, 0].plot(f_theo2, eff_diff_fit(f_theo2, *popt_eff_diff))
    axes[3, 0].set_xlim(0.4, 2.1)
    axes[3, 0].set_ylim(0, 0.1)
    axes[3, 0].set_xlabel("f")
    axes[3, 0].set_ylabel("efficiency contrast")
    axes[3, 0].set_title("efficiency contrast fit vs f")

    axes[3, 1].errorbar(f_list, v_list, yerr=v_err_list, fmt="o")
    axes[3, 1].plot(f_theo2, v_f_fit(f_theo2, *popt_v_f))
    axes[3, 1].set_xlabel("f")
    axes[3, 1].set_ylabel("V")
    axes[3, 1].set_title("visibility fit vs f")

    axes[3, 2].errorbar(f_list, g2_list, yerr=g2_err_list, fmt="o")
    axes[3, 2].plot(f_theo2, g2_f_fit(f_theo2, *popt_g2_f))
    axes[3, 2].set_xlabel("f")
    axes[3, 2].set_ylabel("g2")
    axes[3, 2].set_title("g2 fit vs f")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
