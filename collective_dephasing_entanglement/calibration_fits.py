"""Experimental calibration fits underlying the g^(2)(T_s) measurement of
Y. Li, Y. Mei, H. Nguyen, P. R. Berman, A. Kuzmich, "Dynamics of
Collective-Dephasing-Induced Multiatom Entanglement," Phys. Rev. A 106,
L051701 (2022), Supplemental Material.

Three independent fits calibrate quantities used elsewhere in this package
(N = 273 atoms in g2_storage_time_fit.py and dephasing_dynamics_and_truncation.py):

(a) Two-photon Rabi flopping of the single-atom retrieval probability p1 vs.
    excitation pulse width, fit to a damped-oscillation model; the fitted
    Rabi frequency and decay set the effective atom number N via the
    collective enhancement sqrt(N) of the coupling Rabi frequency.
(b) The excitation-laser spectrum (retrieval efficiency vs. two-photon
    detuning for a fixed pulse), fit to a sinc^2 lineshape set by the
    Fourier-transform-limited excitation pulse.
(c) Electromagnetically-induced-transparency (EIT) transmission spectrum of
    the probe beam, fit to the standard three-level susceptibility, used to
    calibrate the optical depth and coupling Rabi frequency.

Source: Supplement figure.ipynb ("g2 revival" folder), the cells loading
rabi.csv / "exc spec.csv" / eit_35.csv and fitting simple_model / Lor / EIT
(cells ~31-44); the duplicate lmfit-based refit of the same Rabi data
(Parameters/minimize) was dropped as redundant with the curve_fit result it
reproduces. Data: rabi.csv, exc_spec.csv (originally "exc spec.csv"),
eit_35.csv, originally under "Many body rabi oscillation/simulation/dephasing/"
(referenced by relative path from Supplement figure.ipynb).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent / "data"


def simple_model(x, a, alpha, beta, gamma, omega, x0):
    """Damped two-photon Rabi oscillation of the single-atom retrieval
    probability vs. pulse width x (us): a Gaussian envelope a*exp(-alpha*x^2)
    times (1 - a decaying cosine at frequency set by sqrt(omega))."""
    x = x - x0
    return a * np.exp(-alpha * x**2) * (
        1 - np.exp(-beta * x) * np.exp(-gamma * x**2) * np.cos(2 * np.pi * 9.2 * 10.6 / 2 / 480 * np.sqrt(omega) * x)
    )


def lorentzian_sinc2(x, a, x0, sigma, z):
    """sinc^2 lineshape of the excitation efficiency vs. two-photon detuning
    (Fourier-transform-limited square excitation pulse)."""
    return a * np.sinc((x - x0) * sigma) ** 2 + z


def eit_transmission(x, gamma_2, x0, omega_2, delta, a, b, od_0):
    """Standard three-level EIT transmission a*exp(-Im[OD]) + b vs. probe
    detuning x (MHz), with coupling Rabi frequency omega_2, two-photon
    detuning delta, and excited-state linewidth Gamma = 2*pi*6 MHz."""
    Gamma = 2 * np.pi * 6e6
    gamma_2 = 2 * np.pi * gamma_2
    omega_2 = 2 * np.pi * omega_2
    OD = (
        od_0
        * Gamma
        / 2
        * 4
        * (2 * np.pi * (x - x0) * 1e6 + complex(0, gamma_2 / 2))
        / (
            omega_2**2
            - 4 * (2 * np.pi * (x - x0) * 1e6 + 2 * np.pi * delta + complex(0, Gamma / 2))
            * (2 * np.pi * (x - x0) * 1e6 + complex(gamma_2 / 2))
        )
    )
    return a * np.exp(-OD.imag) + b


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Rabi flopping efficiency vs. pulse width
    rabi = pd.read_csv(DATA_DIR / "rabi.csv", dtype="float64")
    x_rabi = (rabi["pulse width"] / 1000).to_numpy()  # ns -> us
    y_rabi = rabi["eff"].to_numpy()
    sigma_rabi = rabi["eff err"].to_numpy()
    popt_rabi, pcov_rabi = curve_fit(
        simple_model,
        x_rabi,
        y_rabi,
        p0=[0.02, 0.015, 0.02, 0.054, 100, 0.1],
        bounds=([0, 0, 0, 0.054, 0, 0], [100, 100, 100, 0.055, 800, 100]),
        sigma=sigma_rabi,
        maxfev=20000,
    )
    print("Rabi-flopping fit [a, alpha, beta, gamma, omega, x0]:", popt_rabi)
    print("Rabi-flopping fit uncertainties:", np.sqrt(np.diag(pcov_rabi)))

    x_fit = np.linspace(min(x_rabi), max(x_rabi) * 1.2, 1000)
    axes[0].errorbar(x_rabi, y_rabi, yerr=sigma_rabi, fmt="o", color="tab:red",
                      markerfacecolor="crimson", label=r"$N = 273 \pm 4$", markersize=4)
    axes[0].plot(x_fit, simple_model(x_fit, *popt_rabi), "--", color="red", lw=1)
    axes[0].set_xlabel(r"$T_p$ ($\mu$s)")
    axes[0].set_ylabel(r"$p_1$")
    axes[0].set_xlim(0, 6)
    axes[0].legend(fontsize=9)
    axes[0].set_title("(a) Two-photon Rabi flopping")

    # (b) excitation spectrum, Lorentzian (sinc^2) fit
    spec = pd.read_csv(DATA_DIR / "exc_spec.csv", dtype="float64")
    x_spec = spec["freq"].to_numpy()
    y_spec = spec["eff"].to_numpy()
    sigma_spec = spec["eff err"].to_numpy()
    popt_spec, _ = curve_fit(lorentzian_sinc2, x_spec, y_spec, p0=[0.005, -1930, 0.1, 0.0001], sigma=sigma_spec)
    print("Excitation-spectrum fit [a, x0, sigma, z]:", popt_spec)

    x0_spec = popt_spec[1]
    x_fit_spec = np.linspace(min(x_spec), max(x_spec), 200)
    axes[1].errorbar(x_spec - x0_spec, y_spec, yerr=sigma_spec, fmt="o", color="tab:brown", label="data")
    axes[1].plot(x_fit_spec - x0_spec, lorentzian_sinc2(x_fit_spec, *popt_spec), "--", lw=1,
                 color="tab:brown", label="fit")
    axes[1].set_xlabel(r"$\delta/2\pi$ (MHz)")
    axes[1].set_ylabel(r"$p_1$")
    axes[1].legend(fontsize=9)
    axes[1].set_title("(b) Excitation spectrum (sinc$^2$)")

    # (c) EIT transmission fit
    df_eit = pd.read_csv(DATA_DIR / "eit_35.csv")
    x_eit = -df_eit["freq"].to_numpy()
    y_eit = df_eit["norm eff"].to_numpy()
    popt_eit, _ = curve_fit(
        eit_transmission,
        x_eit,
        y_eit,
        p0=[0.02e6, -2504, 2e6, 0.1e6, 1, 0, 3],
        bounds=([0, -np.inf, 0, 0, 0, 0, 0], [np.inf, 0, np.inf, 2e6, 2, 1, 5]),
        maxfev=200000,
    )
    print("EIT fit [gamma_2, x0, omega_2, delta, a, b, od_0]:", popt_eit)

    x_fit_eit = np.linspace(min(x_eit), max(x_eit), 200)
    axes[2].plot(x_eit + 2503.97, y_eit, "o", color="tab:cyan", label="EIT")
    axes[2].plot(x_fit_eit + 2503.97, eit_transmission(x_fit_eit, *popt_eit), "--", color="tab:cyan", alpha=0.9, lw=1)
    axes[2].set_xlabel(r"$\delta/2\pi$ (MHz)")
    axes[2].set_ylabel("Transmission")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlim(-4.5, 4.5)
    axes[2].set_title("(c) EIT transmission")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
