"""Effect of atom-number fluctuations on the collective sqrt(N) Rabi oscillation.

Companion analysis to Y. Mei, Y. Li, H. Nguyen, P. R. Berman, A. Kuzmich,
"Trapped Alkali-Metal Rydberg Qubit," Phys. Rev. Lett. 128, 123601 (2022). A
single collective Rydberg spin excitation shared by N atoms is retrieved with
efficiency eff(omega) = (exp(-omega) * sin(sqrt(N) * omega))**2, where omega
is the (dimensionless) two-photon retrieval pulse area; the sqrt(N) collective
enhancement of the Rabi frequency is the qubit's central signature. Because
the trapped atom number fluctuates shot to shot, the observed oscillation
contrast is reduced relative to the fixed-N prediction: this script averages
the efficiency over many shots with N drawn from a Gaussian distribution
around N=50 and compares to the noiseless (fixed-N) curve.

The second part fits the same finite-pulse Fourier lineshape (a sinc, from
the Fourier transform of a square RF/microwave addressing pulse) to a
measured counting-efficiency scan, data/CAvg.txt. The original notebook cell
doing this fit was broken (undefined ``FWHM_list``, ``curve_fit`` called with
swapped arguments, undefined ``p.sinc``); it is fixed here to a standard
scipy curve_fit of an offset, amplitude-scaled sinc.

Source: Effect of N.ipynb. Data: data/CAvg.txt.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def efficiency(omega, n):
    """Retrieval efficiency (exp(-omega) * sin(sqrt(n) * omega))**2 at fixed N=n."""
    return (np.exp(-omega) * np.sin(np.sqrt(n) * omega)) ** 2


def efficiency_n_averaged(omega, n_mean, n_std, n_shots, rng):
    """Monte Carlo average of efficiency() over Gaussian-distributed shot-to-shot N."""
    avg = np.zeros_like(omega)
    for _ in range(n_shots):
        n = max(n_mean + rng.normal(0, n_std), 0)
        avg += efficiency(omega, n)
    return avg / n_shots


def sinc_lineshape(x, amp, a, x0, offset):
    """Amplitude-scaled, offset sinc: the Fourier lineshape of a square pulse."""
    return amp * np.sinc(a * (x - x0)) + offset


def main():
    rng = np.random.default_rng(0)

    # (a) fixed-N vs. Gaussian-N-averaged collective Rabi oscillation
    omega = np.linspace(0, 0.5 * np.pi, 2000)
    n_mean, n_std = 50, 10
    eff_fixed = efficiency(omega, n_mean)
    eff_avg = efficiency_n_averaged(omega, n_mean, n_std, n_shots=400, rng=rng)

    # (b) sinc-lineshape fit to the measured counting-efficiency scan
    cavg = pd.read_csv(
        Path(__file__).parent / "data" / "CAvg.txt", sep=r"\s+", header=None, names=["x", "eff"]
    )
    popt, _ = curve_fit(
        sinc_lineshape, cavg["x"], cavg["eff"], p0=[1, 0.02, 10, 0]
    )
    print(f"sinc fit: amplitude={popt[0]:.3f} a={popt[1]:.4f} x0={popt[2]:.2f} offset={popt[3]:.3f}")
    x_fine = np.linspace(cavg["x"].min(), cavg["x"].max(), 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(omega, eff_fixed, label=f"fixed N={n_mean}")
    axes[0].plot(omega, eff_avg, label=f"N ~ Normal({n_mean}, {n_std}), {400} shots")
    axes[0].set_xlabel(r"$\Omega_{eff} t$ (rad)")
    axes[0].set_ylabel("retrieval efficiency")
    axes[0].set_title("effect of atom-number fluctuations on visibility")
    axes[0].legend(fontsize=8)

    axes[1].plot(cavg["x"], cavg["eff"], "o", label="data")
    axes[1].plot(x_fine, sinc_lineshape(x_fine, *popt), "-", label="sinc fit")
    axes[1].set_xlabel("pulse-sequence parameter (arb. index)")
    axes[1].set_ylabel("counting efficiency")
    axes[1].set_title("sinc lineshape fit")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
