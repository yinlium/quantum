"""Collective sqrt(N) Rabi oscillation: fit of the retrieval efficiency vs. pulse width.

Reproduces the Rabi-oscillation fitting used in Y. Mei, Y. Li, H. Nguyen, P. R.
Berman, A. Kuzmich, "Trapped Alkali-Metal Rydberg Qubit," Phys. Rev. Lett. 128,
123601 (2022). A weak two-photon (EIT-Raman) excitation pulse of variable
width couples the ground state to a single collective Rydberg excitation
shared by N atoms in the trap; because the two-atom-photon coupling scales
with the number of atoms as sqrt(N), the detection ("retrieval") efficiency
oscillates and decays as

    eff(t) = D*exp(-E*(t-t0)) - A*exp(-B*(t-t0))*cos(sqrt(N)*Omega_eff*(t-t0))

with a fixed single-atom two-photon Rabi-frequency scale Omega_eff set by the
control/probe Rabi frequencies (11.9, 14.8 MHz) and single-photon detuning
(480 MHz) of the EIT ladder scheme. Because the trapped atom number N
fluctuates shot to shot (loaded from a Poissonian atom-number distribution),
the observed contrast is reduced relative to the fixed-N curve; a second fit
Poisson-averages the same lineshape over N to capture this.

Source: Rabi fitting.ipynb. Data: data/150uk.csv (150 uK trap depth
retrieval-efficiency scan; columns "pulse width" [ns], "eff" [fraction]).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import poisson

# Fixed two-photon Rabi-frequency scale (MHz) set by the control/probe Rabi
# frequencies (11.9, 14.8 MHz) and the 480 MHz single-photon EIT detuning.
OMEGA_EFF_CONST = 2 * np.pi * 11.9 * 14.8 / (2 * 480)


def rabi_decay(t, a, b, n, d, e, t0):
    """Damped collective Rabi oscillation at fixed atom number n (=N)."""
    return d * np.exp(-e * (t - t0)) - a * np.exp(-b * (t - t0)) * np.cos(
        np.sqrt(n) * OMEGA_EFF_CONST * (t - t0)
    )


def rabi_decay_poisson_avg(t, a, b, n_mean, d, e, t0, omega_eff):
    """rabi_decay Poisson-averaged over the shot-to-shot atom number N.

    omega_eff is fit freely here (rather than fixed to OMEGA_EFF_CONST) since
    the Poisson average itself reshapes the effective envelope.
    """
    y = np.zeros_like(np.asarray(t, dtype=float))
    n_lo = max(int(n_mean - 3 * np.sqrt(n_mean)), 0)
    n_hi = int(n_mean + 3 * np.sqrt(n_mean))
    for j in range(n_lo, n_hi + 1):
        weight = poisson.pmf(j, n_mean)
        y += weight * (
            d * np.exp(-e * (t - t0))
            - a * np.exp(-b * (t - t0)) * np.cos(np.sqrt(j) * omega_eff * (t - t0))
        )
    return y


def main():
    df = pd.read_csv(Path(__file__).parent / "data" / "150uk.csv")
    x = df["pulse width"].to_numpy() / 1000  # ns -> us
    y = df["eff"].to_numpy() * 100  # fraction -> percent

    bounds = ([0, 0, 0, 0, 0, 0], [np.inf] * 6)
    popt1, _ = curve_fit(
        rabi_decay, x, y, p0=[1, 0.1, 4, 0.01, 0.1, 0], bounds=bounds, maxfev=10000
    )

    bounds2 = ([0, 0, 0, 0, 0, 0, 0], [np.inf] * 7)
    popt2, _ = curve_fit(
        rabi_decay_poisson_avg,
        x,
        y,
        p0=[1, 0.1, 200, 0.01, 0.1, 0, 2],
        bounds=bounds2,
        maxfev=10000,
    )

    print(f"fixed-N fit:   A={popt1[0]:.3f} B={popt1[1]:.3f} N={popt1[2]:.1f} "
          f"D={popt1[3]:.3f} E={popt1[4]:.3f} t0={popt1[5]:.3f}")
    print(f"Poisson-N fit: A={popt2[0]:.3f} B={popt2[1]:.3f} <N>={popt2[2]:.1f} "
          f"D={popt2[3]:.3f} E={popt2[4]:.3f} t0={popt2[5]:.3f} "
          f"Omega_eff={popt2[6]:.3f} MHz")

    t_fine = np.linspace(min(x), max(x) * 1.2, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(x, y, "o", label="data")
    axes[0].plot(t_fine, rabi_decay(t_fine, *popt1), "-", label="fit")
    axes[0].set_xlim([0, 6])
    axes[0].set_xlabel(r"retrieval pulse width ($\mu$s)")
    axes[0].set_ylabel("detection efficiency (%)")
    axes[0].set_title(f"fixed N: N={popt1[2]:.0f}")
    axes[0].legend()

    axes[1].plot(x, y, "o", label="data")
    axes[1].plot(t_fine, rabi_decay_poisson_avg(t_fine, *popt2), "-", label="fit")
    axes[1].set_xlim([0, 6])
    axes[1].set_xlabel(r"retrieval pulse width ($\mu$s)")
    axes[1].set_ylabel("detection efficiency (%)")
    axes[1].set_title(f"Poisson-averaged N: <N>={popt2[2]:.0f}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
