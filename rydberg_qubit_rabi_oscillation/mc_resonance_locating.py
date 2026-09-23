"""Monte Carlo model of photon-counting statistics for locating a laser resonance.

Support simulation for Y. Mei, Y. Li, H. Nguyen, P. R. Berman, A. Kuzmich,
"Trapped Alkali-Metal Rydberg Qubit," Phys. Rev. Lett. 128, 123601 (2022).
Locating the two-photon Rydberg excitation resonance experimentally means
scanning a laser frequency and counting retrieved photons at each point. The
underlying (noiseless) lineshape is the sinc**2 Fourier transform of the
finite-duration (pulse_width) square excitation pulse; on top of that, each
shot picks up frequency jitter from the two lasers driving the two-photon
transition (independent Gaussian noise, one contributing twice) before the
mean signal is Poisson-sampled by photon counting statistics. Averaging many
shots per frequency point and fitting a Gaussian to the resulting counts
gives the effective (noise-broadened) linewidth and center actually usable
for locking/locating the resonance in the lab -- broader than the intrinsic
sinc**2 pulse-Fourier width because of the added laser-frequency noise.

Source: MC.ipynb.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

RESONANCE = 402.6  # nominal resonance frequency (arb. units, e.g. GHz offset)
LASER_ERROR = 0.3 / 2.33  # per-shot laser-frequency jitter (1 sigma, same units)
PULSE_WIDTH = 5  # excitation pulse duration -> sets the sinc**2 Fourier width
BACKGROUND_SCALE = 24  # mean background/signal counts scale
POISSON_SCALE = 10  # mean of the Poisson-sampled photon count contribution


def sinc2_lineshape(freq, resonance=RESONANCE, pulse_width=PULSE_WIDTH):
    """Noiseless sinc**2 Fourier lineshape of the finite-duration excitation pulse."""
    return np.sinc((freq - resonance) * pulse_width) ** 2


def single_shot_count(freq, rng, resonance=RESONANCE, laser_error=LASER_ERROR,
                       pulse_width=PULSE_WIDTH):
    """One noisy photon count at nominal laser frequency freq.

    Two independent Gaussian frequency jitters (one laser contributing twice,
    as in the two-photon transition) displace the sinc**2 lineshape before
    the mean signal is Poisson-sampled.
    """
    jitter = rng.normal(0, laser_error) + 2 * rng.normal(0, laser_error)
    lineshape = np.sinc((freq + jitter - resonance) * pulse_width) ** 2
    return lineshape * BACKGROUND_SCALE + rng.poisson(lineshape * POISSON_SCALE)


def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def main():
    rng = np.random.default_rng(0)

    laser_list = np.linspace(402, 403.5, 16)
    n_repeats = 100

    count_list = np.zeros(len(laser_list))
    for i, freq in enumerate(laser_list):
        for _ in range(n_repeats):
            count_list[i] += single_shot_count(freq, rng)

    popt, _ = curve_fit(gaussian, laser_list, count_list, p0=[1, RESONANCE, 0.1])
    print(f"Gaussian fit to MC counts: amplitude={popt[0]:.2f} center={popt[1]:.4f} "
          f"sigma={popt[2]:.4f}")

    freq_fine = np.linspace(402, 403.5, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(laser_list, count_list, label="MC counts (100 shots/pt)")
    axes[0].plot(freq_fine, gaussian(freq_fine, *popt), color="orange", label="Gaussian fit")
    axes[0].set_xlabel("laser frequency (arb. units)")
    axes[0].set_ylabel("summed photon counts")
    axes[0].set_title("noisy resonance scan + Gaussian fit")
    axes[0].legend(fontsize=8)

    axes[1].plot(freq_fine, sinc2_lineshape(freq_fine), color="tab:green")
    axes[1].set_xlabel("laser frequency (arb. units)")
    axes[1].set_ylabel(r"sinc$^2$ lineshape")
    axes[1].set_title(f"intrinsic pulse-Fourier lineshape (pulse width={PULSE_WIDTH})")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
