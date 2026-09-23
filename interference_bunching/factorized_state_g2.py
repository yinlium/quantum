"""Factorized-state interference bunching/antibunching: I(phi) and g2(phi).

Reproduces the factorized-atomic-state result of P.R. Berman, H. Nguyen, Y. Mei,
Y. Li, A. Kuzmich, "Interference bunching and antibunching of coherent atomic
radiation fields," Phys. Rev. A 108, 043713 (2023), Sec. III.A. A single
collective spin excitation shared coherently among N atoms (amplitude b per
atom) radiates a field that interferes with a local-oscillator-like reference
field of relative strength f and phase phi. The efficiency (mean detected
intensity) I(phi) and second-order coherence g2(phi) both carry 1/N finite-
atom-number corrections; g2 is predicted to spike sharply (super-bunching) near
the destructive-interference phase phi = pi for f close to but below 1.

Source: 0223 factorized state.ipynb.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def efficiency(phi, A, f, N, b, phi_0):
    """Mean detected intensity I(phi), Eq. for the factorized state (Sec. III.A)."""
    phi = phi - phi_0
    p = b**2
    I = 1 + 2 * f * np.cos(phi) + f**2 + f**2 * p / (N * (1 - p))
    return I * A


def g2(phi, A, f, N, b, phi_0):
    """Second-order coherence g2(phi) for the factorized state, with 1/N, 1/N^2,
    1/N^3 finite-size corrections."""
    phi = phi - phi_0
    p = b**2
    denom = 1 + 2 * f * np.cos(phi) + f**2 + f**2 * p / (N * (1 - p))
    R_1 = (4 * f**2 * p - 2 * f**4 * (1 - 3 * p)) / (1 - p) - 2 * f**2 * np.cos(
        2 * phi
    ) - 4 * f**3 * (1 - 3 * p) / (1 - p) * np.cos(phi)
    R_2 = -8 * f**3 * p / (1 - p) * np.cos(phi) + f**4 * (1 - 10 * p + 11 * p**2) / (
        1 - p
    ) ** 2
    R_3 = 2 * f**4 * p * (2 - 3 * p) / (1 - p) ** 2
    num = (1 + 2 * f * np.cos(phi) + f**2) ** 2 + R_1 / N + R_2 / N**2 + R_3 / N**3
    return num / denom**2


def main():
    A = 0.025
    N = 100
    b = np.sqrt(0.1)
    phi_0 = 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) intensity vs phase
    f = 1
    phi_list = np.linspace(0, 2 * np.pi, 100)
    I_list = efficiency(phi_list, A, f, N, b, phi_0)
    axes[0].plot(phi_list, I_list)
    axes[0].set_xlabel("phase")
    axes[0].set_ylabel("efficiency")
    axes[0].set_title(f"I(phi), f={f}")

    # (b) g2 vs phase, log scale -- the super-bunching spike near phi = pi
    phi_list = np.linspace(0, 2 * np.pi, 1000)
    g2_list = g2(phi_list, A, f, N, b, phi_0)
    axes[1].plot(phi_list, g2_list)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("phase")
    axes[1].set_ylabel("g2")
    axes[1].set_title(f"g2(phi), f={f}")

    # (c) g2 vs phase near pi with counting-statistics error bars, f=0.9
    f = 0.9
    phi_list = np.linspace(np.pi - 0.3, np.pi + 0.3, 11)
    I_list = N * b**2 * (1 - b**2 + b**2 / N) * efficiency(phi_list, A, f, N, b, phi_0)
    g2_list = g2(phi_list, A, f, N, b, phi_0)
    total = 2000 * 20 * 60  # four hours of acquisition
    g2_err_list = (
        np.sqrt(g2_list * (I_list / 2 * total) ** 2 / total)
        * total
        / (I_list / 2 * total) ** 2
    )
    axes[2].errorbar(phi_list, g2_list, yerr=g2_err_list, fmt="o")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("phase")
    axes[2].set_ylabel("g2")
    axes[2].set_title(f"g2(phi) near pi, f={f}")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
