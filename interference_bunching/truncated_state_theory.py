"""Truncated-state g2 theory: xi/phi' parametrization and f, phi fluctuation averaging.

Reproduces the truncated-collective-state theory of P.R. Berman, H. Nguyen,
Y. Mei, Y. Li, A. Kuzmich, "Interference bunching and antibunching of coherent
atomic radiation fields," Phys. Rev. A 108, 043713 (2023), Sec. III.B-III.C,
neglecting atom-atom interactions. The two-excitation-truncated state is
parametrized by the 1-excitation amplitude beta_1, the pair-correlation
parameter s (which fixes beta_2 = sqrt(s) * beta_1^2 / sqrt(2)), and a phase
offset phi' between the one- and two-excitation contributions to the radiated
field. Besides the bare efficiency eta(phi) and g2(phi)/g2(f) predictions
(and how they depend on beta_1^2 and on f), this reproduces how shot-to-shot
fluctuations of the reference-field ratio f and of the interferometer phase
phi wash out the g2(f) super-bunching peak when averaged over a Gaussian
spread of f or phi.

Source: Dephasing_230519.ipynb (dozens of empty/dead cells and unused
ipywidgets/plotly/imageio/colormap imports from interactive exploration were
dropped).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


def intt2(f, s, b1, phi, phip, c, bg):
    """Intensity of the 2-excitation-truncated state, phase offset phip."""
    b0 = np.sqrt(1 - b1**2 - s * b1**4 / 2)
    intt0 = 1 + 2 * f * (np.abs(b0) * np.cos(phi) + np.sqrt(s) * b1**2 * np.cos(phi + phip)) + f**2 * (1 + s * b1**2)
    return intt0 * (c * b1**2 / f**2) + bg / f**2


def at2(f, s, b1, phi, phip, c, bg):
    """Coincidence rate of the 2-excitation-truncated state."""
    b0 = np.sqrt(1 - b1**2 - s * b1**4 / 2)
    at0 = (
        1
        + 4 * f * (np.abs(b0) * np.cos(phi) + np.sqrt(s) * b1**2 * np.cos(phi + phip))
        + 4 * f**2 * (1 + s * b1**2)
        + 2 * f**2 * np.abs(b0) * np.sqrt(s) * np.cos(2 * phi + phip)
        + 4 * f**3 * np.sqrt(s) * np.cos(phi + phip)
        + f**4 * s
    )
    return at0 * (c * b1**2 / f**2) ** 2 + 2 * bg / f**2 * intt2(f, s, b1, phi, phip, c, bg)


def g2t2(f, s, b1, phi, phip, c, bg):
    return at2(f, s, b1, phi, phip, c, bg) / intt2(f, s, b1, phi, phip, c, bg) ** 2


def eff_calc(f_data, f_theo, s, b1, phi, phip, c, bg, cycle_num):
    counts_data = np.array([intt2(fi, s, b1, phi, phip, c, bg) * cycle_num for fi in f_data])
    eff_data = 2 * counts_data / cycle_num
    eff_data_err = 2 * np.sqrt(counts_data) / cycle_num
    eff_theo = np.array([2 * intt2(fi, s, b1, phi, phip, c, bg) for fi in f_theo])
    return eff_theo, eff_data, eff_data_err


def vis_calc(f_data, f_theo, phi_theo, s, b1, phi, phip, c, bg):
    def _vis(fi):
        counts = intt2(fi, s, b1, phi_theo, phip, c, bg)
        return (counts.max() - counts.min()) / (counts.max() + counts.min())

    return [_vis(fi) for fi in f_theo], [_vis(fi) for fi in f_data]


def g2_calc(f_data, f_theo, s, b1, phi, phip, c, bg, cycle_num):
    double_data = np.array([at2(fi, s, b1, phi, phip, c, bg) * cycle_num for fi in f_data])
    counts_data = np.array([intt2(fi, s, b1, phi, phip, c, bg) * cycle_num for fi in f_data])
    g2_data = double_data / counts_data**2 * cycle_num
    g2_data_err = np.sqrt(double_data) / counts_data**2 * cycle_num
    g2_theo = np.array([g2t2(fi, s, b1, phi, phip, c, bg) for fi in f_theo])
    return g2_theo, g2_data, g2_data_err


def g2t2_avg_f(f, s, b1, phi, phip, c, bg, f_sigma, f_num):
    """g2(f) averaged over a Gaussian spread of the reference-field ratio f."""
    y = 0.0
    weight = np.zeros(f_num)
    for j in range(f_num):
        f_j = f + (j - f_num / 2) * f_sigma * f * 5 / (f_num / 2)
        weight[j] = stats.norm.pdf(f_j, f, f_sigma * f)
        y += weight[j] * at2(f_j, s, b1, phi, phip, c, bg) / intt2(f_j, s, b1, phi, phip, c, bg) ** 2
    return y / np.sum(weight)


def g2t2_avg_phi(f, s, b1, phi, phip, c, bg, phi_sigma, phi_num):
    """g2(f) averaged over a Gaussian spread of the interferometer phase phi."""
    y = np.zeros(np.shape(f))
    weight = np.zeros(phi_num)
    for j in range(phi_num):
        phi_j = phi + (j - phi_num / 2) * phi_sigma * 5 / (phi_num / 2)
        weight[j] = stats.norm.pdf(phi_j, phi, phi_sigma)
        y = y + weight[j] * at2(f, s, b1, phi_j, phip, c, bg) / intt2(f, s, b1, phi_j, phip, c, bg) ** 2
    return y / np.sum(weight)


def main():
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    # (a)/(b): efficiency and g2 vs phase, single (f, beta_1, s) point
    f, s, b1 = 1, 0.15, np.sqrt(0.5)
    phip = np.pi / 2
    c, bg = 0.01, 0.0001
    cycle_num = 2000 * 60 / 3 * 60
    phi_theo = np.linspace(0, 2 * np.pi, 100)
    phi_data = np.arange(0, 2, 0.1) * np.pi

    # eff_calc/g2_calc (defined above) sweep f at fixed phi; here phi itself is
    # swept at fixed f, so the intensity/coincidence formulas are used directly.
    counts_data = intt2(f, s, b1, phi_data, phip, c, bg) * cycle_num
    eff_data = 2 * counts_data / cycle_num
    eff_data_err = 2 * np.sqrt(counts_data) / cycle_num
    eff_theo = 2 * intt2(f, s, b1, phi_theo, phip, c, bg)
    axes[0, 0].plot(phi_theo / np.pi, eff_theo, color="tab:orange")
    axes[0, 0].errorbar(phi_data / np.pi, eff_data, yerr=eff_data_err, fmt="o", color="black")
    axes[0, 0].set_xlabel(r"$\phi~(\pi)$")
    axes[0, 0].set_ylabel(r"$\eta$")
    axes[0, 0].set_title("efficiency vs phase")

    counts_data = intt2(f, s, b1, phi_data, phip, c, bg) * cycle_num
    double_data = at2(f, s, b1, phi_data, phip, c, bg) * cycle_num
    g2_data = double_data / counts_data**2 * cycle_num
    g2_data_err = np.sqrt(double_data) / counts_data**2 * cycle_num
    g2_theo = g2t2(f, s, b1, phi_theo, phip, c, bg)
    axes[0, 1].plot(phi_theo / np.pi, g2_theo, color="tab:blue")
    axes[0, 1].errorbar(phi_data / np.pi, g2_data, yerr=g2_data_err, fmt="o", color="black")
    axes[0, 1].set_xlabel(r"$\phi~(\pi)$")
    axes[0, 1].set_ylabel(r"$g^{(2)}$")
    axes[0, 1].set_title("g2 vs phase")

    # (c): g2 vs |beta_1|^2 at phi=pi
    b1_list = np.sqrt(np.linspace(0.1, 0.8, 100))
    g2f_list = g2t2(f, s, b1_list, np.pi, phip, c, bg / 2)
    axes[0, 2].plot(b1_list**2, g2f_list)
    axes[0, 2].set_xlabel(r"$|\beta_1|^2$")
    axes[0, 2].set_ylabel(r"$g^{(2)}$")
    axes[0, 2].set_title("g2 vs excitation probability")

    # (d)/(e)/(f): efficiency, visibility and g2 vs f, several beta_1^2
    f_v, s_v, phip_v, c_v, bg_v = 1, 0.15, np.pi / 2, 0.02, 0.0001
    phi_v = np.pi
    f_data = np.arange(0.1, 2, 0.1)
    f_theo = np.arange(0.1, 2.5, 0.01)
    phi_theo_v = np.linspace(0, 2 * np.pi, 100)
    beta_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    for beta_sq in beta_list:
        eff_theo, eff_data, eff_data_err = eff_calc(f_data, f_theo, s_v, np.sqrt(beta_sq), phi_v, phip_v, c_v, bg_v, cycle_num)
        axes[1, 0].plot(f_theo, eff_theo, label=f"{beta_sq:.1f}")
        axes[1, 0].errorbar(f_data, eff_data, yerr=eff_data_err, fmt="o", color="black", markersize=3)
    axes[1, 0].set_ylim(0, 0.02)
    axes[1, 0].set_xlabel("f")
    axes[1, 0].set_ylabel(r"$\eta$")
    axes[1, 0].set_title(r"efficiency vs f, several $|\beta_1|^2$")
    axes[1, 0].legend(fontsize=7, loc="upper right")

    for beta_sq in beta_list:
        v_theo, v_data = vis_calc(f_data, f_theo, phi_theo_v, s_v, np.sqrt(beta_sq), phi_v, phip_v, c_v, bg_v)
        axes[1, 1].plot(f_theo, v_theo, label=f"{beta_sq:.1f}")
        axes[1, 1].errorbar(f_data, v_data, fmt="o", color="black", markersize=3)
    axes[1, 1].set_xlabel("f")
    axes[1, 1].set_ylabel(r"$\mathcal{V}$")
    axes[1, 1].set_title(r"visibility vs f, several $|\beta_1|^2$")
    axes[1, 1].legend(fontsize=7, loc="upper right")

    for beta_sq in beta_list:
        g2_theo, g2_data, g2_data_err = g2_calc(f_data, f_theo, s_v, np.sqrt(beta_sq), phi_v, phip_v, c_v, bg_v, cycle_num)
        axes[1, 2].plot(f_theo, g2_theo, label=f"{beta_sq:.1f}")
        axes[1, 2].errorbar(f_data, g2_data, yerr=g2_data_err, fmt="o", color="black", markersize=3)
    axes[1, 2].set_ylim(0, 10)
    axes[1, 2].set_xlim(0, 3)
    axes[1, 2].set_xlabel("f")
    axes[1, 2].set_ylabel(r"$g^{(2)}$")
    axes[1, 2].set_title(r"g2 vs f, several $|\beta_1|^2$")
    axes[1, 2].legend(fontsize=7, loc="upper right")

    # (g)/(h): fluctuation averaging washes out the g2(f) super-bunching peak
    f, s, b1, phip = 1, 0.15, np.sqrt(0.3), np.pi / 2
    phi = np.pi
    c, bg = 0.02, 0.0001
    f_list = np.linspace(0.5, 2.5, 500)

    g2f_list_f = g2t2(f_list, s, b1, phi, phip, c, bg)
    g2f_list_favg = [g2t2_avg_f(fi, s, b1, phi, phip, c, bg, f_sigma=0.1, f_num=100) for fi in f_list]
    axes[2, 0].plot(f_list, g2f_list_f, label="w/o f fluctuation")
    axes[2, 0].plot(f_list, g2f_list_favg, label="w/ f fluctuation (10%)")
    axes[2, 0].set_xlabel("f")
    axes[2, 0].set_ylabel(r"$g^{(2)}$")
    axes[2, 0].set_title("f-fluctuation averaging")
    axes[2, 0].legend(fontsize=8)

    g2f_avg_phi_list = g2t2_avg_phi(f_list, s, b1, phi, phip, c, bg, phi_sigma=np.pi * 0.1, phi_num=100)
    axes[2, 1].plot(f_list, g2f_list_f, label="w/o phase fluctuation")
    axes[2, 1].plot(f_list, g2f_avg_phi_list, "--", label="w/ phase fluctuation")
    axes[2, 1].set_xlabel("f")
    axes[2, 1].set_ylabel(r"$g^{(2)}$")
    axes[2, 1].set_title("phase-fluctuation averaging")
    axes[2, 1].legend(fontsize=8)

    axes[2, 2].axis("off")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
