"""Multiatom entanglement witness from the joint one- and two-excitation
detection probabilities P1, P2, for Y. Li, Y. Mei, H. Nguyen, P. R. Berman,
A. Kuzmich, "Dynamics of Collective-Dephasing-Induced Multiatom
Entanglement," Phys. Rev. A 106, L051701 (2022).

If the retrieved light originated from M independent (unentangled,
single-mode-per-atom-group) emitters, each with its own real excitation
amplitude pair (a_i, b_i), a_i^2 + b_i^2 = 1, then the joint one- and
two-photon detection probabilities of the combined field are

    P1 = (sum_i b_i/a_i)^2 * (prod_i a_i)^2 / M
    P2 = 2 * (prod_i a_i)^2 / M^2 * (sum_{i<j} b_i b_j /(a_i a_j))^2

Sampling (a_i) uniformly at random and evaluating (P1, P2) over many draws
traces out the maximal-P2-at-fixed-P1 boundary achievable by M independent
emitters (a Monte Carlo evaluation, since no closed form is used). A
measured (P1, P2) point that lies *below* the M=1 boundary line P2 = P1^2/2
is inconsistent with a single independent mode, and comparing to the M=2, 3
Monte Carlo boundaries bounds how many independent (unentangled) modes could
have produced it -- the entanglement witness used in the paper (Fig. 5 in
the original analysis).

Measured photon-counting probabilities p1, p2 are converted to atomic
excitation probabilities via the (independently calibrated) total detection
efficiency eta = 0.074: P1 = p1/eta, P2 = p2/eta^2.

Monte Carlo sample counts are reduced from the original notebook (10^8) to
keep runtime short; the boundary is fully vectorized with numpy instead of
the original per-sample Python loops, but the underlying probability
formulas and envelope-tracing procedure (binning in P2 to bound P1, and
binning in P1 to bound P2, to trace both the shallow and steep parts of the
boundary) are unchanged. Source: Monte Carlo.ipynb ("Entanglement/data"
folder), the final annotated version of the P1-P2 plot (cell saving
"fig 5 0913.pdf"); Compare_c_significence.ipynb, Another_Approach.ipynb,
MC_c_included.ipynb and p1_p0.ipynb were early scratch/superseded iterations
of the same P1-P2 analysis (unlabeled axes, dead commented-out code, or
loading intermediate files no longer needed) and were not used. Data:
0909.csv, 0910.csv (measured p1, p2 at several Rydberg n / Rabi frequencies).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
ETA = 0.074  # total (retrieval x collection x detection) efficiency


def sample_p1_p2(M, n_samples, rng):
    """Monte Carlo draws of (P1, P2) for M independent emitters with random
    excitation amplitudes a_i ~ Uniform(0,1), b_i = sqrt(1-a_i^2)."""
    a = rng.uniform(1e-9, 1 - 1e-9, size=(M, n_samples))
    b = np.sqrt(1 - a**2)
    big_A = np.prod(a, axis=0)  # prod_i a_i
    ba = b / a
    sum_ba = np.sum(ba, axis=0)  # sum_i b_i/a_i
    p1 = sum_ba**2 * big_A**2 / M

    # sum_{i<j} b_i b_j/(a_i a_j) = [(sum_i b_i/a_i)^2 - sum_i (b_i/a_i)^2] / 2
    cross_sum = (sum_ba**2 - np.sum(ba**2, axis=0)) / 2
    p2 = 2 * big_A**2 / M**2 * cross_sum**2
    return p1, p2


def envelope_max_y_given_x_bins(x_values, y_values, x_bin_edges):
    """For each bin of x_bin_edges, the max of y_values among samples whose
    x_values falls in that bin -- traces the upper boundary of the (x, y)
    point cloud as a function of x."""
    idx = np.digitize(x_values, x_bin_edges) - 1
    result = np.full(len(x_bin_edges), np.nan)
    valid = (idx >= 0) & (idx < len(x_bin_edges))
    df = pd.DataFrame({"bin": idx[valid], "y": y_values[valid]})
    for b, y_max in df.groupby("bin")["y"].max().items():
        result[b] = y_max
    return result


def main():
    rng = np.random.default_rng(0)
    n_samples = 20_000_000  # reduced from 10^8

    p1_m2, p2_m2 = sample_p1_p2(2, n_samples, rng)
    p1_m3, p2_m3 = sample_p1_p2(3, n_samples, rng)

    # M=2 boundary: max P1 for bins of P2 (the whole accessible P2 range).
    p2_bins_m2 = 10 ** np.linspace(-5, -0.25, 100)
    p1_bound_m2 = envelope_max_y_given_x_bins(p2_m2, p1_m2, p2_bins_m2)

    # M=3 boundary, traced from both directions to resolve the steep part:
    # (i) max P1 for bins of P2 (shallow part of the boundary)
    p2_bins_m3 = 10 ** np.linspace(-5, -0.5, 100)
    p1_bound_m3 = envelope_max_y_given_x_bins(p2_m3, p1_m3, p2_bins_m3)
    # (ii) max P2 for bins of P1 (steep part, underrepresented by (i))
    p1_bins_m3 = 10 ** np.linspace(-2, -0.3, 100)
    p2_bound_m3 = envelope_max_y_given_x_bins(p1_m3, p2_m3, p1_bins_m3)

    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.plot(np.log10(p1_m3), np.log10(p2_m3), ",", color="lightgray", rasterized=True, zorder=-32)

    ax.plot(np.log10(p1_bound_m2), np.log10(p2_bins_m2), color="darkblue", label="M = 2 bound")
    ax.plot(np.log10(p1_bound_m3), np.log10(p2_bins_m3), color="tab:orange", label="M = 3 bound")
    ax.plot(np.log10(p1_bins_m3), np.log10(p2_bound_m3), color="tab:orange")

    # Many acquisitions saw zero two-fold coincidences (p2 = 0, an
    # unremarkable outcome given total counts of order 1e5-1e6); those
    # points are dropped here since log10(0) is undefined, rather than
    # plotted as -inf.
    df_0909 = pd.read_csv(DATA_DIR / "0909.csv")
    first_half, second_half = df_0909.iloc[:11], df_0909.iloc[11:]
    for df_half, color in [(first_half, "tab:red"), (second_half, "darkviolet")]:
        df_half = df_half[df_half["p2"] > 0]
        fmt = "^" if color == "tab:red" else "o"
        ax.errorbar(
            np.log10(df_half["p1"].to_numpy() / ETA), np.log10(df_half["p2"].to_numpy() / ETA**2),
            yerr=0.434 * df_half["p2 err"].to_numpy() / df_half["p2"].to_numpy(),
            color=color, fmt=fmt, markersize=5, alpha=0.5,
        )

    df_0910 = pd.read_csv(DATA_DIR / "0910.csv")
    p1_0910, p2_0910, p2err_0910 = df_0910["p1"].to_numpy(), df_0910["p2"].to_numpy(), df_0910["p2 err"].to_numpy()
    ax.errorbar(
        np.log10(p1_0910[0] / ETA), np.log10(p2_0910[0] / ETA**2),
        yerr=0.434 * p2err_0910[0] / p2_0910[0],
        color="tab:red", fmt="^", markersize=5, label=r"$\Omega = 2\pi \times 26.6$ kHz",
    )
    ax.errorbar(
        np.log10(p1_0910[1] / ETA), np.log10(p2_0910[1] / ETA**2),
        yerr=0.434 * p2err_0910[1] / p2_0910[1],
        color="darkviolet", fmt="o", markersize=5, label=r"$\Omega = 2\pi \times 106$ kHz",
    )

    ax.set_xlim(-2.1, 0)
    ax.set_ylim(-3.6, 0)
    ax.set_xlabel("log(P1)", fontsize=12)
    ax.set_ylabel("log(P2)", fontsize=12)
    ax.tick_params(axis="both", direction="in")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Entanglement witness: measured (P1, P2) vs.\nindependent-emitter Monte Carlo bounds")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
