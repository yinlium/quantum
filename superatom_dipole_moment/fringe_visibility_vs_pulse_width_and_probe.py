"""
Interference-fringe visibility analysis for

    B. Yang et al., "Dipole Moment of a Superatom", PRL 133, 213601 (2024),
    Figs. 3(a) and 4(a).

For each phase-scan data file, the photoelectric detection probability P(phi)
is fit to a sinusoid A*sin(phi + C) + B; the fringe visibility is V = A/B
(cf. Eq. 3a, V = 2*f*beta0/(1+f^2)) and its offset C is the measured phase
shift. Repeating the fit for data taken at different excitation pulse widths
Tp (which sets the single-excitation probability |beta1|^2 via the many-body
Rabi oscillation) reproduces Fig. 3(a); repeating it for data taken at
different probe-attenuator (VCA) settings, which sets the emission/probe
amplitude ratio f, reproduces Fig. 4(a).

Data: phase-scan CSVs (columns phase, A1, A3, total, eff, ...) recorded on
2023-11-01, copied from the lab's "Many Body Rabi" data folder into ./data.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent / "data"


def fringe(phi, A, B, C):
    """Sinusoidal model for the phase-dependent detection probability."""
    return A * np.sin(phi + C) + B


def fit_visibility(csv_path, bounds=((0, 0, -np.pi), (0.1, 0.1, np.pi))):
    """Fit fringe() to one phase-scan CSV; return (V, sigma_V, phase_deg, sigma_phase_deg, popt)."""
    df = pd.read_csv(csv_path)
    phase = df["phase"] * np.pi / 180
    eff = df["eff"]
    eff_err = (np.sqrt(df["A3"]) + np.sqrt(df["A1"])) / df["total"]

    popt, pcov = curve_fit(fringe, phase, eff, bounds=bounds, sigma=eff_err)
    A, B, C = popt
    sigma_A, sigma_B, sigma_C = np.sqrt(np.diag(pcov))
    vis = A / B
    sigma_vis = vis * np.sqrt((sigma_A / A) ** 2 + (sigma_B / B) ** 2)
    return vis, sigma_vis, C * 180 / np.pi, sigma_C * 180 / np.pi, popt, (phase, eff, eff_err)


def f_ratio_from_counts(atom_A3, atom_A1, probe_A3, probe_A1):
    """Emission/probe amplitude ratio f = sqrt(atom photons / probe photons),
    from separate atom-alone and probe-alone photon-count calibration runs
    on two detectors (A1, A3), with Poisson counting errors, as used to
    build Fig. 4(a)."""
    atom_A3, atom_A1 = np.asarray(atom_A3, dtype=float), np.asarray(atom_A1, dtype=float)
    probe_A3, probe_A1 = np.asarray(probe_A3, dtype=float), np.asarray(probe_A1, dtype=float)
    f = np.sqrt((atom_A3 + atom_A1) / (probe_A3 + probe_A1))
    sigma_f = 0.5 * f * np.sqrt(
        1 / (atom_A3 + atom_A1) ** 2 * (np.sqrt(atom_A3) + np.sqrt(atom_A1))
        + 1 / (probe_A3 + probe_A1) ** 2 * (np.sqrt(probe_A3) + np.sqrt(probe_A1))
    )
    return f, sigma_f


def main():
    # --- Fig. 3(a)-style: visibility vs excitation pulse width Tp ---------
    pw_files = {100: "pw100.csv", 120: "pw120.csv", 150: "pw150.csv",
                170: "pw170.csv", 200: "pw200.csv"}
    pw_vals, vis_vs_pw, sigma_vis_vs_pw = [], [], []
    for pw, fname in pw_files.items():
        vis, sigma_vis, *_ = fit_visibility(DATA_DIR / fname)
        pw_vals.append(pw)
        vis_vs_pw.append(vis)
        sigma_vis_vs_pw.append(sigma_vis)

    # --- Fig. 4(a)-style: visibility vs probe-attenuator (VCA) setting ----
    # VCA controls the probe intensity and hence the amplitude ratio f; we
    # don't have the separate atom-alone/probe-alone calibration counts for
    # these particular files (that calibration, f_ratio_from_counts() above,
    # was recorded on a different day in the notebook), so the x axis here
    # is the attenuator setting itself rather than a calibrated f.
    vca_files = {4.7: "vca4.7.csv", 5.4: "vca5.4.csv",
                 9.2: "vca9.2.csv", 12.0: "vca12.0.csv"}
    vca_vals, vis_vs_vca, sigma_vis_vs_vca = [], [], []
    for vca, fname in sorted(vca_files.items()):
        vis, sigma_vis, *_ = fit_visibility(DATA_DIR / fname, bounds=((0, 0, -0.5 * np.pi), (0.1, 0.1, 1.5 * np.pi)))
        vca_vals.append(vca)
        vis_vs_vca.append(vis)
        sigma_vis_vs_vca.append(sigma_vis)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(pw_vals, vis_vs_pw, yerr=sigma_vis_vs_pw, fmt="o")
    axes[0].set_xlabel("excitation pulse width (ns)")
    axes[0].set_ylabel("visibility V")
    axes[0].set_title("V vs pulse width (cf. Fig. 3a)")

    axes[1].errorbar(vca_vals, vis_vs_vca, yerr=sigma_vis_vs_vca, fmt="o", color="tab:orange")
    axes[1].set_xlabel("probe attenuator setting (V)")
    axes[1].set_ylabel("visibility V")
    axes[1].set_title("V vs probe attenuation (cf. Fig. 4a)")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)

    # Illustrative use of f_ratio_from_counts(), with the example atom/probe
    # calibration counts recorded in the original notebook (2023-10-04 VCA
    # scan) -- kept to preserve the f-ratio error-propagation formula.
    atom_A3 = np.array([187, 158, 189, 122, 130, 94, 154, 272])
    atom_A1 = np.array([178, 168, 162, 146, 135, 124, 219, 235])
    probe_A3 = np.array([14, 50, 80, 120, 161, 450, 410, 56])
    probe_A1 = np.array([12, 48, 67, 119, 175, 460, 408, 51])
    f, sigma_f = f_ratio_from_counts(atom_A3, atom_A1, probe_A3, probe_A1)
    print("Example f = emission/probe amplitude ratio (2023-10-04 calibration):")
    print("f =", np.round(f, 3))
    print("sigma_f =", np.round(sigma_f, 3))


if __name__ == "__main__":
    main()
