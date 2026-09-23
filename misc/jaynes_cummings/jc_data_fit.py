"""Fit collective Rabi dynamics of a Rydberg-coupled atom ensemble to fluorescence data.

Unpublished exploratory work (never turned into a paper). Same driven
three-level (Lambda) master-equation model as jc_theory.py -- two fields with
Rabi frequencies x1 (780 nm), x2 (480 nm) detuned by delta from an
intermediate state, collectively coupling N atoms into an effective two-level
Rabi oscillation -- but here the population rho33(t) is Poisson-averaged over
shot-to-shot atom number N and compared/fit against real time-resolved
fluorescence "loading" data (pulse width vs. detection efficiency) taken on
different days of the experiment.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy import stats
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).parent / "data"


def threelevel(t, rho, NN, x1, x2, delta, LW):
    """Master-equation RHS for populations/coherences of the Lambda system."""
    rho11, rho13, rho31, rho33 = rho

    x1 = x1 * 2 * np.pi
    x2 = x2 * 2 * np.pi
    delta = delta * 2 * np.pi
    int_decay = 6 * 2 * np.pi
    LW = LW * 2 * np.pi * 2.355

    decay_cross = complex(0, np.sqrt(NN) * x1 * x2 / (4 * delta))
    decay_33 = (x1**2) * int_decay / (delta**2)
    decay_13 = (x1**2 + x2**2) * int_decay / (4 * delta**2) + LW / 2

    drho11dt = decay_33 * rho33 + decay_cross * (rho13 - rho31)
    drho33dt = -decay_33 * rho33 - decay_cross * (rho13 - rho31)
    drho13dt = -decay_cross * (rho33 - rho11) - decay_13 * rho13
    drho31dt = decay_cross * (rho33 - rho11) - decay_13 * rho31

    return [drho11dt, drho13dt, drho31dt, drho33dt]


def twolevel_Navg(x, a, alpha, x1, x2, delta, LW, N):
    """rho33(t) Poisson-averaged over atom number N, with amplitude/decay envelope.

    The original notebook integrates the ODE out to t=50 regardless of the
    requested x range; here the integration horizon is capped at max(x) to
    avoid wasted work far past the data of interest (same dense-output
    trajectory over the range that matters, much cheaper for curve_fit).
    """
    t_end = max(float(np.max(x)), 1e-3)
    y_list = np.zeros(x.size)
    for j in range(1, int(N + np.sqrt(N) * 3), 1):
        seed = stats.poisson.pmf(j, N)
        sol = integrate.solve_ivp(
            threelevel,
            [0, t_end],
            [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)],
            args=(j, x1, x2, delta, LW),
            dense_output=True,
        )
        y_list += a * np.exp(-alpha * x) * seed * sol.sol(x)[3].real
    return y_list


def fitting_function(x, a, alpha, x1, x2, delta, LW, N):
    """Poisson-averaged rho33(t), scaled to raw (unnormalized) fluorescence counts."""
    return twolevel_Navg(x, a, alpha, x1, x2, delta, LW, N) * 0.002025 * 16000 * 20 + 0.00015 * 16000 * 20


def load_data():
    names = [
        "loading_300_1_6",
        "loading_300_2_3",
        "loading_50",
        "loading_100",
        "0309_verylowN",
        "0311_3",
        "0315",
    ]
    return {name: pd.read_csv(DATA_DIR / f"{name}.csv") for name in names}


def main():
    data = load_data()
    t = np.linspace(0, 20, 3000)

    # Two "loading_300" runs with different (780 nm, 480 nm) power ratios.
    z_2_3 = twolevel_Navg(t, 1, 0.15, 2.2, 3.3, 40, 0.04, 55)
    z_1_6 = twolevel_Navg(t, 1, 0.15, 1.1, 6.6, 40, 0.04, 55)
    # Lower atom-number ("loading_50"/"loading_100") runs.
    z_50 = twolevel_Navg(t, 1, 0.15, 2.6, 2.6, 40, 0.04, 5)
    z_100 = twolevel_Navg(t, 1, 0.15, 2.6, 2.6, 40, 0.04, 12)

    # Very-low-N run (0309_verylowN.csv).
    t_low = np.linspace(0, 3, 1000)
    z_verylowN = twolevel_Navg(t_low, 1, 0.05, 23.72 / 1.29 / 2, 17.27, 480, 0.04, 16)

    # N-scaling comparison against the 0315.csv trace: same theory curve rescaled
    # by trying a few candidate "loaded atom number" multipliers (1.3-2.0).
    t_scan = np.linspace(0, 3, 300)
    scan_curves = {}
    for mult in (1.3, 1.5, 1.7, 2.0):
        scan_curves[mult] = twolevel_Navg(
            t_scan,
            1,
            0.15,
            23.72 / 1.29 / mult,
            18.8 * 1.9 / 2.5,
            480,
            0.04,
            45 * mult**2,
        )

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    d = data["loading_300_2_3"]
    ax.plot(d["pulse width"] / 1000, d["eff"] / d["eff"].max() * 0.85, "o", color="tab:blue", label="(780,480)=(2,3)")
    ax.plot(t, z_2_3, color="tab:blue")
    d = data["loading_300_1_6"]
    ax.plot(d["pulse width"] / 1000, d["eff"] / d["eff"].max() * 0.85, "o", color="tab:red", label="(780,480)=(1,6)")
    ax.plot(t, z_1_6, color="tab:red")
    ax.set_xlim(0, 4)
    ax.set_xlabel("pulse width")
    ax.set_ylabel(r"$\rho_{33}$ (normalized)")
    ax.legend()
    ax.set_title("loading_300: power-ratio comparison")

    ax = axes[0, 1]
    d = data["loading_50"]
    ax.plot(d["pulse width"] / 1000, d["eff"] / d["eff"].max() * 0.85 - 0.3, "o", color="tab:blue", label="N~5")
    ax.plot(t, z_50, color="tab:blue")
    ax.set_xlim(0, 6.1)
    ax.set_xlabel("pulse width")
    ax.legend()
    ax.set_title("loading_50")

    ax = axes[0, 2]
    d = data["loading_100"]
    ax.plot(d["pulse width"] / 1000, d["eff"] / d["eff"].max() * 0.7, "o", color="tab:blue", label="N~12")
    ax.plot(t, z_100, color="tab:blue")
    ax.set_xlim(0, 6.1)
    ax.set_xlabel("pulse width")
    ax.legend()
    ax.set_title("loading_100")

    ax = axes[1, 0]
    d = data["0309_verylowN"]
    ax.plot(d["pulse width"] / 1000, d["eff"] / d["eff"].max(), "o", label="data")
    ax.plot(t_low, z_verylowN, label="theory")
    ax.set_xlabel("pulse width")
    ax.legend()
    ax.set_title("0309_verylowN")

    ax = axes[1, 1]
    d = data["0315"]
    for mult, curve in scan_curves.items():
        ax.plot(t_scan, 1.2 * curve, label=f"N mult={mult}")
    ax.plot(d["pulse width"] / 1000 - 0.02, d["eff"] / d["eff"].max(), "o", color="black", label="data (0315)")
    ax.set_xlabel("pulse width")
    ax.legend(fontsize=8)
    ax.set_title("0315: atom-number scan")

    # Fit the Poisson-averaged model to the 0311_3 raw fluorescence counts
    # (the "counts" column, not the pulse-normalized "eff" column, since
    # fitting_function's scale factors are calibrated to raw counts).
    ax = axes[1, 2]
    d = data["0311_3"].dropna(subset=["counts"])
    xdata = (d["pulse width"] / 1000).to_numpy()
    ydata = d["counts"].to_numpy()
    p0 = [0.3, 0.15, 23.72 / 1.29 / 1.7, 18.8 * 1.9 / 2.5, 480, 0.04, 45 * 1.7**2]
    bounds = (
        [0.01, 0, 1, 1, 300, 0, 20],
        [5, 1, 50, 50, 700, 0.5, 200],
    )
    try:
        popt, _ = curve_fit(fitting_function, xdata, ydata, p0=p0, bounds=bounds, maxfev=150)
        fit_label = f"fit: N={popt[6]:.1f}, delta={popt[4]:.0f}"
        t_fit = np.linspace(0, xdata.max(), 300)
        ax.plot(t_fit, fitting_function(t_fit, *popt), color="tab:red", label=fit_label)
        print("curve_fit result for 0311_3.csv:", popt)
    except RuntimeError as exc:
        print("curve_fit did not converge:", exc)
    ax.plot(xdata, ydata, "o", color="black", label="data (0311_3)")
    ax.set_xlabel("pulse width")
    ax.set_ylabel("counts")
    ax.legend(fontsize=8)
    ax.set_title("0311_3: curve_fit to raw counts")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)
    print(f"Saved figure to {Path(__file__).with_suffix('.png')}")


if __name__ == "__main__":
    main()
