"""Theory-only model of collective Rabi dynamics in a driven three-level system.

Unpublished exploratory work (never turned into a paper). Models a Lambda-type
three-level atom (ground |1>, intermediate, and a collectively-coupled state
|3>) driven by two fields with Rabi frequencies x1, x2 detuned by delta from
the intermediate state, adiabatically eliminated into an effective two-level
Lindblad master equation for rho. The sqrt(N) enhancement of the effective
two-photon coupling models N atoms coupled collectively (e.g. to a shared
Rydberg state or cavity mode). Poisson-averaging over N accounts for shot
to shot atom-number fluctuations in a loaded ensemble.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import stats


def threelevel(t, rho, NN, x1, x2, delta, LW):
    """Master-equation RHS for populations/coherences of the Lambda system."""
    rho11, rho13, rho31, rho33 = rho

    x1 = x1 * 2 * np.pi / 2
    x2 = x2 * 2 * np.pi / 2
    delta = delta * 2 * np.pi
    int_decay = 6 * 2 * np.pi
    LW = LW * 2 * np.pi * 2.355

    decay_cross = complex(0, np.sqrt(NN) * x1 * x2 / delta)
    decay_33 = (x2**2) * int_decay / (delta**2)
    decay_13 = (x1**2 + x2**2) * int_decay / (4 * delta**2) + LW / 2

    drho11dt = decay_33 * rho33 + decay_cross * (rho13 - rho31)
    drho33dt = -decay_33 * rho33 - decay_cross * (rho13 - rho31)
    drho13dt = -decay_cross * (rho33 - rho11) - decay_13 * rho13
    drho31dt = decay_cross * (rho33 - rho11) - decay_13 * rho31

    return [drho11dt, drho13dt, drho31dt, drho33dt]


def twolevel_Navg(x, a, alpha, x1, x2, delta, LW, N):
    """rho33(t) Poisson-averaged over atom number N, with amplitude/decay envelope.

    Integration horizon is capped at max(x) rather than a fixed t=50, which
    gives the same dense-output trajectory over the range of interest much
    more cheaply.
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


def main():
    t_2 = np.linspace(0, 10, 1000)
    z2_theory = twolevel_Navg(t_2, 1, 0.05, 100, 14, 480, 0.04, 10)

    fig = plt.figure(figsize=(12, 6))
    plt.plot(t_2, z2_theory)
    plt.xlabel("t")
    plt.ylabel(r"$\rho_{33}$")
    plt.title(
        r"$\Omega_1 = 2\pi\times 50\,\mathrm{MHz},\ "
        r"\Omega_2 = 2\pi\times 25\,\mathrm{MHz},\ "
        r"\Delta = 2\pi\times 480\,\mathrm{MHz},\ N = 10,\ "
        r"\alpha = 0.05,\ LW = 40\,\mathrm{kHz}$"
    )
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)
    print(f"Saved figure to {Path(__file__).with_suffix('.png')}")


if __name__ == "__main__":
    main()
