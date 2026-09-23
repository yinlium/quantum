"""Interaction-induced dephasing of g2(0) for a box-shaped Rydberg qubit cloud.

Reproduces the coherence-decay analysis of Y. Mei, Y. Li, H. Nguyen,
P. R. Berman, A. Kuzmich, "Trapped Alkali-Metal Rydberg Qubit," Phys. Rev.
Lett. 128, 123601 (2022). N atoms occupy a Gaussian-random cloud of size
Lx x Ly x Lz (box-like trap, as opposed to the spherical cloud used in the
companion PRA 108, 043713 paper -- see interference_bunching/dephasing_g2.py).
Atoms excited to a Rydberg nS state interact pairwise via the near-resonant
dipole-dipole (van der Waals-like) C3/R^3 coupling to the (n-1)P_3/2, nP_3/2
Forster channel; the random pairwise phases accumulated over a storage time
Ts dephase the collective spin-wave correlator and hence the second-order
coherence g2(0)(Ts), which decays from close to 1 (fully coherent, still
super-radiant) toward 0 (fully dephased) as Ts grows and/or the principal
quantum number n (interaction strength) or cloud density increases.
Dephase3()/AtomArray add a hard-sphere Rydberg-blockade exclusion radius,
showing how removing nearby atom pairs suppresses the fast dephasing channel.

Rydberg nS <-> nP3/2, (n-1)P3/2 energies and radial matrix elements use the
same quantum-defect / Kaulakys-formula machinery as
interference_bunching/dephasing_g2.py.

The measured (Ts, g2, g2 error) points below are hardcoded, exactly as in the
original notebook -- no digitized CSV was available in the source material,
and the notebook itself left the two data sets uncaptioned beyond a
color-pairing with one of the two model curves (n=50, n=40).

Source: Dephasing.ipynb (the more complete of two near-duplicate notebooks;
Dephasing_2.ipynb was a strict subset of it with a broken, unused curve_fit
cell and was not ported).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

_HCONST = 6.62606896e-34  # Planck constant
_CLIGHT = 2.99792458e8  # speed of light
_RY = 1.0973731568527e7  # Rydberg constant
_A0 = 0.52917720859e-10  # Bohr radius
_ALPHA = 7.2973525376e-3  # fine-structure constant


@jit(nopython=True, nogil=True)
def rydberg_energy(n, l, j):
    """Energy of an alkali Rydberg level via quantum defects (S, P1/2, P3/2, D3/2, D5/2, F5/2, F7/2)."""
    if l == 0:
        delta, delta2 = 3.1311804, 0.1784
    elif l == 1 and j == 0.5:
        delta, delta2 = 2.6548849, 0.2900
    elif l == 1 and j == 1.5:
        delta, delta2 = 2.6416737, 0.2950
    elif l == 2 and j == 1.5:
        delta, delta2 = 1.34809171, -0.60286
    elif l == 2 and j == 2.5:
        delta, delta2 = 1.34646572, -0.59600
    elif l == 3 and j == 2.5:
        delta, delta2 = 0.0165192, -0.085
    else:  # l == 3, j == 3.5
        delta, delta2 = 0.0165437, -0.086
    nu = n - delta - delta2 / (n - delta) ** 2
    return -(_HCONST * _CLIGHT * _RY) / nu**2


@jit(nopython=True, nogil=True)
def radial_matrix_element(n, l, j, na, la, ja):
    """<na la ja| r |n l j> via the Kaulakys (1995) quasiclassical formula."""
    Z = 1
    if l == 0:
        delta, delta2 = 3.1311804, 0.1784
    elif l == 1 and j == 0.5:
        delta, delta2 = 2.6548849, 0.2900
    else:  # l == 1, j == 1.5
        delta, delta2 = 2.6416737, 0.2950
    nu = n - delta - delta2 / (n - delta) ** 2

    if la == 0:
        deltaa, delta2a = 3.1311804, 0.1784
    elif la == 1 and ja == 0.5:
        deltaa, delta2a = 2.6548849, 0.2900
    else:  # la == 1, ja == 1.5
        deltaa, delta2a = 2.6416737, 0.2950
    nua = na - deltaa - delta2a / (na - deltaa) ** 2

    s_a = nua - nu
    nu_c_a = (2 * (nu * nua) ** 2 / (nu + nua)) ** (1 / 3)
    e_a = np.sqrt(1 - ((l + la + 1) / (2 * nu_c_a)) ** 2)

    csi = np.arange(0, np.pi, 1e-3)
    j_a = (1 / np.pi) * np.sum(np.cos(s_a * csi + s_a * e_a * np.sin(csi))) * 1e-3
    dj_a = (1 / np.pi) * np.sum(-np.sin(s_a * csi + s_a * e_a * np.sin(csi)) * np.sin(csi)) * 1e-3

    if (la - l) == 1:
        D_pa = (1 / s_a) * (dj_a + np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))
    else:
        D_pa = (1 / s_a) * (dj_a - np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))

    return ((-1) ** (n - na)) * nu_c_a**5 / (Z * (nu * nua) ** (3 / 2)) * D_pa


def _c3_and_defect(n):
    """Forster defect (MHz) and C3 (MHz um^3) for the nS <-> nP3/2 + (n-1)P3/2 channel."""
    l, j = 0, 0.5
    n1, l1, j1 = n, 1, 1.5
    n2, l2, j2 = n - 1, 1, 1.5
    defect = (
        rydberg_energy(n1, l1, j1) + rydberg_energy(n2, l2, j2) - 2 * rydberg_energy(n, l, j)
    ) / _HCONST * 1e-6
    C3 = (
        1e18
        * 1e-6
        * (_ALPHA * _CLIGHT / (2 * np.pi) * _A0**2)
        * radial_matrix_element(n, l, j, n1, l1, j1)
        * radial_matrix_element(n, l, j, n2, l2, j2)
    )
    return C3, defect


def dephase(n_atoms, lx, ly, lz, n, time, rng):
    """g2(0) vs. storage time for N atoms in a Gaussian wx=Lx/2 (etc.) cloud.

    Returns (time, g2, g2_num, g2_denom, efficiency, x, y, z).
    """
    wx, wy, wz = lx / 2, ly / 2, lz / 2
    x = rng.normal(0, wx, n_atoms)
    y = rng.normal(0, wy, n_atoms)
    z = rng.normal(0, wz, n_atoms)

    C3, defect = _c3_and_defect(n)

    dist_array = np.zeros((n_atoms, n_atoms - 1))
    dist3D = []
    for cc in range(n_atoms):
        xdiff = x - x[cc]
        xdiff = xdiff[xdiff != 0]
        ydiff = y - y[cc]
        ydiff = ydiff[ydiff != 0]
        zdiff = z - z[cc]
        zdiff = zdiff[zdiff != 0]
        d = np.sqrt(xdiff**2 + ydiff**2 + zdiff**2)
        dist3D.append(d)
        dist_array[cc, :] = d
    dist3D = np.array(dist3D).flatten()

    detuning = np.abs(defect / 2) - np.sqrt((defect / 2) ** 2 + (C3 / dist3D**3) ** 2)

    g2_num = np.zeros(len(time))
    g2_denom = np.zeros(len(time))
    for tt in range(len(time)):
        phase_denom = 0.0
        for cc in range(n_atoms):
            detuning_cc = np.abs(defect / 2) - np.sqrt((defect / 2) ** 2 + (C3 / dist_array[cc, :] ** 3) ** 2)
            phase_denom += (1 / n_atoms**3) * np.abs(np.sum(np.exp(2j * np.pi * detuning_cc * time[tt]))) ** 2
        g2_denom[tt] = (1 + phase_denom) ** 2
        g2_num[tt] = np.abs((1 / (n_atoms * (n_atoms - 1))) * np.sum(np.exp(2j * np.pi * detuning * time[tt]))) ** 2

    g2 = 4 * g2_num / g2_denom
    efficiency = np.sqrt(g2_denom) / np.exp(1)
    return time, g2, g2_num, g2_denom, efficiency, x, y, z


class _Cloud:
    """Gaussian-cloud atom positions with an optional hard-sphere blockade exclusion."""

    def __init__(self, wx, wy, wz, n_atoms, blockade_dist, rng):
        self.blockade_dist = blockade_dist
        self.pos = rng.normal(0, [wx, wy, wz], size=(n_atoms, 3))
        self._update_dist()
        if blockade_dist > 0:
            self._apply_blockade()

    def _update_dist(self):
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        d = np.sqrt(np.sum(diff**2, axis=-1))
        self.dist_array = d[~np.eye(len(self.pos), dtype=bool)].reshape(len(self.pos), -1)

    def _apply_blockade(self):
        while True:
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            d = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] >= self.blockade_dist:
                break
            self.pos = np.delete(self.pos, j, axis=0)
        self._update_dist()


def dephase_with_blockade(n_atoms, lx, ly, lz, n, time, blockade_dist, rng):
    """Like dephase(), but excludes atom pairs closer than blockade_dist (Rydberg blockade)."""
    wx, wy, wz = lx / 2, ly / 2, lz / 2
    C3, defect = _c3_and_defect(n)

    cloud = _Cloud(wx, wy, wz, n_atoms, blockade_dist, rng)
    n_atoms = len(cloud.pos)
    dist_array = cloud.dist_array
    dist3D = dist_array.flatten()

    detuning = np.abs(defect / 2) - np.sqrt((defect / 2) ** 2 + (C3 / dist3D**3) ** 2)

    g2_num = np.zeros(len(time))
    g2_denom = np.zeros(len(time))
    for tt in range(len(time)):
        phase_denom = 0.0
        for cc in range(n_atoms):
            detuning_cc = np.abs(defect / 2) - np.sqrt((defect / 2) ** 2 + (C3 / dist_array[cc, :] ** 3) ** 2)
            phase_denom += (1 / n_atoms**3) * np.abs(np.sum(np.exp(2j * np.pi * detuning_cc * time[tt]))) ** 2
        g2_denom[tt] = (1 + phase_denom) ** 2
        g2_num[tt] = np.abs((1 / (n_atoms * (n_atoms - 1))) * np.sum(np.exp(2j * np.pi * detuning * time[tt]))) ** 2

    g2 = 4 * g2_num / g2_denom
    return time, g2, n_atoms


def main():
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) g2(Ts) vs. principal quantum number n, compared to two measured (hardcoded) datasets
    a = dephase(750, 5, 5, 10, 50, np.linspace(0.1, 10, 60), rng)
    b = dephase(750, 5, 5, 10, 40, np.linspace(0.1, 10, 60), rng)
    data_x = [0.1, 0.3, 0.5, 1, 3, 5, 10]
    data_y = [0.6, 0.42, 0.38, 0.42, 0.24, 0.27, 0]
    data_y_err = [0.15, 0.17, 0.13, 0.14, 0.11, 0.10, 0.08]
    data_x_2 = [0.1, 1, 2, 3, 5, 8, 10, 12, 15]
    data_y_2 = [0.95, 0.75, 0.8, 1.04, 0.97, 0.79, 0.9, 1, 1.03]
    data_y_err_2 = [0.086, 0.116, 0.112, 0.162, 0.182, 0.173, 0.197, 0.267, 0.275]

    # NB: the original notebook left these two hardcoded scatter data sets
    # uncaptioned; only their color-pairing with a given model curve (n=50 or
    # n=40) is preserved here, not a confirmed atom-count/dataset label.
    axes[0].plot(a[0], a[1] * 0.6, linewidth=3, alpha=0.75, color="tab:blue", label="model, n=50")
    axes[0].plot(b[0], b[1], linewidth=3, alpha=0.75, color="tab:red", label="model, n=40")
    axes[0].errorbar(data_x, data_y, yerr=data_y_err, fmt="o", color="tab:blue", label="data set A")
    axes[0].errorbar(data_x_2, data_y_2, yerr=data_y_err_2, fmt="o", color="tab:red", label="data set B")
    axes[0].set_xlim(0, 10.1)
    axes[0].set_ylim(-0.10, 1.1)
    axes[0].set_ylabel(r"$g^{(2)}(0)$")
    axes[0].set_xlabel(r"$T_s$ ($\mu$s)")
    axes[0].set_title("dephasing vs. storage time")
    axes[0].legend(fontsize=7)

    # (b) effect of a Rydberg-blockade exclusion radius on the dephasing rate
    time = np.linspace(0.1, 25, 25)
    no_blockade = dephase(750, 6 * np.sqrt(2), 6 * np.sqrt(2), 5 * np.sqrt(2), 50, time, rng)
    blockade_1um = dephase_with_blockade(750, 6 * np.sqrt(2), 6 * np.sqrt(2), 5 * np.sqrt(2), 50, time, 1, rng)
    blockade_2um = dephase_with_blockade(750, 6 * np.sqrt(2), 6 * np.sqrt(2), 5 * np.sqrt(2), 50, time, 2, rng)
    blockade_4um = dephase_with_blockade(750, 6 * np.sqrt(2), 6 * np.sqrt(2), 5 * np.sqrt(2), 50, time, 4, rng)

    axes[1].plot(no_blockade[0], no_blockade[1], label=f"no blockade, N={750}")
    axes[1].plot(blockade_1um[0], blockade_1um[1], label=f"blockade 1um, N={blockade_1um[2]}")
    axes[1].plot(blockade_2um[0], blockade_2um[1], label=f"blockade 2um, N={blockade_2um[2]}")
    axes[1].plot(blockade_4um[0], blockade_4um[1], label=f"blockade 4um, N={blockade_4um[2]}")
    axes[1].set_ylabel(r"$g^{(2)}(0)$")
    axes[1].set_xlabel(r"$T_s$ ($\mu$s)")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("effect of Rydberg-blockade exclusion radius")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
