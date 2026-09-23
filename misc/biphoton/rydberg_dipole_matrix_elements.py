"""Rydberg radial dipole matrix elements for a two-atom biphoton-generation scheme.

Unpublished exploratory work (never turned into a paper), originally supporting
a proposed biphoton-generation scheme using pairs of Rydberg atoms coupled by
resonant dipole-dipole exchange. Implements, fully from scratch (no external
Rydberg-atom package):
  - quantum-defect energies of Rb nS/nP/nD/nF Rydberg levels,
  - the semiclassical Kaulakys (J. Phys. B 28, 4963 (1995)) radial dipole
    matrix element <n l j| r |n' l' j'>,
  - the van der Waals C3 dispersion coefficient and blockade radius for a
    chosen ns -> n'p, np pair-state transition, and
  - the angle- and distance-dependent dipole-dipole exchange splitting
    Delta(eta, k0 r) between the near-resonant pair states, which sets the
    photon-mediated coupling geometry used by the biphoton scheme.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Physical constants.
A0 = 0.52917720859e-10  # Bohr radius (m)
ALPHA = 7.2973525376e-3  # fine-structure constant
HCONST = 6.62606896e-34  # Planck constant (J s)
CLIGHT = 2.99792458e8  # speed of light (m/s)
RYDBERG_CONST = 1.0973731568527e7  # Rydberg constant (1/m)

# Quantum defects for Rb, from Li et al. PRA 67, 052502 (2003) and
# Han et al. PRA 74, 054502 (2006): {(l, j): (delta0, delta2)}.
QUANTUM_DEFECTS = {
    (0, 0.5): (3.1311804, 0.1784),  # nS_1/2
    (1, 0.5): (2.6548849, 0.2900),  # nP_1/2
    (1, 1.5): (2.6416737, 0.2950),  # nP_3/2
    (2, 1.5): (1.34809171, -0.60286),  # nD_3/2
    (2, 2.5): (1.34646572, -0.59600),  # nD_5/2
    (3, 2.5): (0.0165192, -0.085),  # nF_5/2
    (3, 3.5): (0.0165437, -0.086),  # nF_7/2
}


def effective_n(n, l, j):
    """Effective principal quantum number nu = n - delta(n) for Rb."""
    delta0, delta2 = QUANTUM_DEFECTS[(l, j)]
    return n - delta0 - delta2 / (n - delta0) ** 2


def rydberg_energy(n, l, j):
    """Rydberg-level energy (J) including fine structure, via quantum defects."""
    nu = effective_n(n, l, j)
    return -(HCONST * CLIGHT * RYDBERG_CONST) / nu**2


def radial_matrix_element(n, l, j, na, la, ja):
    """Semiclassical (Kaulakys 1995) radial matrix element <n l j| r |na la ja>.

    Returns <n l j| r |na la ja> in units of the Bohr radius a0 (dimensionless
    prefactor); Z=1 valence electron seen by the Rydberg core.
    """
    Z = 1
    nu = effective_n(n, l, j)
    nua = effective_n(na, la, ja)

    s_a = nua - nu
    nu_c_a = (2 * (nu * nua) ** 2 / (nu + nua)) ** (1 / 3)
    e_a = np.sqrt(1 - ((l + la + 1) / (2 * nu_c_a)) ** 2)  # eccentricity

    # Anger-function-type integrals, evaluated by direct quadrature.
    csi = np.arange(0, np.pi, 1e-3)
    j_a = (1 / np.pi) * np.sum(np.cos(s_a * csi + s_a * e_a * np.sin(csi))) * 1e-3
    dj_a = (1 / np.pi) * np.sum(-np.sin(s_a * csi + s_a * e_a * np.sin(csi)) * np.sin(csi)) * 1e-3

    if (la - l) == 1:
        d_pa = (1 / s_a) * (dj_a + np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))
    elif (la - l) == -1:
        d_pa = (1 / s_a) * (dj_a - np.sqrt(e_a ** (-2) - 1) * (j_a - np.sin(np.pi * s_a) / (np.pi * s_a)))
    else:
        raise ValueError("radial_matrix_element requires |la - l| == 1 (dipole transition)")

    return ((-1) ** (n - na)) * nu_c_a**5 / (Z * (nu * nua) ** 1.5) * d_pa


def dipole_dipole_splitting(eta, k0r):
    """Angle- and distance-dependent resonant dipole-dipole exchange splitting.

    eta is the angle between the quantization axis and the interatomic axis;
    k0r is the dimensionless product of the resonant photon wavevector and the
    interatomic separation. Returns Delta / (2 pi), in the same frequency
    units as the "6" MHz-scale prefactor below (natural units of the
    transition's radiative decay rate).
    """
    return (
        3
        / 4
        * 6
        * (
            -(np.sin(eta) ** 2) * np.cos(k0r) / k0r
            + (1 - 3 * np.cos(eta) ** 2) * (np.sin(k0r) / k0r**2 + np.cos(k0r) / k0r**3)
        )
    )


def van_der_waals_c3(n, l, j, n1, l1, j1, n2, l2, j2, rf_period_us=1.0):
    """C3 dispersion coefficient (MHz um^3) and blockade radius (um) for
    the pair-state resonance |n l j> + |n l j> <-> |n1 l1 j1> + |n2 l2 j2>.
    """
    defect_mhz = (rydberg_energy(n1, l1, j1) + rydberg_energy(n2, l2, j2) - 2 * rydberg_energy(n, l, j)) / HCONST * 1e-6
    c3 = (
        1e18
        * 1e-6
        * (ALPHA * CLIGHT / (2 * np.pi) * A0**2)
        * radial_matrix_element(n, l, j, n1, l1, j1)
        * radial_matrix_element(n, l, j, n2, l2, j2)
    )
    omega = 1 / rf_period_us
    rb = (-c3 / np.sqrt((omega + np.abs(defect_mhz / 2)) ** 2 - (defect_mhz / 2) ** 2)) ** (1 / 3)
    return c3, rb, defect_mhz


def main():
    # Example pair-state transition: 20S_1/2 -> {20P_3/2, 19P_3/2}, a Foerster
    # resonance often used for Rydberg-Rydberg exchange coupling.
    n, l, j = 20, 0, 0.5
    n1, l1, j1 = n, 1, 1.5
    n2, l2, j2 = n - 1, 1, 1.5
    c3, rb, defect_mhz = van_der_waals_c3(n, l, j, n1, l1, j1, n2, l2, j2)
    print(f"{n}S_1/2 -> {n1}P_3/2 + {n2}P_3/2 Foerster defect: {defect_mhz:.3f} MHz")
    print(f"C3 = {c3:.3f} MHz um^3, blockade radius Rb = {rb:.3f} um")

    # Angle/distance dependence of the resonant dipole-dipole exchange
    # splitting, expressed in units of k0*r for a 780 nm photon.
    lambda_nm = 780
    k0r_list = np.linspace(2, 11, 91)
    eta_list = np.linspace(0, 2 * np.pi, 100)
    splitting_grid = np.array([dipole_dipole_splitting(eta, k0r_list) for eta in eta_list])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    r_um_list = k0r_list * lambda_nm * 1e-3 / (2 * np.pi)
    eta_deg_list = eta_list * 180 / np.pi
    X, Y = np.meshgrid(r_um_list, eta_deg_list)
    mesh = ax.pcolormesh(X, Y, splitting_grid, shading="nearest")
    fig.colorbar(mesh, ax=ax, label=r"$\Delta / 2\pi$")
    ax.set_xlabel("r (um)")
    ax.set_ylabel("eta (deg)")
    ax.set_title("Dipole-dipole exchange splitting")

    ax = axes[1]
    k0r_fine = np.linspace(0.1, 21, 1000)
    r_fine_um = k0r_fine * lambda_nm * 1e-3 / (2 * np.pi)
    delta_eta_pi2 = dipole_dipole_splitting(np.pi / 2, k0r_fine)
    ax.plot(r_fine_um, delta_eta_pi2 * 2, label="frequency splitting (eta=pi/2)")
    ax.set_xlim(0.04, 1.0)
    ax.set_ylim(-3, 10)
    ax.set_xlabel("r (um)")
    ax.set_ylabel(r"$\Delta / 2\pi$")
    ax.legend()
    ax.set_title("Splitting vs. separation")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)
    print(f"Saved figure to {Path(__file__).with_suffix('.png')}")


if __name__ == "__main__":
    main()
