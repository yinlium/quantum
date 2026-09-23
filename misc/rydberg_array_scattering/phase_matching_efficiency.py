"""Coherent forward-scattering phase-matching efficiency for a Rydberg atom array.

Unpublished exploratory work (never turned into a paper). Models spontaneous
four-wave-mixing retrieval from an ensemble/array of atoms driven by two
excitation fields (780 nm, 480 nm) and read out with a retrieval field pair,
following the phase-matching formalism of "Phase Matching in Lower
Dimensions" [PRL 125, 163601 (2020)]. Computes the coherent structure factor
(interference of scattered amplitudes from all atoms) combined with a
Debye-Waller disorder factor (thermal/positional spread), and integrates the
resulting angular intensity distribution to get the fraction of light
scattered into a forward collection cone -- the phase-matching efficiency --
as a function of atom number N.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

um = 1e-6
nm = 1e-9

# Wavenumbers of the two excitation fields and the two retrieval fields.
k1 = (2 * np.pi) / (780 * nm)
k2 = (2 * np.pi) / (480 * nm)
k3 = (2 * np.pi) / (480 * nm)
k4 = (2 * np.pi) / (780 * nm)


@jit(nopython=True, nogil=True)
def vec(theta, phi):
    """Unit vector for spherical angles (theta, phi)."""
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


@jit(nopython=True, nogil=True)
def ensemble(natoms, sigmax, sigmay, sigmaz):
    """Gaussian-distributed atom positions (3 x natoms), one fixed random seed."""
    np.random.seed(0)
    atoms = np.zeros((np.int64(3), natoms))
    atoms[0, :] = np.random.normal(0, sigmax, natoms)
    atoms[1, :] = np.random.normal(0, sigmay, natoms)
    atoms[2, :] = np.random.normal(0, sigmaz, natoms)
    return atoms


@jit(nopython=True, nogil=True)
def ktotal(t1, p1, t2, p2, t3, p3, t4, p4):
    """Total (phase-mismatch) wavevector -k1 - k2 + k3 + k4, for a grid of
    retrieval-field angles (t4, p4)."""
    k1vec = k1 * vec(t1, p1)
    k2vec = k2 * vec(t2, p2)
    k3vec = k3 * vec(t3, p3)
    k4vec = np.zeros((len(t4), len(p4), 3))
    for i in range(len(t4)):
        for j in range(len(p4)):
            k4vec[i, j, :] = k4 * vec(t4[i], p4[j])
    return -k1vec - k2vec + k3vec + k4vec


@jit(nopython=True, nogil=True)
def phase_list(atomposlist, t1, p1, t2, p2, t3, p3, t4, p4):
    """Per-atom phase k_total . r_atom, for each (theta4, phi4) direction."""
    ktot = ktotal(t1, p1, t2, p2, t3, p3, t4, p4)
    atomlist = atomposlist.T
    phaselist = np.zeros((len(t4), len(p4), atomlist.shape[0]))
    for i in range(atomlist.shape[0]):
        for j in range(ktot.shape[0]):
            for k in range(ktot.shape[1]):
                phaselist[j, k, i] = np.dot(atomlist[i, :], ktot[j, k, :])
    return phaselist


@jit(nopython=True, nogil=True)
def intensity(atomposlist, t4, p4):
    """Coherent structure factor |sum_atoms exp(i * phase)|^2 (no disorder average)."""
    phase = phase_list(atomposlist, 0, 0, np.pi, np.pi, np.pi, np.pi, t4, p4)
    return np.abs(np.sum(np.exp(1j * phase), axis=2)) ** 2


@jit(nopython=True, nogil=True)
def debye_waller_factor(sigmax, sigmay, sigmaz, t4, p4):
    """Debye-Waller disorder factor from Gaussian positional spread sigma_i."""
    ktot = ktotal(0, 0, np.pi, np.pi, np.pi, np.pi, t4, p4)
    dbf1 = np.exp(-(ktot[:, :, 0] ** 2) * sigmax**2)
    dbf2 = np.exp(-(ktot[:, :, 1] ** 2) * sigmay**2)
    dbf3 = np.exp(-(ktot[:, :, 2] ** 2) * sigmaz**2)
    return dbf1 * dbf2 * dbf3


@jit(nopython=True, nogil=True)
def avg_intensity(atomposlist, sigmax, sigmay, sigmaz, t4, p4):
    """Disorder-averaged intensity: coherent term (weighted by DBF) plus the
    incoherent background N*(1 - DBF)."""
    intens = intensity(atomposlist, t4, p4)
    dbf = debye_waller_factor(sigmax, sigmay, sigmaz, t4, p4)
    return intens * dbf + atomposlist.shape[1] * (1 - dbf)


def phase_matching_efficiency(N, n_theta=1000, n_phi=6):
    """Fraction of scattered light collected within a forward cone of
    ~1.6 degrees half-angle (matching the original notebook's cutoff of
    theta index 36 out of a 4000-point grid spanning [0, pi]), disorder
    averaged over the ensemble's thermal spread.
    """
    theta_cutoff_idx = max(1, round(36 / 4000 * n_theta))
    atoms = ensemble(N, 5.85 / 2 * um, 5.85 / 2 * um, 10.5 / 2 * um)
    sigmax, sigmay, sigmaz = 0.3 * 780 * nm, 0.3 * 780 * nm, 2.4 * 780 * nm
    tdata = np.linspace(0, np.pi, n_theta)
    pdata = np.linspace(0, 2 * np.pi, n_phi)
    idata = avg_intensity(atoms, sigmax, sigmay, sigmaz, tdata, pdata)
    idata_sin = idata * np.sin(tdata)[:, None]
    return np.sum(idata_sin[0:theta_cutoff_idx, :]) / np.sum(idata_sin)


def main():
    # A single-ensemble angular intensity map, for illustration.
    N_demo = 100
    atoms_demo = ensemble(N_demo, 5.85 * um, 5.85 * um, 10.5 / 2 * um)
    t4, p4 = np.linspace(0, np.pi / 30, 200), np.linspace(0, 2 * np.pi, 200)
    intensity_map = intensity(atoms_demo, t4, p4)

    # Phase-matching efficiency vs. atom number.
    N_list = np.arange(100, 1000, 50)
    eff_list = np.array([phase_matching_efficiency(int(N)) for N in N_list])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    thetas_deg = t4 * 180 / np.pi
    pc = ax.pcolormesh(p4 * 180 / np.pi, thetas_deg, intensity_map, shading="nearest")
    fig.colorbar(pc, ax=ax, label="structure factor")
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("theta (deg)")
    ax.set_title(f"Forward-scattering angular intensity, N={N_demo}")

    ax = axes[1]
    ax.plot(N_list, eff_list, "o-")
    ax.set_xlabel("N atoms")
    ax.set_ylabel("phase-matching efficiency")
    ax.set_title("Efficiency vs. atom number")

    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=150)
    print(f"Saved figure to {Path(__file__).with_suffix('.png')}")


if __name__ == "__main__":
    main()
