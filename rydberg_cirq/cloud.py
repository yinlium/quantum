"""
Atomic cloud sampling and interaction-phase matrices.

Reproduces the experimental geometry of Phys. Rev. A 106, L051701:
``N ~ 270`` atoms of 87-Rb in a 1D state-insensitive lattice trap, excited within a
Gaussian volume set by the 780 nm and 480 nm beam waists.

The key output is the **pairwise phase matrix**

    Phi_{mu nu} = kappa_{mu nu} * T_s

which drives :class:`rydberg_cirq.gates.PairwisePhaseGate`.  In the van der Waals
asymptotic regime ``kappa = C6 / R^6``; near a Foerster resonance the full
two-level expression with the energy defect ``delta`` is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CloudParameters", "SHORT_CLOUD", "LONG_CLOUD", "sample_cloud", "phase_matrix", "coherence_eta"]


@dataclass(frozen=True)
class CloudParameters:
    """Gaussian atomic cloud geometry and interaction strength.

    Attributes:
        n_atoms: number of atoms in the excitation volume.
        sigma_x, sigma_y, sigma_z: Gaussian standard deviations in micrometres.
            These are *radii*, i.e. half the quoted diameters.
        c6: van der Waals coefficient in ``GHz * um^6`` (times ``2 pi`` for angular
            frequency; see :func:`phase_matrix`).
        m_bar: mean number of excitations produced by the excitation pulse.
        label: human-readable name.
    """

    n_atoms: int = 270
    sigma_x: float = 2.925
    sigma_y: float = 2.925
    sigma_z: float = 5.25
    c6: float = 15.44
    m_bar: float = 0.79
    label: str = "short cloud (n=50)"

    @property
    def sigmas(self) -> np.ndarray:
        return np.array([self.sigma_x, self.sigma_y, self.sigma_z])


#: MOT -> FORT -> SILT loading, D_z = 10.5 um.  n = 50, C6/h = 15.44 GHz um^6.
SHORT_CLOUD = CloudParameters()

#: Direct MOT -> SILT loading, D_z ~ 230 um.  n = 40, C6/h = 1.00 GHz um^6.
LONG_CLOUD = CloudParameters(
    n_atoms=270,
    sigma_z=115.0,
    c6=1.00,
    m_bar=1.63,
    label="long cloud (n=40)",
)


def sample_cloud(params: CloudParameters, n: int | None = None, rng=None) -> np.ndarray:
    """Draw ``n`` atomic positions from the Gaussian excitation volume.

    Returns:
        An ``(n, 3)`` array of coordinates in micrometres.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = params.n_atoms if n is None else n
    return rng.normal(0.0, 1.0, size=(n, 3)) * params.sigmas[None, :]


def phase_matrix(
    positions: np.ndarray,
    storage_time: float,
    c6: float,
    r_min: float = 0.3,
) -> np.ndarray:
    """Pairwise interaction phases ``Phi_{mu nu} = (2 pi C6 / R^6) T_s``.

    Args:
        positions: ``(n, 3)`` atomic coordinates in micrometres.
        storage_time: ``T_s`` in microseconds.
        c6: van der Waals coefficient ``C6 / h`` in ``GHz * um^6``.
        r_min: separations below this (in um) are clipped, regularising the
            ``R^-6`` divergence for the rare pair of nearly coincident sampled atoms.

    Returns:
        A symmetric ``(n, n)`` array with a zero diagonal.  Units work out as
        ``GHz * us = 10^3 rad``, times ``2 pi`` for the angular frequency.
    """
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, np.inf)
    r = np.maximum(r, r_min)

    phi = 2.0 * np.pi * c6 * 1e3 * storage_time / r**6
    np.fill_diagonal(phi, 0.0)
    return phi


def coherence_eta(phi: np.ndarray) -> float:
    """Elementary two-atom phase-coherence factor ``eta = |<exp(-i dPhi)>|``.

    This is the single number that controls the scaling ansatz
    ``X_m ~ eta^(2m-3)``, ``Y_m ~ eta^(m-1)`` of the supplemental material.
    """
    n = phi.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.abs(np.mean(np.exp(-1j * phi[iu]))))
