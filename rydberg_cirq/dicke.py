"""
Dicke (permutation-symmetric) basis utilities.

Convention used throughout this package
---------------------------------------
Each atom is a qubit with

    |0> == |g>   (ground state)
    |1> == |r>   (Rydberg state)

so the *number of excitations* ``m`` equals the Hamming weight of the computational
basis label.  The collective spin projection is

    M = m - N/2,        m = 0, 1, ..., N

and the symmetric Dicke state is

    |J = N/2, M>  =  binom(N, m)^(-1/2)  sum_{|s| = m} |s>.

With |0> = |g> the Pauli-Z eigenvalue of an unexcited atom is +1, hence

    Jz = -(1/2) sum_j Z_j          (so that Jz |g...g> = -(N/2) |g...g>)
    Jx = +(1/2) sum_j X_j
    Jy = -(1/2) sum_j Y_j

The sign of Jy is fixed by requiring the angular-momentum algebra [Jx, Jy] = i Jz;
this is verified numerically in ``tests/test_metrics.py``.

Reference
---------
Y. Li, Y. Mei, H. Nguyen, P. R. Berman, A. Kuzmich,
Phys. Rev. A 106, L051701 (2022).
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import gammaln

__all__ = [
    "get_dicke_operators",
    "hamming_weights",
    "dicke_projector",
    "dicke_state_vector",
    "w_state_vector",
    "excitation_masks",
    "coherent_spin_state",
    "poisson_excitation_amplitudes",
]


@functools.lru_cache(maxsize=64)
def _hamming_weights_cached(num_qubits: int) -> tuple:
    return tuple(int(s).bit_count() for s in range(2**num_qubits))


def hamming_weights(num_qubits: int) -> np.ndarray:
    """Excitation number ``m`` of every computational basis state of ``num_qubits``."""
    return np.asarray(_hamming_weights_cached(num_qubits), dtype=int)


def get_dicke_operators(N: int):
    """Collective spin operators in the symmetric (Dicke) basis.

    The basis is ``|J = N/2, M>`` with ``M in [-N/2, ..., N/2]`` and dimension
    ``D = N + 1``.  Basis index ``m = M + N/2 in {0, ..., N}`` counts excitations.

    Returns:
        ``(Jx, Jy, Jz, M_vals)`` where the operators are ``(N+1, N+1)`` complex
        arrays and ``M_vals`` holds the spin projections.
    """
    D = N + 1
    J = N / 2.0
    m_indices = np.arange(D)
    M_vals = m_indices - J

    # Jz is diagonal: Jz|m> = (m - J)|m>
    Jz = np.diag(M_vals).astype(complex)

    # J+|m> = sqrt((N - m) * (m + 1)) |m+1>
    J_plus = np.zeros((D, D), dtype=complex)
    for m in range(N):
        J_plus[m + 1, m] = np.sqrt((N - m) * (m + 1))

    J_minus = J_plus.T.conj()
    Jx = 0.5 * (J_plus + J_minus)
    Jy = -0.5j * (J_plus - J_minus)

    return Jx, Jy, Jz, M_vals


def excitation_masks(N: int) -> dict:
    """Boolean masks over the ``2**N`` computational basis selecting excitation sectors.

    Returns a dict with keys ``'m0'``, ``'m1'``, ``'multi'`` (m >= 2) and
    ``'by_m'`` (list indexed by ``m``).
    """
    w = hamming_weights(N)
    return {
        "m0": w == 0,
        "m1": w == 1,
        "multi": w >= 2,
        "by_m": [w == m for m in range(N + 1)],
    }


def dicke_state_vector(N: int, m: int) -> np.ndarray:
    """Symmetric Dicke state ``|J = N/2, M = m - N/2>`` as a ``2**N`` statevector."""
    if not 0 <= m <= N:
        raise ValueError(f"excitation number m={m} out of range [0, {N}]")
    w = hamming_weights(N)
    mask = w == m
    psi = np.zeros(2**N, dtype=complex)
    psi[mask] = 1.0 / np.sqrt(int(mask.sum()))
    return psi


def w_state_vector(N: int) -> np.ndarray:
    """Single-excitation symmetric superatom state ``|W>``."""
    return dicke_state_vector(N, 1)


def dicke_projector(N: int) -> np.ndarray:
    """Isometry ``P`` of shape ``(N+1, 2**N)`` projecting onto the symmetric subspace.

    Row ``m`` is the bra ``<J = N/2, M = m - N/2|``.  For a permutation-symmetric
    state this is norm preserving; for a general state it discards the
    non-symmetric components.
    """
    w = hamming_weights(N)
    P = np.zeros((N + 1, 2**N), dtype=complex)
    for m in range(N + 1):
        mask = w == m
        P[m, mask] = 1.0 / np.sqrt(int(mask.sum()))
    return P


def coherent_spin_state(N: int, theta: float = np.pi / 2.0) -> np.ndarray:
    """Coherent spin state in the Dicke basis, tipped by ``theta`` from ``-z``.

    ``theta = pi/2`` gives the equatorial ``|+x>^{\\otimes N}`` state.  Computed via
    log-gamma functions so it is numerically stable for large ``N``.
    """
    D = N + 1
    log_fact = gammaln(np.arange(D) + 1)
    log_binom = log_fact[N] - log_fact - log_fact[::-1]
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    with np.errstate(divide="ignore"):
        log_amp = (
            0.5 * log_binom
            + np.arange(D) * np.log(np.abs(s) + 1e-300)
            + (N - np.arange(D)) * np.log(np.abs(c) + 1e-300)
        )
    psi = np.exp(log_amp).astype(complex)
    return psi / np.linalg.norm(psi)


def poisson_excitation_amplitudes(N: int, m_bar: float, m_max: int | None = None) -> np.ndarray:
    """Binomial/Poissonian spin-wave amplitudes ``c_m`` after a short excitation pulse.

    Implements ``c_m = sqrt(binom(N, m)) a^(N-m) b^m`` with ``|b|^2 = m_bar / N``,
    i.e. Eq. (2) of Phys. Rev. A 106, L051701.  For ``N >> m_bar`` the resulting
    ``|c_m|^2`` is Poissonian with mean ``m_bar``.
    """
    m_max = N if m_max is None else min(m_max, N)
    b2 = m_bar / N
    if not 0.0 <= b2 <= 1.0:
        raise ValueError(f"mean excitation m_bar={m_bar} incompatible with N={N}")
    a2 = 1.0 - b2

    m = np.arange(m_max + 1)
    log_fact = gammaln(np.arange(N + 1) + 1)
    log_binom = log_fact[N] - log_fact[: m_max + 1] - gammaln(N - m + 1)
    with np.errstate(divide="ignore"):
        log_p = log_binom + (N - m) * np.log(max(a2, 1e-300)) + m * np.log(max(b2, 1e-300))
    return np.exp(log_p)  # returns |c_m|^2
