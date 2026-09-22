"""
Simulation of Dynamics of Collective-Dephasing-Induced Multiatom Entanglement
Reference: Y. Li, Y. Mei, H. Nguyen, P. R. Berman, A. Kuzmich, Phys. Rev. A 106, L051701 (2022).
"""

import numpy as np

def get_dicke_operators(N):
    """
    Construct collective spin operators Jx, Jy, Jz in the symmetric (Dicke) basis
    |J=N/2, M> where M in [-N/2, ..., N/2], dimension D = N + 1.
    Basis index m = M + N/2 in {0, 1, ..., N}.
    """
    D = N + 1
    J = N / 2.0
    m_indices = np.arange(D)  # m = 0, ..., N
    M_vals = m_indices - J    # M = -J, ..., +J
    
    # Jz is diagonal: Jz|m> = (m - J)|m>
    Jz = np.diag(M_vals).astype(complex)
    
    # J+|m> = sqrt((N - m)*(m + 1)) |m+1>
    J_plus = np.zeros((D, D), dtype=complex)
    for m in range(N):
        J_plus[m + 1, m] = np.sqrt((N - m) * (m + 1))
    
    J_minus = J_plus.T.conj()
    Jx = 0.5 * (J_plus + J_minus)
    Jy = -0.5j * (J_plus - J_minus)
    
    return Jx, Jy, Jz, M_vals
