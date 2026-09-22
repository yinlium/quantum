"""
Entanglement metrics for permutation-symmetric multiatom states.
Includes:
- Mean collective spin vector <J>
- Minimum transverse variance (Delta J_perp)^2
- Wineland squeezing parameter xi_R^2
- Kitagawa-Ueda squeezing parameter xi_S^2
- Permutation-symmetric two-atom reduced density matrix rho_12
- Wootters concurrence C(rho_12)
- Quantum Fisher Information (QFI)
"""

import numpy as np
import scipy.linalg as la

def compute_spin_observables(rho, Jx, Jy, Jz):
    """
    Compute first and second moments of collective spin.
    """
    exp_Jx = np.real(np.trace(rho @ Jx))
    exp_Jy = np.real(np.trace(rho @ Jy))
    exp_Jz = np.real(np.trace(rho @ Jz))
    J_mean = np.array([exp_Jx, exp_Jy, exp_Jz])
    J_len = np.linalg.norm(J_mean)
    
    # Second moments
    exp_Jx2 = np.real(np.trace(rho @ (Jx @ Jx)))
    exp_Jy2 = np.real(np.trace(rho @ (Jy @ Jy)))
    exp_Jz2 = np.real(np.trace(rho @ (Jz @ Jz)))
    
    exp_JxJy = np.real(np.trace(rho @ (0.5 * (Jx @ Jy + Jy @ Jx))))
    exp_JxJz = np.real(np.trace(rho @ (0.5 * (Jx @ Jz + Jz @ Jx))))
    exp_JyJz = np.real(np.trace(rho @ (0.5 * (Jy @ Jz + Jz @ Jy))))
    
    return {
        'J_mean': J_mean,
        'J_len': J_len,
        'Jx': exp_Jx, 'Jy': exp_Jy, 'Jz': exp_Jz,
        'Jx2': exp_Jx2, 'Jy2': exp_Jy2, 'Jz2': exp_Jz2,
        'JxJy': exp_JxJy, 'JxJz': exp_JxJz, 'JyJz': exp_JyJz
    }

def compute_squeezing(obs, N):
    """
    Compute minimum transverse variance and squeezing parameters.
    """
    J_mean = obs['J_mean']
    J_len = obs['J_len']
    if J_len < 1e-10:
        return {'xi_R2': np.nan, 'xi_S2': np.nan, 'var_perp_min': np.nan}
    
    n0 = J_mean / J_len
    # Construct orthonormal basis {n1, n2} perpendicular to n0
    if abs(n0[2]) < 0.9:
        n1 = np.cross(n0, np.array([0, 0, 1.0]))
    else:
        n1 = np.cross(n0, np.array([1.0, 0, 0]))
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(n0, n1)
    
    # Covariance matrix in the orthogonal plane
    # Cov(Ja, Jb) = <1/2(Ja Jb + Jb Ja)> - <Ja><Jb>
    # Note: <Ja> = 0 in orthogonal plane by construction
    cov_3d = np.array([
        [obs['Jx2'] - obs['Jx']**2, obs['JxJy'] - obs['Jx']*obs['Jy'], obs['JxJz'] - obs['Jx']*obs['Jz']],
        [obs['JxJy'] - obs['Jx']*obs['Jy'], obs['Jy2'] - obs['Jy']**2, obs['JyJz'] - obs['Jy']*obs['Jz']],
        [obs['JxJz'] - obs['Jx']*obs['Jz'], obs['JyJz'] - obs['Jy']*obs['Jz'], obs['Jz2'] - obs['Jz']**2]
    ])
    
    # Project cov into 2D subspace spanned by n1, n2
    P = np.column_stack([n1, n2])  # 3 x 2
    cov_2d = P.T @ cov_3d @ P
    
    # Minimum eigenvalue of cov_2d is the minimal transverse variance
    eigvals = np.linalg.eigvalsh(cov_2d)
    var_perp_min = max(0.0, eigvals[0])
    
    # Wineland parameter xi_R^2 = N * (Delta J_perp)^2 / |<J>|^2
    xi_R2 = (N * var_perp_min) / (J_len**2)
    
    # Kitagawa-Ueda parameter xi_S^2 = 4 * (Delta J_perp)^2 / N
    xi_S2 = (4.0 * var_perp_min) / N
    
    return {
        'var_perp_min': var_perp_min,
        'xi_R2': xi_R2,
        'xi_S2': xi_S2
    }

def get_two_atom_reduced_dm(obs, N):
    """
    Compute 4x4 reduced density matrix for any pair of atoms in the symmetric state.
    Basis: |00>, |01>, |10>, |11>
    """
    sx = 2.0 * obs['Jx'] / N
    sy = 2.0 * obs['Jy'] / N
    sz = 2.0 * obs['Jz'] / N
    
    cxx = (4.0 * obs['Jx2'] - N) / (N * (N - 1))
    cyy = (4.0 * obs['Jy2'] - N) / (N * (N - 1))
    czz = (4.0 * obs['Jz2'] - N) / (N * (N - 1))
    
    cxy = 4.0 * obs['JxJy'] / (N * (N - 1))
    cxz = 4.0 * obs['JxJz'] / (N * (N - 1))
    cyz = 4.0 * obs['JyJz'] / (N * (N - 1))
    
    # Pauli matrices
    s0 = np.eye(2, dtype=complex)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    
    paulis = [s1, s2, s3]
    s_vec = [sx, sy, sz]
    c_mat = np.array([
        [cxx, cxy, cxz],
        [cxy, cyy, cyz],
        [cxz, cyz, czz]
    ])
    
    rho12 = np.kron(s0, s0).astype(complex)
    for a in range(3):
        rho12 += s_vec[a] * (np.kron(paulis[a], s0) + np.kron(s0, paulis[a]))
        for b in range(3):
            rho12 += c_mat[a, b] * np.kron(paulis[a], paulis[b])
            
    rho12 *= 0.25
    return rho12

def compute_concurrence(rho12):
    """
    Compute Wootters concurrence for a 4x4 density matrix.
    """
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sy_sy = np.kron(sy, sy)
    
    rho_tilde = sy_sy @ rho12.conj() @ sy_sy
    R = rho12 @ rho_tilde
    
    eigvals = la.eigvals(R)
    # Numerical stability: take absolute values and square roots
    lambdas = np.sort(np.sqrt(np.maximum(0.0, np.real(eigvals))))[::-1]
    
    concurrence = max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])
    return concurrence

def compute_qfi(rho, J_op):
    """
    Quantum Fisher Information for observable J_op:
    F_Q = 2 sum_{j, k: p_j + p_k > 0} (p_j - p_k)^2 / (p_j + p_k) |<j|J_op|k>|^2
    """
    eigvals, eigvecs = la.eigh(rho)
    D = len(eigvals)
    F_Q = 0.0
    for j in range(D):
        pj = eigvals[j]
        if pj < 1e-14:
            continue
        for k in range(D):
            pk = eigvals[k]
            if pj + pk > 1e-14:
                mat_elem = np.abs(eigvecs[:, j].conj() @ (J_op @ eigvecs[:, k]))**2
                F_Q += 2.0 * ((pj - pk)**2 / (pj + pk)) * mat_elem
    return F_Q
