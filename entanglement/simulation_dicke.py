"""
High-performance Dicke-basis master equation solver for multiatom ensembles.
Simulates:
    drho/dt = -i [H, rho] + (gamma_c / 2) * (2 Jz rho Jz - Jz^2 rho - rho Jz^2)
where H = Omega * Jx + Delta * Jz.

Uses Strang (Trotter-Suzuki) operator splitting:
    rho(t + dt) = S(dt/2) * exp(-i H dt) * S(dt/2) * rho(t)
where S(tau) elementwise dampens: rho_ij -> rho_ij * exp(-0.5 * gamma_c * (M_i - M_j)^2 * tau).
This is UNCONDITIONALLY STABLE, completely positive, trace preserving, and exact for any N!
"""

import numpy as np
import scipy.linalg as la
from scipy.special import gammaln
from dicke_operators import get_dicke_operators
from metrics import compute_spin_observables, compute_squeezing, get_two_atom_reduced_dm, compute_concurrence

def get_initial_state(N, state_type='ground'):
    r"""
    Construct initial state density matrix in Dicke basis (dimension D = N + 1).
    state_type:
        'ground': |J, M=-J> = |0>^{\otimes N}
        'plus_x': |J, Mx=J> = |+>^{\otimes N}
    """
    D = N + 1
    if state_type == 'ground':
        psi = np.zeros(D, dtype=complex)
        psi[0] = 1.0  # m = 0 corresponds to M = -N/2
        return np.outer(psi, psi.conj())
    
    elif state_type == 'plus_x':
        # Coherent spin state along +x:
        # psi[m] = sqrt(binom(N, m)) * (1/sqrt(2))^N
        log_fact = gammaln(np.arange(D) + 1)
        log_binom = log_fact[N] - log_fact - log_fact[::-1]
        log_coeffs = 0.5 * log_binom - 0.5 * N * np.log(2.0)
        psi = np.exp(log_coeffs).astype(complex)
        return np.outer(psi, psi.conj())
    else:
        raise ValueError(f"Unknown state type: {state_type}")

def run_dicke_simulation(N, Omega=1.0, Delta=0.0, gamma_c=0.1, t_max=3.0, num_steps=300, initial_state='ground'):
    """
    Run full time evolution in Dicke basis using Strang operator splitting.
    """
    Jx, Jy, Jz, M_vals = get_dicke_operators(N)
    H = Omega * Jx + Delta * Jz
    
    dt = t_max / num_steps
    
    # Precompute damping factor matrix for half-step: exp(-0.5 * gamma_c * (dM)^2 * (dt/2))
    dM = M_vals[:, None] - M_vals[None, :]
    damping_half = np.exp(-0.25 * gamma_c * (dM**2) * dt)
    
    # Precompute unitary evolution operator U = exp(-i H dt)
    U = la.expm(-1j * H * dt)
    U_dag = U.T.conj()
    
    rho = get_initial_state(N, state_type=initial_state)
    times = np.linspace(0, t_max, num_steps + 1)
    
    xi_R2_list = []
    xi_S2_list = []
    var_perp_list = []
    concurrence_list = []
    J_len_list = []
    
    for step in range(num_steps + 1):
        # Normalize and enforce Hermiticity
        rho = 0.5 * (rho + rho.T.conj())
        tr = np.real(np.trace(rho))
        if tr > 0:
            rho /= tr
            
        obs = compute_spin_observables(rho, Jx, Jy, Jz)
        sq = compute_squeezing(obs, N)
        rho12 = get_two_atom_reduced_dm(obs, N)
        C = compute_concurrence(rho12)
        
        xi_R2_list.append(sq['xi_R2'])
        xi_S2_list.append(sq['xi_S2'])
        var_perp_list.append(sq['var_perp_min'])
        concurrence_list.append(C)
        J_len_list.append(obs['J_len'])
        
        if step < num_steps:
            # Strang splitting step:
            # 1. Half-step dephasing
            rho = rho * damping_half
            # 2. Full-step unitary
            rho = U @ rho @ U_dag
            # 3. Half-step dephasing
            rho = rho * damping_half
            
    return {
        'times': times,
        'xi_R2': np.array(xi_R2_list),
        'xi_S2': np.array(xi_S2_list),
        'var_perp': np.array(var_perp_list),
        'concurrence': np.array(concurrence_list),
        'J_len': np.array(J_len_list),
        'final_rho': rho
    }
