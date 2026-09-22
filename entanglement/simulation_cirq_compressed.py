"""
Cirq simulation of macroscopic atomic ensembles using logarithmic qubit encoding.
Maps the (N+1)-dimensional Dicke subspace onto k = ceil(log2(N+1)) Cirq qubits.

For N = 100 atoms  -> 7 Cirq qubits (2^7 = 128)
For N = 1000 atoms -> 10 Cirq qubits (2^10 = 1024)

Allows native Cirq quantum circuits to simulate massive atomic ensembles!
"""

import numpy as np
import scipy.linalg as la
import cirq
from dicke_operators import get_dicke_operators
from metrics import (
    compute_spin_observables, compute_squeezing,
    get_two_atom_reduced_dm, compute_concurrence
)

def get_k_qubit_operators(N):
    """
    Construct Jx, Jy, Jz padded to 2^k x 2^k for k = ceil(log2(N+1)).
    """
    D = N + 1
    k = int(np.ceil(np.log2(D)))
    dim_k = 2**k
    
    Jx, Jy, Jz, M_vals = get_dicke_operators(N)
    
    # Pad to dim_k
    def pad(mat):
        P = np.zeros((dim_k, dim_k), dtype=complex)
        P[:D, :D] = mat
        return P
        
    return k, dim_k, D, Jx, Jy, Jz, M_vals, pad(Jx), pad(Jy), pad(Jz)

def run_cirq_compressed_simulation(N, Omega=1.0, Delta=0.0, gamma_c=0.1, t_max=3.0, num_steps=100, num_trajectories=150):
    """
    Simulate collective dephasing dynamics for large N using k Cirq qubits.
    """
    k, dim_k, D, Jx, Jy, Jz, M_vals, Jx_k, Jy_k, Jz_k = get_k_qubit_operators(N)
    qubits = cirq.LineQubit.range(k)
    dt = t_max / num_steps
    sim = cirq.Simulator()
    
    # Coherent step unitary U = exp(-i (Omega*Jx + Delta*Jz) dt)
    H = Omega * Jx + Delta * Jz
    H_k = Omega * Jx_k + Delta * Jz_k
    U_step = la.expm(-1j * H_k * dt)
    u_gate = cirq.MatrixGate(U_step)
    
    # Trajectory-averaged density matrix in the (N+1) Dicke subspace
    rho_history = [np.zeros((D, D), dtype=complex) for _ in range(num_steps + 1)]
    
    for traj in range(num_trajectories):
        # Initial state |J, M=-J> = |m=0> = |00...0>
        state = np.zeros(dim_k, dtype=complex)
        state[0] = 1.0
        
        rho_history[0] += np.outer(state[:D], state[:D].conj())
        
        for step in range(num_steps):
            delta_phi = np.random.normal(0.0, np.sqrt(gamma_c * dt))
            
            # Diagonal dephasing unitary V(delta_phi) = exp(-i delta_phi Jz)
            V_diag = np.ones(dim_k, dtype=complex)
            V_diag[:D] = np.exp(-1j * delta_phi * M_vals)
            v_gate = cirq.MatrixGate(np.diag(V_diag))
            
            # Build circuit for one time step
            step_circuit = cirq.Circuit()
            step_circuit.append(u_gate.on(*qubits))
            step_circuit.append(v_gate.on(*qubits))
            
            result = sim.simulate(step_circuit, initial_state=state)
            state = result.final_state_vector
            
            rho_history[step + 1] += np.outer(state[:D], state[:D].conj())
            
    for step in range(num_steps + 1):
        rho_history[step] /= num_trajectories
        
    times = np.linspace(0, t_max, num_steps + 1)
    xi_R2_list = []
    concurrence_list = []
    J_len_list = []
    
    for step in range(num_steps + 1):
        rho = rho_history[step]
        # Normalize trace within Dicke subspace
        tr = np.real(np.trace(rho))
        if tr > 0:
            rho = rho / tr
        obs = compute_spin_observables(rho, Jx, Jy, Jz)
        sq = compute_squeezing(obs, N)
        rho12 = get_two_atom_reduced_dm(obs, N)
        C = compute_concurrence(rho12)
        
        xi_R2_list.append(sq['xi_R2'])
        concurrence_list.append(C)
        J_len_list.append(obs['J_len'])
        
    return {
        'times': times,
        'k_qubits': k,
        'xi_R2': np.array(xi_R2_list),
        'concurrence': np.array(concurrence_list),
        'J_len': np.array(J_len_list)
    }
