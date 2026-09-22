"""
Direct N-qubit Cirq circuit simulation of collective dephasing and multiatom entanglement.
Uses Cirq quantum circuits with:
1. Coherent collective drive gates: cirq.rx(Omega * dt).on_each(*qubits)
2. Collective dephasing via stochastic trajectory ensembles:
   Each trajectory applies a shared random phase Rz(delta_phi) to all qubits,
   where delta_phi ~ Normal(0, gamma_c * dt).
"""

import numpy as np
import cirq
from metrics import (
    compute_spin_observables, compute_squeezing,
    get_two_atom_reduced_dm, compute_concurrence
)
from dicke_operators import get_dicke_operators

def build_cirq_step(qubits, Omega, dt, delta_phi):
    """
    Construct a single Trotter step in Cirq:
    1. Drive: R_x(Omega * dt) on all qubits.
    2. Collective dephasing: R_z(delta_phi) on all qubits with identical angle delta_phi.
    """
    circuit = cirq.Circuit()
    # Rx(theta) applies exp(-i theta/2 X). For H = Omega * X/2, theta = Omega * dt
    circuit.append(cirq.rx(Omega * dt).on_each(qubits))
    # Rz(theta) applies exp(-i theta/2 Z). Collective phase delta_phi
    circuit.append(cirq.rz(delta_phi).on_each(qubits))
    return circuit

def run_cirq_trajectories(N, Omega=1.0, gamma_c=0.1, t_max=3.0, num_steps=150, num_trajectories=200):
    """
    Simulate the collective dephasing Lindbladian using Cirq statevector simulation
    averaged over quantum trajectories.
    """
    qubits = cirq.LineQubit.range(N)
    dt = t_max / num_steps
    sim = cirq.Simulator()
    
    # Pre-generate Dicke operators for computing observables from symmetric density matrix
    Jx, Jy, Jz, M_vals = get_dicke_operators(N)
    
    # We will record density matrix rho(t) in the (N+1) Dicke basis
    # A symmetric state |psi> in 2^N can be projected to Dicke basis:
    # |J, M> states
    # Let's construct projection matrix P of shape ((N+1), 2^N)
    # where P[m, :] = <J, M = m - N/2|
    dim_full = 2**N
    P_dicke = np.zeros((N + 1, dim_full), dtype=complex)
    for idx in range(dim_full):
        # Hamming weight (number of 1s in binary representation)
        hw = bin(idx).count('1')
        # m = hw, M = hw - N/2
        # Normalization: state |J, M> is uniform superposition over all bin(idx) with weight hw
        # Number of such states is binom(N, hw)
        P_dicke[hw, idx] = 1.0
    # Normalize rows
    for m in range(N + 1):
        norm = np.linalg.norm(P_dicke[m, :])
        if norm > 0:
            P_dicke[m, :] /= norm

    # We track rho_dicke(t) for each time step
    rho_dicke_history = [np.zeros((N + 1, N + 1), dtype=complex) for _ in range(num_steps + 1)]
    
    for traj in range(num_trajectories):
        # Initial state |0>^N
        current_state = np.zeros(dim_full, dtype=complex)
        current_state[0] = 1.0
        
        # Project initial state to Dicke basis
        psi_d = P_dicke @ current_state
        rho_dicke_history[0] += np.outer(psi_d, psi_d.conj())
        
        for step in range(num_steps):
            delta_phi = np.random.normal(0.0, np.sqrt(gamma_c * dt))
            step_circuit = build_cirq_step(qubits, Omega, dt, delta_phi)
            
            # Simulate one step in Cirq
            result = sim.simulate(step_circuit, initial_state=current_state)
            current_state = result.final_state_vector
            
            psi_d = P_dicke @ current_state
            rho_dicke_history[step + 1] += np.outer(psi_d, psi_d.conj())
            
    # Average over trajectories
    for step in range(num_steps + 1):
        rho_dicke_history[step] /= num_trajectories

    # Compute metrics over time
    times = np.linspace(0, t_max, num_steps + 1)
    xi_R2_list = []
    concurrence_list = []
    
    for step in range(num_steps + 1):
        rho = rho_dicke_history[step]
        obs = compute_spin_observables(rho, Jx, Jy, Jz)
        sq = compute_squeezing(obs, N)
        rho12 = get_two_atom_reduced_dm(obs, N)
        C = compute_concurrence(rho12)
        
        xi_R2_list.append(sq['xi_R2'])
        concurrence_list.append(C)
        
    return {
        'times': times,
        'xi_R2': np.array(xi_R2_list),
        'concurrence': np.array(concurrence_list)
    }
