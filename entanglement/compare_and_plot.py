"""
Comparison and visualization script:
1. Compares Direct Cirq circuit simulation vs. Dicke master equation for small N (N=4).
2. Runs Compressed Cirq circuit simulation on 7 qubits for macroscopic N=100 atoms vs. Dicke master equation.
3. Generates plots of:
   - Wineland squeezing parameter xi_R^2(t) (xi_R^2 < 1 indicates entanglement)
   - Pairwise concurrence C(t) between atoms
   - Collective spin length |<J>|(t)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation_dicke import run_dicke_simulation
from simulation_cirq_direct import run_cirq_trajectories
from simulation_cirq_compressed import run_cirq_compressed_simulation

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_file = os.path.join(script_dir, "entanglement_dynamics_comparison.png")
    
    print("=" * 60)
    print("1. Running Small N=4 Validation (Direct Cirq vs Dicke ME)")
    print("=" * 60)
    t_max = 3.0
    num_steps = 100
    Omega = 1.0
    gamma_c = 0.1
    
    # Dicke ME
    res_dicke_4 = run_dicke_simulation(N=4, Omega=Omega, gamma_c=gamma_c, t_max=t_max, num_steps=num_steps)
    # Cirq direct trajectories
    res_cirq_4 = run_cirq_trajectories(N=4, Omega=Omega, gamma_c=gamma_c, t_max=t_max, num_steps=num_steps, num_trajectories=300)
    
    print(f"N=4 | Dicke Min xi_R^2: {np.nanmin(res_dicke_4['xi_R2']):.4f} | Cirq Min xi_R^2: {np.nanmin(res_cirq_4['xi_R2']):.4f}")
    print(f"N=4 | Dicke Max Concurrence: {np.nanmax(res_dicke_4['concurrence']):.6f} | Cirq Max Concurrence: {np.nanmax(res_cirq_4['concurrence']):.6f}")

    print("\n" + "=" * 60)
    print("2. Running Macroscopic N=100 Atoms (Compressed Cirq 7-Qubit Circuit vs Dicke ME)")
    print("=" * 60)
    res_dicke_100 = run_dicke_simulation(N=100, Omega=Omega, gamma_c=gamma_c, t_max=t_max, num_steps=num_steps)
    res_cirq_100 = run_cirq_compressed_simulation(N=100, Omega=Omega, gamma_c=gamma_c, t_max=t_max, num_steps=num_steps, num_trajectories=250)
    
    print(f"N=100 | Dicke Min xi_R^2: {np.nanmin(res_dicke_100['xi_R2']):.4f} | Cirq (7 qubits) Min xi_R^2: {np.nanmin(res_cirq_100['xi_R2']):.4f}")
    print(f"N=100 | Dicke Max Concurrence: {np.nanmax(res_dicke_100['concurrence']):.6f} | Cirq (7 qubits) Max Concurrence: {np.nanmax(res_cirq_100['concurrence']):.6f}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    
    # (0, 0): N=4 Squeezing
    axes[0, 0].plot(res_dicke_4['times'], res_dicke_4['xi_R2'], 'k-', lw=2.5, label='Dicke ME (Exact)')
    axes[0, 0].plot(res_cirq_4['times'], res_cirq_4['xi_R2'], 'r--', lw=1.8, label='Cirq 4-Qubit Circuit (300 traj)')
    axes[0, 0].axhline(1.0, color='gray', ls=':', label='Standard Quantum Limit (SQL)')
    axes[0, 0].set_title("N = 4 Atoms: Wineland Squeezing Parameter $\\xi_R^2$")
    axes[0, 0].set_xlabel("Time $\\Omega t$")
    axes[0, 0].set_ylabel("$\\xi_R^2$")
    axes[0, 0].set_ylim(0, 3)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # (0, 1): N=4 Concurrence
    axes[0, 1].plot(res_dicke_4['times'], res_dicke_4['concurrence'], 'k-', lw=2.5, label='Dicke ME (Exact)')
    axes[0, 1].plot(res_cirq_4['times'], res_cirq_4['concurrence'], 'r--', lw=1.8, label='Cirq 4-Qubit Circuit')
    axes[0, 1].set_title("N = 4 Atoms: Pairwise Concurrence $C(\\rho_{12})$")
    axes[0, 1].set_xlabel("Time $\\Omega t$")
    axes[0, 1].set_ylabel("Concurrence $C$")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # (1, 0): N=100 Squeezing
    axes[1, 0].plot(res_dicke_100['times'], res_dicke_100['xi_R2'], 'b-', lw=2.5, label='Dicke ME (Exact N=100)')
    axes[1, 0].plot(res_cirq_100['times'], res_cirq_100['xi_R2'], 'm--', lw=1.8, label='Cirq Log-Qubit Circuit (k=7 qubits)')
    axes[1, 0].axhline(1.0, color='gray', ls=':', label='Standard Quantum Limit (SQL)')
    axes[1, 0].set_title("N = 100 Atoms: Wineland Squeezing Parameter $\\xi_R^2$")
    axes[1, 0].set_xlabel("Time $\\Omega t$")
    axes[1, 0].set_ylabel("$\\xi_R^2$")
    axes[1, 0].set_ylim(0, 2)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # (1, 1): N=100 Concurrence
    axes[1, 1].plot(res_dicke_100['times'], res_dicke_100['concurrence'], 'b-', lw=2.5, label='Dicke ME (Exact N=100)')
    axes[1, 1].plot(res_cirq_100['times'], res_cirq_100['concurrence'], 'm--', lw=1.8, label='Cirq Log-Qubit Circuit (k=7 qubits)')
    axes[1, 1].set_title("N = 100 Atoms: Pairwise Concurrence $C(\\rho_{12})$")
    axes[1, 1].set_xlabel("Time $\\Omega t$")
    axes[1, 1].set_ylabel("Concurrence $C$")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    print(f"\nPlot successfully saved to: {plot_file}")

if __name__ == "__main__":
    main()
