import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cirq

class CollectiveLaserDriveStep(cirq.Gate):
    """
    Custom Cirq composite gate representing a Trotter step of uniform laser driving
    H_drive = (Omega_1 / 2) * sum_{j=1}^N X_j across an N-atom Rydberg ensemble.
    Demonstrates Cirq hierarchical gate decomposition (`_decompose_`).
    """
    def __init__(self, num_qubits: int, omega_1: float, dt: float):
        super().__init__()
        self._num_qubits = num_qubits
        self.omega_1 = omega_1
        self.dt = dt

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits):
        # Rotate each atom around X by angle theta = Omega_1 * dt
        exponent = (self.omega_1 * self.dt) / np.pi
        for q in qubits:
            yield cirq.XPowGate(exponent=exponent).on(q)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return [f"Drive(Ω₁dt)"] * self._num_qubits


class RydbergBlockadeStep(cirq.Gate):
    """
    Custom Cirq composite gate representing a Trotter step of pairwise Rydberg
    dipole blockade interactions: H_vdW = sum_{j < k} V_vdW * n_j * n_k.
    Decomposes into native all-to-all cirq.CZPowGate operations.
    """
    def __init__(self, num_qubits: int, v_vdw: float, dt: float):
        super().__init__()
        self._num_qubits = num_qubits
        self.v_vdw = v_vdw
        self.dt = dt

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def _decompose_(self, qubits):
        exponent = -(self.v_vdw * self.dt) / np.pi
        for i in range(self._num_qubits):
            for j in range(i + 1, self._num_qubits):
                yield cirq.CZPowGate(exponent=exponent).on(qubits[i], qubits[j])

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs):
        return [f"Blockade(V)"] * self._num_qubits


def simulate_superatom_rabi_cirq(N: int, omega_1: float = 1.0, v_vdw: float = 40.0, t_max: float = 2.0 * np.pi, num_steps: int = 80):
    """
    Simulate many-body collective Rabi oscillations of an N-atom Rydberg superatom
    using second-order Strang-Trotterized Cirq quantum circuits.
    Returns: times, pop_W (single excitation Dicke state), pop_multi (m >= 2 leakage).
    """
    qubits = cirq.LineQubit.range(N)
    sim = cirq.Simulator()
    times = np.linspace(0.0, t_max, num_steps)
    dt = times[1] - times[0]
    
    # Precompute excitation number masks for fast projection
    num_states = 2**N
    bit_counts = np.array([bin(s).count('1') for s in range(num_states)])
    mask_m1 = (bit_counts == 1)
    mask_multi = (bit_counts >= 2)
    
    # Symmetric Dicke state |W> = (1/sqrt(N)) sum_{|s|=1} |s>
    w_state = np.zeros(num_states, dtype=complex)
    w_state[mask_m1] = 1.0 / np.sqrt(N)
    
    pop_W = np.zeros(num_steps)
    pop_multi = np.zeros(num_steps)
    
    # Sub-step Trotterization for high precision under large V_vdW
    sub_steps = 8
    dt_sub = dt / sub_steps
    
    step_circuit = cirq.Circuit()
    for _ in range(sub_steps):
        step_circuit.append(CollectiveLaserDriveStep(N, omega_1, dt_sub / 2.0).on(*qubits))
        step_circuit.append(RydbergBlockadeStep(N, v_vdw, dt_sub).on(*qubits))
        step_circuit.append(CollectiveLaserDriveStep(N, omega_1, dt_sub / 2.0).on(*qubits))
        
    current_state = np.zeros(num_states, dtype=complex)
    current_state[0] = 1.0  # Initial state |00...0>
    
    for it in range(num_steps):
        if it > 0:
            res = sim.simulate(step_circuit, initial_state=current_state)
            current_state = res.final_state_vector
            
        # Fidelity with single-excitation symmetric superatom state |W>
        pop_W[it] = np.abs(np.vdot(w_state, current_state))**2
        # Total population leakage into multiply excited states (m >= 2)
        pop_multi[it] = np.sum(np.abs(current_state[mask_multi])**2)
        
    return times, pop_W, pop_multi


def run_superatom_study():
    print("=" * 75)
    print("Cirq Collective Superatom Rabi Oscillations (Phys. Rev. Lett. 128, 123601)")
    print("=" * 75)
    
    # 1. Print sample decomposed Cirq circuit for N = 3 superatom
    q3 = cirq.LineQubit.range(3)
    demo_circuit = cirq.Circuit(
        CollectiveLaserDriveStep(3, 1.0, 0.1).on(*q3),
        RydbergBlockadeStep(3, 20.0, 0.1).on(*q3)
    )
    print("\n[High-Level Modular Superatom Circuit (N=3)]:")
    print(demo_circuit)
    print("\n[Decomposed Native Gate Circuit via cirq.decompose_once]:")
    print(cirq.Circuit(cirq.decompose_once(op) for op in demo_circuit.all_operations()))
    
    # 2. Simulate Collective Rabi Oscillations for N = 1, 2, 4, 9
    N_list = [1, 2, 4, 9]
    omega_1 = 1.0
    v_vdw = 45.0  # Strong dipole blockade V_vdW >> sqrt(N)*Omega_1
    t_max = 1.5 * np.pi
    
    results = {}
    print("\nSimulating collective sqrt(N) Rabi enhancement in Cirq...")
    for N in N_list:
        t0 = time.time()
        times, p_W, p_multi = simulate_superatom_rabi_cirq(N, omega_1=omega_1, v_vdw=v_vdw, t_max=t_max, num_steps=75)
        results[N] = {'times': times, 'p_W': p_W, 'p_multi': p_multi}
        print(f"  N = {N:2d} (sqrt(N) = {np.sqrt(N):.2f}x speed-up) | Max leakage P(m>=2) = {np.max(p_multi):.4f} | Done in {time.time()-t0:.2f}s")
        
    # 3. Simulate Blockade Leakage Sweep vs V_vdW / Omega_N for N = 4
    print("\nSimulating blockade leakage suppression vs V_vdW / Omega_N (N=4)...")
    v_ratios = np.logspace(-0.5, 1.8, 18)
    max_leakage = []
    max_fidelity = []
    N_ref = 4
    omega_N = np.sqrt(N_ref) * omega_1
    t_pi = np.pi / omega_N  # Collective pi-pulse time
    
    for ratio in v_ratios:
        v_val = ratio * omega_N
        _, pW, pM = simulate_superatom_rabi_cirq(N_ref, omega_1=omega_1, v_vdw=v_val, t_max=t_pi * 1.2, num_steps=35)
        max_leakage.append(np.max(pM))
        max_fidelity.append(np.max(pW))
        
    # 4. Generate Publication-Grade Figure
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=250)
    colors = {1: '#1f77b4', 2: '#ff7f0e', 4: '#2ca02c', 9: '#d62728'}
    
    # Panel (a): Collective sqrt(N) Rabi Oscillations
    ax = axes[0]
    for N in N_list:
        t_norm = results[N]['times'] * omega_1 / (2.0 * np.pi)
        pW = results[N]['p_W']
        c = colors[N]
        ax.plot(t_norm, pW, color=c, lw=2.3, label=f'Cirq $N={N}$ ($\\Omega_N = {np.sqrt(N):.1f}\\,\\Omega_1$)')
        # Plot theoretical analytical curve sin^2(sqrt(N) * Omega_1 * t / 2)
        th_curve = np.sin(np.sqrt(N) * results[N]['times'] * omega_1 / 2.0)**2
        ax.plot(t_norm, th_curve, color=c, ls=':', lw=1.5, alpha=0.75)
        
    ax.set_title("(a) Collective Many-Body Rabi Oscillations ($\\Omega_N = \\sqrt{N}\\,\\Omega_1$)\n(Solid: Cirq Trotterized Circuit | Dotted: Ideal $\\sqrt{N}$ Theory)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Single-Atom Pulse Area $\\Omega_1 t / (2\\pi)$", fontsize=11)
    ax.set_ylabel("Superatom $|W\\rangle$ State Population $P(|W\\rangle)$", fontsize=11)
    ax.set_ylim(-0.03, 1.08)
    ax.legend(loc='upper right', fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    
    # Panel (b): Dipole Blockade Qubit Isolation & Leakage Suppression
    ax = axes[1]
    ax.plot(v_ratios, np.array(max_fidelity) * 100, 'b-o', lw=2.2, markersize=5, label='Peak Superatom $|W\\rangle$ Fidelity (%)')
    ax.plot(v_ratios, np.array(max_leakage) * 100, 'r--s', lw=2.2, markersize=5, label='Multi-Excitation Leakage $P(m \\geq 2)$ (%)')
    ax.axvline(10.0, color='purple', ls=':', lw=1.5, label='Strong Blockade Regime ($V_{\\mathrm{vdW}} \\geq 10\\,\\Omega_N$)')
    ax.set_xscale('log')
    ax.set_title("(b) Dipole Blockade Qubit Subspace Isolation ($N = 4$)\n(Suppression of Doubly-Excited Leakage $P(m \\geq 2) \\to 0$)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Blockade-to-Rabi Ratio $V_{\\mathrm{vdW}} / \\Omega_N$", fontsize=11)
    ax.set_ylabel("Population Percentage (%)", fontsize=11)
    ax.set_ylim(-3, 105)
    ax.legend(loc='center right', fontsize=9, frameon=True)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cirq_prl2022_collective_rabi.png")
    plt.savefig(out_img, dpi=250)
    print(f"\nSaved publication-grade figure to: {out_img}")
    print("=" * 75)

if __name__ == '__main__':
    run_superatom_study()
