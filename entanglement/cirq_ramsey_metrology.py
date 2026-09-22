"""
Ramsey Interferometry and Quantum Metrology Beyond the Standard Quantum Limit.

Reference
---------
Y. Li, Y. Mei, H. Nguyen, P. R. Berman, and A. Kuzmich,
Phys. Rev. A 106, L051701 (2022).

Physics & Cirq Showcase
-----------------------
Demonstrates that collective-dephasing-purified Dicke states and symmetric
|W> states provide metrologically useful multiparticle entanglement that beats
the Standard Quantum Limit (SQL).

Cirq APIs showcased:
- Symbolic parameter sweeps with ``sympy.Symbol`` and ``cirq.Linspace``
- Explicit Ramsey interferometer circuits (pi/2 -> phase accumulation -> pi/2)
- ``rydberg_cirq.CollectiveDephasingChannel`` density-matrix evolution
- Quantum Fisher Information (QFI) via ``rydberg_cirq.quantum_fisher_information``
  and verification of the Quantum Cramer-Rao Bound: Delta phi >= 1 / sqrt(F_Q)
"""

from __future__ import annotations

import numpy as np
import sympy
import cirq
import matplotlib.pyplot as plt

import rydberg_cirq as rc


def build_ramsey_circuit(qubits, phi_symbol: sympy.Symbol) -> cirq.Circuit:
    """Build a collective Ramsey interferometer circuit with symbolic phase phi."""
    return cirq.Circuit(
        cirq.rx(np.pi / 2).on_each(*qubits),
        cirq.rz(phi_symbol).on_each(*qubits),
        cirq.rx(-np.pi / 2).on_each(*qubits),
    )


def prepare_states(N: int, m_bar: float = 1.6, gamma_tau: float = 4.0) -> dict:
    """Prepare initial density matrices for four benchmark states of N qubits."""
    qubits = cirq.LineQubit.range(N)
    dm_sim = cirq.DensityMatrixSimulator()
    dim = 2**N

    # 1. Coherent Spin State (SQL reference): |+y>^{\otimes N} or |0>^{\otimes N}
    psi_css = np.zeros(dim, dtype=complex)
    psi_css[0] = 1.0
    rho_css = np.outer(psi_css, psi_css.conj())

    # 2. Symmetric single-excitation Dicke state |W>
    psi_w = rc.w_state_vector(N)
    rho_w = np.outer(psi_w, psi_w.conj())

    # 3. Retrieved spin wave purified by interaction-induced dephasing:
    #    During storage Ts, pairwise van der Waals interactions (PairwisePhaseGate)
    #    dephase m >= 2 symmetric Dicke components into orthogonal non-retrieved modes
    #    while leaving the m = 1 |W> state intact (Dicke purity p1 / (1 - p0) -> 1).
    theta = 2.0 * np.arcsin(np.sqrt(min(m_bar / N, 0.95)))
    rng = np.random.default_rng(42)
    pos = rc.sample_cloud(rc.SHORT_CLOUD, n=N, rng=rng)
    phi_mat = rc.phase_matrix(pos, storage_time=gamma_tau, c6=rc.SHORT_CLOUD.c6)
    prep_circuit = cirq.Circuit(
        cirq.ry(theta).on_each(*qubits),
        rc.PairwisePhaseGate(phi_mat).on(*qubits),
        rc.CollectiveDephasingChannel(N, gamma_c=1.0, tau=gamma_tau).on(*qubits),
    )
    rho_raw = dm_sim.simulate(prep_circuit).final_density_matrix
    # Project onto the phase-matched retrieved Dicke manifold (m >= 1):
    P_sym = rc.dicke_projector(N)[1:, :]  # excited symmetric Dicke states m = 1..N
    Pi_ret = P_sym.conj().T @ P_sym
    rho_deph = Pi_ret @ rho_raw @ Pi_ret
    rho_deph /= max(float(np.real(np.trace(rho_deph))), 1e-15)

    # 4. GHZ state (Heisenberg limit reference)
    psi_ghz = np.zeros(dim, dtype=complex)
    psi_ghz[0] = 1.0 / np.sqrt(2.0)
    psi_ghz[-1] = 1.0 / np.sqrt(2.0)
    rho_ghz = np.outer(psi_ghz, psi_ghz.conj())

    return {
        "Coherent Spin State (SQL)": rho_css,
        "Dephased Spin Wave": rho_deph,
        "Symmetric |W> State": rho_w,
        "GHZ State (Heisenberg)": rho_ghz,
    }


def main():
    rc.plotting.apply_style()
    print("=" * 78)
    print("Cirq Ramsey Metrology & Quantum Fisher Information (Phys. Rev. A 106, L051701)")
    print("=" * 78)

    N_demo = 6
    qubits = cirq.LineQubit.range(N_demo)
    phi_sym = sympy.Symbol("phi")
    ramsey_circuit = build_ramsey_circuit(qubits, phi_sym)
    sweep = cirq.Linspace("phi", 0.0, 2.0 * np.pi, 65)

    states = prepare_states(N_demo)
    J_ops = rc.collective_spin_paulisums(qubits)
    Jx_mat = J_ops["Jx"].matrix(qubits)
    Jy_mat = J_ops["Jy"].matrix(qubits)
    Jz_mat = J_ops["Jz"].matrix(qubits)

    print(f"\n[1] Quantum Fisher Information Validation (N = {N_demo}):")
    print(f"  {'State':<28} | {'QFI F_Q':>10} | {'F_Q / N':>10} | {'Cramer-Rao 1/sqrt(F_Q)':>22}")
    print("-" * 78)

    qfi_results = {}
    for name, rho in states.items():
        # Evaluate maximum QFI over collective spin generators {Jx, Jy, Jz}
        fq_vals = [
            rc.quantum_fisher_information(rho, Jx_mat),
            rc.quantum_fisher_information(rho, Jy_mat),
            rc.quantum_fisher_information(rho, Jz_mat),
        ]
        fq = max(fq_vals)
        qfi_results[name] = fq
        cr_bound = 1.0 / np.sqrt(max(fq, 1e-12))
        print(f"  {name:<28} | {fq:10.4f} | {fq / N_demo:10.4f} | {cr_bound:22.5f}")

    # Sweep phi using cirq.ParamResolver from cirq.Linspace
    phi_vals = np.array([resolver.value_of(phi_sym) for resolver in sweep])
    fringes = {}
    for name, rho0 in states.items():
        curve = []
        for resolver in sweep:
            U = cirq.unitary(cirq.resolve_parameters(ramsey_circuit, resolver))
            rho_f = U @ rho0 @ U.conj().T
            # Parity / collective population contrast signal
            if "GHZ" in name:
                # Parity observable prod_j Z_j for GHZ Ramsey fringe
                parity_diag = np.array([(-1) ** (int(s).bit_count()) for s in range(2**N_demo)])
                sig = float(np.real(np.sum(np.diag(rho_f) * parity_diag)))
            else:
                sig = float(np.real(np.trace(rho_f @ Jz_mat))) / (N_demo / 2.0)
            curve.append(sig)
        fringes[name] = np.array(curve)

    # Scaling study across N = 2..8
    N_list = [2, 3, 4, 5, 6, 7, 8]
    fq_over_n = {k: [] for k in ["SQL", "Dephased", "W", "GHZ"]}
    sens_vs_n = {k: [] for k in ["SQL", "Dephased", "W", "GHZ"]}

    print("\n[2] Scaling Study of QFI/N and Phase Sensitivity across N = 2..8:")
    for N in N_list:
        qs = cirq.LineQubit.range(N)
        J_n = rc.collective_spin_paulisums(qs)
        Jx_n, Jy_n, Jz_n = J_n["Jx"].matrix(qs), J_n["Jy"].matrix(qs), J_n["Jz"].matrix(qs)
        st_n = prepare_states(N)

        for key, full_name in [
            ("SQL", "Coherent Spin State (SQL)"),
            ("Dephased", "Dephased Spin Wave"),
            ("W", "Symmetric |W> State"),
            ("GHZ", "GHZ State (Heisenberg)"),
        ]:
            rho = st_n[full_name]
            fq = max(
                rc.quantum_fisher_information(rho, Jx_n),
                rc.quantum_fisher_information(rho, Jy_n),
                rc.quantum_fisher_information(rho, Jz_n),
            )
            fq_over_n[key].append(fq / N)
            sens_vs_n[key].append(1.0 / np.sqrt(max(fq, 1e-12)))

    print(
        f"  N={N_list[-1]}: F_Q/N -> SQL={fq_over_n['SQL'][-1]:.3f}, "
        f"Dephased={fq_over_n['Dephased'][-1]:.3f}, "
        f"|W>={fq_over_n['W'][-1]:.3f} (analytic 3-2/N={3 - 2/N_list[-1]:.3f}), "
        f"GHZ={fq_over_n['GHZ'][-1]:.3f} (analytic N={N_list[-1]:.3f})"
    )

    # Plot 3-panel publication figure
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    colors = {
        "Coherent Spin State (SQL)": rc.PALETTE["gray"],
        "Dephased Spin Wave": rc.PALETTE["blue"],
        "Symmetric |W> State": rc.PALETTE["green"],
        "GHZ State (Heisenberg)": rc.PALETTE["red"],
    }

    # Panel (a): Ramsey fringes
    ax = axes[0]
    for name, curve in fringes.items():
        ls = "--" if "SQL" in name else "-"
        ax.plot(phi_vals / np.pi, curve, label=name, color=colors[name], linestyle=ls)
    ax.set_title(f"(a) Cirq Parametrized Ramsey Fringes ($N = {N_demo}$)")
    ax.set_xlabel(r"Interferometer Phase $\phi / \pi$")
    ax.set_ylabel(r"Normalized Ramsey Signal $\langle \hat{\mathcal{O}}(\phi) \rangle$")
    ax.legend(loc="lower left", fontsize=8)

    # Panel (b): Phase sensitivity vs N
    ax = axes[1]
    N_arr = np.array(N_list, dtype=float)
    ax.loglog(N_arr, sens_vs_n["SQL"], "o--", color=rc.PALETTE["gray"], label=r"Coherent Spin State ($1/\sqrt{N}$ SQL)")
    ax.loglog(N_arr, sens_vs_n["Dephased"], "s-", color=rc.PALETTE["blue"], label="Dephased Spin Wave (Cirq Channel)")
    ax.loglog(N_arr, sens_vs_n["W"], "^-", color=rc.PALETTE["green"], label=r"Symmetric $|W\rangle$ State")
    ax.loglog(N_arr, sens_vs_n["GHZ"], "d-", color=rc.PALETTE["red"], label=r"GHZ State ($1/N$ Heisenberg Limit)")
    ax.set_title(r"(b) Cramer-Rao Phase Sensitivity $\Delta\phi_{\min} = 1/\sqrt{F_Q}$")
    ax.set_xlabel("Atom Number $N$")
    ax.set_ylabel(r"Minimum Phase Uncertainty $\Delta\phi_{\min}$ (rad)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel (c): Metrological gain F_Q / N
    ax = axes[2]
    ax.axhline(1.0, color=rc.PALETTE["black"], ls=":", lw=1.8, label="SQL Threshold ($F_Q / N = 1$)")
    ax.plot(N_arr, fq_over_n["SQL"], "o--", color=rc.PALETTE["gray"], label="Coherent Spin State")
    ax.plot(N_arr, fq_over_n["Dephased"], "s-", color=rc.PALETTE["blue"], label="Dephased Spin Wave")
    ax.plot(N_arr, fq_over_n["W"], "^-", color=rc.PALETTE["green"], label=r"Symmetric $|W\rangle$ ($3 - 2/N$)")
    ax.plot(N_arr, fq_over_n["GHZ"], "d-", color=rc.PALETTE["red"], label="GHZ State ($N$)")
    ax.set_title("(c) Multipartite Entanglement Witness ($F_Q / N > 1$)")
    ax.set_xlabel("Atom Number $N$")
    ax.set_ylabel(r"Normalized Quantum Fisher Information $F_Q / N$")
    ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Google Cirq Ramsey Interferometry & Quantum Fisher Information (Phys. Rev. A 106, L051701)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    rc.plotting.save_figure(fig, __file__, "cirq_ramsey_metrology.png")


if __name__ == "__main__":
    main()
