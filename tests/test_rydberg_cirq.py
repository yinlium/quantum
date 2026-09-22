"""
Validation suite for :mod:`rydberg_cirq`.

Philosophy: every Cirq construct is checked against an *independent* reference --
either a closed-form analytic result or a direct NumPy computation -- so that the
Cirq layer is verified rather than merely exercised.

Run with::

    .venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import cirq

import rydberg_cirq as rc


# --------------------------------------------------------------------------- #
# Conventions: the collective spin algebra
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_angular_momentum_algebra(N):
    """The PauliSum spin operators must satisfy [Jx, Jy] = i Jz."""
    q = cirq.LineQubit.range(N)
    J = rc.collective_spin_paulisums(q)
    Jx, Jy, Jz = (J[k].matrix(q) for k in ("Jx", "Jy", "Jz"))
    np.testing.assert_allclose(Jx @ Jy - Jy @ Jx, 1j * Jz, atol=1e-12)
    np.testing.assert_allclose(Jy @ Jz - Jz @ Jy, 1j * Jx, atol=1e-12)
    np.testing.assert_allclose(Jz @ Jx - Jx @ Jz, 1j * Jy, atol=1e-12)


@pytest.mark.parametrize("N", [1, 2, 3, 5])
def test_ground_state_is_lowest_weight(N):
    """|g...g> must be the M = -N/2 eigenstate, fixing the sign of Jz."""
    q = cirq.LineQubit.range(N)
    Jz = rc.collective_spin_paulisums(q)["Jz"].matrix(q)
    psi = np.zeros(2**N, dtype=complex)
    psi[0] = 1.0
    assert np.isclose(np.real(psi.conj() @ Jz @ psi), -N / 2.0)


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_paulisum_matches_dicke_operators(N):
    """Projected onto the symmetric subspace, the PauliSums equal the Dicke matrices."""
    q = cirq.LineQubit.range(N)
    J = rc.collective_spin_paulisums(q)
    P = rc.dicke_projector(N)
    Jx_d, Jy_d, Jz_d, _ = rc.get_dicke_operators(N)
    for key, ref in (("Jx", Jx_d), ("Jy", Jy_d), ("Jz", Jz_d)):
        np.testing.assert_allclose(P @ J[key].matrix(q) @ P.conj().T, ref, atol=1e-12)


@pytest.mark.parametrize("N", [1, 2, 3, 6])
def test_dicke_states_normalised_and_orthogonal(N):
    states = [rc.dicke_state_vector(N, m) for m in range(N + 1)]
    gram = np.array([[np.vdot(a, b) for b in states] for a in states])
    np.testing.assert_allclose(gram, np.eye(N + 1), atol=1e-12)


def test_w_state_is_single_excitation_dicke_state():
    np.testing.assert_allclose(rc.w_state_vector(4), rc.dicke_state_vector(4, 1))


@pytest.mark.parametrize("N", [4, 8])
def test_hamming_weights_and_masks_agree(N):
    w = rc.hamming_weights(N)
    masks = rc.excitation_masks(N)
    np.testing.assert_array_equal(masks["m0"], w == 0)
    np.testing.assert_array_equal(masks["m1"], w == 1)
    np.testing.assert_array_equal(masks["multi"], w >= 2)
    assert int(masks["m0"].sum()) == 1
    assert int(masks["m1"].sum()) == N


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N", [1, 2, 3])
def test_collective_drive_matches_exact_exponential(N):
    """CollectiveLaserDriveStep must equal exp(-i (Omega/2) sum_j X_j dt)."""
    omega, dt = 1.3, 0.37
    q = cirq.LineQubit.range(N)
    got = cirq.unitary(cirq.Circuit(rc.CollectiveLaserDriveStep(N, omega, dt).on(*q)))

    X = cirq.unitary(cirq.X)
    H = np.zeros((2**N, 2**N), dtype=complex)
    for j in range(N):
        ops = [np.eye(2, dtype=complex)] * N
        ops[j] = X
        term = ops[0]
        for o in ops[1:]:
            term = np.kron(term, o)
        H += 0.5 * omega * term
    from scipy.linalg import expm

    want = expm(-1j * H * dt)
    assert cirq.allclose_up_to_global_phase(got, want, atol=1e-10)


@pytest.mark.parametrize("N", [2, 3, 4])
def test_blockade_step_matches_exact_exponential(N):
    """RydbergBlockadeStep must equal exp(-i V sum_{j<k} n_j n_k dt)."""
    v, dt = 2.1, 0.23
    q = cirq.LineQubit.range(N)
    got = cirq.unitary(cirq.Circuit(rc.RydbergBlockadeStep(N, v, dt).on(*q)))

    w = rc.hamming_weights(N)
    # sum_{j<k} n_j n_k = m(m-1)/2 for a basis state of weight m.
    pairs = w * (w - 1) / 2.0
    want = np.diag(np.exp(-1j * v * dt * pairs))
    assert cirq.allclose_up_to_global_phase(got, want, atol=1e-10)


def test_pairwise_phase_gate_is_exact_and_diagonal():
    """PairwisePhaseGate reproduces prod_{j<k} exp(-i Phi_jk n_j n_k) with no Trotter error."""
    rng = np.random.default_rng(7)
    N = 4
    phi = rng.uniform(0, 2 * np.pi, size=(N, N))
    phi = np.triu(phi, 1)
    phi = phi + phi.T

    q = cirq.LineQubit.range(N)
    got = cirq.unitary(cirq.Circuit(rc.PairwisePhaseGate(phi).on(*q)))

    diag = np.ones(2**N, dtype=complex)
    for s in range(2**N):
        bits = [(s >> (N - 1 - j)) & 1 for j in range(N)]
        total = sum(
            phi[j, k] for j in range(N) for k in range(j + 1, N) if bits[j] and bits[k]
        )
        diag[s] = np.exp(-1j * total)
    assert cirq.allclose_up_to_global_phase(got, np.diag(diag), atol=1e-10)


def test_storage_gate_decomposition_matches_unitary():
    g = rc.MagicLatticeStorageGate(3.7, 0.42, 0.25, 0.2, t_start=1.1)
    q = cirq.LineQubit(0)
    decomposed = cirq.unitary(cirq.Circuit(cirq.decompose_once(g.on(q))))
    assert cirq.allclose_up_to_global_phase(decomposed, cirq.unitary(g), atol=1e-12)


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("N,gamma,tau", [(2, 1.0, 0.5), (3, 0.3, 2.0), (4, 2.0, 0.1)])
def test_collective_dephasing_is_cptp(N, gamma, tau):
    K = rc.collective_dephasing_kraus(N, gamma, tau)
    total = sum(k.conj().T @ k for k in K)
    np.testing.assert_allclose(total, np.eye(2**N), atol=1e-10)


@pytest.mark.parametrize("N,gamma,tau", [(2, 1.0, 0.5), (4, 0.8, 1.3)])
def test_collective_dephasing_matches_master_equation(N, gamma, tau):
    """The channel must damp rho_{ss'} by exp(-gamma tau (m_s - m_s')^2 / 2) exactly."""
    rng = np.random.default_rng(3)
    dim = 2**N
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    rho /= np.trace(rho)

    K = rc.collective_dephasing_kraus(N, gamma, tau)
    got = sum(k @ rho @ k.conj().T for k in K)

    m = rc.hamming_weights(N).astype(float)
    damping = np.exp(-0.5 * gamma * tau * (m[:, None] - m[None, :]) ** 2)
    np.testing.assert_allclose(got, damping * rho, atol=1e-10)


@pytest.mark.parametrize("N", [3, 4])
def test_fixed_excitation_sectors_are_invariant(N):
    """Collective dephasing cannot touch a state of definite excitation number.

    This is the physical heart of Phys. Rev. A 106, L051701: |W> survives while
    superpositions across different m decohere.
    """
    K = rc.collective_dephasing_kraus(N, 1.5, 3.0)
    for m in range(N + 1):
        psi = rc.dicke_state_vector(N, m)
        rho = np.outer(psi, psi.conj())
        out = sum(k @ rho @ k.conj().T for k in K)
        np.testing.assert_allclose(out, rho, atol=1e-10)


def test_collective_dephasing_equals_random_global_phase_average():
    """Equivalence to averaging exp(-i phi Jz) with phi ~ N(0, gamma tau)."""
    N, gamma, tau = 3, 0.9, 1.1
    q = cirq.LineQubit.range(N)
    Jz = rc.collective_spin_paulisums(q)["Jz"].matrix(q)

    psi = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
    rho = np.outer(psi, psi.conj())

    rng = np.random.default_rng(11)
    phis = rng.normal(0.0, np.sqrt(gamma * tau), 200_000)
    from scipy.linalg import expm

    # Diagonalise once; Jz is diagonal in the computational basis.
    jz_diag = np.real(np.diag(Jz))
    acc = np.zeros_like(rho)
    for phi in phis:
        u = np.exp(-1j * phi * jz_diag)
        acc += (u[:, None] * rho) * u.conj()[None, :]
    acc /= len(phis)

    K = rc.collective_dephasing_kraus(N, gamma, tau)
    exact = sum(k @ rho @ k.conj().T for k in K)
    assert np.max(np.abs(acc - exact)) < 5e-3


def test_collective_dephasing_runs_in_density_matrix_simulator():
    N = 3
    q = cirq.LineQubit.range(N)
    circuit = cirq.Circuit(
        cirq.H.on_each(*q), rc.CollectiveDephasingChannel(N, 1.0, 0.7).on(*q)
    )
    rho = cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-6)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-6)
    assert np.min(np.linalg.eigvalsh(rho)) > -1e-6  # positive semi-definite


def test_zero_time_channel_is_identity():
    K = rc.collective_dephasing_kraus(3, 1.0, 0.0)
    assert len(K) == 1
    np.testing.assert_allclose(K[0], np.eye(8), atol=1e-12)


def test_rydberg_decay_channel_probability():
    ch = rc.RydbergDecayChannel(t1=100.0, tau=10.0)
    assert np.isclose(ch.probability, 1 - np.exp(-0.1))
    K = cirq.kraus(ch)
    total = sum(k.conj().T @ k for k in K)
    np.testing.assert_allclose(total, np.eye(2), atol=1e-12)


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #


def test_device_rejects_gates_outside_blockade_radius():
    dev = rc.RydbergTweezerDevice.square_array(3, spacing=1.0, blockade_radius=1.5)
    qs = dev.qubits
    dev.validate_operation(cirq.CZ(qs[0], qs[1]))  # nearest neighbour: fine
    with pytest.raises(ValueError, match="blockade radius"):
        dev.validate_operation(cirq.CZ(qs[0], qs[-1]))


def test_device_rejects_more_than_two_qubit_gates():
    dev = rc.RydbergTweezerDevice.chain(3, spacing=1.0, blockade_radius=10.0)
    qs = dev.qubits
    with pytest.raises(ValueError):
        dev.validate_operation(cirq.TOFFOLI(*qs))


def test_device_rejects_foreign_qubits():
    dev = rc.RydbergTweezerDevice.chain(2, blockade_radius=10.0)
    with pytest.raises(ValueError, match="not a tweezer site"):
        dev.validate_operation(cirq.X(cirq.LineQubit(99)))


def test_device_connectivity_graph_tracks_blockade_radius():
    tight = rc.RydbergTweezerDevice.square_array(3, spacing=1.0, blockade_radius=1.01)
    loose = rc.RydbergTweezerDevice.square_array(3, spacing=1.0, blockade_radius=5.0)
    assert tight.connectivity_graph.number_of_edges() == 12  # 4-neighbour grid
    assert loose.connectivity_graph.number_of_edges() == 36  # all 9 choose 2 pairs


def test_device_accepts_composite_gate_within_radius():
    dev = rc.RydbergTweezerDevice.chain(4, spacing=1.0, blockade_radius=10.0)
    dev.validate_operation(rc.RydbergBlockadeStep(4, 20.0, 0.1).on(*dev.qubits))


def test_device_rejects_composite_gate_spanning_too_far():
    dev = rc.RydbergTweezerDevice.chain(4, spacing=1.0, blockade_radius=1.5)
    with pytest.raises(ValueError, match="blockade radius"):
        dev.validate_operation(rc.RydbergBlockadeStep(4, 20.0, 0.1).on(*dev.qubits))


# --------------------------------------------------------------------------- #
# Noise model
# --------------------------------------------------------------------------- #


def test_noise_model_rates_match_exponential_decay():
    nm = rc.RydbergNoiseModel(t1=100.0, t2_star=4.0, moment_duration=0.5)
    assert np.isclose(nm.p_amplitude_damp, 1 - np.exp(-0.5 / 100.0))
    assert np.isclose(nm.p_phase_damp, 1 - np.exp(-2 * 0.5 / 4.0))


def test_noise_model_disabled_reproduces_noiseless_simulation():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q[0]), cirq.CZ(*q), cirq.X(q[1]))
    clean = cirq.DensityMatrixSimulator().simulate(circuit).final_density_matrix
    nm = rc.RydbergNoiseModel(t1=None, t2_star=None)
    noisy = cirq.DensityMatrixSimulator(noise=nm).simulate(circuit).final_density_matrix
    np.testing.assert_allclose(clean, noisy, atol=1e-6)


def test_noise_model_preserves_trace():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H.on_each(*q), cirq.CZ(*q))
    nm = rc.RydbergNoiseModel(t1=10.0, t2_star=2.0, moment_duration=1.0)
    rho = cirq.DensityMatrixSimulator(noise=nm).simulate(circuit).final_density_matrix
    assert np.isclose(np.trace(rho).real, 1.0, atol=1e-5)


def test_noise_model_damps_coherence():
    q = cirq.LineQubit.range(1)
    circuit = cirq.Circuit(cirq.H(q[0]))
    nm = rc.RydbergNoiseModel(t2_star=1.0, moment_duration=1.0)
    rho = cirq.DensityMatrixSimulator(noise=nm).simulate(circuit).final_density_matrix
    assert abs(rho[0, 1]) < 0.5 * 0.5  # coherence strictly reduced from 0.5


# --------------------------------------------------------------------------- #
# Transformers
# --------------------------------------------------------------------------- #


def test_dd_refocuses_static_inhomogeneous_shift():
    """A Hahn echo must perfectly undo a purely static detuning."""
    q = cirq.LineQubit(0)
    raw = cirq.Circuit(
        cirq.H(q),
        rc.MagicLatticeStorageGate(10.0, 0.5, delta_osc=0.0, omega_trap=0.0).on(q),
        cirq.H(q),
    )
    sim = cirq.Simulator()
    free = sim.simulate(raw).final_state_vector
    echo = sim.simulate(rc.compile_dynamical_decoupling(raw, num_pulses=1)).final_state_vector

    vis_free = 2 * abs(free[0]) ** 2 - 1
    vis_echo = 2 * abs(echo[0]) ** 2 - 1
    assert vis_echo > 0.999
    assert vis_free < 0.5


@pytest.mark.parametrize("num_pulses", [1, 2, 4, 8])
def test_dd_preserves_readout_basis(num_pulses):
    """Net rotation of the inserted pulse train must be the identity."""
    q = cirq.LineQubit(0)
    raw = cirq.Circuit(
        cirq.H(q),
        rc.MagicLatticeStorageGate(4.0, 0.0, delta_osc=0.0, omega_trap=0.0).on(q),
        cirq.H(q),
    )
    out = rc.compile_dynamical_decoupling(raw, num_pulses=num_pulses)
    st = cirq.Simulator().simulate(out).final_state_vector
    assert abs(st[0]) ** 2 > 0.999


def test_dd_with_zero_pulses_is_a_noop():
    q = cirq.LineQubit(0)
    raw = cirq.Circuit(cirq.H(q), rc.MagicLatticeStorageGate(5.0, 0.3).on(q), cirq.H(q))
    assert rc.compile_dynamical_decoupling(raw, num_pulses=0) == raw


def test_hardware_compilation_preserves_unitary():
    """Lowering to the native gateset must not change the physics."""
    N = 3
    q = cirq.LineQubit.range(N)
    circuit = cirq.Circuit(
        rc.CollectiveLaserDriveStep(N, 1.0, 0.3).on(*q),
        rc.RydbergBlockadeStep(N, 5.0, 0.3).on(*q),
        rc.CollectiveLaserDriveStep(N, 1.0, 0.3).on(*q),
    )
    compiled = rc.compile_to_rydberg_hardware(circuit)
    assert cirq.allclose_up_to_global_phase(
        cirq.unitary(circuit), cirq.unitary(compiled), atol=1e-8
    )


def test_circuit_stats_counts_two_qubit_gates():
    N = 4
    q = cirq.LineQubit.range(N)
    circuit = cirq.Circuit(cirq.decompose_once(rc.RydbergBlockadeStep(N, 1.0, 0.1).on(*q)))
    stats = rc.circuit_stats(circuit)
    assert stats["2q_ops"] == N * (N - 1) // 2


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_product_state_has_zero_concurrence():
    N = 4
    q = cirq.LineQubit.range(N)
    state = cirq.Simulator().simulate(cirq.Circuit(cirq.ry(0.7).on_each(*q))).final_state_vector
    obs = rc.spin_observables(state, q)
    rho12 = rc.two_atom_reduced_dm(obs, N)
    assert rc.concurrence(rho12) < 1e-6


def test_w_state_has_expected_pairwise_concurrence():
    """For |W_N> the pairwise concurrence is exactly 2/N."""
    N = 4
    q = cirq.LineQubit.range(N)
    obs = rc.spin_observables(rc.w_state_vector(N), q)
    rho12 = rc.two_atom_reduced_dm(obs, N)
    assert np.isclose(rc.concurrence(rho12), 2.0 / N, atol=1e-8)


def test_two_atom_rdm_matches_explicit_partial_trace():
    N = 4
    q = cirq.LineQubit.range(N)
    psi = rc.w_state_vector(N)
    obs = rc.spin_observables(psi, q)
    from_moments = rc.two_atom_reduced_dm(obs, N)

    rho_full = np.outer(psi, psi.conj()).reshape((2,) * (2 * N))
    traced = cirq.partial_trace(rho_full, [0, 1]).reshape(4, 4)
    np.testing.assert_allclose(from_moments, traced, atol=1e-10)


def test_two_atom_rdm_is_a_valid_density_matrix():
    N = 5
    q = cirq.LineQubit.range(N)
    obs = rc.spin_observables(rc.dicke_state_vector(N, 2), q)
    rho12 = rc.two_atom_reduced_dm(obs, N)
    assert np.isclose(np.trace(rho12).real, 1.0, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(rho12)) > -1e-10


def test_qfi_of_product_state_is_at_the_standard_quantum_limit():
    """A coherent spin state has F_Q = N, i.e. F_Q / N = 1 exactly."""
    N = 4
    q = cirq.LineQubit.range(N)
    state = cirq.Simulator().simulate(
        cirq.Circuit(cirq.ry(np.pi / 2).on_each(*q))
    ).final_state_vector
    rho = np.outer(state, np.conj(state))
    Jz = rc.collective_spin_paulisums(q)["Jz"].matrix(q)
    assert np.isclose(rc.quantum_fisher_information(rho, Jz) / N, 1.0, atol=1e-6)


def test_qfi_of_ghz_state_reaches_heisenberg_limit():
    """GHZ has F_Q = N^2 with respect to Jz."""
    N = 4
    q = cirq.LineQubit.range(N)
    psi = np.zeros(2**N, dtype=complex)
    psi[0] = psi[-1] = 1 / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    Jz = rc.collective_spin_paulisums(q)["Jz"].matrix(q)
    assert np.isclose(rc.quantum_fisher_information(rho, Jz), N**2, atol=1e-6)


def test_qfi_of_pure_state_equals_four_times_variance():
    N = 3
    q = cirq.LineQubit.range(N)
    psi = cirq.Simulator().simulate(
        cirq.Circuit(cirq.ry(0.9).on_each(*q), cirq.CZ(q[0], q[1]))
    ).final_state_vector
    rho = np.outer(psi, np.conj(psi))
    Jz = rc.collective_spin_paulisums(q)["Jz"].matrix(q)
    var = np.real(psi.conj() @ Jz @ Jz @ psi) - np.real(psi.conj() @ Jz @ psi) ** 2
    assert np.isclose(rc.quantum_fisher_information(rho, Jz), 4 * var, atol=1e-6)


def test_coherent_spin_state_is_unsqueezed():
    """xi_R^2 = 1 exactly for a coherent spin state."""
    N = 6
    q = cirq.LineQubit.range(N)
    state = cirq.Simulator().simulate(
        cirq.Circuit(cirq.ry(np.pi / 2).on_each(*q))
    ).final_state_vector
    obs = rc.spin_observables(state, q)
    assert np.isclose(rc.squeezing_parameters(obs, N)["xi_R2"], 1.0, atol=1e-6)


def test_dicke_purity_limits():
    p = np.array([0.5, 0.5, 0.0])
    assert np.isclose(rc.dicke_purity(p), 1.0)
    p = np.array([0.5, 0.25, 0.25])
    assert np.isclose(rc.dicke_purity(p), 0.5)


def test_g2_of_poissonian_is_one():
    """With no dephasing (X_m = Y_m = 1) a Poissonian spin wave gives g2 = 1."""
    N, m_bar = 200, 1.5
    p = rc.poisson_excitation_amplitudes(N, m_bar, m_max=20)
    ones = np.ones_like(p)
    assert np.isclose(rc.g2_from_populations(p, ones, ones), 1.0, atol=2e-2)


# --------------------------------------------------------------------------- #
# Dicke / cloud helpers
# --------------------------------------------------------------------------- #


def test_poisson_amplitudes_normalised_with_correct_mean():
    N, m_bar = 270, 1.63
    p = rc.poisson_excitation_amplitudes(N, m_bar)
    assert np.isclose(p.sum(), 1.0, atol=1e-9)
    assert np.isclose((p * np.arange(len(p))).sum(), m_bar, atol=1e-6)


def test_coherent_spin_state_normalised():
    psi = rc.coherent_spin_state(10, np.pi / 3)
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_dicke_projector_is_an_isometry():
    N = 5
    P = rc.dicke_projector(N)
    np.testing.assert_allclose(P @ P.conj().T, np.eye(N + 1), atol=1e-12)


def test_phase_matrix_is_symmetric_with_zero_diagonal():
    rng = np.random.default_rng(1)
    pos = rc.sample_cloud(rc.SHORT_CLOUD, n=25, rng=rng)
    phi = rc.phase_matrix(pos, storage_time=1.0, c6=rc.SHORT_CLOUD.c6)
    np.testing.assert_allclose(phi, phi.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(phi), 0.0, atol=1e-12)
    assert np.all(np.isfinite(phi))


def test_phase_matrix_scales_linearly_with_storage_time():
    rng = np.random.default_rng(2)
    pos = rc.sample_cloud(rc.SHORT_CLOUD, n=12, rng=rng)
    a = rc.phase_matrix(pos, 1.0, c6=10.0)
    b = rc.phase_matrix(pos, 3.0, c6=10.0)
    np.testing.assert_allclose(3.0 * a, b, rtol=1e-12)


def test_coherence_eta_decays_with_storage_time():
    rng = np.random.default_rng(4)
    pos = rc.sample_cloud(rc.SHORT_CLOUD, n=60, rng=rng)
    etas = [
        rc.coherence_eta(rc.phase_matrix(pos, t, c6=rc.SHORT_CLOUD.c6))
        for t in [0.0, 0.05, 0.2, 1.0]
    ]
    assert np.isclose(etas[0], 1.0, atol=1e-12)
    assert etas[0] >= etas[1] >= etas[2]


def test_sample_cloud_reproduces_requested_widths():
    rng = np.random.default_rng(5)
    pos = rc.sample_cloud(rc.SHORT_CLOUD, n=200_000, rng=rng)
    got = pos.std(axis=0)
    np.testing.assert_allclose(got, rc.SHORT_CLOUD.sigmas, rtol=0.02)
