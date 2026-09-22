"""
Hardware-aware Rydberg blockade constraints with a custom ``cirq.Device``.

Reference
---------
Y. Mei, Y. Li, H. Nguyen, P. R. Berman, and A. Kuzmich,
*Trapped Alkali-Metal Rydberg Qubit*, Phys. Rev. Lett. **128**, 123601 (2022).

Physics
-------
A neutral-atom quantum processor is defined by one hard geometric constraint: two
atoms can be entangled **only** if the van der Waals shift of the doubly excited
pair state exceeds the laser coupling that would populate it,

    V(R) = C6 / R^6  >>  Omega ,

which defines the *blockade radius*

    R_b = (C6 / hbar Omega)^(1/6) .

With the realistic n = 50 coefficient ``C6/h = 15.44 GHz um^6`` and a single-atom
Rabi frequency ``Omega_1/2pi = 1 MHz`` this gives ``R_b = 4.99 um`` -- a few lattice
sites of a typical tweezer array.  Inside a single blockade volume the ensemble
behaves as one *superatom*: the laser can only ever create the single symmetric
excitation

    |W> = N^(-1/2) sum_j |g ... r_j ... g>

and does so at the collectively enhanced rate ``Omega_N = sqrt(N) Omega_1``
(PRL 128, 123601).  If some pairs fall *outside* ``R_b`` the blockade fails for
those pairs, the ``m >= 2`` sectors come back into resonance, and the superatom
qubit leaks out of its two-level subspace.  That is precisely the failure mode the
device model is designed to catch *before* the circuit is ever simulated.

Cirq APIs showcased
-------------------
* :class:`rydberg_cirq.RydbergTweezerDevice` -- a ``cirq.Device`` subclass whose
  ``validate_operation`` enforces the blockade radius, the two-atom arity limit and
  the native gateset (``rydberg_cirq.RYDBERG_NATIVE_GATESET``).
* ``device.metadata`` (:class:`cirq.DeviceMetadata`) and ``device.connectivity_graph``
  (a ``networkx.Graph``) -- the blockade graph, analysed with ``networkx`` to show
  how connectivity *percolates* as ``R_b`` grows relative to the lattice spacing.
* Composite ensemble gates ``rydberg_cirq.CollectiveLaserDriveStep`` /
  ``PairwisePhaseGate`` validated *through their decomposition*, so an illegal
  long-range ``cirq.CZPowGate`` hidden inside an ensemble gate is still rejected.
* ``cirq.Simulator.simulate_moment_steps`` for a Strang-Trotterised evolution under
  the *position-dependent* interaction matrix, cross-checked against a dense
  ``scipy.linalg.expm`` propagator.

Run:  python manybody/cirq_device_blockade.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.linalg as sla
import cirq

import rydberg_cirq as rc
from rydberg_cirq import plotting as rplt  # submodule is not re-exported by the package __init__

# --------------------------------------------------------------------------- units
# All frequencies are angular (rad/us); distances are micrometres.
MHZ = 2.0 * np.pi  # 1 MHz  ->  rad/us
UM = 1.0

#: van der Waals coefficient of the ``n = 50`` S Rydberg level, C6/h in MHz um^6.
C6_OVER_H_MHZ = 15.44e3
#: ...converted to angular frequency units (rad/us) x um^6.
C6 = C6_OVER_H_MHZ * MHZ

#: Single-atom Rabi frequency of the two-photon Rydberg drive.
OMEGA_1 = 1.0 * MHZ


def blockade_radius(omega: float, c6: float = C6) -> float:
    """``R_b = (C6 / Omega)^(1/6)``: separation at which ``V(R) = Omega``."""
    return float((c6 / omega) ** (1.0 / 6.0))


def rabi_for_radius(r_b: float, c6: float = C6) -> float:
    """Inverse of :func:`blockade_radius`: the drive strength giving this ``R_b``."""
    return float(c6 / r_b**6)


R_B = blockade_radius(OMEGA_1)  # 4.99 um


# ------------------------------------------------------------------ local helpers
def pair_interaction_matrix(positions: np.ndarray, c6: float = C6) -> np.ndarray:
    """Upper-triangular ``V_jk = C6 / R_jk^6`` (rad/us) for a set of atom positions.

    Local helper (not part of ``rydberg_cirq``): the shared package exposes only a
    *uniform* ``RydbergBlockadeStep``; here every pair needs its own shift.
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    v = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    v[iu] = c6 / d[iu] ** 6
    return v


def graph_metrics(device: rc.RydbergTweezerDevice) -> dict:
    """Percolation diagnostics of the blockade graph, via ``networkx``."""
    g = device.connectivity_graph
    n = g.number_of_nodes()
    components = list(nx.connected_components(g))
    largest = max((len(c) for c in components), default=0)
    degrees = [d for _, d in g.degree()]
    return {
        "edges": g.number_of_edges(),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "components": len(components),
        "largest_frac": largest / n,
        "connected": nx.is_connected(g) if n else False,
    }


def trotter_superatom_evolution(
    positions: np.ndarray,
    omega_1: float = OMEGA_1,
    t_max: float | None = None,
    n_out: int = 60,
    max_phase_per_step: float = 0.05,
):
    """Strang-Trotterised Cirq evolution of a driven, interacting atom cluster.

    Hamiltonian (``|0> = |g>``, ``|1> = |r>``, ``n_j = |r><r|_j``)::

        H = (Omega_1 / 2) sum_j X_j  +  sum_{j<k} V_jk n_j n_k

    Each Trotter step is
    ``Drive(dt/2) . PairwisePhase(V dt) . Drive(dt/2)`` built from
    :class:`rydberg_cirq.CollectiveLaserDriveStep` (-> ``cirq.XPowGate``) and
    :class:`rydberg_cirq.PairwisePhaseGate` (-> ``cirq.CZPowGate``), i.e. entirely
    out of the package's composite gates.  ``dt`` is chosen so the largest pair phase
    per step is ``max_phase_per_step``.

    Returns:
        ``(times, states, n_steps)`` with ``states`` of shape ``(n_out, 2**N)``.
    """
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    v_mat = pair_interaction_matrix(positions)
    omega_n = np.sqrt(n_atoms) * omega_1
    t_max = 2.0 * np.pi / omega_n if t_max is None else t_max

    # Resolve the fastest phase in the problem.
    fastest = max(v_mat.max(), omega_n)
    n_steps = int(np.ceil(t_max * fastest / max_phase_per_step))
    n_steps = int(np.ceil(max(n_steps, 2 * n_out) / n_out) * n_out)
    dt = t_max / n_steps

    qubits = cirq.LineQubit.range(n_atoms)
    circuit = cirq.Circuit()
    for _ in range(n_steps):
        circuit.append(rc.CollectiveLaserDriveStep(n_atoms, omega_1, dt / 2).on(*qubits))
        circuit.append(rc.PairwisePhaseGate(v_mat * dt).on(*qubits))
        circuit.append(rc.CollectiveLaserDriveStep(n_atoms, omega_1, dt / 2).on(*qubits))

    moments_per_sample = len(circuit) // n_out
    sim = cirq.Simulator(dtype=np.complex128)
    states = []
    for i, step in enumerate(sim.simulate_moment_steps(circuit)):
        if (i + 1) % moments_per_sample == 0 and len(states) < n_out:
            states.append(step.state_vector(copy=True))

    times = np.arange(1, n_out + 1) * (t_max / n_out)
    times = np.concatenate([[0.0], times])
    psi0 = np.zeros(2**n_atoms, dtype=complex)
    psi0[0] = 1.0
    states = np.vstack([psi0[None, :], np.array(states)])
    return times, states, n_steps


def exact_evolution(positions: np.ndarray, t: float, omega_1: float = OMEGA_1) -> np.ndarray:
    """Dense reference propagator ``exp(-i H t)|g...g>`` via ``scipy.linalg.expm``."""
    positions = np.asarray(positions, dtype=float)
    n_atoms = len(positions)
    v_mat = pair_interaction_matrix(positions)
    dim = 2**n_atoms
    weights = rc.hamming_weights(n_atoms)

    # Diagonal interaction term.
    diag = np.zeros(dim)
    bits = ((np.arange(dim)[:, None] >> np.arange(n_atoms - 1, -1, -1)[None, :]) & 1).astype(float)
    for j in range(n_atoms):
        for k in range(j + 1, n_atoms):
            diag += v_mat[j, k] * bits[:, j] * bits[:, k]
    ham = np.diag(diag).astype(complex)

    # Transverse drive term (Omega_1 / 2) sum_j X_j.
    x_single = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    for j in range(n_atoms):
        op = np.array([[1.0]], dtype=complex)
        for k in range(n_atoms):
            op = np.kron(op, x_single if k == j else np.eye(2, dtype=complex))
        ham += 0.5 * omega_1 * op

    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0
    del weights
    return sla.expm(-1j * ham * t) @ psi0


def sector_populations(states: np.ndarray, n_atoms: int) -> dict:
    """``P(|W>)``, ``P(m = 1)`` and the ``P(m >= 2)`` leakage of each sampled state."""
    masks = rc.excitation_masks(n_atoms)
    w = rc.w_state_vector(n_atoms)
    return {
        "p_w": np.abs(states @ w.conj()) ** 2,
        "p_m1": np.sum(np.abs(states[:, masks["m1"]]) ** 2, axis=1),
        "p_multi": np.sum(np.abs(states[:, masks["multi"]]) ** 2, axis=1),
    }


# --------------------------------------------------------------------------- study
LATTICE_SPACING = 3.5 * UM
ARRAY_SIDE = 5
GRAPH_RATIOS = (1.05, 1.45, 2.10)  # R_b / a shown in the top row of the figure

#: Three four-atom clusters with identical 2 um nearest-neighbour spacing but very
#: different *global* geometry -- hence very different blockade connectivity.
CLUSTERS = {
    "square": {
        "label": "Fully blockaded\n(2 um square)",
        "positions": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]]),
        "color": rplt.PALETTE["blue"],
    },
    "chain": {
        "label": "Partially blockaded\n(2 um chain)",
        "positions": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]),
        "color": rplt.PALETTE["orange"],
    },
    "dimers": {
        "label": "Broken blockade\n(two split dimers)",
        "positions": np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [12.0, 0.0, 0.0], [14.0, 0.0, 0.0]]),
        "color": rplt.PALETTE["red"],
    },
}


def report_header() -> None:
    print("=" * 86)
    print("Rydberg blockade as a cirq.Device constraint")
    print("Phys. Rev. Lett. 128, 123601 (2022) -- Trapped Alkali-Metal Rydberg Qubit")
    print("=" * 86)
    print(f"  C6 / h                = {C6_OVER_H_MHZ / 1e3:.2f} GHz um^6   (n = 50 S state)")
    print(f"  Omega_1 / 2pi         = {OMEGA_1 / MHZ:.2f} MHz")
    print(f"  Blockade radius R_b   = (C6/Omega_1)^(1/6) = {R_B:.3f} um")
    print(f"  Lattice spacing a     = {LATTICE_SPACING:.2f} um  ->  R_b / a = {R_B / LATTICE_SPACING:.2f}")


def study_connectivity():
    """Blockade-graph percolation of a square tweezer array vs ``R_b``."""
    print("\n" + "-" * 86)
    print("[1] Blockade connectivity graph of a %dx%d tweezer array (networkx)" % (ARRAY_SIDE, ARRAY_SIDE))
    print("-" * 86)

    ratios = np.linspace(0.6, 2.6, 81)
    metrics = []
    for ratio in ratios:
        dev = rc.RydbergTweezerDevice.square_array(
            ARRAY_SIDE, spacing=LATTICE_SPACING, blockade_radius=ratio * LATTICE_SPACING
        )
        metrics.append(graph_metrics(dev))

    print(f"  {'R_b/a':>7} {'R_b[um]':>9} {'Omega/2pi[MHz]':>15} {'edges':>7} {'<deg>':>7} "
          f"{'#comp':>6} {'largest':>8}  connected")
    for ratio in (0.80, *GRAPH_RATIOS, 2.40):
        r_b = ratio * LATTICE_SPACING
        dev = rc.RydbergTweezerDevice.square_array(ARRAY_SIDE, spacing=LATTICE_SPACING, blockade_radius=r_b)
        m = graph_metrics(dev)
        print(f"  {ratio:7.2f} {r_b:9.2f} {rabi_for_radius(r_b) / MHZ:15.3f} {m['edges']:7d} "
              f"{m['mean_degree']:7.2f} {m['components']:6d} {m['largest_frac']:8.2f}"
              f"  {'YES' if m['connected'] else 'no'}")
    print("  (R_b is tuned experimentally through the drive strength: R_b = (C6/Omega)^(1/6),")
    print("   so a *weaker* laser blockades a *larger* volume and the graph percolates.)")

    dev_nom = rc.RydbergTweezerDevice.square_array(
        ARRAY_SIDE, spacing=LATTICE_SPACING, blockade_radius=R_B
    )
    print(f"\n  Nominal device: {dev_nom}")
    print(f"  cirq.DeviceMetadata: {len(dev_nom.metadata.qubit_set)} qubits, "
          f"{dev_nom.metadata.nx_graph.number_of_edges()} entangling edges")
    return ratios, metrics, dev_nom


def study_validation(dev: rc.RydbergTweezerDevice) -> None:
    """``validate_operation``: what the hardware accepts and what it refuses."""
    print("\n" + "-" * 86)
    print("[2] cirq.Device.validate_operation -- legal vs illegal operations")
    print("-" * 86)
    qs = dev.qubits
    near_a, near_b = qs[0], qs[1]            # nearest neighbours, R = a
    far_a, far_b = qs[0], qs[-1]             # opposite corners

    cases = [
        ("nearest-neighbour CZ (R = %.2f um)" % dev.distance(near_a, near_b),
         cirq.CZ(near_a, near_b)),
        ("diagonal CZ**0.37 (R = %.2f um)" % dev.distance(qs[0], qs[ARRAY_SIDE + 1]),
         cirq.CZ(qs[0], qs[ARRAY_SIDE + 1]) ** 0.37),
        ("single-atom PhasedXZ (always native)",
         cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0.1, axis_phase_exponent=0.2).on(near_a)),
        ("corner-to-corner CZ (R = %.2f um > R_b)" % dev.distance(far_a, far_b),
         cirq.CZ(far_a, far_b)),
        ("three-atom CCZ (non-native, arity 3)", cirq.CCZ(qs[0], qs[1], qs[2])),
        ("two-atom ISWAP (non-native gate)", cirq.ISWAP(near_a, near_b)),
        ("three-atom identity (arity > 2)", cirq.IdentityGate(3).on(qs[0], qs[1], qs[2])),
    ]
    for label, op in cases:
        try:
            dev.validate_operation(op)
            print(f"  ACCEPTED  {label}")
        except ValueError as err:
            msg = " ".join(str(err).split())
            print(f"  REJECTED  {label}")
            print(f"            ValueError: {msg}")


def study_cluster_devices() -> dict:
    """Composite ensemble gates are validated through their decomposition."""
    print("\n" + "-" * 86)
    print("[3] Ensemble composite gates validated through cirq decomposition")
    print("-" * 86)
    devices = {}
    for key, spec in CLUSTERS.items():
        pos = spec["positions"]
        dev = rc.RydbergTweezerDevice.from_cloud(pos, R_B)
        devices[key] = dev
        v_mat = pair_interaction_matrix(pos)
        iu = np.triu_indices(len(pos), 1)
        v_pairs = v_mat[iu] / MHZ
        dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)[iu]
        omega_n = np.sqrt(len(pos)) * OMEGA_1

        n_pairs = len(v_pairs)
        n_blockaded = int(np.sum(dists <= R_B))
        print(f"\n  {key:>7}: {dev}")
        print(f"           pair separations [um] : {np.array2string(dists, precision=2)}")
        print(f"           V/2pi [MHz]           : {np.array2string(v_pairs, precision=3)}")
        print(f"           V / Omega_N           : {np.array2string(v_mat[iu] / omega_n, precision=2)}")
        print(f"           blockaded pairs       : {n_blockaded}/{n_pairs}")

        q = dev.qubits
        # The uniform ensemble blockade step touches *every* pair -> it can only be
        # scheduled on a cluster whose atoms are mutually inside R_b.
        op = rc.RydbergBlockadeStep(len(pos), 40.0 * MHZ, 0.01).on(*q)
        try:
            dev.validate_operation(op)
            print("           RydbergBlockadeStep   : ACCEPTED (all pairs inside R_b)")
        except ValueError as err:
            msg = " ".join(str(err).split())
            print(f"           RydbergBlockadeStep   : REJECTED -> {msg[:150]}")
        # A collective drive is a product of single-atom rotations: always legal.
        drive = rc.CollectiveLaserDriveStep(len(pos), OMEGA_1, 0.01).on(*q)
        dev.validate_operation(drive)
        print("           CollectiveLaserDrive  : ACCEPTED (single-atom rotations)")
    return devices


def study_superatom_physics() -> dict:
    """|W> preparation on fully vs partially blockaded clusters."""
    print("\n" + "-" * 86)
    print("[4] Consequence for the physics: collective pi-pulse into |W>")
    print("-" * 86)
    results = {}
    n_atoms = 4
    omega_n = np.sqrt(n_atoms) * OMEGA_1
    t_pi = np.pi / omega_n
    print(f"  N = {n_atoms}, Omega_N = sqrt(N) Omega_1 = {omega_n / MHZ:.2f} x 2pi MHz, "
          f"collective pi-time t_pi = {t_pi * 1e3:.1f} ns")

    for key, spec in CLUSTERS.items():
        t0 = time.time()
        times, states, n_steps = trotter_superatom_evolution(spec["positions"])
        pops = sector_populations(states, n_atoms)
        i_pi = int(np.argmin(np.abs(times - t_pi)))
        i_best = int(np.argmax(pops["p_w"]))

        # Validate the Trotter circuit against a dense matrix exponential.
        psi_exact = exact_evolution(spec["positions"], times[i_pi])
        overlap = abs(np.vdot(psi_exact, states[i_pi])) ** 2

        results[key] = {
            "times": times,
            "p_w": pops["p_w"],
            "p_multi": pops["p_multi"],
            "p_w_pi": float(pops["p_w"][i_pi]),
            "leak_pi": float(pops["p_multi"][i_pi]),
            "p_w_best": float(pops["p_w"][i_best]),
            "leak_best": float(pops["p_multi"][i_best]),
            "leak_max": float(pops["p_multi"].max()),
            "trotter_fidelity": float(overlap),
            "n_steps": n_steps,
        }
        r = results[key]
        print(f"\n  {key:>7}  ({n_steps} Trotter steps, {time.time() - t0:.1f} s)")
        print(f"           F(|W>) at t_pi        = {r['p_w_pi'] * 100:7.3f} %")
        print(f"           leakage P(m>=2) @t_pi = {r['leak_pi'] * 100:7.3f} %")
        print(f"           best F(|W>) over t    = {r['p_w_best'] * 100:7.3f} %")
        print(f"           worst-case P(m>=2)    = {r['leak_max'] * 100:7.3f} %")
        print(f"           Trotter vs expm       = {r['trotter_fidelity']:.8f} (state overlap)")

    fid_ratio = results["square"]["p_w_pi"] / max(results["dimers"]["p_w_pi"], 1e-12)
    print(f"\n  => the fully blockaded square reaches {results['square']['p_w_pi'] * 100:.2f} % |W> fidelity")
    print(f"     with only {results['square']['leak_pi'] * 100:.3f} % leakage, while the split-dimer cluster")
    print(f"     manages {results['dimers']['p_w_pi'] * 100:.2f} % ({fid_ratio:.1f}x worse) and leaks "
          f"{results['dimers']['leak_pi'] * 100:.1f} % into m >= 2.")
    print("     The two clusters have identical nearest-neighbour spacing: only the")
    print("     blockade *graph* differs -- exactly what the cirq.Device refuses to schedule.")
    return results


# ---------------------------------------------------------------------- figure
def make_figure(ratios, metrics, clusters_result, devices) -> str:
    rplt.apply_style()
    fig = plt.figure(figsize=(15.0, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.34, wspace=0.26)

    # ---------------- (a) blockade graphs at three R_b values
    for i, ratio in enumerate(GRAPH_RATIOS):
        ax = fig.add_subplot(gs[0, i])
        r_b = ratio * LATTICE_SPACING
        dev = rc.RydbergTweezerDevice.square_array(
            ARRAY_SIDE, spacing=LATTICE_SPACING, blockade_radius=r_b
        )
        g = dev.connectivity_graph
        pos = {q: dev.position(q)[:2] for q in dev.qubits}
        m = graph_metrics(dev)
        colour = rplt.SERIES_COLORS[i]

        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=colour, width=1.6, alpha=0.75)
        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_color="white", edgecolors="black", node_size=110, linewidths=1.2
        )
        # Blockade disc around the central atom.
        centre = dev.position(dev.qubits[len(dev.qubits) // 2])[:2]
        ax.add_patch(plt.Circle(centre, r_b, fill=True, color=colour, alpha=0.10))
        ax.add_patch(plt.Circle(centre, r_b, fill=False, color=colour, ls="--", lw=1.4))

        ax.set_aspect("equal")
        ax.set_xlim(-r_b * 0.55 - 1, (ARRAY_SIDE - 1) * LATTICE_SPACING + r_b * 0.55 + 1)
        ax.set_ylim(-r_b * 0.55 - 1, (ARRAY_SIDE - 1) * LATTICE_SPACING + r_b * 0.55 + 1)
        # networkx's draw helpers switch the tick labels off; put them back.
        ax.tick_params(axis="both", which="both", bottom=True, left=True,
                       labelbottom=True, labelleft=True, labelsize=8.5)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15])
        panel = "abc"[i]
        ax.set_title(
            f"({panel}) $R_b/a = {ratio:.2f}$  ($\\Omega/2\\pi = {rabi_for_radius(r_b) / MHZ:.2f}$ MHz)\n"
            f"{m['edges']} blockaded pairs, "
            f"{'percolated' if m['connected'] else str(m['components']) + ' clusters'}",
            fontsize=10.5,
        )
        ax.set_xlabel("$x$ ($\\mu$m)")
        if i == 0:
            ax.set_ylabel("$y$ ($\\mu$m)")
        ax.grid(alpha=0.2)

    # ---------------- (d) percolation metrics
    ax = fig.add_subplot(gs[1, 0])
    edges = np.array([m["edges"] for m in metrics], dtype=float)
    largest = np.array([m["largest_frac"] for m in metrics])
    mean_deg = np.array([m["mean_degree"] for m in metrics])

    ax.plot(ratios, largest, color=rplt.PALETTE["green"], lw=2.4,
            label="largest connected component / $N$")
    ax.plot(ratios, edges / edges.max(), color=rplt.PALETTE["blue"], lw=2.0, ls="--",
            label="blockaded pairs (normalised)")
    ax.plot(ratios, mean_deg / mean_deg.max(), color=rplt.PALETTE["purple"], lw=2.0, ls=":",
            label="mean degree (normalised)")
    for i, ratio in enumerate(GRAPH_RATIOS):
        ax.axvline(ratio, color=rplt.SERIES_COLORS[i], lw=1.3, alpha=0.8)
        ax.text(ratio, 1.03, "(%s)" % "abc"[i], ha="center", fontsize=9,
                color=rplt.SERIES_COLORS[i], fontweight="bold")
    ax.axvline(1.0, color="gray", lw=1.0, ls="-.", alpha=0.7)
    ax.text(1.0, 0.06, " $R_b = a$", fontsize=8.5, color="gray")
    ax.set_xlabel("blockade radius / lattice spacing  $R_b / a$")
    ax.set_ylabel("normalised graph metric")
    ax.set_ylim(-0.04, 1.12)
    ax.set_title("(d) Percolation of the blockade graph\n($5 \\times 5$ tweezer array, "
                 f"$a = {LATTICE_SPACING:.1f}\\ \\mu$m)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5)

    # ---------------- (e) blockade shift vs separation
    ax = fig.add_subplot(gs[1, 1])
    r_grid = np.logspace(np.log10(1.2), np.log10(20.0), 400)
    v_grid = C6_OVER_H_MHZ / r_grid**6
    ax.loglog(r_grid, v_grid, color=rplt.PALETTE["black"], lw=2.4,
              label="$V(R)/h = C_6/R^6$,  $C_6/h = 15.44$ GHz $\\mu$m$^6$")
    ax.axhline(OMEGA_1 / MHZ, color=rplt.PALETTE["red"], ls="--", lw=1.8,
               label="$\\Omega_1/2\\pi = 1$ MHz")
    ax.axhline(2.0 * OMEGA_1 / MHZ, color=rplt.PALETTE["orange"], ls=":", lw=1.8,
               label="$\\Omega_N = \\sqrt{N}\\,\\Omega_1$ ($N=4$)")
    ax.axvline(R_B, color=rplt.PALETTE["blue"], ls="-.", lw=1.8,
               label=f"$R_b = (C_6/\\Omega_1)^{{1/6}} = {R_B:.2f}\\ \\mu$m")
    ax.fill_betweenx([1e-4, 1e6], 0.5, R_B, color=rplt.PALETTE["blue"], alpha=0.08)
    ax.text(1.35, 3e-4, "blockaded\n($V > \\Omega$)", fontsize=9,
            color=rplt.PALETTE["blue"], fontweight="bold")
    ax.text(7.0, 3e-4, "independent atoms\n($V < \\Omega$)", fontsize=9,
            color=rplt.PALETTE["gray"], fontweight="bold")

    marks = [(2.0, "cluster n.n."), (2.83, "square diag."), (LATTICE_SPACING, "lattice $a$"),
             (6.0, "chain ends"), (12.0, "dimer gap")]
    for r_val, lab in marks:
        ax.plot([r_val], [C6_OVER_H_MHZ / r_val**6], "o", ms=6,
                color=rplt.PALETTE["purple"], zorder=5)
        ax.annotate(lab, xy=(r_val, C6_OVER_H_MHZ / r_val**6), xytext=(5, 7),
                    textcoords="offset points", fontsize=8, color=rplt.PALETTE["purple"])
    ax.text(3.9, 1.5e3, "blockaded\n($V > \\Omega$)", fontsize=9, ha="right",
            color=rplt.PALETTE["blue"], fontweight="bold")
    ax.text(6.2, 1.5e3, "independent\natoms ($V < \\Omega$)", fontsize=9,
            color=rplt.PALETTE["gray"], fontweight="bold")
    ax.set_xlim(1.2, 20.0)
    ax.set_ylim(1e-4, 1e5)
    ax.set_xticks([1.5, 2, 3, 5, 7, 10, 15, 20])
    ax.set_xticklabels(["1.5", "2", "3", "5", "7", "10", "15", "20"])
    ax.minorticks_off()
    ax.set_xlabel("interatomic separation $R$ ($\\mu$m)")
    ax.set_ylabel("van der Waals shift $V/h$ (MHz)")
    ax.set_title("(e) Blockade shift sets the entangling range\n($n = 50$ Rydberg level)", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8)

    # inset: the three four-atom clusters and their blockade graphs
    inset = ax.inset_axes([0.05, 0.06, 0.45, 0.30])
    for row, (key, spec) in enumerate(CLUSTERS.items()):
        pos = spec["positions"]
        dev = devices[key]
        xs = pos[:, 0]
        ys = (2 - row) * 3.0 + pos[:, 1] * 0.8
        for a, b in dev.connectivity_graph.edges():
            ia, ib = dev.qubits.index(a), dev.qubits.index(b)
            inset.plot([xs[ia], xs[ib]], [ys[ia], ys[ib]], color=spec["color"], lw=1.2, alpha=0.85)
        inset.plot(xs, ys, "o", ms=4, color=spec["color"])
    inset.set_xlim(-1.5, 15.5)
    inset.set_ylim(-1.2, 8.6)
    inset.set_xticks([0, 5, 10, 15])
    inset.set_yticks([])
    inset.tick_params(labelsize=6.5)
    inset.set_title("$N=4$ clusters & blockade graph ($\\mu$m)", fontsize=6.8)
    inset.grid(alpha=0.15)

    # ---------------- (f) |W> fidelity and leakage
    ax = fig.add_subplot(gs[1, 2])
    omega_n = 2.0 * OMEGA_1
    t_pi = np.pi / omega_n
    for key, spec in CLUSTERS.items():
        res = clusters_result[key]
        x = res["times"] / t_pi
        ax.plot(x, res["p_w"], color=spec["color"], lw=2.4,
                label=f"{spec['label'].splitlines()[0]}: $F_{{|W\\rangle}}$")
        ax.plot(x, res["p_multi"], color=spec["color"], lw=1.6, ls="--", alpha=0.85,
                label="     leakage $P(m\\geq 2)$")
    ax.axvline(1.0, color="gray", ls=":", lw=1.4)
    ax.text(1.02, 1.16, "collective $\\pi$-pulse", rotation=90, fontsize=8, color="gray",
            va="bottom")
    ax.set_xlabel("time  $t / t_\\pi$,   $t_\\pi = \\pi/\\Omega_N$")
    ax.set_ylabel("population")
    ax.set_ylim(-0.04, 1.62)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("(f) Superatom $|W\\rangle$ preparation, $N = 4$\n"
                 "identical 2 $\\mu$m n.n. spacing, different blockade graph", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=7.2, ncol=2, framealpha=0.95, columnspacing=0.9,
              handlelength=1.6)

    rows = "\n".join(
        f"{spec['label'].splitlines()[0]:<20s} {clusters_result[key]['p_w_pi'] * 100:6.2f} "
        f"{clusters_result[key]['leak_pi'] * 100:7.2f}"
        for key, spec in CLUSTERS.items()
    )
    ax.text(
        0.03, 0.30,
        "at $t_\\pi$:              $F_{|W\\rangle}$ [%]  $P(m\\geq2)$ [%]\n" + rows,
        transform=ax.transAxes, fontsize=7.6, family="monospace", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=rplt.PALETTE["gray"], lw=1.0, alpha=0.95),
    )

    fig.suptitle(
        "Rydberg blockade radius as a hardware constraint: a custom $\\mathtt{cirq.Device}$ for a tweezer array\n"
        "Phys. Rev. Lett. 128, 123601 (2022) -- Trapped Alkali-Metal Rydberg Qubit",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    return rplt.save_figure(fig, __file__, "cirq_prl2022_device_blockade.png")


def main() -> None:
    t_start = time.time()
    report_header()
    ratios, metrics, dev_nom = study_connectivity()
    study_validation(dev_nom)
    devices = study_cluster_devices()
    results = study_superatom_physics()

    print("\n" + "-" * 86)
    print("[5] Figure")
    print("-" * 86)
    make_figure(ratios, metrics, results, devices)
    print(f"\nTotal runtime: {time.time() - t_start:.1f} s")
    print("=" * 86)


if __name__ == "__main__":
    main()
