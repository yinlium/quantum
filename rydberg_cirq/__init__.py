"""
``rydberg_cirq`` -- a Google Cirq toolkit for trapped Rydberg atomic ensembles.

Shared library backing the reproductions of two experiments from the Kuzmich group:

* **Phys. Rev. Lett. 128, 123601 (2022)** -- *Trapped Alkali-Metal Rydberg Qubit*
  (collective ``sqrt(N)`` Rabi oscillations, magic-wavelength lattice, dynamical
  decoupling).  See the ``manybody/`` directory.
* **Phys. Rev. A 106, L051701 (2022)** -- *Dynamics of Collective-Dephasing-Induced
  Multiatom Entanglement* (interaction-induced dephasing, ``g^(2)``, Dicke-state
  purification).  See the ``entanglement/`` directory.

Module map
----------
============================  =====================================================
:mod:`~rydberg_cirq.gates`     custom ``cirq.Gate`` subclasses with ``_decompose_``
:mod:`~rydberg_cirq.channels`  CPTP ``_kraus_`` channels (collective dephasing, decay)
:mod:`~rydberg_cirq.device`    ``cirq.Device`` enforcing the blockade radius
:mod:`~rydberg_cirq.noise`     ``cirq.NoiseModel`` from physical T1/T2 times
:mod:`~rydberg_cirq.transformers``@cirq.transformer`` passes (DD, hardware lowering)
:mod:`~rydberg_cirq.metrics`   observables as ``cirq.PauliSum``; concurrence, QFI
:mod:`~rydberg_cirq.dicke`     symmetric-basis helpers and sign conventions
:mod:`~rydberg_cirq.cloud`     Gaussian cloud sampling and interaction phases
:mod:`~rydberg_cirq.plotting`  shared figure styling
============================  =====================================================
"""

from __future__ import annotations

__version__ = "0.1.0"

from .channels import (
    CollectiveDephasingChannel,
    RydbergDecayChannel,
    collective_dephasing_kraus,
    dicke_dephasing_factors,
)
from .cloud import (
    LONG_CLOUD,
    SHORT_CLOUD,
    CloudParameters,
    coherence_eta,
    phase_matrix,
    sample_cloud,
)
from .device import RYDBERG_NATIVE_GATESET, RydbergTweezerDevice
from .dicke import (
    coherent_spin_state,
    dicke_projector,
    dicke_state_vector,
    excitation_masks,
    get_dicke_operators,
    hamming_weights,
    poisson_excitation_amplitudes,
    w_state_vector,
)
from .gates import (
    CollectiveLaserDriveStep,
    MagicLatticeStorageGate,
    PairwisePhaseGate,
    RydbergBlockadeStep,
)
from .metrics import (
    collective_spin_paulisums,
    concurrence,
    dicke_purity,
    g2_from_populations,
    quantum_fisher_information,
    spin_observables,
    squeezing_parameters,
    two_atom_reduced_dm,
)
from .noise import RydbergNoiseModel
from . import plotting
from .plotting import (
    PALETTE,
    SERIES_COLORS,
    apply_style,
    figure_path,
    save_figure,
)
from .transformers import (
    circuit_stats,
    compile_dynamical_decoupling,
    compile_to_rydberg_hardware,
)

__all__ = [
    "__version__",
    # gates
    "CollectiveLaserDriveStep",
    "RydbergBlockadeStep",
    "PairwisePhaseGate",
    "MagicLatticeStorageGate",
    # channels
    "CollectiveDephasingChannel",
    "RydbergDecayChannel",
    "collective_dephasing_kraus",
    "dicke_dephasing_factors",
    # device / noise
    "RydbergTweezerDevice",
    "RYDBERG_NATIVE_GATESET",
    "RydbergNoiseModel",
    # transformers
    "compile_dynamical_decoupling",
    "compile_to_rydberg_hardware",
    "circuit_stats",
    # metrics
    "collective_spin_paulisums",
    "spin_observables",
    "squeezing_parameters",
    "two_atom_reduced_dm",
    "concurrence",
    "quantum_fisher_information",
    "dicke_purity",
    "g2_from_populations",
    # dicke
    "get_dicke_operators",
    "hamming_weights",
    "dicke_projector",
    "dicke_state_vector",
    "w_state_vector",
    "excitation_masks",
    "coherent_spin_state",
    "poisson_excitation_amplitudes",
    # cloud
    "CloudParameters",
    "SHORT_CLOUD",
    "LONG_CLOUD",
    "sample_cloud",
    "phase_matrix",
    "coherence_eta",
]
