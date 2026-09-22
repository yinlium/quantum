"""
A physically-motivated ``cirq.NoiseModel`` for trapped Rydberg qubits.

Three decoherence mechanisms dominate the experiments of
Phys. Rev. Lett. 128, 123601 and Phys. Rev. A 106, L051701:

======================  =======================================  ====================
Mechanism               Physical origin                          Cirq primitive
======================  =======================================  ====================
Rydberg decay (T1)      spontaneous + blackbody |r> -> |g>       ``amplitude_damp``
Laser phase noise (T2)  finite excitation-laser linewidth        ``phase_damp``
Doppler / collective    residual thermal motion, common-mode     custom Kraus channel
======================  =======================================  ====================

:class:`RydbergNoiseModel` injects these after every moment, with probabilities
derived from a per-moment gate duration so the decay is *time-based* rather than
an arbitrary per-gate error rate.
"""

from __future__ import annotations

import numpy as np
import cirq

from .channels import CollectiveDephasingChannel

__all__ = ["RydbergNoiseModel"]


class RydbergNoiseModel(cirq.NoiseModel):
    """Time-based decoherence for a trapped Rydberg ensemble.

    Args:
        t1: Rydberg state lifetime (same time units as ``moment_duration``).
            ``n = 50`` at 300 K has ``T1 ~ 100 us``.  ``None`` disables decay.
        t2_star: laser/dephasing coherence time driving single-atom phase damping.
            ``None`` disables it.
        moment_duration: physical duration assigned to one circuit moment.
        gamma_collective: rate of *correlated* dephasing across the whole ensemble
            (Doppler / common-mode laser phase).  ``None`` or 0 disables it.
        include_collective: whether to append the ensemble-wide correlated channel.
            Requires a density-matrix simulation over all qubits at once.
        prepend: if ``True`` noise is applied *before* the moment instead of after.

    Example:
        >>> import cirq
        >>> from rydberg_cirq.noise import RydbergNoiseModel
        >>> q = cirq.LineQubit.range(2)
        >>> noise = RydbergNoiseModel(t1=100.0, t2_star=3.1, moment_duration=0.5)
        >>> circuit = cirq.Circuit(cirq.H(q[0]), cirq.CZ(*q))
        >>> rho = cirq.DensityMatrixSimulator(noise=noise).simulate(circuit)
        >>> bool(abs(rho.final_density_matrix.trace().real - 1.0) < 1e-6)
        True
    """

    def __init__(
        self,
        t1: float | None = None,
        t2_star: float | None = None,
        moment_duration: float = 1.0,
        gamma_collective: float | None = None,
        include_collective: bool = False,
        prepend: bool = False,
    ):
        self.t1 = t1
        self.t2_star = t2_star
        self.moment_duration = float(moment_duration)
        self.gamma_collective = gamma_collective
        self.include_collective = include_collective
        self.prepend = prepend

    # -------------------------------------------------------------- derived rates

    @property
    def p_amplitude_damp(self) -> float:
        """``1 - exp(-dt / T1)`` -- probability of a Rydberg decay per moment."""
        if not self.t1:
            return 0.0
        return 1.0 - np.exp(-self.moment_duration / self.t1)

    @property
    def p_phase_damp(self) -> float:
        """Phase-damping probability reproducing a coherence decay ``exp(-dt / T2*)``.

        ``cirq.phase_damp(p)`` multiplies off-diagonal elements by ``sqrt(1 - p)``,
        so matching ``exp(-dt / T2*)`` requires ``p = 1 - exp(-2 dt / T2*)``.
        """
        if not self.t2_star:
            return 0.0
        return 1.0 - np.exp(-2.0 * self.moment_duration / self.t2_star)

    # ------------------------------------------------------------------ interface

    def noisy_moment(self, moment: cirq.Moment, system_qubits):
        # Never add noise on top of noise, or around measurements.
        if self.is_virtual_moment(moment):
            return moment

        noise_moments = []

        p_amp = self.p_amplitude_damp
        if p_amp > 0.0:
            noise_moments.append(
                cirq.Moment(cirq.amplitude_damp(p_amp).on_each(*system_qubits))
            )

        p_ph = self.p_phase_damp
        if p_ph > 0.0:
            noise_moments.append(cirq.Moment(cirq.phase_damp(p_ph).on_each(*system_qubits)))

        if self.include_collective and self.gamma_collective:
            channel = CollectiveDephasingChannel(
                len(system_qubits), self.gamma_collective, self.moment_duration
            )
            noise_moments.append(cirq.Moment(channel.on(*sorted(system_qubits))))

        if not noise_moments:
            return moment

        # Tag the injected moments so nested application is a no-op.
        tagged = [
            cirq.Moment(op.with_tags(cirq.VirtualTag()) for op in m.operations)
            for m in noise_moments
        ]
        return tagged + [moment] if self.prepend else [moment] + tagged

    def __repr__(self) -> str:
        return (
            f"RydbergNoiseModel(t1={self.t1!r}, t2_star={self.t2_star!r}, "
            f"moment_duration={self.moment_duration!r}, "
            f"gamma_collective={self.gamma_collective!r})"
        )
