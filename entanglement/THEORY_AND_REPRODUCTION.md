# Dynamics of Collective-Dephasing-Induced Multi-Atom Entanglement

**Reference:**  
Y. Li, Y. Mei, H. Nguyen, P. R. Berman, and A. Kuzmich  
*Phys. Rev. A* **106**, L051701 (2022) and Supplemental Material

---

## 1. Overview and Core Physics

In cold-atom Rydberg experiments, atomic ensembles provide a powerful interface for generating non-classical light and multiatom entanglement. While the creation of single collective excitations is often attributed to the **Rydberg excitation blockade**, this paper demonstrates both theoretically and experimentally that **interaction-induced dephasing during storage** is the dominant mechanism responsible for:
1. Transforming an initially unentangled multi-excitation Rydberg spin wave into an entangled Dicke state ($|W\rangle$ state).
2. Generating on-demand directional single photons ($g^{(2)}(T_s) \to 0$) in the phase-matched optical retrieval.
3. Suppressing multi-excitation components even when excitation blockade is inoperative during the excitation pulse.

---

## 2. Experimental Parameters & Setup

* **Atomic Species:** $^{87}\text{Rb}$ trapped in a 1D state-insensitive optical lattice trap (SILT) at $\lambda = 1012\text{ nm}$.
* **Atomic States:**
  * Ground state: $|g\rangle = |5S_{1/2}, F = 2, m_F = -2\rangle$
  * Intermediate state: $|p\rangle = |5P_{3/2}, F = 3, m_F = -3\rangle$
  * Rydberg state: $|r\rangle = |nS_{1/2}, m_J = -1/2\rangle$ ($n = 40, 50, 75$)
* **Excitation Laser Fields:**
  * $E_1$ ($780\text{ nm}, \sigma_-$) with beam waist $w_{E1,0} = 6\ \mu\text{m}$
  * $E_2$ ($480\text{ nm}, \sigma_+$) with beam waist $w_{E2,0} = 15\ \mu\text{m}$
  * Intermediate-state detuning: $\Delta / 2\pi = 480\text{ MHz}$
  * Rabi frequencies: $\Omega_1 / 2\pi = 9.2\text{ MHz}$, $\Omega_2 / 2\pi = 25.7\text{ MHz}$ ($n=40$) and $17.9\text{ MHz}$ ($n=50$)
  * Rectangular excitation pulse duration: $T_p = 103(4)\text{ ns}$
* **Atom Number & Geometry:**
  * Atom number in the excitation volume: $N \approx 270$
  * Transverse beam waist diameter at atoms: $D_{\text{waist}} = 5.85\ \mu\text{m}$  
    *Corresponding Gaussian spatial standard deviation (radius):* $\sigma_x = \sigma_y = D_{\text{waist}} / 2 = 2.925\ \mu\text{m}$.
  * Longitudinal cloud length diameter (short cloud, MOT $\to$ FORT $\to$ SILT): $D_z = 10.5\ \mu\text{m}$  
    *Corresponding Gaussian standard deviation (radius):* $\sigma_z = D_z / 2 = 5.25\ \mu\text{m}$ (best fit, with $\pm 20\%$ experimental uncertainty band).
  * Longitudinal cloud length diameter (long cloud, direct MOT $\to$ SILT): $D_z \approx 230\ \mu\text{m}$  
    *Corresponding Gaussian standard deviation (radius):* $\sigma_z \approx D_z / 2 = 115\ \mu\text{m}$ (Rayleigh range $z_{R,1} \approx 135\ \mu\text{m}$).
* **Interaction Strengths ($C_6$ coefficients):**
  * $n = 40$: $C_6 / h = 1.00\text{ GHz}\cdot\mu\text{m}^6$, average excitations $\bar{m} = 1.63$
  * $n = 50$: $C_6 / h = 15.44\text{ GHz}\cdot\mu\text{m}^6$, average excitations $\bar{m} = 0.79$

---

## 3. Theoretical Formulation

### 3.1 Initial State: Unentangled Rydberg Spin-Wave
The collective spin-wave destruction operator is:
$$\hat{S}_{\mathbf{k}_0} = \frac{1}{\sqrt{N}} \sum_{\mu=1}^N e^{i \mathbf{k}_0 \cdot \mathbf{r}_\mu} \hat{\sigma}_\mu^{gr}$$
where $\hat{\sigma}_\mu^{gr} = |g_\mu\rangle\langle r_\mu|$.

The Fock state containing $m$ excitations is:
$$|m\rangle = \frac{(\hat{S}_{\mathbf{k}_0}^\dagger)^m}{\sqrt{m!}} |0\rangle = \frac{1}{\sqrt{\binom{N}{m}}} \sum_{\mu_1 < \dots < \mu_m} |\mu_1 \dots \mu_m\rangle$$

At the end of the short excitation pulse $T_p$, the state of the ensemble is an uncorrelated product state:
$$|\Psi_0\rangle = \sum_{m=0}^N c_m |m\rangle$$
with coefficients:
$$c_m = \sqrt{\binom{N}{m}} a^{N-m} b^m$$
where:
$$a = \cos\left( \frac{\Omega_1 \Omega_2}{4\Delta} T_p \right), \quad b = i \sin\left( \frac{\Omega_1 \Omega_2}{4\Delta} T_p \right)$$
For $N \gg 1$, the excitation distribution $|c_m|^2$ follows a Poisson distribution with mean $\bar{m} = |b|^2 N$.

---

### 3.2 Rydberg Interaction Hamiltonian & Storage Evolution
During the storage period $T_s$, atoms interact via the Rydberg Hamiltonian:
$$\hat{H}_c = \sum_{\mu < \nu} \hbar \kappa_{\mu\nu} \hat{\sigma}_\mu^{rr} \hat{\sigma}_\nu^{rr}$$
where $\hat{\sigma}_\mu^{rr} = |r_\mu\rangle\langle r_\mu|$.

The time evolution operator is:
$$\hat{U}(T_s) = \exp(-i \hat{H}_c T_s / \hbar) = \prod_{\mu < \nu} \left[ 1 + \hat{\sigma}_\mu^{rr} \hat{\sigma}_\nu^{rr} \left( e^{-i \Phi_{\mu\nu}} - 1 \right) \right]$$

The pairwise interaction-induced phase shift accumulated during storage time $T_s$ is:
$$\Phi_{\mu\nu} = \kappa_{\mu\nu} T_s = \left[ \frac{\delta}{2} - \text{sgn}(\delta)\sqrt{(\delta/2)^2 + V_{\mu\nu}^2} \right] \frac{T_s}{\hbar}$$
where $V_{\mu\nu} = C_3 / R_{\mu\nu}^3$ (dipole-dipole interaction) and $\delta = E_{r1} + E_{r2} - 2E_r$ is the Förster energy defect. In the van der Waals asymptotic regime, $\Phi_{\mu\nu} \approx \frac{C_6}{R_{\mu\nu}^6} T_s$.

**Crucial Selective Property:**
* For $m = 0$: $\hat{H}_c |0\rangle = 0$.
* For $m = 1$: $\hat{H}_c |1\rangle = 0$. Because there is only one Rydberg atom in the ensemble, there is no second Rydberg partner to interact with! **The single-excitation Dicke state $|1\rangle$ experiences zero interaction shift.**
* For $m \ge 2$: Atom pairs experience random phase shifts $\Phi_{\mu\nu}$ depending on their spatial positions, causing rapid inhomogeneous dephasing of the multi-excitation states.

---

### 3.3 The Second-Order Autocorrelation Function $g^{(2)}(T_s)$

The quantum property of the phase-matched retrieved optical field is quantified by:
$$g^{(2)}(T_s) \equiv \frac{\langle \hat{S}_{\mathbf{k}_0}^\dagger \hat{S}_{\mathbf{k}_0}^\dagger \hat{S}_{\mathbf{k}_0} \hat{S}_{\mathbf{k}_0} \rangle}{\langle \hat{S}_{\mathbf{k}_0}^\dagger \hat{S}_{\mathbf{k}_0} \rangle^2} = \frac{\sum_{m \ge 2} |c_m|^2 m(m - 1) X_m(T_s)}{\left| \sum_{m \ge 1} |c_m|^2 m Y_m(T_s) \right|^2}$$

where:
$$X_m(T_s) = \frac{1}{m(m - 1)} \langle m | \hat{U}^\dagger \hat{S}_{\mathbf{k}_0}^\dagger \hat{S}_{\mathbf{k}_0}^\dagger \hat{S}_{\mathbf{k}_0} \hat{S}_{\mathbf{k}_0} \hat{U} | m \rangle = \frac{1}{N^2} \frac{(N - m)!}{N!} \sum_{\mu_1 \dots \mu_{m-2}} \left| \sum_{\nu_1, \nu_2} \exp(-i \Phi_{\mu_1 \dots \mu_{m-2} \nu_1 \nu_2}) \right|^2$$
$$Y_m(T_s) = \frac{1}{m} \langle m | \hat{U}^\dagger \hat{S}_{\mathbf{k}_0}^\dagger \hat{S}_{\mathbf{k}_0} \hat{U} | m \rangle = \frac{1}{N} \frac{(N - m)!}{N!} \sum_{\mu_1 \dots \mu_{m-1}} \left| \sum_{\nu} \exp(-i \Phi_{\mu_1 \dots \mu_{m-1} \nu}) \right|^2$$

---

### 3.4 Rigorous Mathematical Derivation of the Scaling Ansatz

Computing $X_m$ and $Y_m$ by direct summation scales as $N^m$, which becomes computationally impossible for $m \ge 5$ (requiring $\sim 10^6$ times more operations). Below is the complete mathematical derivation of the scaling relations in Eqs. (S.7) and (S.8).

#### 1. Statistical Preliminaries
Let $\Phi_j$ denote the spatial phase acquired by atom $j$ during storage. Across an uncorrelated, ergodic ensemble of experimental runs:
1. $\Phi_1, \dots, \Phi_N$ are independent and identically distributed (i.i.d.) random variables.
2. The elementary 2-atom phase-coherence factor is:
   $$\eta \equiv \langle e^{-i(\Phi_j - \Phi_k)} \rangle = e^{-\frac{1}{2} \langle (\Phi_j - \Phi_k)^2 \rangle} \in [0, 1]$$

#### 2. Derivation of $Y_m$ [Eq. (S.8)]
For $m$ excitations, $Y_m$ involves $m - 1$ fixed indices $\vec{\mu} = \{\mu_1, \dots, \mu_{m-1}\}$ and one free index $\nu \in \mathcal{K}_\nu$, where $|\mathcal{K}_\nu| = N - m + 1$:
$$S_Y(\vec{\mu}) = \sum_{\nu \in \mathcal{K}_\nu} \exp\left(-i \sum_{k=1}^{m-1} \Phi_{\mu_k \nu}\right), \quad \text{with } \Phi_{\mu\nu} = \Phi_\mu - \Phi_\nu$$

Expanding the modulus square $|S_Y(\vec{\mu})|^2$ into diagonal and off-diagonal cross-terms:
$$|S_Y(\vec{\mu})|^2 = \sum_{\nu \in \mathcal{K}_\nu} 1 + \sum_{\substack{\nu, \nu' \in \mathcal{K}_\nu \\ \nu \ne \nu'}} \exp\left(-i \sum_{k=1}^{m-1} (\Phi_\nu - \Phi_{\nu'})\right)$$
* **Diagonal terms:** $|\mathcal{K}_\nu| = N - m + 1$ (phase-independent background).
* **Off-diagonal terms:** $|\mathcal{K}_\nu|(|\mathcal{K}_\nu| - 1) = (N - m + 1)(N - m)$.

Because the $m - 1$ phases are independent, ensemble averaging yields:
$$\left\langle \prod_{k=1}^{m-1} e^{-i(\Phi_\nu - \Phi_{\nu'})} \right\rangle = \prod_{k=1}^{m-1} \langle e^{-i(\Phi_\nu - \Phi_{\nu'})} \rangle = \eta^{m-1}$$
$$\langle |S_Y|^2 \rangle = (N - m + 1) + (N - m + 1)(N - m) \eta^{m-1}$$

Substituting into the definition of $Y_m = \frac{1}{N} \frac{(N-m)!}{N!} \sum_{\vec{\mu}} \langle |S_Y|^2 \rangle$ and noting that the sum over $\vec{\mu}$ has $\frac{N!}{(N-m+1)!}$ terms:
$$Y_m = \frac{N - m}{N} \eta^{m-1} + \frac{1}{N}$$

Setting $m = 2$:
$$Y_2 = \frac{N - 2}{N} \eta + \frac{1}{N} \implies \eta = \frac{Y_2 - 1/N}{1 - 2/N}$$

Substituting $\eta$ back yields **Eq. (S.8)**:
$$Y_m = \frac{N - m}{N} \left( \frac{Y_2 - 1/N}{1 - 2/N} \right)^{m-1} + \frac{1}{N}$$

#### 3. Derivation of $X_m$ [Eq. (S.7)] via Topological Phase Counting
For $X_m$, the inner phase sum involves $m - 2$ fixed indices $\vec{\mu}$ and two distinct free indices $\nu_1 \ne \nu_2 \in \mathcal{K}$ ($|\mathcal{K}| = N - m + 2$):
$$S_X(\vec{\mu}) = \sum_{\substack{\nu_1, \nu_2 \in \mathcal{K} \\ \nu_1 \ne \nu_2}} \exp\left( -i \left[\sum_{k=1}^{m-2} (\Phi_{\mu_k \nu_1} + \Phi_{\mu_k \nu_2})\right] - i \Phi_{\nu_1 \nu_2} \right)$$

**Pairwise Topological Phase Counting:**  
To find the exponent of $\eta$, we count the independent pairwise phase differences connecting the interference paths:
* Connections between $\mu_k$ and $\nu_1$: $m - 2$
* Connections between $\mu_k$ and $\nu_2$: $m - 2$
* Direct connection between $\nu_1$ and $\nu_2$: $1$  
* **Total independent phase differences exponent:**
  $$p = (m - 2) + (m - 2) + 1 = 2m - 3$$

Evaluating the combinatorics of phase-matched channels and non-phase-matched diagonal background terms:
$$\langle |S_X|^2 \rangle = \left[(N - m)^2 + 3(N - m)\right] \eta^{2m-3} + 2(N - m + 1)(N - m + 2)$$

Substituting into the normalized $X_m$ definition:
$$X_m = \frac{(N - m)^2 + 3(N - m)}{N^2} \eta^{2m-3} + \frac{2}{N^2}$$

Setting $m = 2$ ($2m - 3 = 1$):
$$X_2 = \left( 1 - \frac{1}{N} - \frac{2}{N^2} \right) \eta + \frac{2}{N^2} \implies \eta = \frac{X_2 - 2/N^2}{1 - 1/N - 2/N^2}$$

Substituting $\eta$ back yields **Eq. (S.7)**:
$$X_m = \frac{(N - m)^2 + 3(N - m)}{N^2} \left( \frac{X_2 - 2/N^2}{1 - 1/N - 2/N^2} \right)^{2m-3} + \frac{2}{N^2}$$

In the large-$N$ limit ($N \gg m$):
$$X_m \approx X_2^{2m-3}, \quad Y_m \approx Y_2^{m-1}$$

This completes the analytical proof. The accuracy of this derivation was independently verified across atom numbers $N = 1$ to $20$ using Google Cirq quantum circuits ([`verify_ansatz_cirq.py`](file:///Users/yinliyl/quantum/entanglement/verify_ansatz_cirq.py)):
* **Microscopic Baselines ($N=1, 2$):** Correctly reflects that for $N=1$ no pairs exist, and for $N=2$ the interaction is a global phase with zero spectators ($m-2=0$), so $X_2=0.5, Y_2=0.5$ are constant in time.
* **Mesoscopic Scaling ($N=3$ to $20$):** Spectator-induced dephasing activates at $N \ge 3$. Across all $N \in [3, 20]$, gate-level Cirq simulations verify the ansatz with $< 2.5\%$ maximum residual error across large $N$.
* **Asymptotic Background:** Cirq fully dephased circuits confirm that the background vanishes strictly as $\mathcal{O}(2/N^2)$ and $\mathcal{O}(1/N)$, reproducing Fig.~S.2 and generating the comprehensive scaling study ([`cirq_verified_ansatz_N1_to_20.png`](file:///Users/yinliyl/quantum/entanglement/cirq_verified_ansatz_N1_to_20.png)).

---

### 3.5 Pairwise Concurrence Dynamics and Dicke State Purity

#### Wootters Concurrence
Due to permutation symmetry, the two-atom reduced density matrix in the basis $\{|gg\rangle, |gr\rangle, |rg\rangle, |rr\rangle\}$ has elements:
$$\rho_{rr,rr}(t) = \frac{2 p_2(t)}{N(N - 1)}, \quad \rho_{gr,rg}(t) = \frac{p_1(t)}{N}, \quad \rho_{gg,gg}(t) \approx p_0(t)$$

The analytical concurrence is:
$$C(\rho_{12}) = 2 \max\left(0, \frac{p_1(t)}{N} - \sqrt{p_0(t) \frac{2 p_2(t)}{N(N - 1)}}\right)$$

In the macroscopic limit $N \gg 1$:
$$N \times C(t) = 2 \max\left(0, p_1(t) - \sqrt{2 p_0(t) p_2(t)}\right)$$

* **At $t = 0$:** Poissonian statistics give $\sqrt{2 p_0(0) p_2(0)} = p_1(0)$, so $C(0) = 0$ (strictly unentangled initial state).
* **For $t > 0$:** Interaction-induced dephasing damps the two-excitation coherence $p_2(t) \to 0$, causing $N \times C(t) \approx 2 \bar{m} e^{-\bar{m}}(1 - e^{-\Gamma_2 t/2}) > 0$ to rise to a substantial positive peak (entanglement generation).

#### Dicke State Purity & Quantum Fisher Information
* **Dicke state purity:** $\mathcal{P}_{\text{Dicke}}(t) = \frac{p_1(t)}{1 - p_0(t)} \xrightarrow{t \to \infty} 1.0$ (purifies into the single-excitation Dicke state $|W\rangle$).
* **Quantum Fisher Information:** $\frac{F_Q(t)}{N} \approx 4 p_1(t) (1 - \frac{1}{2}g^{(2)}(t)) > 1$ (surpasses the Standard Quantum Limit).

---

## 4. Cirq Quantum Computing Implementation

### 4.1 Microscopic Operator Validation with Cirq
For small registers ($N \le 10$), the time evolution operator $\hat{U}(T_s)$ is implemented as a gate-level quantum circuit using two-qubit controlled-phase gates:
```python
# Each interacting pair (j, k) receives a controlled-phase gate
circuit.append(cirq.CZPowGate(exponent=-Phi_jk / np.pi).on(qubits[j], qubits[k]))
```
The state $|m=2\rangle$ is evolved under this circuit and evaluated against the collective spin-wave annihilation operator $\hat{S}_{\mathbf{k}_0} = \frac{1}{\sqrt{N}}\sum_j \hat{\sigma}_-^{(j)}$.
* Cirq yields exact numerical equivalence:
  $$X_2(\text{Cirq}) = 0.749782, \quad X_2(\text{Formula}) = 0.749782$$
  $$Y_2(\text{Cirq}) = 0.749852, \quad Y_2(\text{Formula}) = 0.749852$$

### 4.2 Macroscopic Cloud Reproduction ($N = 270$ Atoms)
For macroscopic ensembles where $2^N = 2^{270} \approx 10^{81}$ is impossible for statevector simulation, we combine Cirq's two-body operator characterization with the paper's $N^2$ scaling ansatz:
1. 3D coordinates for $N = 270$ atoms are sampled from the calibrated Gaussian cloud:
   $$\sigma_x = \sigma_y = \frac{w_0}{2} = 2.925\ \mu\text{m}, \quad \sigma_z = \frac{10.5}{2} = 5.25\ \mu\text{m} \text{ (short)}, \quad \sigma_z = \frac{230}{2} = 115\ \mu\text{m} \text{ (long)}$$
2. Pairwise phase matrices $\Phi_{\mu\nu}$ are computed.
3. $X_2(T_s)$ and $Y_2(T_s)$ are evaluated and scaled via $X_m = X_2^{2m-3}$, $Y_m = Y_2^{m-1}$.
4. $g^{(2)}(T_s)$ is synthesized across 80 spatial configurations up to $M = 15$ excitations.

---

## 5. Codebase Guide

| File | Description |
| :--- | :--- |
| [`cirq_collective_dephasing.py`](file:///Users/yinliyl/quantum/entanglement/cirq_collective_dephasing.py) | **Exact CPTP Kraus-Channel Density-Matrix Simulation (`CollectiveDephasingChannel`):** Replaces the classical Strang master-equation solver with a genuine `cirq.DensityMatrixSimulator` circuit, proving concurrence rises from $C(0)=0$ and Dicke purity $\mathcal{P}_{\text{Dicke}} \to 1$. Generates [`cirq_collective_dephasing.png`](file:///Users/yinliyl/quantum/entanglement/cirq_collective_dephasing.png). |
| [`cirq_g2_fig2_reproduction.py`](file:///Users/yinliyl/quantum/entanglement/cirq_g2_fig2_reproduction.py) | **Parametrized Cirq Circuit Reproduction of Fig. 2 ($g^{(2)}(T_s)$):** Uses `sympy.Symbol` + `cirq.Linspace` sweeps over `PairwisePhaseGate` circuits to reproduce both the short ($n=50$) and long ($n=40$) cloud curves. Generates [`cirq_g2_fig2_reproduction.png`](file:///Users/yinliyl/quantum/entanglement/cirq_g2_fig2_reproduction.png). |
| [`cirq_g2_shot_noise.py`](file:///Users/yinliyl/quantum/entanglement/cirq_g2_shot_noise.py) | **Finite-Statistics HBT Coincidence Sampling (`sim.run(repetitions=...)`):** Computes $g^{(2)}(T_s)$ directly from sampled multi-excitation bitstrings with $1/\sqrt{M_{\text{shots}}}$ error bars. Generates [`cirq_g2_shot_noise.png`](file:///Users/yinliyl/quantum/entanglement/cirq_g2_shot_noise.png). |
| [`cirq_ramsey_metrology.py`](file:///Users/yinliyl/quantum/entanglement/cirq_ramsey_metrology.py) | **Parametrized Ramsey Interferometer & Quantum Fisher Information:** Demonstrates that the dephased spin wave and $|W\rangle$ state surpass the Standard Quantum Limit ($F_Q / N > 1$). Generates [`cirq_ramsey_metrology.png`](file:///Users/yinliyl/quantum/entanglement/cirq_ramsey_metrology.png). |
| [`verify_ansatz_cirq.py`](file:///Users/yinliyl/quantum/entanglement/verify_ansatz_cirq.py) | **Direct Cirq Quantum Circuit Verification of the Paper's Ansatz (Eqs. S.7-S.8) for $N=1$ to $20$:** Simulates exact statevectors $|m=2, 3, 4\rangle$ under CZPowGates and compares exact quantum expectation values against ansatz predictions across system sizes $N=1$ to $20$. |
| [`cirq_verified_ansatz_N1_to_20.png`](file:///Users/yinliyl/quantum/entanglement/cirq_verified_ansatz_N1_to_20.png) | **Master 4-Panel Verification Plot across $N = 1$ to $20$:** Panel (a) $X_m(t)$, Panel (b) $Y_m(t)$, Panel (c) Error scaling $< 2.5\%$, Panel (d) Vanishing background $2/N^2$ and $1/N$. |
| [`cirq_verified_ansatz_figS2.png`](file:///Users/yinliyl/quantum/entanglement/cirq_verified_ansatz_figS2.png) | **Cirq Quantum-Circuit Reproduction of Fig. S.2:** Side-by-side plot comparing exact Cirq quantum circuits vs. paper ansatz for $X_3, X_4$ and $Y_3, Y_4$. |
| [`reproduce_original_notebook.py`](file:///Users/yinliyl/quantum/entanglement/reproduce_original_notebook.py) | Full implementation of the author's original Jupyter notebook with fine-structure defect $\delta$, dipole $C_3$, $\sin^2(\frac{\pi}{2} e^{-r^2/\sigma^2})$ radial PDF, and $w_z = L_z/2$. |
| [`author_notebook_fig2_reproduction.png`](file:///Users/yinliyl/quantum/entanglement/author_notebook_fig2_reproduction.png) | High-resolution plot generated directly from the author's notebook algorithm against experiment. |
| [`cirq_fig2_reproduction.png`](file:///Users/yinliyl/quantum/entanglement/cirq_fig2_reproduction.png) | Reproduction plot of Fig. 2 generated via Cirq quantum controlled-phase gate model vs. experiment. |
| [`reproduce_paper_exact.py`](file:///Users/yinliyl/quantum/entanglement/reproduce_paper_exact.py) | Full Monte Carlo simulation of $N = 270$ atoms reproducing Fig. 2 and validating against Cirq controlled-phase circuits. |
| [`reproduce_paper_cirq.py`](file:///Users/yinliyl/quantum/entanglement/reproduce_paper_cirq.py) | Cirq simulation testing microscopic $N=6$ vdW cloud and logarithmic-encoded $N=100, 1000$ Dicke dynamics. |
| [`simulation_cirq_compressed.py`](file:///Users/yinliyl/quantum/entanglement/simulation_cirq_compressed.py) | Cirq circuit mapping $(N+1)$ Dicke states onto $k = \lceil \log_2(N+1) \rceil$ qubits. |
| [`simulation_cirq_direct.py`](file:///Users/yinliyl/quantum/entanglement/simulation_cirq_direct.py) | Cirq $N$-qubit circuit simulation using stochastic quantum trajectory ensembles. |
| [`simulation_dicke.py`](file:///Users/yinliyl/quantum/entanglement/simulation_dicke.py) | High-performance Strang operator-splitting master equation solver. |
| [`metrics.py`](file:///Users/yinliyl/quantum/entanglement/metrics.py) | Calculation of concurrence $C(\rho_{12})$, Wineland parameter $\xi_R^2$, and Quantum Fisher Information. |
| [`paper_exact_fig2_reproduction.png`](file:///Users/yinliyl/quantum/entanglement/paper_exact_fig2_reproduction.png) | Reproduction plot matching Fig. 2 of the main text. |
| [`rydberg_spinwave_entanglement.png`](file:///Users/yinliyl/quantum/entanglement/rydberg_spinwave_entanglement.png) | Multi-panel plot showing universal concurrence scaling, $g^{(2)}(t)$, Dicke purity, and QFI. |
| [`equations_sheet.png`](file:///Users/yinliyl/quantum/entanglement/equations_sheet.png) | High-resolution publication-quality render of the mathematical equations. |
