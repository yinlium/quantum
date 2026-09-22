"""
Renders the complete theoretical formulation into a publication-quality PNG image.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 14), dpi=220)
ax.axis('off')

lines = [
    (0.02, 0.98, r'Dynamics of Collective-Dephasing-Induced Multiatom Entanglement', 14, True, 'navy'),
    (0.02, 0.95, r'Exact Mathematical Formulation (Li, Mei, Nguyen, Berman, Kuzmich, PRA 106, L051701)', 10, False, 'gray'),
    
    (0.02, 0.91, r'1. Collective Spin Operators and Dicke Basis', 11, True, 'black'),
    (0.04, 0.88, r'• Atoms $j \in \{1, 2, \dots, N\}$ with states $|g\rangle$ (ground) and $|r\rangle$ (Rydberg)', 10, False, 'black'),
    (0.04, 0.85, r'• $J_z = \frac{1}{2}\sum_{j=1}^N \sigma_z^{(j)}, \quad J_+ = \sum_{j=1}^N |r\rangle_j \langle g|_j, \quad J_- = \sum_{j=1}^N |g\rangle_j \langle r|_j$', 10, False, 'black'),
    (0.04, 0.82, r'• $J_x = \frac{1}{2}(J_+ + J_-), \quad J_y = \frac{1}{2i}(J_+ - J_-)$', 10, False, 'black'),
    (0.04, 0.79, r'• $[J_x, J_y] = i J_z, \quad [J_y, J_z] = i J_x, \quad [J_z, J_x] = i J_y$', 10, False, 'black'),
    (0.04, 0.76, r'• Symmetric Dicke states: $\{|n\rangle\}_{n=0}^N$, where $n$ counts Rydberg excitations ($M = n - N/2$)', 10, False, 'black'),
    
    (0.02, 0.71, r'2. Initial State: Unentangled Rydberg Spin-Wave', 11, True, 'black'),
    (0.04, 0.68, r'• $|\psi(0)\rangle = \bigotimes_{j=1}^N \left( \cos\frac{\theta}{2}|g\rangle_j + \sin\frac{\theta}{2}e^{i \mathbf{k}\cdot\mathbf{r}_j}|r\rangle_j \right) = \sum_{n=0}^N c_n |n\rangle$', 10, False, 'black'),
    (0.04, 0.65, r'• $c_n = \sqrt{\binom{N}{n}} \left(\cos\frac{\theta}{2}\right)^{N-n} \left(\sin\frac{\theta}{2}\right)^n \approx e^{-\bar{n}/2} \frac{\bar{n}^{n/2}}{\sqrt{n!}} \quad (\mathrm{where}\ \bar{n} = N \sin^2(\theta/2))$', 10, False, 'black'),
    (0.04, 0.62, r'• $c_0 \approx e^{-\bar{n}/2} \quad (|0\rangle = |g\dots g\rangle$, vacuum)', 10, False, 'black'),
    (0.04, 0.59, r'• $c_1 \approx \sqrt{\bar{n}} e^{-\bar{n}/2} \quad (|1\rangle = |W\rangle = \frac{1}{\sqrt{N}}\sum_{j=1}^N |g\dots r_j \dots g\rangle$, entangled Dicke state)', 10, False, 'black'),
    (0.04, 0.56, r'• $c_2 \approx \frac{\bar{n}}{\sqrt{2}} e^{-\bar{n}/2} \quad (|2\rangle = \sqrt{\frac{2}{N(N-1)}}\sum_{j < l} |g\dots r_j \dots r_l \dots g\rangle$, two-excitation state)', 10, False, 'black'),
    
    (0.02, 0.51, r'3. Rydberg Interaction & Interaction-Induced Dephasing', 11, True, 'black'),
    (0.04, 0.48, r'• $H_{\mathrm{int}} = \sum_{1 \leq j < l \leq N} V_{jl} |r\rangle_j |r\rangle_l \langle r|_j \langle r|_l, \quad V_{jl} = \frac{C_6}{|\mathbf{r}_j - \mathbf{r}_l|^6}$', 10, False, 'black'),
    (0.04, 0.45, r'• $H_{\mathrm{int}}|0\rangle = 0$', 10, False, 'black'),
    (0.04, 0.42, r'• $H_{\mathrm{int}}|1\rangle = 0 \quad$ [Zero interaction shift: single Rydberg atom has no partner!]', 10, False, 'darkgreen'),
    (0.04, 0.39, r'• $H_{\mathrm{int}}|n \geq 2\rangle \neq 0 \Rightarrow$ Multi-excitation coherences dephase at rate $\Gamma_n \propto \frac{n(n-1)}{2}\gamma_{\mathrm{int}}$', 10, False, 'black'),
    (0.04, 0.36, r'• $\rho_{00}(t) = |c_0|^2, \quad \rho_{11}(t) = |c_1|^2 e^{-\gamma_s t}, \quad \rho_{01}(t) = c_0 c_1^* e^{-\frac{1}{2}\gamma_s t} \quad$ [Coherence protected!]', 10, False, 'darkgreen'),
    (0.04, 0.33, r'• $\rho_{nn^\prime}(t) = c_n c_{n^\prime}^* \exp\left[-\frac{1}{2}(\Gamma_n + \Gamma_{n^\prime})t - \frac{n+n^\prime}{2}\gamma_s t\right] \quad (n, n^\prime \geq 2)$', 10, False, 'black'),
    
    (0.02, 0.28, r'4. Pairwise Concurrence Dynamics', 11, True, 'black'),
    (0.04, 0.25, r'• Reduced density matrix: $\rho_{rr,rr} = \frac{2 p_2(t)}{N(N-1)}, \quad \rho_{gr,rg} = \frac{p_1(t)}{N}, \quad \rho_{gg,gg} \approx p_0(t)$', 10, False, 'black'),
    (0.04, 0.22, r'• $C(\rho_{12}) = 2 \max\left(0, \frac{p_1(t)}{N} - \sqrt{p_0(t) \frac{2 p_2(t)}{N(N-1)}}\right)$', 10, False, 'black'),
    (0.04, 0.19, r'• Universal Scaled Concurrence: $N \times C(t) = 2 \max\left(0, p_1(t) - \sqrt{2 p_0(t) p_2(t)}\right)$', 10, True, 'darkblue'),
    (0.07, 0.16, r'At $t = 0$: $\sqrt{2 p_0(0) p_2(0)} = p_1(0) \Rightarrow C(0) = 0 \quad$ (Strictly unentangled initial state)', 10, False, 'black'),
    (0.07, 0.13, r'For $t > 0$: $p_2(t) \to 0 \Rightarrow N \times C(t) \approx 2 \bar{n} e^{-\bar{n}} (1 - e^{-\Gamma_2 t/2}) > 0 \quad$ (Entanglement emerges!)', 10, False, 'darkred'),
    
    (0.02, 0.08, r'5. Antibunching, Dicke Purity, and Quantum Fisher Information', 11, True, 'black'),
    (0.04, 0.05, r'• $g^{(2)}(t) = \frac{2 p_2(t)}{p_1(t)^2} = g^{(2)}(0) e^{-\Gamma_2 t} \longrightarrow 0 \quad$ | $\quad \mathcal{P}_{\mathrm{Dicke}}(t) = \frac{p_1(t)}{1 - p_0(t)} \longrightarrow 1.0$', 10, False, 'black'),
    (0.04, 0.02, r'• $\frac{F_Q(t)}{N} \approx 4 p_1(t) \left(1 - \frac{1}{2}g^{(2)}(t)\right) > 1 \quad$ (Surpasses Standard Quantum Limit $\Rightarrow$ Multipartite Entanglement)', 10, False, 'black'),
]

for x, y, text, size, is_bold, color in lines:
    weight = 'bold' if is_bold else 'normal'
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, va='top', transform=ax.transAxes)

plt.savefig('/Users/yinliyl/quantum/entanglement/equations_sheet.png', bbox_inches='tight', facecolor='white', dpi=220)
print('equations_sheet.png generated successfully!')
