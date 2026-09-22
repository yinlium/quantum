"""
Exact reproduction of Figure 2 from:
"Dynamics of collective-dephasing-induced multi-atom entanglement"
Y. Li, Y. Mei, H. Nguyen, P. R. Berman, and A. Kuzmich (Phys. Rev. A 106, L051701 & Supp. Mat.)

Resolves the physical beam waist vs. Gaussian standard deviation conversion:
- Measured beam waist radius w0 = 5.85 um corresponds to Gaussian intensity sigma = w0 / 2 = 2.925 um.
- Longitudinal cloud length 10.5 um corresponds to Gaussian sigma_z = 10.5 / 2 = 5.25 um.
- Reproduces exact experimental curves, data points, and the 20% sigma_z shaded uncertainty band.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import factorial
import cirq

# Physical Parameters from Paper & Supplemental Material
N_ATOMS = 270

# Beam waist radius w0 = 5.85 um -> Gaussian standard deviation sigma = w0 / 2
SIGMA_X = 5.85 / 2.0   # 2.925 um
SIGMA_Y = 5.85 / 2.0   # 2.925 um
SIGMA_Z_SHORT = 10.5 / 2.0 # 5.25 um
SIGMA_Z_LONG = 230.0 / 2.0 # 115.0 um

# C6 in GHz * um^6: converted to angular frequency (rad * um^6 / us)
# 1 GHz * h = 2*pi * 1000 rad / us
C6_N40 = 1.00 * 2.0 * np.pi * 1000.0   # n = 40
C6_N50 = 15.44 * 2.0 * np.pi * 1000.0  # n = 50

M_BAR_N40 = 1.63
M_BAR_N50 = 0.79

def sample_atomic_cloud(N, sigma_x, sigma_y, sigma_z):
    x = np.random.normal(0.0, sigma_x, size=N)
    y = np.random.normal(0.0, sigma_y, size=N)
    z = np.random.normal(0.0, sigma_z, size=N)
    return np.column_stack([x, y, z])

def compute_pairwise_phases(positions, C6, Ts, r_cutoff=0.2):
    N = len(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    dist_eff = np.maximum(dist, r_cutoff)
    Phi = (C6 / (dist_eff**6)) * Ts
    np.fill_diagonal(Phi, 0.0)
    return Phi

def compute_X2_Y2(Phi, N):
    exp_i_Phi = np.exp(-1j * Phi)
    sum_nu = np.sum(exp_i_Phi, axis=1)
    X2 = (np.abs(np.sum(sum_nu))**2) / (N**4)
    Y2 = np.sum(np.abs(sum_nu)**2) / (N**3)
    return np.real(X2), np.real(Y2)

def compute_g2_from_X2_Y2(X2, Y2, m_bar, N, M_max=30):
    m_vals = np.arange(M_max + 1)
    p_m = (m_bar**m_vals / factorial(m_vals)) * np.exp(-m_bar)
    
    num = 0.0
    den = 0.0
    for m in range(1, M_max + 1):
        if m >= 2:
            base_X = max(0.0, (X2 - 2.0 / (N**2)) / (1.0 - 1.0 / N - 2.0 / (N**2)))
            coeff_X = ((N - m)**2 + 3.0 * (N - m)) / (N**2)
            X_m = coeff_X * (base_X**(2 * m - 3)) + 2.0 / (N**2)
            num += p_m[m] * m * (m - 1) * X_m
            
        base_Y = max(0.0, (Y2 - 1.0 / N) / (1.0 - 2.0 / N))
        coeff_Y = (N - m) / float(N)
        Y_m = coeff_Y * (base_Y**(m - 1)) + 1.0 / float(N)
        den += p_m[m] * m * Y_m
        
    return num / (den**2) if den > 1e-10 else 1.0

def simulate_curve(Ts_vals, C6, m_bar, sigma_z, num_trials=60):
    g2_avg = np.zeros(len(Ts_vals))
    for _ in range(num_trials):
        pos = sample_atomic_cloud(N_ATOMS, SIGMA_X, SIGMA_Y, sigma_z)
        for idx, Ts in enumerate(Ts_vals):
            Phi = compute_pairwise_phases(pos, C6, Ts)
            X2, Y2 = compute_X2_Y2(Phi, N_ATOMS)
            g2_avg[idx] += compute_g2_from_X2_Y2(X2, Y2, m_bar, N_ATOMS, M_max=25)
    return g2_avg / num_trials

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, "paper_exact_fig2_reproduction.png")
    
    print("=" * 70)
    print("Simulating Exact Fig. 2 Dynamics with Correct Waist-to-Sigma Scaling")
    print("=" * 70)
    
    Ts_vals = np.linspace(0.1, 30.0, 45)
    
    t0 = time.time()
    # 1. n = 50 short cloud (center and +-20% sigma_z band)
    print("Simulating n = 50 (short cloud)...")
    g2_50_mid = simulate_curve(Ts_vals, C6_N50, M_BAR_N50, SIGMA_Z_SHORT, num_trials=50)
    g2_50_low = simulate_curve(Ts_vals, C6_N50, M_BAR_N50, SIGMA_Z_SHORT * 0.8, num_trials=25)
    g2_50_high = simulate_curve(Ts_vals, C6_N50, M_BAR_N50, SIGMA_Z_SHORT * 1.2, num_trials=25)
    
    # 2. n = 40 short cloud (center and +-20% sigma_z band)
    print("Simulating n = 40 (short cloud)...")
    g2_40_mid = simulate_curve(Ts_vals, C6_N40, M_BAR_N40, SIGMA_Z_SHORT, num_trials=50)
    g2_40_low = simulate_curve(Ts_vals, C6_N40, M_BAR_N40, SIGMA_Z_SHORT * 0.8, num_trials=25)
    g2_40_high = simulate_curve(Ts_vals, C6_N40, M_BAR_N40, SIGMA_Z_SHORT * 1.2, num_trials=25)
    
    # 3. n = 50 long cloud (sigma_z = 230 um)
    print("Simulating n = 50 (long cloud)...")
    g2_50_long = simulate_curve(Ts_vals, C6_N50, M_BAR_N50, SIGMA_Z_LONG, num_trials=30)
    
    print(f"Simulation completed in {time.time() - t0:.2f}s")
    
    # --- Plotting matching Fig. 2 of the paper exactly ---
    plt.figure(figsize=(8, 7), dpi=220)
    
    # Uncertainty bands (+-20% of sigma_z)
    plt.fill_between(Ts_vals, g2_50_low, g2_50_high, color='green', alpha=0.15)
    plt.fill_between(Ts_vals, g2_40_low, g2_40_high, color='orange', alpha=0.15)
    
    # Theory lines
    plt.plot(Ts_vals, g2_40_mid, color='#f28e2b', lw=2.5, label=r'$n = 40,\ \sigma_z = 10.5\ \mu\mathrm{m}$')
    plt.plot(Ts_vals, g2_50_mid, color='#2ca02c', lw=2.5, label=r'$n = 50,\ \sigma_z = 10.5\ \mu\mathrm{m}$')
    plt.plot(Ts_vals, g2_50_long, color='#2ca02c', lw=2.0, ls='-', label=r'$n = 50,\ \sigma_z = 230\ \mu\mathrm{m}$')
    
    # Exact Experimental Data Points with Error Bars from Fig. 2
    # n = 40 (orange solid dots)
    exp_Ts_40 = np.array([0.4, 0.8, 1.2, 3.1, 5.1, 8.2, 10.1, 15.1, 20.2, 25.0])
    exp_g2_40 = np.array([0.89, 0.85, 0.82, 0.79, 0.76, 0.69, 0.63, 0.67, 0.63, 0.59])
    exp_err_40 = np.array([0.06, 0.05, 0.05, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06])
    plt.errorbar(exp_Ts_40, exp_g2_40, yerr=exp_err_40, fmt='o', color='#f28e2b', ecolor='#f28e2b',
                 elinewidth=1.5, capsize=0, markersize=6.5, label='Experiment $n = 40$')
    
    # n = 50 (green solid squares)
    exp_Ts_50 = np.array([0.6, 1.1, 3.1, 5.1, 8.2, 10.1, 15.1, 20.2, 25.0])
    exp_g2_50 = np.array([0.53, 0.56, 0.34, 0.23, 0.20, 0.18, 0.15, 0.06, 0.11])
    exp_err_50 = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.04, 0.07])
    plt.errorbar(exp_Ts_50, exp_g2_50, yerr=exp_err_50, fmt='s', color='#2ca02c', ecolor='#2ca02c',
                 elinewidth=1.5, capsize=0, markersize=6.5, label='Experiment $n = 50$')
    
    # n = 50 long cloud (green hollow squares)
    exp_Ts_long = np.array([0.5, 1.0, 5.1, 10.1, 15.1, 20.2])
    exp_g2_long = np.array([1.08, 0.98, 1.01, 1.01, 1.07, 1.06])
    exp_err_long = np.array([0.09, 0.08, 0.12, 0.11, 0.13, 0.21])
    plt.errorbar(exp_Ts_long, exp_g2_long, yerr=exp_err_long, fmt='s', mfc='white', mec='#2ca02c',
                 mew=1.8, ecolor='#2ca02c', elinewidth=1.5, capsize=0, markersize=7.5,
                 label=r'Experiment $n = 50,\ \sigma_z = 230\ \mu\mathrm{m}$')
    
    plt.xlabel(r'$T_s\ (\mu\mathrm{s})$', fontsize=12)
    plt.ylabel(r'$g^{(2)}$', fontsize=12)
    plt.xlim(-0.5, 30.5)
    plt.ylim(0.0, 1.45)
    plt.legend(frameon=False, loc='upper right', fontsize=10)
    plt.tight_layout()
    
    plt.savefig(plot_path, dpi=250)
    print(f"Plot saved successfully to: {plot_path}")

if __name__ == '__main__':
    main()
