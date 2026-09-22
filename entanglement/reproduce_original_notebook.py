"""
Exact implementation of the author's original Jupyter notebook simulation:
- Uses the author's exact custom radial PDF: P(r) ~ sin^2(pi/2 * exp(-r^2 / sigma^2))
- Uses wz = Lz / 2 for the longitudinal Gaussian distribution (Lz = 10 um -> wz = 5 um)
- Uses exact fine-structure Rydberg defect delta and dipole-dipole C3 via Kaulakys radial integrals
- Uses the square-root crossover detuning: Delta_nu = |delta/2| - sqrt((delta/2)^2 + (C3/R^3)^2)
- Evaluates g^(2)(Ts) = 4 * g2_num / g2_denom scaled by g^(2)(0) = 0.91 (n=40) and 0.80 (n=50)
"""

import os
import time
import numpy as np
import scipy.stats as sp_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Physical constants
hconst = 6.62606896e-34     # Planck constant
clight = 2.99792458e8       # Speed of light
Ry = 1.0973731568527e7      # Rydberg constant
a_0 = 0.52917720859e-10     # Bohr radius
alpha = 7.2973525376e-3     # Fine-structure constant

def Rydberg_energy(n, l, j):
    if l == 0:
        delta_s = 3.1311804
        delta_s2 = 0.1784
        nu = n - delta_s - (delta_s2) / (n - delta_s)**2
    elif (l == 1) and (j == 0.5):
        delta_p = 2.6548849
        delta_p2 = 0.2900
        nu = n - delta_p - (delta_p2) / (n - delta_p)**2
    elif (l == 1) and (j == 1.5):
        delta_p = 2.6416737
        delta_p2 = 0.2950
        nu = n - delta_p - (delta_p2) / (n - delta_p)**2
    elif (l == 2) and (j == 1.5):
        delta_d = 1.34809171
        delta_d2 = -0.60286
        nu = n - delta_d - (delta_d2) / (n - delta_d)**2
    elif (l == 2) and (j == 2.5):
        delta_d = 1.34646572
        delta_d2 = -0.59600
        nu = n - delta_d - (delta_d2) / (n - delta_d)**2
    elif (l == 3) and (j == 2.5):
        delta_f = 0.0165192
        delta_f2 = -0.085
        nu = n - delta_f - (delta_f2) / (n - delta_f)**2
    elif (l == 3) and (j == 3.5):
        delta_f = 0.0165437
        delta_f2 = -0.086
        nu = n - delta_f - (delta_f2) / (n - delta_f)**2
    else:
        nu = float(n)
    return - (hconst * clight * Ry) / (nu**2)

def radial(n, l, j, na, la, ja):
    Z = 1
    def get_nu(n_val, l_val, j_val):
        if l_val == 0:
            ds = 3.1311804; ds2 = 0.1784
            return n_val - ds - ds2 / (n_val - ds)**2
        elif (l_val == 1) and (j_val == 0.5):
            dp = 2.6548849; dp2 = 0.2900
            return n_val - dp - dp2 / (n_val - dp)**2
        elif (l_val == 1) and (j_val == 1.5):
            dp = 2.6416737; dp2 = 0.2950
            return n_val - dp - dp2 / (n_val - dp)**2
        elif (l_val == 2) and (j_val == 1.5):
            dd = 1.34809171; dd2 = -0.60286
            return n_val - dd - dd2 / (n_val - dd)**2
        elif (l_val == 2) and (j_val == 2.5):
            dd = 1.34646572; dd2 = -0.59600
            return n_val - dd - dd2 / (n_val - dd)**2
        else:
            return float(n_val)

    nu = get_nu(n, l, j)
    nua = get_nu(na, la, ja)

    s_a = nua - nu
    nu_c_a = (2.0 * (nu * nua)**2 / (nu + nua))**(1.0 / 3.0)
    e_a = np.sqrt(max(0.0, 1.0 - ((l + la + 1.0) / (2.0 * nu_c_a))**2))

    csi = np.arange(0, np.pi, 1e-3)
    j_a = (1.0 / np.pi) * np.sum(np.cos(s_a * csi + s_a * e_a * np.sin(csi))) * 1e-3
    dj_a = (1.0 / np.pi) * np.sum((-np.sin(s_a * csi + s_a * e_a * np.sin(csi)) * np.sin(csi))) * 1e-3

    if (la - l) == 1:
        D_pa = (1.0 / s_a) * (dj_a + (np.sqrt(max(0.0, e_a**(-2) - 1.0)) * (j_a - (np.sin(np.pi * s_a) / (np.pi * s_a)))))
    elif (la - l) == -1:
        D_pa = (1.0 / s_a) * (dj_a - (np.sqrt(max(0.0, e_a**(-2) - 1.0)) * (j_a - (np.sin(np.pi * s_a) / (np.pi * s_a)))))
    else:
        D_pa = 0.0

    er_element = ((-1)**(n - na)) * (nu_c_a**5) / (Z * (nu * nua)**(1.5)) * D_pa
    return er_element

# Fast sampler for the author's custom radial distribution
class RadialSampler:
    def __init__(self, sigma=5.85, r_max=25.0, num_pts=5000):
        self.r_vals = np.linspace(0.0, r_max, num_pts)
        pdf_vals = (np.sin(np.pi / 2.0 * np.exp(-self.r_vals**2 / (sigma**2))))**2
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]
        self.cdf_vals = cdf_vals

    def sample(self, size):
        u = np.random.uniform(0.0, 1.0, size)
        return np.interp(u, self.cdf_vals, self.r_vals)

radial_sampler = RadialSampler(sigma=5.85)

def Dephase_fast(N, Lz, n, time_array):
    wz = Lz / 2.0
    rvec = radial_sampler.sample(N)
    thetavec = np.random.uniform(0.0, 2.0 * np.pi, N)
    zvec = np.random.normal(0.0, wz, N)

    # Rydberg defect and C3
    defect = (Rydberg_energy(n, 1, 1.5) + Rydberg_energy(n - 1, 1, 1.5) - 2.0 * Rydberg_energy(n, 0, 0.5)) / hconst * 1e-6
    C3 = (1e18) * (1e-6) * ((alpha * clight / (2.0 * np.pi)) * (a_0**2)) * radial(n, 0, 0.5, n, 1, 1.5) * radial(n, 0, 0.5, n - 1, 1, 1.5)

    x = rvec * np.cos(thetavec)
    y = rvec * np.sin(thetavec)
    z = zvec
    pos = np.column_stack([x, y, z])

    diff = pos[:, None, :] - pos[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_matrix, np.inf)

    # 1D upper triangle distances for numerator
    i_idx, j_idx = np.triu_indices(N, k=1)
    dist1D = dist_matrix[i_idx, j_idx]

    detuning_1D = (np.abs(defect / 2.0) - np.sqrt((defect / 2.0)**2 + (C3 / (dist1D**3))**2))

    g2_num = np.zeros(len(time_array))
    g2_denom = np.zeros(len(time_array))

    detuning_matrix = (np.abs(defect / 2.0) - np.sqrt((defect / 2.0)**2 + (C3 / (dist_matrix**3))**2))
    np.fill_diagonal(detuning_matrix, 0.0)

    for tt, t in enumerate(time_array):
        # Numerator: pair sum over distinct pairs
        exp_sum = np.sum(np.exp(2j * np.pi * detuning_1D * t))
        g2_num[tt] = np.abs((2.0 / (N * (N - 1.0))) * exp_sum)**2

        # Denominator
        exp_matrix = np.exp(2j * np.pi * detuning_matrix * t)
        np.fill_diagonal(exp_matrix, 0.0)
        row_sums = np.sum(exp_matrix, axis=1)
        phase_denom = (1.0 / (N**3)) * np.sum(np.abs(row_sums)**2)
        g2_denom[tt] = (1.0 + phase_denom)**2

    g2 = 4.0 * g2_num / g2_denom
    return g2

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, "author_notebook_fig2_reproduction.png")
    
    print("=" * 70)
    print("Running Author's Exact Original Notebook Simulation")
    print("=" * 70)
    
    time_pts = np.linspace(0.01, 25.0, 45)
    num_trials = 10
    N = 800
    
    # Storage arrays
    L5_40 = np.zeros(len(time_pts))
    L10_40 = np.zeros(len(time_pts))
    L15_40 = np.zeros(len(time_pts))
    
    L5_50 = np.zeros(len(time_pts))
    L10_50 = np.zeros(len(time_pts))
    L15_50 = np.zeros(len(time_pts))
    
    t0 = time.time()
    print("Simulating across 10 Monte Carlo ensembles...")
    for i in range(num_trials):
        print(f"Trial {i+1}/{num_trials}...")
        L5_40 += 0.91 * Dephase_fast(N, 5.0, 40, time_pts)
        L10_40 += 0.91 * Dephase_fast(N, 10.0, 40, time_pts)
        L15_40 += 0.91 * Dephase_fast(N, 15.0, 40, time_pts)
        
        L5_50 += 0.80 * Dephase_fast(N, 5.0, 50, time_pts)
        L10_50 += 0.80 * Dephase_fast(N, 10.0, 50, time_pts)
        L15_50 += 0.80 * Dephase_fast(N, 15.0, 50, time_pts)
        
    L5_40 /= num_trials; L10_40 /= num_trials; L15_40 /= num_trials
    L5_50 /= num_trials; L10_50 /= num_trials; L15_50 /= num_trials
    print(f"Completed in {time.time() - t0:.2f}s")
    
    # Plotting
    plt.figure(figsize=(8, 7), dpi=220)
    
    # n = 40 (orange)
    plt.fill_between(time_pts, L15_40, L5_40, color='#f28e2b', alpha=0.18)
    plt.plot(time_pts, L10_40, color='#f28e2b', lw=2.5, label=r'Notebook $n = 40,\ L_z = 10\ \mu\mathrm{m}$')
    
    # n = 50 (green)
    plt.fill_between(time_pts, L15_50, L5_50, color='#2ca02c', alpha=0.18)
    plt.plot(time_pts, L10_50, color='#2ca02c', lw=2.5, label=r'Notebook $n = 50,\ L_z = 10\ \mu\mathrm{m}$')
    
    # Experimental Data
    exp_Ts_40 = np.array([0.4, 0.8, 1.2, 3.1, 5.1, 8.2, 10.1, 15.1, 20.2, 25.0])
    exp_g2_40 = np.array([0.89, 0.85, 0.82, 0.79, 0.76, 0.69, 0.63, 0.67, 0.63, 0.59])
    exp_err_40 = np.array([0.06, 0.05, 0.05, 0.04, 0.04, 0.05, 0.05, 0.05, 0.05, 0.06])
    plt.errorbar(exp_Ts_40, exp_g2_40, yerr=exp_err_40, fmt='o', color='#f28e2b', ecolor='#f28e2b',
                 elinewidth=1.5, capsize=0, markersize=6.5, label='Experiment $n = 40$')
    
    exp_Ts_50 = np.array([0.6, 1.1, 3.1, 5.1, 8.2, 10.1, 15.1, 20.2, 25.0])
    exp_g2_50 = np.array([0.53, 0.56, 0.34, 0.23, 0.20, 0.18, 0.15, 0.06, 0.11])
    exp_err_50 = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.06, 0.04, 0.07])
    plt.errorbar(exp_Ts_50, exp_g2_50, yerr=exp_err_50, fmt='s', color='#2ca02c', ecolor='#2ca02c',
                 elinewidth=1.5, capsize=0, markersize=6.5, label='Experiment $n = 50$')
    
    plt.xlabel(r'$T_s\ (\mu\mathrm{s})$', fontsize=12)
    plt.ylabel(r'$g^{(2)}$', fontsize=12)
    plt.xlim(-0.5, 26.0)
    plt.ylim(0.0, 1.2)
    plt.title(r"Author's Original Notebook Code vs. Experiment", fontsize=13)
    plt.legend(frameon=False, loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=250)
    print(f"Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
