"""
Benchmark script for Dicke master equation solver.
Tests N=10, 100, and 1000.
"""

import time
import numpy as np
from simulation_dicke import run_dicke_simulation

for N in [10, 100, 500, 1000]:
    t0 = time.time()
    res = run_dicke_simulation(N=N, Omega=1.0, Delta=0.0, gamma_c=0.1, t_max=3.0, num_steps=150, initial_state='ground')
    dt = time.time() - t0
    min_xi_R2 = np.nanmin(res['xi_R2'])
    max_C = np.nanmax(res['concurrence'])
    print(f"N = {N:4d} | Elapsed: {dt:6.2f}s | Min Wineland xi_R^2: {min_xi_R2:.4f} | Max Concurrence: {max_C:.6f}")
