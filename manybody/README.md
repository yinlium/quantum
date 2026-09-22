# Trapped Alkali-Metal Rydberg Qubit (Google Cirq Implementation)

**Reference:**  
Y. Mei, Y. Li, H. Nguyen, P. R. Berman, and A. Kuzmich  
*Phys. Rev. Lett.* **128**, 123601 (2022) — *"Trapped Alkali-Metal Rydberg Qubit"*

---

## 1. Overview & Separation of Repositories

This directory is dedicated to the **trapped collective Rydberg superatom qubit**, **many-body $\sqrt{N}$ Rabi oscillations**, **magic-wavelength optical lattice dynamical decoupling**, **hardware blockade constraints**, **noise modeling**, **adiabatic preparation**, and **tensor-network scaling** reported in *Phys. Rev. Lett.* **128**, 123601 (2022).  
*(Note: The companion repository [`../entanglement/`](../entanglement/) focuses on interaction-induced spin-wave dephasing and $g^{(2)}(T_s)$ photon statistics from Phys. Rev. A **106**, L051701, and both share the core [`../rydberg_cirq/`](../rydberg_cirq/) library).*

---

## 2. Core Physics & Google Cirq Modules

| Module | Cirq Features Showcased | Generated Figure |
| :--- | :--- | :--- |
| [`cirq_superatom_rabi_oscillation.py`](./cirq_superatom_rabi_oscillation.py) | Custom `cirq.Gate` subclasses (`CollectiveLaserDriveStep`, `RydbergBlockadeStep`) with hierarchical `_decompose_` | [`cirq_prl2022_collective_rabi.png`](./cirq_prl2022_collective_rabi.png) |
| [`cirq_dd_transformer_prl2022.py`](./cirq_dd_transformer_prl2022.py) | Custom `@cirq.transformer` compiler pass (`compile_dynamical_decoupling`) inserting Hahn Echo / CPMG-$N$ sequences | [`cirq_prl2022_dd_coherence.png`](./cirq_prl2022_dd_coherence.png) |
| [`cirq_device_blockade.py`](./cirq_device_blockade.py) | Custom `RydbergTweezerDevice(cirq.Device)` with `validate_operation` enforcing the dipole-blockade radius $R_b$ | [`cirq_prl2022_device_blockade.png`](./cirq_prl2022_device_blockade.png) |
| [`cirq_noise_model.py`](./cirq_noise_model.py) | Custom `RydbergNoiseModel(cirq.NoiseModel)` combining $T_1$ amplitude damping, $T_2^*$ laser phase damping, and collective dephasing | [`cirq_prl2022_noise_model.png`](./cirq_prl2022_noise_model.png) |
| [`cirq_transformer_pipeline.py`](./cirq_transformer_pipeline.py) | End-to-end hardware lowering (`expand_composite`, `merge_single_qubit_moments_to_phxz`, gate-count scaling, unitary invariance) | [`cirq_prl2022_transformer_pipeline.png`](./cirq_prl2022_transformer_pipeline.png) |
| [`cirq_adiabatic_w_state.py`](./cirq_adiabatic_w_state.py) | Chirped adiabatic rapid passage circuit preparing $|W\rangle$ with robustness against Rabi pulse-area errors | [`cirq_prl2022_adiabatic_w_state.png`](./cirq_prl2022_adiabatic_w_state.png) |
| [`cirq_mps_large_n.py`](./cirq_mps_large_n.py) | Tensor-network / MPS simulation (`cirq.contrib.quimb`) scaling collective $\sqrt{N}$ Rabi flopping up to $N = 20$ qubits | [`cirq_prl2022_mps_large_n.png`](./cirq_prl2022_mps_large_n.png) |

---

## 3. Running All Simulations

From the repository root (`/Users/yinliyl/quantum`):

```bash
.venv/bin/python run_all.py --only manybody
```
