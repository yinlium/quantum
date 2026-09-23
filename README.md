# Rydberg Atom Research Code

Original research code behind five publications on trapped Rydberg atomic ensembles
from the Kuzmich group, lightly cleaned up (dead cells, duplicate notebooks, and
lab-machine-specific paths removed) into standalone scripts — not rewritten into any
particular framework.

| Paper | Physics | Directory |
| :--- | :--- | :--- |
| [*Phys. Rev. Lett.* **128**, 123601 (2022)](https://doi.org/10.1103/PhysRevLett.128.123601)<br>*Trapped Alkali-Metal Rydberg Qubit* | Collective $\sqrt{N}$ Rabi oscillation fitting, atom-number-fluctuation Monte Carlo, laser-resonance locating, van-der-Waals dephasing of $g^{(2)}$ | [`rydberg_qubit_rabi_oscillation/`](./rydberg_qubit_rabi_oscillation) |
| [*Phys. Rev. A* **106**, L051701 (2022)](https://doi.org/10.1103/PhysRevA.106.L051701)<br>*Dynamics of Collective-Dephasing-Induced Multiatom Entanglement* | Dephasing Monte Carlo fit to real $g^{(2)}(T_s)$ data, Fock-truncation convergence, calibration fits, and the multiatom entanglement witness | [`collective_dephasing_entanglement/`](./collective_dephasing_entanglement) |
| [*Phys. Rev. A* **108**, 043713 (2023)](https://doi.org/10.1103/PhysRevA.108.043713)<br>*Interference Bunching and Antibunching of Coherent Atomic Radiation Fields* | $g^{(2)}(\phi)$ for factorized/truncated collective atomic states interfering with a reference field; interaction-induced dephasing for uniform and Gaussian clouds | [`interference_bunching/`](./interference_bunching) |
| [*Phys. Rev. Lett.* **133**, 213601 (2024)](https://doi.org/10.1103/PhysRevLett.133.213601)<br>*Dipole Moment of a Superatom* | Homodyne measurement of the collective atomic dipole moment: interference fringe visibility and $g^{(2)}_\text{max}$ vs. excitation probability and probe amplitude | [`superatom_dipole_moment/`](./superatom_dipole_moment) |
| [*Phys. Rev. A* **97**, 043401 (2018)](https://doi.org/10.1103/PhysRevA.97.043401)<br>*Expansion of an Ultracold Rydberg Plasma*, and the author's 2019 Colby College honors thesis of the same title | Monte Carlo kinetic simulation — three-body recombination, electron-Rydberg scattering, radiative $n,l$-cascade decay, coupled to an RK2 hydrodynamic expansion | [`plasma_expansion/`](./plasma_expansion) |

`rydberg_qubit_rabi_oscillation/`, `collective_dephasing_entanglement/`,
`interference_bunching/`, and `superatom_dipole_moment/` need the optional `legacy`
extra (`pandas`, `numba`, `ARC-Alkali-Rydberg-Calculator` — see Setup below).
`plasma_expansion/` is C++; build it with `make` inside that directory.

The PRL 128, 123601 and PRA 106, L051701 papers above were *also* reproduced
gate-level in Google Cirq, as full from-scratch reimplementations built on a shared
Cirq library — but that code lives in a private companion repository rather than
here (kept separate given it's built on Google's own open-source framework).

---

## Unpublished / exploratory work

[`misc/`](./misc) holds cleaned-up but unpublished code from grad school: a
Jaynes-Cummings collective-dynamics model, Rydberg dipole matrix elements for a
biphoton-generation scheme, and Rydberg-array phase-matched scattering efficiency.
See [`misc/README.md`](./misc/README.md) for details; polish bar is lower there since
there's no published figure to validate against.

---

## Setup from a fresh clone

The virtualenv is **not** tracked by git, so recreate it once after cloning
(Python 3.10+ required):

```bash
git clone https://github.com/yinlium/quantum.git
cd quantum

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[legacy]"
```

The `legacy` extra pulls in `pandas` + `numba` + `ARC-Alkali-Rydberg-Calculator`,
needed by `interference_bunching/` and `superatom_dipole_moment/`.

---

## Quick start

```bash
.venv/bin/python run_all.py --list        # catalogue of every simulation
.venv/bin/python run_all.py               # regenerate every figure
.venv/bin/python run_all.py --only bunching  # run a subset
```

`plasma_expansion/` is not part of this catalogue (it's C++, not Python) — build and
run it separately with `make` inside that directory.

---

## Repository layout

```
quantum/
├── rydberg_qubit_rabi_oscillation/    PRL 128, 123601 (original code, cleaned up)
├── collective_dephasing_entanglement/ PRA 106, L051701 (original code, cleaned up)
├── interference_bunching/             PRA 108, 043713 (original code, cleaned up)
├── superatom_dipole_moment/           PRL 133, 213601 (original code, cleaned up)
├── plasma_expansion/                  PRA 97, 043401 + honors thesis (C++)
├── misc/                               unpublished/exploratory code
├── docs/                               GitHub Pages portfolio site
├── run_all.py                         regenerate every Python figure
├── pyproject.toml
├── LICENSE
└── README.md
```

`.venv/` exists locally but is **not tracked** (see [`.gitignore`](./.gitignore)) —
recreate it with the setup command above.

---

## License

Released under the [MIT License](./LICENSE).

This is the author's own original research code; please cite the corresponding
papers if you use it:
[*Phys. Rev. Lett.* **128**, 123601 (2022)](https://doi.org/10.1103/PhysRevLett.128.123601),
[*Phys. Rev. A* **106**, L051701 (2022)](https://doi.org/10.1103/PhysRevA.106.L051701),
[*Phys. Rev. A* **108**, 043713 (2023)](https://doi.org/10.1103/PhysRevA.108.043713),
[*Phys. Rev. Lett.* **133**, 213601 (2024)](https://doi.org/10.1103/PhysRevLett.133.213601),
and [*Phys. Rev. A* **97**, 043401 (2018)](https://doi.org/10.1103/PhysRevA.97.043401).
