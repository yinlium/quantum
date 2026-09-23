# misc

Unpublished, exploratory research code from grad school that never turned
into a paper: collective-Rabi modeling for a Jaynes-Cummings-type experiment,
Rydberg dipole matrix elements for a proposed biphoton-generation scheme, and
phase-matched scattering efficiency for a Rydberg atom array. These are light
cleanups of the original NumPy/SciPy notebooks (dead cells removed, wrapped
in `main()`), not full reimplementations like `manybody/` and `entanglement/`
-- the polish bar here is lower since there's no published result to validate
against.

- [`jaynes_cummings/`](./jaynes_cummings) -- three-level (Lambda-system)
  master-equation model of collective Rabi dynamics; a theory-only demo and a
  version that Poisson-averages over atom number and fits real time-resolved
  fluorescence data.
- [`biphoton/`](./biphoton) -- Rydberg radial dipole matrix elements (Kaulakys
  semiclassical formula + Rb quantum defects) supporting a proposed
  two-atom biphoton-generation scheme.
- [`rydberg_array_scattering/`](./rydberg_array_scattering) -- coherent
  forward-scattering phase-matching efficiency for an atom ensemble/array,
  plus a script computing Rydberg reduced dipole matrix elements via the
  [`ARC-Alkali-Rydberg-Calculator`](https://pypi.org/project/ARC-Alkali-Rydberg-Calculator/)
  package (`dipole_matrix_elements_arc.py` depends on it).
