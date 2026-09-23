# Plasma Expansion

A Monte Carlo kinetic simulation of the expansion of an ultracold Rydberg
plasma. Starting from a Gaussian cloud of cold Rydberg atoms plus a seed
population of ions and electrons, each time step stochastically applies
three-body recombination (Muller-Wolf), electron-Rydberg scattering /
excitation / deexcitation / ionization (Mansbach-Keck), l-changing
collisions, and radiative n,l-cascade decay via tabulated hydrogenic rates.
The resulting energy exchange between electrons, ions, and the Rydberg
reservoir is coupled to a self-similar RK2 hydrodynamic integration of the
expanding Gaussian ion/electron cloud, giving the plasma's expansion
velocity, temperature, and Coulomb coupling parameter over time.

This code (originally by F. Robicheaux, adapted by G. Forest and D. Tate,
further modified by Y. Li) is the numerical model behind:

- G. T. Forest, Y. Li, E. D. Ward, A. L. Goodsell, D. A. Tate, "Expansion
  of an ultracold Rydberg plasma," Phys. Rev. A 97, 043401 (2018).
- Y. Li, "Expansion of an Ultracold Neutral Plasma," Colby College Honors
  Thesis (2019).

## Build

    make

Requires only g++ and the standard library — no external dependencies.

## Run

Reads `data/energiesH.dat` and `data/radratesH.dat` via relative paths, so
it must be run from inside `plasma_expansion/`:

    ./plasma_expansion

Writes two output files per run (`plasparRb_*.dat`, `timoutRb_*.dat`),
named from the density/temperature/scale, into the current directory, and
prints periodic progress to stdout.

## Configuring a run

All physical run parameters (temperature, cloud size, density, run
duration, Rydberg coarse-graining factor, RNG seed) are named constants
near the top of `plasma_expansion.cpp`, under the "run parameters" block.
There are no command-line arguments — inherited from the original code,
not a design choice here. Edit the constants and rebuild to change a run.
