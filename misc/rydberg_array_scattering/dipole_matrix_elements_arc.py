"""Reduced dipole matrix elements for the Rb 5P3/2 -> nS1/2 Rydberg transition.

Unpublished exploratory work (never turned into a paper). Uses the ARC
(Alkali Rydberg Calculator) package to compute the reduced dipole matrix
element <5P3/2||er||nS1/2> for Rubidium-87, for principal quantum numbers
n = 20..99, converts it to SI units, and writes the squared dipole moments
to a text file (as used e.g. in magic-detuning and coherence-time
calculations for a Rydberg-array scattering experiment).

Requires the ARC-Alkali-Rydberg-Calculator package.
"""

from pathlib import Path

import numpy as np
from arc import Rubidium87

E_CHARGE = 1.6021766e-19  # elementary charge (C)
A0 = 5.291772e-11  # Bohr radius (m)

OUTPUT_FILE = Path(__file__).with_name("DipoleSquared5p32.txt")


def compute_dipole_squared(n_min=20, n_max=100):
    """<5P3/2||er||nS1/2>^2 in SI units (C^2 m^2), for n in [n_min, n_max)."""
    atom = Rubidium87()
    n_list = np.arange(n_min, n_max, 1)
    dipole_squared = []
    for n in n_list:
        dme = atom.getReducedMatrixElementJ(5, 1, 1.5, n, 0, 0.5, 0.5)
        dipole_squared.append((n, (dme * E_CHARGE * A0) ** 2))
    return dipole_squared


def main():
    dipole_squared = compute_dipole_squared()

    with open(OUTPUT_FILE, "w") as fp:
        for n, value in dipole_squared:
            fp.write(f"{n} {value}\n")
    print(f"Wrote {len(dipole_squared)} entries to {OUTPUT_FILE}")

    for n, value in dipole_squared[:5]:
        print(f"n={n}: |<5P3/2|er|{n}S1/2>|^2 = {value:.4e} C^2 m^2")


if __name__ == "__main__":
    main()
