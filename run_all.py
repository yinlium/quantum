#!/usr/bin/env python
"""
Regenerate every figure in the repository, end to end.

Usage::

    .venv/bin/python run_all.py                # run everything
    .venv/bin/python run_all.py --list         # show the catalogue
    .venv/bin/python run_all.py --only rabi    # run modules matching a substring
    .venv/bin/python run_all.py --skip mps     # skip slow modules
    .venv/bin/python run_all.py --tests        # run the pytest suite first

Each module is executed in a subprocess so that one failure cannot abort the rest;
a summary table of status and runtime is printed at the end.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

REPO = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(REPO, ".venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


@dataclass(frozen=True)
class Module:
    path: str
    paper: str
    description: str


#: Every executable simulation in the repo, in a sensible reading order.
MODULES: list[Module] = [
    # ---- Phys. Rev. A 108, 043713 (2023): interference bunching/antibunching
    # Original NumPy/SciPy research code, lightly cleaned up -- not a Cirq
    # reimplementation. Needs the "legacy" extra (pandas, numba, ARC).
    Module(
        "interference_bunching/factorized_state_g2.py",
        "PRA 108, 043713",
        "Factorized-state I(phi)/g2(phi); log-scale g2 super-bunching spike",
    ),
    Module(
        "interference_bunching/truncated_state_fit.py",
        "PRA 108, 043713",
        "Truncated-state g2 theory fit to real phase-scan data via curve_fit",
    ),
    Module(
        "interference_bunching/dephasing_g2.py",
        "PRA 108, 043713",
        "Interaction-induced dephasing, MC atom-position averaging (uniform/Gaussian)",
    ),
    Module(
        "interference_bunching/truncated_state_theory.py",
        "PRA 108, 043713",
        "Truncated-state xi/phi' theory and shot-to-shot fluctuation averaging",
    ),
    # ---- Phys. Rev. Lett. 133, 213601 (2024): dipole moment of a superatom
    # Original NumPy/SciPy research code, lightly cleaned up.
    Module(
        "superatom_dipole_moment/interaction_phase_and_g2_dynamics.py",
        "PRL 133, 213601",
        "MC dephasing simulation of the interaction-induced phase phi' and g2(t)",
    ),
    Module(
        "superatom_dipole_moment/g2_truncated_fock_estimator.py",
        "PRL 133, 213601",
        "Truncated-Fock-state g2 estimator sensitivity vs mean excitation number",
    ),
    Module(
        "superatom_dipole_moment/fringe_visibility_vs_pulse_width_and_probe.py",
        "PRL 133, 213601",
        "Interference fringe visibility fit to real phase-scan photon-count data",
    ),
    # ---- Unpublished / exploratory work (see misc/README.md)
    Module(
        "misc/jaynes_cummings/jc_theory.py",
        "unpublished",
        "Theory-only Lambda-system master-equation model of collective Rabi dynamics",
    ),
    Module(
        "misc/jaynes_cummings/jc_data_fit.py",
        "unpublished",
        "Same model, Poisson-averaged over N and fit to real fluorescence data",
    ),
    Module(
        "misc/biphoton/rydberg_dipole_matrix_elements.py",
        "unpublished",
        "Kaulakys semiclassical Rydberg dipole matrix elements for a biphoton scheme",
    ),
    Module(
        "misc/rydberg_array_scattering/phase_matching_efficiency.py",
        "unpublished",
        "Structure-factor + Debye-Waller phase-matched forward-scattering efficiency",
    ),
    Module(
        "misc/rydberg_array_scattering/dipole_matrix_elements_arc.py",
        "unpublished",
        "Rb87 5P3/2->nS1/2 reduced dipole matrix elements via ARC",
    ),
]


def run_module(mod: Module, verbose: bool) -> tuple[bool, float, str]:
    path = os.path.join(REPO, mod.path)
    if not os.path.exists(path):
        return False, 0.0, "file not found"

    start = time.time()
    proc = subprocess.run(
        [PYTHON, path],
        cwd=os.path.dirname(path),
        capture_output=not verbose,
        text=True,
    )
    elapsed = time.time() - start

    if proc.returncode == 0:
        return True, elapsed, ""

    detail = ""
    if not verbose and proc.stderr:
        lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        detail = lines[-1][:160] if lines else ""
    return False, elapsed, detail or f"exit code {proc.returncode}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list modules and exit")
    ap.add_argument("--only", action="append", default=[], help="substring filter (repeatable)")
    ap.add_argument("--skip", action="append", default=[], help="substring to skip (repeatable)")
    ap.add_argument("--tests", action="store_true", help="run the pytest suite first")
    ap.add_argument("-v", "--verbose", action="store_true", help="stream module output")
    args = ap.parse_args()

    if args.list:
        print(f"{'module':<52} {'paper':<18} description")
        print("-" * 120)
        for m in MODULES:
            print(f"{m.path:<52} {m.paper:<18} {m.description}")
        return 0

    selected = MODULES
    if args.only:
        selected = [m for m in selected if any(s in m.path for s in args.only)]
    if args.skip:
        selected = [m for m in selected if not any(s in m.path for s in args.skip)]

    if not selected:
        print("No modules matched the given filters.")
        return 1

    if args.tests:
        print("=" * 100)
        print("Running the rydberg_cirq validation suite")
        print("=" * 100)
        rc = subprocess.run([PYTHON, "-m", "pytest", "tests/", "-q"], cwd=REPO).returncode
        if rc != 0:
            print("\nTest suite FAILED -- aborting before regenerating figures.")
            return rc
        print()

    print("=" * 100)
    print(f"Regenerating {len(selected)} simulation module(s)")
    print("=" * 100)

    results = []
    for i, mod in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}] {mod.path}\n    {mod.paper} -- {mod.description}")
        ok, elapsed, detail = run_module(mod, args.verbose)
        status = "OK" if ok else "FAILED"
        print(f"    -> {status} in {elapsed:.1f}s" + (f"  ({detail})" if detail else ""))
        results.append((mod, ok, elapsed, detail))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'status':<8} {'time':>8}  module")
    print("-" * 100)
    for mod, ok, elapsed, detail in results:
        print(f"{'OK' if ok else 'FAIL':<8} {elapsed:>7.1f}s  {mod.path}")
        if detail:
            print(f"{'':<8} {'':>8}  ^ {detail}")

    n_ok = sum(1 for _, ok, _, _ in results if ok)
    total = sum(t for _, _, t, _ in results)
    print("-" * 100)
    print(f"{n_ok}/{len(results)} modules succeeded in {total:.1f}s total")

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
