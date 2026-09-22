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
    # ---- Phys. Rev. Lett. 128, 123601 (2022): trapped Rydberg superatom qubit
    Module(
        "manybody/cirq_superatom_rabi_oscillation.py",
        "PRL 128, 123601",
        "Collective sqrt(N) Rabi oscillations and blockade leakage suppression",
    ),
    Module(
        "manybody/cirq_dd_transformer_prl2022.py",
        "PRL 128, 123601",
        "@cirq.transformer dynamical-decoupling compiler pass, T2 extension",
    ),
    Module(
        "manybody/cirq_device_blockade.py",
        "PRL 128, 123601",
        "cirq.Device enforcing the blockade radius; connectivity and leakage",
    ),
    Module(
        "manybody/cirq_noise_model.py",
        "PRL 128, 123601",
        "cirq.NoiseModel: damped Rabi oscillations from T1/T2 decoherence",
    ),
    Module(
        "manybody/cirq_transformer_pipeline.py",
        "PRL 128, 123601",
        "Circuit compilation report and semantics-preservation proof",
    ),
    Module(
        "manybody/cirq_adiabatic_w_state.py",
        "PRL 128, 123601",
        "Chirped adiabatic |W> preparation vs resonant pi-pulse robustness",
    ),
    Module(
        "manybody/cirq_mps_large_n.py",
        "PRL 128, 123601",
        "Tensor-network (MPS) simulation beyond the statevector limit",
    ),
    # ---- Phys. Rev. A 106, L051701 (2022): collective-dephasing entanglement
    Module(
        "entanglement/cirq_collective_dephasing.py",
        "PRA 106, L051701",
        "Density-matrix simulation: dephasing-generated entanglement, Dicke purity",
    ),
    Module(
        "entanglement/cirq_g2_fig2_reproduction.py",
        "PRA 106, L051701",
        "Fig. 2 reproduction: g2(Ts) for the short and long clouds",
    ),
    Module(
        "entanglement/cirq_g2_shot_noise.py",
        "PRA 106, L051701",
        "g2 from sampled photon coincidences with shot-noise error bars",
    ),
    Module(
        "entanglement/cirq_ramsey_metrology.py",
        "PRA 106, L051701",
        "Ramsey interferometry, QFI, and beating the standard quantum limit",
    ),
    Module(
        "entanglement/verify_ansatz_cirq.py",
        "PRA 106, L051701",
        "Gate-level Cirq verification of the Eq. (S.7)/(S.8) scaling ansatz",
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
