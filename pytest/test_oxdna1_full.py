"""Stage 5 capstone: the complete oxDNA1 model.

Validates all seven energy terms simultaneously against split_energy.dat for the
averaged and sequence-dependent models, then checks the duplex stays base-paired
under a short Langevin trajectory.
"""

import numpy as np

import oxdna  # noqa: F401
from oxdna.model import dna1
import _helpers as H

TERMS = ["fene", "b_exc", "stack", "n_exc", "hb", "cr_stack", "cx_stack"]


def _all_terms(case, average):
    snap = H.snapshot(case)
    n = snap.particles.N
    forces = dna1.forces(average=average)[0]

    def read(fs):
        bonded, excvol, hb, crst, coax = fs
        return {
            "fene": bonded.fene_energy / n,
            "b_exc": bonded.bonded_excvol_energy / n,
            "stack": bonded.stacking_energy / n,
            "n_exc": excvol.energy / n,
            "hb": hb.energy / n,
            "cr_stack": crst.energy / n,
            "cx_stack": coax.energy / n,
        }

    return H.single_point(snap, forces, read), H.split_energy(case)


def _check(case, average):
    vals, ref = _all_terms(case, average)
    for name in TERMS:
        assert abs(vals[name] - ref[H.COL[name]]) < 1e-4, (
            f"{case} {name}: {vals[name]:.6f} vs {ref[H.COL[name]]:.6f}"
        )


def test_all_terms_simple_helix():
    _check("simple-helix", average=True)


def test_all_terms_simple_coax():
    _check("simple-coax", average=True)


def test_all_terms_sequence_dependent():
    _check("simple-helix-ss", average=False)


def test_duplex_stays_bound():
    snap = H.snapshot("simple-helix")
    n = snap.particles.N
    forces = dna1.forces()[0]
    sim = H.run_langevin(snap, forces, kT=dna1.kT_from_temperature(296.15), steps=2000, seed=3)
    hb = forces[2].energy / n
    pos = np.array(sim.state.get_snapshot().particles.position)
    assert np.isfinite(pos).all()
    assert hb < -0.15, f"duplex appears to have melted: hb/nuc={hb:.4f}"
