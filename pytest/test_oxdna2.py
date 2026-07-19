"""Stage 6 validation: the complete oxDNA2 model.

Validates all eight energy terms (adds Debye-Huckel) against split_energy.dat,
including the half-charged-ends variant, a Debye force/torque finite-difference
check, and a duplex-stability run.
"""

import numpy as np

import oxdna  # noqa: F401
from oxdna.model import dna1, dna2
import _helpers as H

TERMS = ["fene", "b_exc", "stack", "n_exc", "hb", "cr_stack", "cx_stack", "debye"]


def _all_terms(case, half_charged):
    snap = H.snapshot(case, half_charged_ends=half_charged)
    n = snap.particles.N
    forces = dna2.forces()[0]

    def read(fs):
        bonded, excvol, hb, crst, coax, debye = fs
        return {
            "fene": bonded.fene_energy / n,
            "b_exc": bonded.bonded_excvol_energy / n,
            "stack": bonded.stacking_energy / n,
            "n_exc": excvol.energy / n,
            "hb": hb.energy / n,
            "cr_stack": crst.energy / n,
            "cx_stack": coax.energy / n,
            "debye": debye.energy / n,
        }

    return H.single_point(snap, forces, read), H.split_energy(case)


def _check(case, half_charged, terms=TERMS):
    vals, ref = _all_terms(case, half_charged)
    for name in terms:
        assert abs(vals[name] - ref[H.COL[name]]) < 1e-4, (
            f"{case} {name}: {vals[name]:.6f} vs {ref[H.COL[name]]:.6f}"
        )


def test_oxdna2_all_terms_simple_helix():
    _check("simple-helix-oxdna2", half_charged=False)


def test_oxdna2_all_terms_simple_coax():
    _check("simple-coax-oxdna2", half_charged=False)


def test_oxdna2_half_charged_ends():
    _check("simple-helix-oxdna2-half-charged-ends", half_charged=True, terms=["debye"])


def test_debye_force_and_torque_fd():
    def build():
        nlist = dna2.make_neighbor_list()
        return [dna2.debye_huckel(nlist)]

    H.fd_check(build, H.load_frame("simple-helix-oxdna2"))


def test_oxdna2_duplex_stays_bound():
    snap = H.snapshot("simple-helix-oxdna2")
    n = snap.particles.N
    forces = dna2.forces()[0]
    sim = H.run_langevin(snap, forces, kT=dna1.kT_from_temperature(296.15), steps=2000, seed=5)
    hb = forces[2].energy / n
    pos = np.array(sim.state.get_snapshot().particles.position)
    assert np.isfinite(pos).all()
    assert hb < -0.15, f"duplex melted: hb/nuc={hb:.4f}"
