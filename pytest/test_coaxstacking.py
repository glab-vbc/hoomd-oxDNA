"""Stage 5 validation: oxDNA1 coaxial stacking energy + force/torque.

Nonzero on simple-coax.
"""

import oxdna  # noqa: F401
from oxdna.model import dna1
import _helpers as H


def _cx_energy(case):
    snap = H.snapshot(case)
    n = snap.particles.N
    nlist = dna1.make_neighbor_list()
    coax = dna1.coaxial_stacking(nlist)
    e = H.single_point(snap, [coax], lambda fs: fs[0].energy)
    return e / n, H.split_energy(case)[H.COL["cx_stack"]]


def test_coaxial_energy():
    for case in ("simple-coax", "simple-helix"):
        e, ref = _cx_energy(case)
        assert abs(e - ref) < 1e-4, f"{case}: cx_stack/nuc={e:.6f} ref={ref:.6f}"


def test_coaxial_force_and_torque_fd():
    def build():
        nlist = dna1.make_neighbor_list()
        return [dna1.coaxial_stacking(nlist)]

    H.fd_check(build, H.load_frame("simple-coax"))
