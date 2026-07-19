"""Stage 4 validation: oxDNA1 cross-stacking energy + force/torque."""

import oxdna  # noqa: F401
from oxdna.model import dna1
import _helpers as H


def _cx_energy(case):
    snap = H.snapshot(case)
    n = snap.particles.N
    nlist = dna1.make_neighbor_list()
    cx = dna1.cross_stacking(nlist)
    e = H.single_point(snap, [cx], lambda fs: fs[0].energy)
    return e / n, H.split_energy(case)[H.COL["cr_stack"]]


def test_cross_stacking_energy():
    for case in ("simple-helix", "simple-coax"):
        e, ref = _cx_energy(case)
        assert abs(e - ref) < 1e-4, f"{case}: cr_stack/nuc={e:.6f} ref={ref:.6f}"


def test_cross_stacking_force_and_torque_fd():
    def build():
        nlist = dna1.make_neighbor_list()
        return [dna1.cross_stacking(nlist)]

    H.fd_check(build, H.load_frame("simple-helix"))
