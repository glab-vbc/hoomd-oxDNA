"""Stage 3 validation: oxDNA1 hydrogen bonding energy + force/torque."""

import oxdna  # noqa: F401
from oxdna.model import dna1
import _helpers as H


def _hb_energy(case, average):
    snap = H.snapshot(case)
    n = snap.particles.N
    nlist = dna1.make_neighbor_list()
    hb = dna1.hydrogen_bonding(nlist, average=average)
    e = H.single_point(snap, [hb], lambda fs: fs[0].energy)
    return e / n, H.split_energy(case)[H.COL["hb"]]


def test_hb_energy_averaged():
    e, ref = _hb_energy("simple-helix", average=True)
    assert abs(e - ref) < 1e-4, f"hb/nuc={e:.6f} ref={ref:.6f}"


def test_hb_energy_sequence_dependent():
    e, ref = _hb_energy("simple-helix-ss", average=False)
    assert abs(e - ref) < 1e-4, f"hb/nuc={e:.6f} ref={ref:.6f}"


def test_hb_complementarity():
    assert dna1.hb_epsilon("A", "T") > 0 and dna1.hb_epsilon("G", "C") > 0
    for a, b in [("A", "A"), ("A", "G"), ("A", "C"), ("G", "G"), ("G", "T"), ("C", "T")]:
        assert dna1.hb_epsilon(a, b) == 0.0


def test_hb_force_and_torque_fd():
    def build():
        nlist = dna1.make_neighbor_list()
        return [dna1.hydrogen_bonding(nlist)]

    H.fd_check(build, H.load_frame("simple-helix"))
