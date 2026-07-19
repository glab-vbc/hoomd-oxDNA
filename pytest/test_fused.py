"""Validation: the fused HBondCross force equals hydrogen bonding + cross-stacking
run as two separate forces, and matches the reference hb + cr_stack columns."""

import oxdna  # noqa: F401
from oxdna.model import dna1, dna2
import _helpers as H


def test_fused_matches_separate_dna1():
    snap = H.snapshot("simple-helix")
    n = snap.particles.N
    nl = dna1.make_neighbor_list()
    e_sep = H.single_point(
        snap, [dna1.hydrogen_bonding(nl), dna1.cross_stacking(nl)],
        lambda fs: fs[0].energy + fs[1].energy) / n
    nl2 = dna1.make_neighbor_list()
    e_fused = H.single_point(snap, [dna1.hbond_cross(nl2)], lambda fs: fs[0].energy) / n
    assert abs(e_sep - e_fused) < 1e-5, f"sep={e_sep:.6f} fused={e_fused:.6f}"


def test_fused_matches_reference_dna1():
    snap = H.snapshot("simple-helix")
    n = snap.particles.N
    nl = dna1.make_neighbor_list()
    e = H.single_point(snap, [dna1.hbond_cross(nl)], lambda fs: fs[0].energy) / n
    ref = H.split_energy("simple-helix")
    target = ref[H.COL["hb"]] + ref[H.COL["cr_stack"]]
    assert abs(e - target) < 1e-4, f"fused={e:.6f} ref(hb+cr)={target:.6f}"


def test_fused_matches_separate_dna2():
    snap = H.snapshot("simple-helix-oxdna2")
    n = snap.particles.N
    nl = dna2.make_neighbor_list()
    e_sep = H.single_point(
        snap, [dna2.hydrogen_bonding(nl), dna1.cross_stacking(nl)],
        lambda fs: fs[0].energy + fs[1].energy) / n
    nl2 = dna2.make_neighbor_list()
    e_fused = H.single_point(snap, [dna2.hbond_cross(nl2)], lambda fs: fs[0].energy) / n
    assert abs(e_sep - e_fused) < 1e-5, f"sep={e_sep:.6f} fused={e_fused:.6f}"


def test_fused_force_and_torque_fd():
    def build():
        nl = dna1.make_neighbor_list()
        return [dna1.hbond_cross(nl)]

    H.fd_check(build, H.load_frame("simple-helix"))
