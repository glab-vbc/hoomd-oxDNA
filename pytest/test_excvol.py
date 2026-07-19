"""Stage 1 validation: oxDNA1 excluded volume (nonbonded n_exc, bonded b_exc).

simple-coax has a large n_exc at frame 0 and nonzero b_exc in later frames.
"""

import os
import numpy as np

import oxdna  # noqa: F401
from oxdna import io
from oxdna.model import dna1
import _helpers as H


def _per_nuc(top, conf, frame):
    snap = io.build_snapshot(top, conf, frame=frame)
    n = snap.particles.N
    nlist = dna1.make_neighbor_list()
    excvol = dna1.excluded_volume(nlist)
    bonded = dna1.bonded_force()
    return H.single_point(
        snap, [excvol, bonded],
        lambda fs: (fs[0].energy / n, fs[1].bonded_excvol_energy / n),
    )


def test_nonbonded_excvol_startconf():
    ref = H.split_energy("simple-coax")
    top, start = H.case_paths("simple-coax")
    nexc, _ = _per_nuc(top, start, 0)
    assert abs(nexc - ref[H.COL["n_exc"]]) < 1e-4


def test_excvol_trajectory():
    top, start = H.case_paths("simple-coax")
    out = os.path.join(H.REF_DIR, "simple-coax", "output.dat")
    ref = np.loadtxt(os.path.join(H.REF_DIR, "simple-coax", "split_energy.dat"))
    nexc0, bexc0 = _per_nuc(top, start, 0)
    max_n = abs(nexc0 - ref[0, H.COL["n_exc"]])
    max_b = abs(bexc0 - ref[0, H.COL["b_exc"]])
    for k in range(20):  # 20 frames is plenty
        nexc, bexc = _per_nuc(top, out, k)
        max_n = max(max_n, abs(nexc - ref[k + 1, H.COL["n_exc"]]))
        max_b = max(max_b, abs(bexc - ref[k + 1, H.COL["b_exc"]]))
    assert max_n < 2e-4, f"max n_exc error {max_n:.2e}"
    assert max_b < 2e-4, f"max b_exc error {max_b:.2e}"


def test_excvol_force_and_torque_fd():
    def build():
        nlist = dna1.make_neighbor_list()
        return [dna1.excluded_volume(nlist)]

    H.fd_check(build, H.load_frame("simple-coax"))
