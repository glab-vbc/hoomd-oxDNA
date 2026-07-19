"""Stage 0 validation: oxDNA1 FENE backbone energy vs the oxDNA reference."""

import oxdna  # noqa: F401
from oxdna.model import dna1
import _helpers as H


def test_fene_matches_reference():
    snap = H.snapshot("simple-helix")
    n = snap.particles.N
    force = dna1.bonded_force()
    per_nuc = H.single_point(snap, [force], lambda fs: fs[0].fene_energy) / n
    ref = H.split_energy("simple-helix")[H.COL["fene"]]
    assert abs(per_nuc - ref) < 1e-4, f"FENE/nuc={per_nuc:.6f} ref={ref:.6f}"
