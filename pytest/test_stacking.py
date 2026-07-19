"""Stage 2 validation: oxDNA1 stacking energy + force/torque.

Validated for the sequence-averaged (simple-helix) and sequence-dependent
(simple-helix-ss) models, with a finite-difference force/torque check.
"""

import oxdna  # noqa: F401
from oxdna.forces import OxDNABonded
from oxdna.model import dna1
import _helpers as H


def _stack_energy(case, average):
    snap = H.snapshot(case)
    n = snap.particles.N
    force = dna1.bonded_force(average=average)
    e = H.single_point(snap, [force], lambda fs: fs[0].stacking_energy)
    return e / n, H.split_energy(case)[H.COL["stack"]]


def test_stacking_energy_averaged():
    e, ref = _stack_energy("simple-helix", average=True)
    assert abs(e - ref) < 1e-4, f"stack/nuc={e:.6f} ref={ref:.6f}"


def test_stacking_energy_sequence_dependent():
    e, ref = _stack_energy("simple-helix-ss", average=False)
    assert abs(e - ref) < 1e-4, f"stack/nuc={e:.6f} ref={ref:.6f}"


def _stacking_only():
    """A bonded force with FENE (epsilon=0) and excluded volume (rc=0) disabled."""
    force = OxDNABonded()
    params = dict(
        epsilon=0.0, r0=0.7525, delta=0.25,
        pos_back=(-0.4, 0.0, 0.0), pos_base=(0.4, 0.0, 0.0),
        excl_eps=0.0,
        bexc_base_base=(0.0, 0.0, 0.0, 0.0),
        bexc_base_back=(0.0, 0.0, 0.0, 0.0),
        bexc_back_base=(0.0, 0.0, 0.0, 0.0),
    )
    params.update(dna1.stacking_params())
    force.params["backbone"] = params
    return force


def test_stacking_force_and_torque_fd():
    H.fd_check(lambda: [_stacking_only()], H.load_frame("simple-helix"))
