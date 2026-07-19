"""Stage 0: finite-difference checks of the FENE force AND torque.

Uses a bonded force with the excluded volume disabled (rc=0) so only FENE
contributes, and validates the analytic force/torque against central differences
of the energy on a real configuration. This is the key guard for the lab-frame
torque convention.
"""

import oxdna  # noqa: F401
from oxdna.forces import OxDNABonded
import _helpers as H


def _fene_only():
    force = OxDNABonded()
    force.params["backbone"] = dict(
        epsilon=2.0, r0=0.7525, delta=0.25,
        pos_back=(-0.4, 0.0, 0.0), pos_base=(0.4, 0.0, 0.0),
        excl_eps=0.0,
        bexc_base_base=(0.0, 0.0, 0.0, 0.0),
        bexc_base_back=(0.0, 0.0, 0.0, 0.0),
        bexc_back_base=(0.0, 0.0, 0.0, 0.0),
    )
    return force


def test_fene_force_and_torque_fd():
    H.fd_check(lambda: [_fene_only()], H.load_frame("simple-helix"))
