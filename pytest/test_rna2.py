"""oxRNA2 validation: the reuse terms + cross-stacking against the oxRNA2 reference.

FENE, bonded/nonbonded excluded volume, hydrogen bonding, cross-stacking (with the
RNA theta4-dropped form) and Debye-Huckel. Energies are checked per-nucleotide against
split_energy.dat; forces/torques of the base-base terms are finite-difference checked
(the cross-stacking t4_enabled=False path changes the force/torque, so energy parity is
not enough). Stacking and coaxial stacking (RNA-specific new C++) are added separately.

The oxRNA reference cases use sys.top + a trajectory in output.dat, and split_energy.dat
row (f+1) corresponds to output.dat frame f.
"""

import oxdna  # noqa: F401
from oxdna.model import rna2
import _helpers as H

CASE = "simple-helix-rna2-12bp"
COAX_CASE = "simple-coax-rna2"  # exercises coaxial stacking (cx_stack != 0)
TOP, CONF = "sys.top", "output.dat"
COL = dict(fene=1, b_exc=2, n_exc=4, hb=5, cr_stack=6, cx_stack=7, debye=8)


def _energies(snap):
    n = snap.particles.N
    nl = rna2.make_neighbor_list()
    bonded = rna2.bonded_force()
    excl = rna2.excluded_volume(nl)
    hb = rna2.hydrogen_bonding(nl)
    cross = rna2.cross_stacking(nl)
    dh = rna2.debye_huckel(nl, salt=1.0)

    def read(_):
        return dict(fene=bonded.fene_energy / n, b_exc=bonded.bonded_excvol_energy / n,
                    n_exc=excl.energy / n, hb=hb.energy / n,
                    cr_stack=cross.energy / n, debye=dh.energy / n)

    return H.single_point(snap, [bonded, excl, hb, cross, dh], read)


def _check(frame, terms):
    snap = H.snapshot(CASE, top_name=TOP, conf_name=CONF, frame=frame)
    got = _energies(snap)
    ref = H.split_energy(CASE, frame=frame + 1)  # row f+1 <-> output.dat frame f
    for t in terms:
        assert abs(got[t] - ref[COL[t]]) < 1e-4, (
            f"{CASE} f{frame} {t}: {got[t]:.6f} vs {ref[COL[t]]:.6f}"
        )


def test_rna2_fene_hb_cross_debye():
    _check(0, ["fene", "hb", "cr_stack", "debye"])


def test_rna2_bonded_excluded_volume():
    _check(47, ["b_exc"])  # frame with nonzero bonded excluded volume


def test_rna2_nonbonded_excluded_volume():
    _check(54, ["n_exc"])  # frame with nonzero nonbonded excluded volume


def test_rna2_debye_half_charged_ends():
    # same geometry as the full-charge case, only terminal charges = 0.5 differ, so
    # this specifically exercises the per-particle-charge Debye path.
    case = "simple-helix-rna2-12bp-half-charged-ends"
    snap = H.snapshot(case, half_charged_ends=True, top_name="sys.top", conf_name="init.conf")
    n = snap.particles.N
    nl = rna2.make_neighbor_list()
    dh = rna2.debye_huckel(nl, salt=1.0)
    e = H.single_point(snap, [dh], lambda fs: fs[0].energy) / n
    ref = H.split_energy(case, frame=0)[COL["debye"]]
    assert abs(e - ref) < 1e-4, f"debye(half-charged): {e:.6f} vs ref {ref:.6f}"


def test_rna2_coaxial_stacking():
    # simple-coax-rna2 uses the DNA-style generated.top / start.conf (the defaults)
    snap = H.snapshot(COAX_CASE)
    n = snap.particles.N
    nl = rna2.make_neighbor_list()
    coax = rna2.coaxial_stacking(nl)
    e = H.single_point(snap, [coax], lambda fs: fs[0].energy) / n
    ref = H.split_energy(COAX_CASE, frame=0)[COL["cx_stack"]]
    assert abs(e - ref) < 1e-4, f"cx_stack: {e:.6f} vs ref {ref:.6f}"


def test_rna2_coaxial_stacking_fd():
    def build():
        nl = rna2.make_neighbor_list()
        return [rna2.coaxial_stacking(nl)]

    H.fd_check(build, H.load_frame(COAX_CASE, frame=0))


def test_rna2_stacking():
    snap = H.snapshot(CASE, top_name="sys.top", conf_name="init.conf")
    n = snap.particles.N
    bonded = rna2.bonded_force()
    e = H.single_point(snap, [bonded], lambda fs: fs[0].stacking_energy) / n
    ref = H.split_energy(CASE, frame=0)[3]  # stack column (init.conf == row 0)
    assert abs(e - ref) < 1e-4, f"stack: {e:.6f} vs ref {ref:.6f}"


def test_rna2_bonded_force_fd():
    # FD-check the full bonded force (FENE + bonded excl vol + RNA stacking) together;
    # FENE/bexc are shared with the validated DNA path, so this exercises the new
    # stacking force/torque.
    def build():
        return [rna2.bonded_force()]

    H.fd_check(build, H.load_frame(CASE, frame=0, top_name="sys.top", conf_name="init.conf"))


def test_rna2_sequence_dependent():
    # per-pair HB (incl. G-U wobble) + stacking tables (seq_rna.txt), mixed sequence
    case = "simple-helix-rna2-12bp-ss"
    snap = H.snapshot(case, top_name="sys.top", conf_name="init.conf")
    n = snap.particles.N
    nl = rna2.make_neighbor_list()
    bonded = rna2.bonded_force(average=False)
    hb = rna2.hydrogen_bonding(nl, average=False)
    got = H.single_point(snap, [bonded, hb],
                         lambda fs: (fs[0].stacking_energy / n, fs[1].energy / n))
    ref = H.split_energy(case, frame=0)
    assert abs(got[0] - ref[3]) < 1e-4, f"stack: {got[0]:.6f} vs {ref[3]:.6f}"  # stack col
    assert abs(got[1] - ref[COL['hb']]) < 1e-4, f"hb: {got[1]:.6f} vs {ref[COL['hb']]:.6f}"


def _frame():
    return H.load_frame(CASE, frame=0, top_name=TOP, conf_name=CONF)


def test_rna2_cross_stacking_fd():
    def build():
        nl = rna2.make_neighbor_list()
        return [rna2.cross_stacking(nl)]

    H.fd_check(build, _frame())


def test_rna2_hydrogen_bonding_fd():
    def build():
        nl = rna2.make_neighbor_list()
        return [rna2.hydrogen_bonding(nl)]

    H.fd_check(build, _frame())
