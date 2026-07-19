"""oxDNA1 model: geometry, parameters, and force wiring.

Constants follow oxDNA/src/model.h (the analytic ``_nomesh`` reference). Stage 0
added the FENE backbone; Stage 1 adds the bonded and nonbonded excluded volume.
Further terms are layered in later stages.
"""

import itertools
import math

import hoomd

from ..forces import (
    OxDNABonded,
    ExcludedVolume,
    HydrogenBonding,
    CrossStacking,
    HBondCross,
    CoaxialStacking,
)
from ..io import DNA_BASES

# --- geometry (oxDNA/src/model.h; oxDNA1 = ungrooved, all sites along a1) ---
POS_BACK = -0.40
POS_STACK = 0.34
POS_BASE = 0.40
BACK_SITE = (POS_BACK, 0.0, 0.0)
BASE_SITE = (POS_BASE, 0.0, 0.0)

# --- FENE backbone (oxDNA/src/model.h) ---
FENE_EPS = 2.0
FENE_R0 = 0.7525  # FENE_R0_OXDNA
FENE_DELTA = 0.25

# --- excluded volume (oxDNA/src/model.h) ---
EXCL_EPS = 2.0
# (sigma, r_star, b, r_cut) for each site pair, indexed 1..4 as in model.h:
#   1: BACK-BACK, 2: BASE-BASE, 3/4: BASE-BACK / BACK-BASE (identical).
EXCL = {
    1: (0.70, 0.675, 892.016223343, 0.711879214356),
    2: (0.33, 0.32, 4119.70450017, 0.335388426126),
    3: (0.515, 0.50, 1707.30627298, 0.52329943261),
    4: (0.515, 0.50, 1707.30627298, 0.52329943261),
}

#: Conservative COM cutoff for the nonbonded excluded volume: max site r_cut
#: (BACK-BACK, ~0.712) plus the two BASE offsets (0.4 each).
EXCVOL_R_CUT = 2.0

# --- stacking (oxDNA/src/model.h) ---
STACK_SITE = (POS_STACK, 0.0, 0.0)
STACK_BACK_REF = (POS_BACK, 0.0, 0.0)  # ungrooved back reference for the phi angles
GAMMA = POS_STACK - POS_BACK  # 0.74
STCK_BASE_EPS = 1.3448  # STCK_BASE_EPS_OXDNA
STCK_FACT_EPS = 2.6568  # STCK_FACT_EPS_OXDNA
STCK_A, STCK_RC, STCK_R0 = 6.0, 0.9, 0.4
STCK_BLOW, STCK_BHIGH = -68.1857, -3.12992
STCK_RLOW, STCK_RHIGH, STCK_RCLOW, STCK_RCHIGH = 0.32, 0.75, 0.23239, 0.956
# f4: (A, B, T0, TS, TC);  f5: (A, B, XC, XS)
STCK_F4_THETA4 = (1.3, 6.4381, 0.0, 0.8, 0.961538)
STCK_F4_THETA5 = (0.9, 3.89361, 0.0, 0.95, 1.16959)  # theta6 uses the same set
STCK_F5_PHI1 = (2.0, 10.9032, -0.769231, -0.65)      # phi2 uses the same set


# Sequence-dependent stacking strengths (oxDNA1_sequence_dependent_parameters;
# keys are STCK_<base>_<base>). Used when average=False.
STCK_FACT_EPS_SEQ = 0.18
STCK_SEQ = {
    ("G", "C"): 1.684, ("C", "G"): 1.737, ("G", "G"): 1.604, ("C", "C"): 1.604,
    ("G", "A"): 1.590, ("T", "C"): 1.590, ("A", "G"): 1.610, ("C", "T"): 1.610,
    ("T", "G"): 1.654, ("C", "A"): 1.654, ("G", "T"): 1.671, ("A", "C"): 1.671,
    ("A", "T"): 1.553, ("T", "A"): 1.634, ("A", "A"): 1.709, ("T", "T"): 1.709,
}


def kT_from_temperature(t_kelvin):
    """oxDNA reduced temperature: _T = kT = 0.1 * T[K] / 300."""
    return 0.1 * t_kelvin / 300.0


def sequence_stacking_eps_table(t_kelvin):
    """4x4 sequence-dependent stacking epsilon, indexed by oxDNA base order (A,G,C,T).

    Follows DNAInteraction::init: eps[i][j] = STCK_<b_i>_<b_j> * (1 - f + T*9*f),
    with f = STCK_FACT_EPS and T = kT.
    """
    from ..io import DNA_BASES
    kT = kT_from_temperature(t_kelvin)
    f = STCK_FACT_EPS_SEQ
    factor = 1.0 - f + kT * 9.0 * f
    table = [[0.0] * 4 for _ in range(4)]
    for i, bi in enumerate(DNA_BASES):
        for j, bj in enumerate(DNA_BASES):
            table[i][j] = STCK_SEQ[(bi, bj)] * factor
    return table


def _stack_shift_factor():
    import math
    return (1.0 - math.exp(-(STCK_RC - STCK_R0) * STCK_A)) ** 2


def stacking_params(t_kelvin=296.15, eps_table=None):
    """Build the stacking sub-dict. Sequence-averaged unless eps_table (4x4) given."""
    kT = kT_from_temperature(t_kelvin)
    if eps_table is None:
        eps_avg = STCK_BASE_EPS + STCK_FACT_EPS * kT
        eps_flat = tuple([eps_avg] * 16)
    else:
        eps_flat = tuple(float(eps_table[i][j]) for i in range(4) for j in range(4))
    f1 = (STCK_A, STCK_R0, STCK_RLOW, STCK_RHIGH, STCK_RCLOW, STCK_RCHIGH,
          STCK_BLOW, STCK_BHIGH, _stack_shift_factor())
    return dict(
        stack_enabled=True,
        stack_site=STACK_SITE,
        stack_back_ref=STACK_BACK_REF,
        stack_gamma=GAMMA,
        stack_f1=f1,
        stack_eps=eps_flat,
        stack_f4_t4=STCK_F4_THETA4,
        stack_f4_t5=STCK_F4_THETA5,
        stack_f5_p1=STCK_F5_PHI1,
    )


def bonded_force(bond_type="backbone", temperature=296.15, stacking=True, average=True):
    """Build the oxDNA1 bonded backbone force (FENE + bonded excluded volume + stacking).

    ``average=False`` uses sequence-dependent stacking strengths.
    """
    force = OxDNABonded()
    params = dict(
        epsilon=FENE_EPS,
        r0=FENE_R0,
        delta=FENE_DELTA,
        pos_back=BACK_SITE,
        pos_base=BASE_SITE,
        excl_eps=EXCL_EPS,
        bexc_base_base=EXCL[2],
        bexc_base_back=EXCL[3],
        bexc_back_base=EXCL[4],
    )
    if stacking:
        eps_table = None if average else sequence_stacking_eps_table(temperature)
        params.update(stacking_params(temperature, eps_table=eps_table))
    force.params[bond_type] = params
    return force


def excluded_volume(nlist, types=DNA_BASES, r_cut=EXCVOL_R_CUT):
    """Build the oxDNA1 nonbonded excluded-volume pair force (all type pairs)."""
    force = ExcludedVolume(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair_params = dict(
        eps=EXCL_EPS,
        base_site=BASE_SITE,
        back_site=BACK_SITE,
        back_back=EXCL[1],
        base_base=EXCL[2],
        base_back=EXCL[3],
        back_base=EXCL[4],
    )
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair_params)
    return force


# --- hydrogen bonding (oxDNA/src/model.h) ---
HYDR_EPS = 1.077
HYDR_A, HYDR_RC, HYDR_R0 = 8.0, 0.75, 0.4
HYDR_BLOW, HYDR_BHIGH = -126.243, -7.87708
HYDR_RLOW, HYDR_RHIGH, HYDR_RCLOW, HYDR_RCHIGH = 0.34, 0.7, 0.276908, 0.783775
HB_F4_THETA1 = (1.5, 4.16038, 0.0, 0.7, 0.952381)
HB_F4_THETA23 = (1.5, 4.16038, 0.0, 0.7, 0.952381)  # theta2 == theta3
HB_F4_THETA4 = (0.46, 0.133855, math.pi, 0.7, 3.10559)
HB_F4_THETA78 = (4.0, 17.0526, math.pi / 2.0, 0.45, 0.555556)  # theta7 == theta8
HB_R_CUT = HYDR_RCHIGH + 2.0 * POS_BASE  # ~1.584
# sequence-dependent HB strengths (oxDNA1 seq-specific parameters)
HYDR_SEQ = {("A", "T"): 0.893, ("T", "A"): 0.893, ("C", "G"): 1.243, ("G", "C"): 1.243}
_BASE_IDX = {"A": 0, "G": 1, "C": 2, "T": 3}


def hb_shift_factor():
    return (1.0 - math.exp(-(HYDR_RC - HYDR_R0) * HYDR_A)) ** 2


def is_complementary(b1, b2):
    return _BASE_IDX[b1] + _BASE_IDX[b2] == 3


def hb_epsilon(b1, b2, average=True):
    """HB epsilon for a base pair: 0 unless Watson-Crick complementary."""
    if not is_complementary(b1, b2):
        return 0.0
    return HYDR_EPS if average else HYDR_SEQ[(b1, b2)]


def hydrogen_bonding(nlist, types=DNA_BASES, average=True, r_cut=HB_R_CUT):
    """Build the oxDNA1 hydrogen-bonding pair force (all type pairs)."""
    force = HydrogenBonding(nlist=nlist, default_r_cut=r_cut, mode="none")
    sf = hb_shift_factor()
    f1 = (HYDR_A, HYDR_R0, HYDR_RLOW, HYDR_RHIGH, HYDR_RCLOW, HYDR_RCHIGH, HYDR_BLOW, HYDR_BHIGH)
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(
            eps=hb_epsilon(t1, t2, average),
            shift_factor=sf,
            base_site=BASE_SITE,
            f1=f1,
            f4_t1=HB_F4_THETA1,
            f4_t23=HB_F4_THETA23,
            f4_t4=HB_F4_THETA4,
            f4_t78=HB_F4_THETA78,
        )
    return force


# --- cross-stacking (oxDNA/src/model.h) ---
# f2: (K, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh)
CRST_F2 = (47.5, 0.575, 0.675, 0.495, 0.655, 0.45, 0.7, -0.888889, -0.888889)
CRST_F4_THETA1 = (2.25, 7.00545, math.pi - 2.35, 0.58, 0.766284)
CRST_F4_THETA23 = (1.70, 6.2469, 1.0, 0.68, 0.865052)  # theta2 == theta3
CRST_F4_THETA4 = (1.50, 2.59556, 0.0, 0.65, 1.02564)
CRST_F4_THETA78 = (1.70, 6.2469, 0.875, 0.68, 0.865052)  # theta7 == theta8
CRST_RCHIGH = 0.7
CROSS_R_CUT = CRST_RCHIGH + 2.0 * POS_BASE  # 1.5


def cross_stacking(nlist, types=DNA_BASES, r_cut=CROSS_R_CUT):
    """Build the oxDNA1 cross-stacking pair force (all type pairs, sequence-independent)."""
    force = CrossStacking(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair = dict(
        base_site=BASE_SITE,
        f2=CRST_F2,
        f4_t1=CRST_F4_THETA1,
        f4_t23=CRST_F4_THETA23,
        f4_t4=CRST_F4_THETA4,
        f4_t78=CRST_F4_THETA78,
    )
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


def hbond_cross(nlist, types=DNA_BASES, average=True, r_cut=None):
    """Fused hydrogen-bonding + cross-stacking (one kernel; see forces.HBondCross).

    Numerically identical to hydrogen_bonding() + cross_stacking() run separately,
    but shares their common BASE-BASE geometry. Uses the larger of the two cutoffs;
    each term self-limits to its own range via its radial factor.
    """
    if r_cut is None:
        r_cut = max(HB_R_CUT, CROSS_R_CUT)
    force = HBondCross(nlist=nlist, default_r_cut=r_cut, mode="none")
    sf = hb_shift_factor()
    f1 = (HYDR_A, HYDR_R0, HYDR_RLOW, HYDR_RHIGH, HYDR_RCLOW, HYDR_RCHIGH, HYDR_BLOW, HYDR_BHIGH)
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(
            base_site=BASE_SITE,
            eps=hb_epsilon(t1, t2, average),
            shift_factor=sf,
            f1=f1,
            hb_f4_t1=HB_F4_THETA1,
            hb_f4_t23=HB_F4_THETA23,
            hb_f4_t4=HB_F4_THETA4,
            hb_f4_t78=HB_F4_THETA78,
            f2=CRST_F2,
            cs_f4_t1=CRST_F4_THETA1,
            cs_f4_t23=CRST_F4_THETA23,
            cs_f4_t4=CRST_F4_THETA4,
            cs_f4_t78=CRST_F4_THETA78,
        )
    return force


# --- coaxial stacking (oxDNA1; oxDNA/src/model.h) ---
# f2: (K, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh)
CXST_F2 = (46.0, 0.400, 0.6, 0.22, 0.58, 0.177778, 0.6222222, -2.13158, -2.13158)
CXST_F4_THETA1 = (2.0, 10.9032, math.pi - 0.60, 0.65, 0.769231)  # modified f4(t)+f4(2pi-t)
CXST_F4_THETA4 = (1.3, 6.4381, 0.0, 0.8, 0.961538)
CXST_F4_THETA56 = (0.9, 3.89361, 0.0, 0.95, 1.16959)  # theta5 == theta6
CXST_F5_PHI3 = (2.0, 10.9032, -0.769231, -0.65)  # phi4 == phi3
CXST_RCHIGH = 0.6222222
COAX_R_CUT = CXST_RCHIGH + 2.0 * POS_STACK  # ~1.302


def coaxial_stacking(nlist, types=DNA_BASES, r_cut=COAX_R_CUT):
    """Build the oxDNA1 coaxial-stacking pair force (all type pairs, sequence-independent)."""
    force = CoaxialStacking(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair = dict(
        stack_site=STACK_SITE,
        stack_back_ref=STACK_BACK_REF,
        gamma=GAMMA,
        f2=CXST_F2,
        f4_t1=CXST_F4_THETA1,
        f4_t4=CXST_F4_THETA4,
        f4_t56=CXST_F4_THETA56,
        f5_phi3=CXST_F5_PHI3,
        t1_mode=0,  # oxDNA1: reflection f4(t)+f4(2pi-t)
        t1_sa=0.0,
        t1_sb=0.0,
        phi3_enabled=True,
    )
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


def make_neighbor_list(buffer=0.4):
    """Cell neighbor list with bonded pairs excluded (as oxDNA's is_bonded check)."""
    return hoomd.md.nlist.Cell(buffer=buffer, exclusions=("bond",))


def forces(nlist=None, temperature=296.15, average=True, fused=False):
    """Assemble the complete oxDNA1 force field.

    Returns ``(force_list, nlist)``. The force list is
    [bonded (FENE + bonded exc-vol + stacking), excluded volume, hydrogen bonding,
    cross-stacking, coaxial stacking] — the seven oxDNA1 energy terms.

    With ``fused=True`` the hydrogen-bonding and cross-stacking terms are merged into
    one HBondCross force (one GPU kernel sharing their common geometry) instead of two
    separate forces; the physics is identical.
    """
    if nlist is None:
        nlist = make_neighbor_list()
    base_base = ([hbond_cross(nlist, average=average)] if fused
                 else [hydrogen_bonding(nlist, average=average), cross_stacking(nlist)])
    return (
        [
            bonded_force(temperature=temperature, average=average),
            excluded_volume(nlist),
            *base_base,
            coaxial_stacking(nlist),
        ],
        nlist,
    )
