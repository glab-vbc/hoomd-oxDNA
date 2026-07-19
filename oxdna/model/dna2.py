"""oxDNA2 model: grooved geometry, updated constants, and Debye-Huckel.

oxDNA2 differs from oxDNA1 (see DNA2Interaction) in:
  * a grooved BACK site (POS_MM_BACK1 a1 + POS_MM_BACK2 a2) used by FENE, excluded
    volume and Debye-Huckel (stacking/coaxial keep the ungrooved back reference);
  * FENE r0, stacking/HB epsilons;
  * a modified coaxial term (harmonic theta1, no phi dihedral, stiffer K);
  * an added Debye-Huckel electrostatic term with half-charged strand ends.

Everything else (cross-stacking, the f1..f6 shapes, the bonded/nonbonded geometry
machinery) is reused from oxDNA1.
"""

import itertools
import math

import hoomd

from ..forces import (
    OxDNABonded, ExcludedVolume, HydrogenBonding, HBondCross, CoaxialStacking, DebyeHuckel)
from ..io import DNA_BASES
from . import dna1

# --- grooved geometry (oxDNA/src/model.h) ---
POS_MM_BACK1, POS_MM_BACK2 = -0.3400, 0.3408
BACK_SITE = (POS_MM_BACK1, POS_MM_BACK2, 0.0)

# --- updated constants ---
FENE_R0 = 0.7564  # FENE_R0_OXDNA2
STCK_BASE_EPS = 1.3523  # STCK_BASE_EPS_OXDNA2
STCK_FACT_EPS = 2.6717  # STCK_FACT_EPS_OXDNA2
HYDR_EPS = 1.0678       # HYDR_EPS_OXDNA2

# --- sequence-dependent tables (seq_oxdna2.txt; DNAInteraction.cpp:345,368) ---
# HB: per-pair epsilon directly. Stacking: STCK_<pair> * (1 - f + kT*9*f), f=0.18.
STCK_FACT_EPS_SEQ = 0.18
STCK_SEQ = {
    ("G", "C"): 1.69339, ("C", "G"): 1.74669, ("G", "G"): 1.61295, ("C", "C"): 1.61295,
    ("G", "A"): 1.59887, ("T", "C"): 1.59887, ("A", "G"): 1.61898, ("C", "T"): 1.61898,
    ("T", "G"): 1.66322, ("C", "A"): 1.66322, ("G", "T"): 1.68032, ("A", "C"): 1.68032,
    ("A", "T"): 1.56166, ("T", "A"): 1.64311, ("A", "A"): 1.84642, ("T", "T"): 1.58952,
}
HYDR_SEQ = {("A", "T"): 0.88537, ("T", "A"): 0.88537,
            ("C", "G"): 1.23238, ("G", "C"): 1.23238}

# --- coaxial (oxDNA2 form: harmonic theta1, no phi3, stiffer K) ---
CXST_K = 58.5  # CXST_K_OXDNA2
CXST_F4_THETA1 = (2.0, 10.9032, math.pi - 0.25, 0.65, 0.769231)  # T0 = PI - 0.25
CXST_THETA1_SA = 20.0
CXST_THETA1_SB = math.pi - 0.1 * (math.pi - (math.pi - 0.25))  # PI - 0.025

# --- Debye-Huckel ---
DH_LAMBDA_FACTOR = 0.3616455
DH_PREFACTOR = 0.0543
DEFAULT_SALT = 0.5


def stacking_eps_table(t_kelvin=296.15):
    """oxDNA2 sequence-averaged stacking epsilon table (uniform 4x4)."""
    kT = dna1.kT_from_temperature(t_kelvin)
    eps = STCK_BASE_EPS + STCK_FACT_EPS * kT
    return [[eps] * 4 for _ in range(4)]


def sequence_stacking_eps_table(t_kelvin=296.15):
    """oxDNA2 sequence-dependent stacking epsilon (DNAInteraction.cpp:345)."""
    kT = dna1.kT_from_temperature(t_kelvin)
    factor = 1.0 - STCK_FACT_EPS_SEQ + kT * 9.0 * STCK_FACT_EPS_SEQ
    return [[STCK_SEQ[(DNA_BASES[i], DNA_BASES[j])] * factor for j in range(4)]
            for i in range(4)]


def _stacking_params(t_kelvin, average=True):
    sp = dna1.stacking_params(t_kelvin)  # f1/f4/f5 shapes are shared with oxDNA1
    table = stacking_eps_table(t_kelvin) if average else sequence_stacking_eps_table(t_kelvin)
    sp["stack_eps"] = tuple(table[i][j] for i in range(4) for j in range(4))
    return sp


def bonded_force(bond_type="backbone", temperature=296.15, stacking=True, average=True):
    """oxDNA2 bonded force (grooved BACK, FENE r0, oxDNA2 stacking eps)."""
    force = OxDNABonded()
    params = dict(
        epsilon=dna1.FENE_EPS,
        r0=FENE_R0,
        delta=dna1.FENE_DELTA,
        pos_back=BACK_SITE,
        pos_base=dna1.BASE_SITE,
        excl_eps=dna1.EXCL_EPS,
        bexc_base_base=dna1.EXCL[2],
        bexc_base_back=dna1.EXCL[3],
        bexc_back_base=dna1.EXCL[4],
    )
    if stacking:
        params.update(_stacking_params(temperature, average=average))
    force.params[bond_type] = params
    return force


def excluded_volume(nlist, types=DNA_BASES, r_cut=dna1.EXCVOL_R_CUT):
    """oxDNA2 nonbonded excluded volume (grooved BACK site)."""
    force = ExcludedVolume(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair = dict(
        eps=dna1.EXCL_EPS,
        base_site=dna1.BASE_SITE,
        back_site=BACK_SITE,
        back_back=dna1.EXCL[1],
        base_base=dna1.EXCL[2],
        base_back=dna1.EXCL[3],
        back_base=dna1.EXCL[4],
    )
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


def hb_epsilon(t1, t2, average=True):
    """oxDNA2 HB epsilon for a type pair (0 unless Watson-Crick complementary)."""
    if average:
        return HYDR_EPS if dna1.is_complementary(t1, t2) else 0.0
    return HYDR_SEQ.get((t1, t2), 0.0)


def hydrogen_bonding(nlist, types=DNA_BASES, average=True, r_cut=dna1.HB_R_CUT):
    """oxDNA2 hydrogen bonding (HYDR_EPS_OXDNA2 averaged, or per-pair when average=False)."""
    force = HydrogenBonding(nlist=nlist, default_r_cut=r_cut, mode="none")
    sf = dna1.hb_shift_factor()
    f1 = (dna1.HYDR_A, dna1.HYDR_R0, dna1.HYDR_RLOW, dna1.HYDR_RHIGH, dna1.HYDR_RCLOW,
          dna1.HYDR_RCHIGH, dna1.HYDR_BLOW, dna1.HYDR_BHIGH)
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(
            eps=hb_epsilon(t1, t2, average), shift_factor=sf, base_site=dna1.BASE_SITE, f1=f1,
            f4_t1=dna1.HB_F4_THETA1, f4_t23=dna1.HB_F4_THETA23,
            f4_t4=dna1.HB_F4_THETA4, f4_t78=dna1.HB_F4_THETA78,
        )
    return force


def hbond_cross(nlist, types=DNA_BASES, average=True, r_cut=None):
    """Fused oxDNA2 hydrogen-bonding + (oxDNA1) cross-stacking; see forces.HBondCross."""
    if r_cut is None:
        r_cut = max(dna1.HB_R_CUT, dna1.CROSS_R_CUT)
    force = HBondCross(nlist=nlist, default_r_cut=r_cut, mode="none")
    sf = dna1.hb_shift_factor()
    f1 = (dna1.HYDR_A, dna1.HYDR_R0, dna1.HYDR_RLOW, dna1.HYDR_RHIGH, dna1.HYDR_RCLOW,
          dna1.HYDR_RCHIGH, dna1.HYDR_BLOW, dna1.HYDR_BHIGH)
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        eps = hb_epsilon(t1, t2, average)
        force.params[(t1, t2)] = dict(
            base_site=dna1.BASE_SITE, eps=eps, shift_factor=sf, f1=f1,
            hb_f4_t1=dna1.HB_F4_THETA1, hb_f4_t23=dna1.HB_F4_THETA23,
            hb_f4_t4=dna1.HB_F4_THETA4, hb_f4_t78=dna1.HB_F4_THETA78,
            f2=dna1.CRST_F2, cs_f4_t1=dna1.CRST_F4_THETA1, cs_f4_t23=dna1.CRST_F4_THETA23,
            cs_f4_t4=dna1.CRST_F4_THETA4, cs_f4_t78=dna1.CRST_F4_THETA78,
        )
    return force


def coaxial_stacking(nlist, types=DNA_BASES, r_cut=dna1.COAX_R_CUT):
    """oxDNA2 coaxial stacking (harmonic theta1, no phi3, stiffer K)."""
    force = CoaxialStacking(nlist=nlist, default_r_cut=r_cut, mode="none")
    f2 = (CXST_K,) + dna1.CXST_F2[1:]
    pair = dict(
        stack_site=dna1.STACK_SITE,
        stack_back_ref=dna1.STACK_BACK_REF,
        gamma=dna1.GAMMA,
        f2=f2,
        f4_t1=CXST_F4_THETA1,
        f4_t4=dna1.CXST_F4_THETA4,
        f4_t56=dna1.CXST_F4_THETA56,
        f5_phi3=dna1.CXST_F5_PHI3,  # unused (phi3 disabled)
        t1_mode=1,  # oxDNA2: harmonic theta1
        t1_sa=CXST_THETA1_SA,
        t1_sb=CXST_THETA1_SB,
        phi3_enabled=False,
    )
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


def debye_params(t_kelvin=296.15, salt=DEFAULT_SALT):
    """Derived Debye-Huckel parameters (DNA2Interaction::init)."""
    kT = dna1.kT_from_temperature(t_kelvin)
    lam = DH_LAMBDA_FACTOR * math.sqrt(kT / 0.1) / math.sqrt(salt)
    minus_kappa = -1.0 / lam
    rhigh = 3.0 * lam
    x, q, l = rhigh, DH_PREFACTOR, lam
    b_smooth = -(math.exp(-x / l) * q * q * (x + l) * (x + l)) / (-4.0 * x * x * x * l * l * q)
    rc = x * (q * x + 3.0 * q * l) / (q * (x + l))
    return minus_kappa, DH_PREFACTOR, b_smooth, rc, rhigh


def debye_huckel(nlist, types=DNA_BASES, t_kelvin=296.15, salt=DEFAULT_SALT, r_cut=None):
    """oxDNA2 Debye-Huckel electrostatics (BACK-BACK, half-charged ends)."""
    mk, pref, b_smooth, rc, rhigh = debye_params(t_kelvin, salt)
    if r_cut is None:
        back_off = math.sqrt(POS_MM_BACK1 ** 2 + POS_MM_BACK2 ** 2)
        r_cut = rc + 2.0 * back_off
    force = DebyeHuckel(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair = dict(back_site=BACK_SITE, debye=(mk, pref, b_smooth, rc, rhigh))
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


def make_neighbor_list(buffer=0.4):
    return hoomd.md.nlist.Cell(buffer=buffer, exclusions=("bond",))


def forces(nlist=None, temperature=296.15, salt=DEFAULT_SALT, fused=False, average=True):
    """Assemble the complete oxDNA2 force field (eight terms).

    Returns ``(force_list, nlist)`` with
    [bonded, excluded volume, hydrogen bonding, cross-stacking, coaxial stacking,
    Debye-Huckel]. ``average=False`` uses the oxDNA2 sequence-dependent HB and stacking
    tables.

    With ``fused=True`` hydrogen bonding and cross-stacking are merged into one
    HBondCross force (one GPU kernel sharing their geometry); the physics is identical.
    """
    if nlist is None:
        nlist = make_neighbor_list()
    base_base = ([hbond_cross(nlist, average=average)] if fused
                 else [hydrogen_bonding(nlist, average=average), dna1.cross_stacking(nlist)])
    return (
        [
            bonded_force(temperature=temperature, average=average),
            excluded_volume(nlist),
            *base_base,
            coaxial_stacking(nlist),
            debye_huckel(nlist, t_kelvin=temperature, salt=salt),
        ],
        nlist,
    )
