"""oxRNA2 model: RNA geometry/constants over the shared oxDNA force engine.

oxRNA2 (oxDNA/src/Interactions/RNAInteraction2 = RNAInteraction + Debye-Huckel) shares
the oxDNA term structure. Five of the eight terms are identical in FORM to oxDNA2 and
reuse the same evaluators with RNA constants and geometry (this module): FENE backbone,
bonded and nonbonded excluded volume, hydrogen bonding, and Debye-Huckel. The RNA
variants of stacking, cross-stacking and coaxial stacking differ functionally and are
layered in separately (they need new/modified C++).

Constants follow oxDNA/src/Interactions/rna_model.h and RNAInteraction2.cpp. Base U is
handled as T (same type index) by oxdna.io.
"""

import itertools
import math

import hoomd

from ..forces import (
    OxDNABonded, ExcludedVolume, HydrogenBonding, CrossStacking, DebyeHuckel)
from ..io import DNA_BASES
from . import dna1

# --- geometry (rna_model.h) ---
# The RNA backbone site is tilted into the base normal (a3): -0.4 a1 + 0.2 a3.
POS_BASE = 0.40
POS_STACK = 0.34
BACK_SITE = (-0.40, 0.0, 0.20)      # RNA_POS_BACK_a1 / a2 / a3
BASE_SITE = (POS_BASE, 0.0, 0.0)    # RNA_POS_BASE * a1
STACK_SITE = (POS_STACK, 0.0, 0.0)  # RNA_POS_STACK * a1 (coaxial)

# --- FENE backbone (rna_model.h) ---
FENE_EPS = 2.0
FENE_R0 = 0.761070781051  # RNA_FENE_R0
FENE_DELTA = 0.25

# --- excluded volume: RNA_EXCL_* are numerically identical to oxDNA's EXCL ---
EXCL_EPS = 2.0
EXCL = dna1.EXCL
EXCVOL_R_CUT = 2.0

# --- hydrogen bonding (rna_model.h; f1/f4 shapes identical to DNA, eps differs) ---
HYDR_EPS = 0.870439  # RNA_HYDR_EPS (sequence-averaged)
HYDR_A, HYDR_RC, HYDR_R0 = 8.0, 0.75, 0.4
HYDR_RLOW, HYDR_RHIGH, HYDR_RCLOW, HYDR_RCHIGH = 0.34, 0.7, 0.276908, 0.783775
HYDR_BLOW, HYDR_BHIGH = -126.243, -7.87708
HB_F4_THETA1 = (1.5, 4.16038, 0.0, 0.7, 0.952381)
HB_F4_THETA23 = (1.5, 4.16038, 0.0, 0.7, 0.952381)  # theta2 == theta3
HB_F4_THETA4 = (0.46, 0.133855, math.pi, 0.7, 3.10559)
HB_F4_THETA78 = (4.0, 17.0526, math.pi / 2.0, 0.45, 0.555556)  # theta7 == theta8
HB_R_CUT = HYDR_RCHIGH + 2.0 * POS_BASE  # ~1.5

# --- cross-stacking (rna_model.h; RNA drops the f4(theta4) factor) ---
# f2: (K, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh)
CRST_F2 = (59.9626, 0.5, 0.6, 0.42, 0.58, 0.375, 0.625, -0.888889, -0.888889)
CRST_F4_THETA1 = (2.25, 7.00545, 0.505, 0.58, 0.766284)
CRST_F4_THETA23 = (1.70, 6.2469, 1.266, 0.68, 0.865052)  # theta2 == theta3
CRST_F4_THETA4 = (1.50, 2.59556, 0.0, 0.65, 1.02564)     # present but disabled (t4_enabled=False)
CRST_F4_THETA78 = (1.70, 6.2469, 0.309, 0.68, 0.865052)  # theta7 == theta8
CRST_RCHIGH = 0.625
CROSS_R_CUT = CRST_RCHIGH + 2.0 * POS_BASE  # ~1.425


def cross_stacking(nlist, types=DNA_BASES, r_cut=CROSS_R_CUT):
    """oxRNA2 cross-stacking (BASE-BASE, theta4 factor dropped: t4_enabled=False)."""
    force = CrossStacking(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair = dict(
        base_site=BASE_SITE, f2=CRST_F2, f4_t1=CRST_F4_THETA1, f4_t23=CRST_F4_THETA23,
        f4_t4=CRST_F4_THETA4, f4_t78=CRST_F4_THETA78, t4_enabled=False,
    )
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


# --- Debye-Huckel (RNAInteraction2.cpp; same form as oxDNA2, RNA constants) ---
DH_LAMBDA_FACTOR = 0.3667258  # dh_lambda
DH_PREFACTOR = 0.0858         # dh_strength
DEFAULT_SALT = 1.0


def bonded_force(bond_type="backbone", temperature=296.15):
    """oxRNA2 bonded force: FENE + bonded excluded volume.

    Stacking is a functionally different RNA term (asymmetric sites + backbone
    angles) added separately, so it is left disabled here (stack_enabled defaults
    False in the C++ params).
    """
    force = OxDNABonded()
    force.params[bond_type] = dict(
        epsilon=FENE_EPS, r0=FENE_R0, delta=FENE_DELTA,
        pos_back=BACK_SITE, pos_base=BASE_SITE,
        excl_eps=EXCL_EPS,
        bexc_base_base=EXCL[2], bexc_base_back=EXCL[3], bexc_back_base=EXCL[4],
    )
    return force


def excluded_volume(nlist, types=DNA_BASES, r_cut=EXCVOL_R_CUT):
    """oxRNA2 nonbonded excluded volume (BACK/BASE site repulsions)."""
    force = ExcludedVolume(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair = dict(
        eps=EXCL_EPS, base_site=BASE_SITE, back_site=BACK_SITE,
        back_back=EXCL[1], base_base=EXCL[2], base_back=EXCL[3], back_base=EXCL[4],
    )
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


def hydrogen_bonding(nlist, types=DNA_BASES, r_cut=HB_R_CUT):
    """oxRNA2 hydrogen bonding (sequence-averaged, Watson-Crick complementary only)."""
    force = HydrogenBonding(nlist=nlist, default_r_cut=r_cut, mode="none")
    sf = (1.0 - math.exp(-(HYDR_RC - HYDR_R0) * HYDR_A)) ** 2
    f1 = (HYDR_A, HYDR_R0, HYDR_RLOW, HYDR_RHIGH, HYDR_RCLOW, HYDR_RCHIGH, HYDR_BLOW, HYDR_BHIGH)
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        eps = HYDR_EPS if dna1.is_complementary(t1, t2) else 0.0
        force.params[(t1, t2)] = dict(
            eps=eps, shift_factor=sf, base_site=BASE_SITE, f1=f1,
            f4_t1=HB_F4_THETA1, f4_t23=HB_F4_THETA23, f4_t4=HB_F4_THETA4, f4_t78=HB_F4_THETA78,
        )
    return force


def debye_params(t_kelvin=296.15, salt=DEFAULT_SALT):
    """Derived Debye-Huckel parameters (RNAInteraction2::_debye_huckel init)."""
    kT = dna1.kT_from_temperature(t_kelvin)
    lam = DH_LAMBDA_FACTOR * math.sqrt(kT / 0.1) / math.sqrt(salt)
    minus_kappa = -1.0 / lam
    rhigh = 3.0 * lam
    x, q, l = rhigh, DH_PREFACTOR, lam
    b_smooth = -(math.exp(-x / l) * q * q * (x + l) * (x + l)) / (-4.0 * x * x * x * l * l * q)
    rc = x * (q * x + 3.0 * q * l) / (q * (x + l))
    return minus_kappa, DH_PREFACTOR, b_smooth, rc, rhigh


def debye_huckel(nlist, types=DNA_BASES, t_kelvin=296.15, salt=DEFAULT_SALT, r_cut=None):
    """oxRNA2 Debye-Huckel electrostatics (BACK-BACK screened Coulomb)."""
    mk, pref, b_smooth, rc, rhigh = debye_params(t_kelvin, salt)
    if r_cut is None:
        back_off = math.sqrt(sum(c * c for c in BACK_SITE))
        r_cut = rc + 2.0 * back_off
    force = DebyeHuckel(nlist=nlist, default_r_cut=r_cut, mode="none")
    pair = dict(back_site=BACK_SITE, debye=(mk, pref, b_smooth, rc, rhigh))
    for t1, t2 in itertools.combinations_with_replacement(types, 2):
        force.params[(t1, t2)] = dict(pair)
    return force


def make_neighbor_list(buffer=0.4):
    return hoomd.md.nlist.Cell(buffer=buffer, exclusions=("bond",))


def forces(nlist=None, temperature=296.15, salt=DEFAULT_SALT):
    """oxRNA2 forces. PHASE A: the five reuse terms (FENE, bonded/nonbonded excluded
    volume, hydrogen bonding, Debye-Huckel). The RNA-specific stacking, cross-stacking
    and coaxial-stacking terms are added in later phases.
    """
    if nlist is None:
        nlist = make_neighbor_list()
    return (
        [
            bonded_force(temperature=temperature),
            excluded_volume(nlist),
            hydrogen_bonding(nlist),
            cross_stacking(nlist),
            debye_huckel(nlist, t_kelvin=temperature, salt=salt),
        ],
        nlist,
    )
