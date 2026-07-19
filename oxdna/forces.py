"""Thin HOOMD Python wrappers over the compiled oxDNA force engine.

These mirror the ``hoomd.md.bond.Bond`` / ``hoomd.md.pair.aniso`` wrapper idiom
but bind to this plugin's ``_engine`` module instead of ``hoomd.md._md``.
"""

from hoomd.md.force import Force
from hoomd.md.pair.aniso import AnisotropicPair
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict

from . import _engine


class OxDNABonded(Force):
    """oxDNA bonded backbone force between 3'-5' neighbours.

    Implements the FENE backbone spring and the bonded excluded volume. Parameters
    are keyed by bond type:

        * ``epsilon``, ``r0``, ``delta`` - FENE energy scale, rest length, range
        * ``pos_back``, ``pos_base`` - body-frame BACK / BASE site offsets
        * ``excl_eps`` - excluded-volume energy scale (EXCL_EPS)
        * ``bexc_base_base`` / ``bexc_base_back`` / ``bexc_back_base`` -
          (sigma, r_star, b, r_cut) for each bonded excluded-volume site pair
    """

    _ext_module = _engine
    _cpp_class_name = "OxDNABondedForceCompute"

    def __init__(self):
        super().__init__()
        params = TypeParameter(
            "params",
            "bond_types",
            TypeParameterDict(
                epsilon=float,
                r0=float,
                delta=float,
                pos_back=(float, float, float),
                pos_base=(float, float, float),
                excl_eps=float,
                bexc_base_base=(float, float, float, float),
                bexc_base_back=(float, float, float, float),
                bexc_back_base=(float, float, float, float),
                # stacking (flattened; disabled by default)
                stack_enabled=bool,
                stack_site=(float, float, float),
                stack_back_ref=(float, float, float),
                stack_gamma=float,
                stack_f1=(float,) * 9,
                stack_eps=(float,) * 16,
                stack_f4_t4=(float,) * 5,
                stack_f4_t5=(float,) * 5,
                stack_f5_p1=(float,) * 4,
                # oxRNA stacking (asymmetric sites + backbone angles); off by default
                stack_rna=bool,
                stack_3_site=(float, float, float),
                stack_5_site=(float, float, float),
                stack_bbvector_3=(float, float, float),
                stack_bbvector_5=(float, float, float),
                stack_f4_tB1=(float,) * 5,
                stack_f4_tB2=(float,) * 5,
                len_keys=1,
                _defaults={
                    "stack_enabled": False,
                    "stack_site": (0.0, 0.0, 0.0),
                    "stack_back_ref": (0.0, 0.0, 0.0),
                    "stack_gamma": 0.0,
                    "stack_f1": (0.0,) * 9,
                    "stack_eps": (0.0,) * 16,
                    "stack_f4_t4": (0.0,) * 5,
                    "stack_f4_t5": (0.0,) * 5,
                    "stack_f5_p1": (0.0,) * 4,
                    "stack_rna": False,
                    "stack_3_site": (0.0, 0.0, 0.0),
                    "stack_5_site": (0.0, 0.0, 0.0),
                    "stack_bbvector_3": (0.0, 0.0, 0.0),
                    "stack_bbvector_5": (0.0, 0.0, 0.0),
                    "stack_f4_tB1": (0.0,) * 5,
                    "stack_f4_tB2": (0.0,) * 5,
                },
            ),
        )
        self._add_typeparam(params)

    def _attach_hook(self):
        import hoomd

        name = self._cpp_class_name
        if isinstance(self._simulation.device, hoomd.device.GPU):
            name += "GPU"
        cpp_cls = getattr(self._ext_module, name)
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)

    @property
    def fene_energy(self):
        """Total FENE backbone energy (computed on read)."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getFeneEnergy()

    @property
    def bonded_excvol_energy(self):
        """Total bonded excluded-volume energy (computed on read)."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getBondedExcVolEnergy()

    @property
    def stacking_energy(self):
        """Total stacking energy (computed on read)."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getStackingEnergy()


class ExcludedVolume(AnisotropicPair):
    """oxDNA nonbonded excluded volume (BACK/BASE site-site repulsions).

    Per-type-pair parameters:

        * ``eps`` - EXCL_EPS
        * ``base_site`` / ``back_site`` - body-frame BASE / BACK offsets
        * ``back_back`` / ``base_base`` / ``base_back`` / ``back_base`` -
          (sigma, r_star, b, r_cut) for each of the four site pairs
    """

    _ext_module = _engine
    _cpp_class_name = "AnisoPotentialPairOxDNAExcVol"

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                eps=float,
                base_site=(float, float, float),
                back_site=(float, float, float),
                back_back=(float, float, float, float),
                base_base=(float, float, float, float),
                base_back=(float, float, float, float),
                back_base=(float, float, float, float),
                len_keys=2,
            ),
        )
        self._add_typeparam(params)


class HydrogenBonding(AnisotropicPair):
    """oxDNA Watson-Crick hydrogen bonding (BASE-BASE, complementary pairs only).

    Per-type-pair parameters: ``eps`` (0 for non-complementary pairs),
    ``shift_factor``, ``base_site``, ``f1`` (8-tuple), and the four f4 angular sets
    ``f4_t1`` / ``f4_t23`` / ``f4_t4`` / ``f4_t78`` (each a 5-tuple A,B,T0,TS,TC).
    """

    _ext_module = _engine
    _cpp_class_name = "AnisoPotentialPairOxDNAHBond"

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                eps=float,
                shift_factor=float,
                base_site=(float, float, float),
                f1=(float,) * 8,
                f4_t1=(float,) * 5,
                f4_t23=(float,) * 5,
                f4_t4=(float,) * 5,
                f4_t78=(float,) * 5,
                len_keys=2,
            ),
        )
        self._add_typeparam(params)


class CrossStacking(AnisotropicPair):
    """oxDNA cross-stacking (BASE-BASE). Per-type-pair parameters:

        * ``base_site`` - body-frame BASE offset
        * ``f2`` (9-tuple K, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh)
        * ``f4_t1`` / ``f4_t23`` / ``f4_t4`` / ``f4_t78`` (each a 5-tuple)
    """

    _ext_module = _engine
    _cpp_class_name = "AnisoPotentialPairOxDNACrossStacking"

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                base_site=(float, float, float),
                f2=(float,) * 9,
                f4_t1=(float,) * 5,
                f4_t23=(float,) * 5,
                f4_t4=(float,) * 5,
                f4_t78=(float,) * 5,
                t4_enabled=bool,  # oxDNA keeps f4(theta4); oxRNA drops it
                len_keys=2,
                _defaults={"t4_enabled": True},
            ),
        )
        self._add_typeparam(params)


class HBondCross(AnisotropicPair):
    """Fused hydrogen-bonding + cross-stacking (BASE-BASE).

    Both terms act between the same BASE sites and share their entire six-angle
    geometry, so one evaluator reconstructs it once and applies both — one GPU kernel
    and one neighbour-list traversal instead of two. Numerically identical to running
    ``HydrogenBonding`` and ``CrossStacking`` separately (they still exist for
    per-term validation). Per-type-pair parameters are the union of the two: the H-bond
    ``eps`` / ``shift_factor`` / ``f1`` / ``hb_f4_*`` and the cross ``f2`` / ``cs_f4_*``,
    plus the shared ``base_site``.
    """

    _ext_module = _engine
    _cpp_class_name = "AnisoPotentialPairOxDNAHBondCross"

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                base_site=(float, float, float),
                eps=float,
                shift_factor=float,
                f1=(float,) * 8,
                hb_f4_t1=(float,) * 5,
                hb_f4_t23=(float,) * 5,
                hb_f4_t4=(float,) * 5,
                hb_f4_t78=(float,) * 5,
                f2=(float,) * 9,
                cs_f4_t1=(float,) * 5,
                cs_f4_t23=(float,) * 5,
                cs_f4_t4=(float,) * 5,
                cs_f4_t78=(float,) * 5,
                len_keys=2,
            ),
        )
        self._add_typeparam(params)


class CoaxialStacking(AnisotropicPair):
    """oxDNA1 coaxial stacking (STACK-STACK). Per-type-pair parameters:

        * ``stack_site`` / ``stack_back_ref`` - STACK and ungrooved-BACK offsets
        * ``gamma`` - POS_STACK - POS_BACK
        * ``f2`` (9-tuple) and f4 sets ``f4_t1`` / ``f4_t4`` / ``f4_t56`` (5-tuples)
        * ``f5_phi3`` (4-tuple A, B, XC, XS)
    """

    _ext_module = _engine
    _cpp_class_name = "AnisoPotentialPairOxDNACoaxStacking"

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                stack_site=(float, float, float),
                stack_back_ref=(float, float, float),
                gamma=float,
                f2=(float,) * 9,
                f4_t1=(float,) * 5,
                f4_t4=(float,) * 5,
                f4_t56=(float,) * 5,
                f5_phi3=(float,) * 4,
                f5_phi4=(float,) * 4,  # oxRNA second dihedral (defaults to phi3)
                t1_mode=int,
                t1_sa=float,
                t1_sb=float,
                phi3_enabled=bool,
                rna_coax=bool,  # oxRNA two-dihedral f5(phi3)*f5(phi4) form
                len_keys=2,
                _defaults={"f5_phi4": (0.0,) * 4, "rna_coax": False},
            ),
        )
        self._add_typeparam(params)


class DebyeHuckel(AnisotropicPair):
    """oxDNA2 Debye-Huckel electrostatics (BACK-BACK screened Coulomb).

    Per-type-pair parameters ``back_site`` (grooved BACK offset) and ``debye``
    (5-tuple minus_kappa, prefactor, b_smooth, rc, rhigh). Terminal nucleotides use
    half charge, supplied as a per-particle charge in the snapshot.
    """

    _ext_module = _engine
    _cpp_class_name = "AnisoPotentialPairOxDNADebye"

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            "params",
            "particle_types",
            TypeParameterDict(
                back_site=(float, float, float),
                debye=(float, float, float, float, float),
                len_keys=2,
            ),
        )
        self._add_typeparam(params)
