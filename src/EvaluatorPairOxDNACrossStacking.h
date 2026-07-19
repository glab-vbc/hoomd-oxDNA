// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file EvaluatorPairOxDNACrossStacking.h
    \brief oxDNA cross-stacking evaluator for AnisoPotentialPair.

    DNAInteraction::_cross_stacking, between the BASE sites:
        U = f2(r) * f4(t1) f4(t2) f4(t3) f4(t4) f4(t7) f4(t8),
    with t4/t7/t8 symmetrized. Shares the six-angle energy/force/torque math with
    hydrogen bonding via evaluate_base_base(); this term supplies the f2 radial
    factor and DOES symmetrize t4/t7/t8. Not sequence dependent.
*/

#ifndef __EVALUATOR_PAIR_OXDNA_CROSSSTACKING_H__
#define __EVALUATOR_PAIR_OXDNA_CROSSSTACKING_H__

#include "OxDNAAnisoPair.h"
#include "OxDNABaseBase.h"
#ifndef __HIPCC__
#include "OxDNAParamIO.h"
#include <string>
#endif

namespace hoomd
    {
namespace md
    {

//! Parameters for one type pair of the oxDNA cross-stacking term.
struct OxDNACrossStackingParams
    {
    vec3<Scalar> base_site;
    // f2: k, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh
    Scalar f2_k, f2_r0, f2_rc, f2_rlow, f2_rhigh, f2_rclow, f2_rchigh, f2_blow, f2_bhigh;
    Scalar t1[5], t23[5], t4[5], t78[5];
    bool t4_enabled; //!< oxDNA keeps the f4(theta4) factor; oxRNA drops it (false)

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    DEVICE void load_shared(char*&, unsigned int&) { }
    HOSTDEVICE void allocate_shared(char*&, unsigned int&) const { }

    HOSTDEVICE OxDNACrossStackingParams()
        : base_site(0, 0, 0), f2_k(0), f2_r0(0), f2_rc(0), f2_rlow(0), f2_rhigh(0),
          f2_rclow(0), f2_rchigh(0), f2_blow(0), f2_bhigh(0), t4_enabled(true)
        {
        for (int k = 0; k < 5; k++)
            t1[k] = t23[k] = t4[k] = t78[k] = 0;
        }

#ifndef __HIPCC__
    OxDNACrossStackingParams(pybind11::dict v, bool managed)
        {
        base_site = oxdna_io::read_vec3(v["base_site"]);
        pybind11::tuple f2(v["f2"]);
        f2_k = f2[0].cast<Scalar>();
        f2_r0 = f2[1].cast<Scalar>();
        f2_rc = f2[2].cast<Scalar>();
        f2_rlow = f2[3].cast<Scalar>();
        f2_rhigh = f2[4].cast<Scalar>();
        f2_rclow = f2[5].cast<Scalar>();
        f2_rchigh = f2[6].cast<Scalar>();
        f2_blow = f2[7].cast<Scalar>();
        f2_bhigh = f2[8].cast<Scalar>();
        oxdna_io::read_scalars<5>(v["f4_t1"], t1);
        oxdna_io::read_scalars<5>(v["f4_t23"], t23);
        oxdna_io::read_scalars<5>(v["f4_t4"], t4);
        oxdna_io::read_scalars<5>(v["f4_t78"], t78);
        t4_enabled = v.contains("t4_enabled") ? v["t4_enabled"].cast<bool>() : true;
        }

    pybind11::object toPython()
        {
        pybind11::dict v;
        v["t4_enabled"] = t4_enabled;
        v["base_site"] = oxdna_io::pack_vec3(base_site);
        v["f2"] = pybind11::make_tuple(f2_k, f2_r0, f2_rc, f2_rlow, f2_rhigh, f2_rclow,
                                       f2_rchigh, f2_blow, f2_bhigh);
        v["f4_t1"] = oxdna_io::pack_scalars<5>(t1);
        v["f4_t23"] = oxdna_io::pack_scalars<5>(t23);
        v["f4_t4"] = oxdna_io::pack_scalars<5>(t4);
        v["f4_t78"] = oxdna_io::pack_scalars<5>(t78);
        return std::move(v);
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

class EvaluatorPairOxDNACrossStacking
    : public OxDNAAnisoPair<EvaluatorPairOxDNACrossStacking, OxDNACrossStackingParams>
    {
    public:
    typedef OxDNAAnisoPair<EvaluatorPairOxDNACrossStacking, OxDNACrossStackingParams> Base;
    using Base::Base;

    HOSTDEVICE bool evaluate_scalar(Scalar3& force,
                                    Scalar& pair_eng,
                                    bool /*energy_shift*/,
                                    Scalar3& torque_i,
                                    Scalar3& torque_j)
        {
        oxdna::F2Radial radial {p.f2_k,     p.f2_r0,     p.f2_rc,   p.f2_rlow, p.f2_rhigh,
                                p.f2_rclow,  p.f2_rchigh, p.f2_blow, p.f2_bhigh};
        return oxdna::evaluate_base_base(dr, quat_i, quat_j, rcutsq, p.base_site,
                                         p.t1, p.t23, p.t4, p.t78, /*symmetrize=*/true,
                                         radial, force, pair_eng, torque_i, torque_j,
                                         p.t4_enabled);
        }

#ifndef __HIPCC__
    static std::string getName() { return "oxdna_crossstacking"; }
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_OXDNA_CROSSSTACKING_H__
