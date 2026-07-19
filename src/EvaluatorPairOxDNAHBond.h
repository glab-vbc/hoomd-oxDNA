// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file EvaluatorPairOxDNAHBond.h
    \brief oxDNA hydrogen-bonding evaluator for AnisoPotentialPair.

    Watson-Crick base pairing (DNAInteraction::_hydrogen_bonding): between the BASE
    sites, U = f1(r_hb) * f4(t1) f4(t2) f4(t3) f4(t4) f4(t7) f4(t8), nonzero only for
    complementary pairs (encoded as a zero per-type-pair epsilon). The six-angle
    energy/force/torque math is shared with cross-stacking via evaluate_base_base();
    this term supplies the f1 radial factor and does NOT symmetrize t4/t7/t8.
*/

#ifndef __EVALUATOR_PAIR_OXDNA_HBOND_H__
#define __EVALUATOR_PAIR_OXDNA_HBOND_H__

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

//! Parameters for one type pair of the oxDNA hydrogen-bonding term.
struct OxDNAHBondParams
    {
    Scalar eps;             // HB epsilon for this type pair (0 => no bonding)
    Scalar shift_factor;    // f1 shift = eps * shift_factor
    vec3<Scalar> base_site; // body-frame BASE offset
    // f1: a, r0, rlow, rhigh, rclow, rchigh, blow, bhigh
    Scalar f1_a, f1_r0, f1_rlow, f1_rhigh, f1_rclow, f1_rchigh, f1_blow, f1_bhigh;
    // f4 sets (a,b,t0,ts,tc): t1, t23 (theta2==theta3), t4, t78 (theta7==theta8)
    Scalar t1[5], t23[5], t4[5], t78[5];

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    DEVICE void load_shared(char*&, unsigned int&) { }
    HOSTDEVICE void allocate_shared(char*&, unsigned int&) const { }

    HOSTDEVICE OxDNAHBondParams()
        : eps(0), shift_factor(0), base_site(0, 0, 0),
          f1_a(0), f1_r0(0), f1_rlow(0), f1_rhigh(0), f1_rclow(0), f1_rchigh(0),
          f1_blow(0), f1_bhigh(0)
        {
        for (int k = 0; k < 5; k++)
            t1[k] = t23[k] = t4[k] = t78[k] = 0;
        }

#ifndef __HIPCC__
    OxDNAHBondParams(pybind11::dict v, bool managed)
        {
        eps = v["eps"].cast<Scalar>();
        shift_factor = v["shift_factor"].cast<Scalar>();
        base_site = oxdna_io::read_vec3(v["base_site"]);
        pybind11::tuple f1(v["f1"]);
        f1_a = f1[0].cast<Scalar>();
        f1_r0 = f1[1].cast<Scalar>();
        f1_rlow = f1[2].cast<Scalar>();
        f1_rhigh = f1[3].cast<Scalar>();
        f1_rclow = f1[4].cast<Scalar>();
        f1_rchigh = f1[5].cast<Scalar>();
        f1_blow = f1[6].cast<Scalar>();
        f1_bhigh = f1[7].cast<Scalar>();
        oxdna_io::read_scalars<5>(v["f4_t1"], t1);
        oxdna_io::read_scalars<5>(v["f4_t23"], t23);
        oxdna_io::read_scalars<5>(v["f4_t4"], t4);
        oxdna_io::read_scalars<5>(v["f4_t78"], t78);
        }

    pybind11::object toPython()
        {
        pybind11::dict v;
        v["eps"] = eps;
        v["shift_factor"] = shift_factor;
        v["base_site"] = oxdna_io::pack_vec3(base_site);
        v["f1"] = pybind11::make_tuple(f1_a, f1_r0, f1_rlow, f1_rhigh, f1_rclow, f1_rchigh,
                                       f1_blow, f1_bhigh);
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

class EvaluatorPairOxDNAHBond : public OxDNAAnisoPair<EvaluatorPairOxDNAHBond, OxDNAHBondParams>
    {
    public:
    typedef OxDNAAnisoPair<EvaluatorPairOxDNAHBond, OxDNAHBondParams> Base;
    using Base::Base;

    HOSTDEVICE bool evaluate_scalar(Scalar3& force,
                                    Scalar& pair_eng,
                                    bool /*energy_shift*/,
                                    Scalar3& torque_i,
                                    Scalar3& torque_j)
        {
        oxdna::F1Radial radial {p.f1_a,     p.f1_r0,   p.f1_rlow, p.f1_rhigh,
                                p.f1_rclow,  p.f1_rchigh, p.f1_blow, p.f1_bhigh,
                                p.eps,       p.eps * p.shift_factor};
        return oxdna::evaluate_base_base(dr, quat_i, quat_j, rcutsq, p.base_site,
                                         p.t1, p.t23, p.t4, p.t78, /*symmetrize=*/false,
                                         radial, force, pair_eng, torque_i, torque_j);
        }

#ifndef __HIPCC__
    static std::string getName() { return "oxdna_hbond"; }
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_OXDNA_HBOND_H__
