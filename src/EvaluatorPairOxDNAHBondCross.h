// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file EvaluatorPairOxDNAHBondCross.h
    \brief Fused hydrogen-bonding + cross-stacking evaluator for AnisoPotentialPair.

    Hydrogen bonding and cross-stacking act between the SAME two BASE sites and share
    their entire six-angle geometry (body axes, sites, the six angle cosines); they
    differ only in the radial factor (f1 Morse vs f2 harmonic), the f4 parameters, and
    whether t4/t7/t8 are symmetrized. This evaluator reconstructs that geometry ONCE
    (via base_base_geometry) and applies both terms (via base_base_term), instead of
    HOOMD running two separate AnisoPotentialPair kernels that each re-traverse the
    neighbour list and re-rotate the quaternions. The per-term physics is the exact
    same shared code the standalone EvaluatorPairOxDNAHBond / ...CrossStacking use, so
    there is no duplicated math — both the fused and unfused paths remain available.

    The cutoff is the larger of the two terms' COM cutoffs; each term self-limits to
    its own range through its radial factor (f1/f2 vanish beyond their rchigh).
*/

#ifndef __EVALUATOR_PAIR_OXDNA_HBONDCROSS_H__
#define __EVALUATOR_PAIR_OXDNA_HBONDCROSS_H__

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

//! Parameters for one type pair of the fused HBond + cross-stacking term.
struct OxDNAHBondCrossParams
    {
    vec3<Scalar> base_site; //!< body-frame BASE offset (shared by both terms)
    // --- hydrogen bonding (f1 Morse) ---
    Scalar eps;          //!< HB epsilon for this type pair (0 => no bonding)
    Scalar shift_factor; //!< f1 shift = eps * shift_factor
    Scalar f1_a, f1_r0, f1_rlow, f1_rhigh, f1_rclow, f1_rchigh, f1_blow, f1_bhigh;
    Scalar hb_t1[5], hb_t23[5], hb_t4[5], hb_t78[5];
    // --- cross-stacking (f2 harmonic) ---
    Scalar f2_k, f2_r0, f2_rc, f2_rlow, f2_rhigh, f2_rclow, f2_rchigh, f2_blow, f2_bhigh;
    Scalar cs_t1[5], cs_t23[5], cs_t4[5], cs_t78[5];

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    DEVICE void load_shared(char*&, unsigned int&) { }
    HOSTDEVICE void allocate_shared(char*&, unsigned int&) const { }

    HOSTDEVICE OxDNAHBondCrossParams()
        : base_site(0, 0, 0), eps(0), shift_factor(0), f1_a(0), f1_r0(0), f1_rlow(0),
          f1_rhigh(0), f1_rclow(0), f1_rchigh(0), f1_blow(0), f1_bhigh(0), f2_k(0),
          f2_r0(0), f2_rc(0), f2_rlow(0), f2_rhigh(0), f2_rclow(0), f2_rchigh(0),
          f2_blow(0), f2_bhigh(0)
        {
        for (int k = 0; k < 5; k++)
            {
            hb_t1[k] = hb_t23[k] = hb_t4[k] = hb_t78[k] = 0;
            cs_t1[k] = cs_t23[k] = cs_t4[k] = cs_t78[k] = 0;
            }
        }

#ifndef __HIPCC__
    OxDNAHBondCrossParams(pybind11::dict v, bool managed)
        {
        base_site = oxdna_io::read_vec3(v["base_site"]);
        eps = v["eps"].cast<Scalar>();
        shift_factor = v["shift_factor"].cast<Scalar>();
        pybind11::tuple f1(v["f1"]);
        f1_a = f1[0].cast<Scalar>();
        f1_r0 = f1[1].cast<Scalar>();
        f1_rlow = f1[2].cast<Scalar>();
        f1_rhigh = f1[3].cast<Scalar>();
        f1_rclow = f1[4].cast<Scalar>();
        f1_rchigh = f1[5].cast<Scalar>();
        f1_blow = f1[6].cast<Scalar>();
        f1_bhigh = f1[7].cast<Scalar>();
        oxdna_io::read_scalars<5>(v["hb_f4_t1"], hb_t1);
        oxdna_io::read_scalars<5>(v["hb_f4_t23"], hb_t23);
        oxdna_io::read_scalars<5>(v["hb_f4_t4"], hb_t4);
        oxdna_io::read_scalars<5>(v["hb_f4_t78"], hb_t78);
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
        oxdna_io::read_scalars<5>(v["cs_f4_t1"], cs_t1);
        oxdna_io::read_scalars<5>(v["cs_f4_t23"], cs_t23);
        oxdna_io::read_scalars<5>(v["cs_f4_t4"], cs_t4);
        oxdna_io::read_scalars<5>(v["cs_f4_t78"], cs_t78);
        }

    pybind11::object toPython()
        {
        pybind11::dict v;
        v["base_site"] = oxdna_io::pack_vec3(base_site);
        v["eps"] = eps;
        v["shift_factor"] = shift_factor;
        v["f1"] = pybind11::make_tuple(f1_a, f1_r0, f1_rlow, f1_rhigh, f1_rclow, f1_rchigh,
                                       f1_blow, f1_bhigh);
        v["hb_f4_t1"] = oxdna_io::pack_scalars<5>(hb_t1);
        v["hb_f4_t23"] = oxdna_io::pack_scalars<5>(hb_t23);
        v["hb_f4_t4"] = oxdna_io::pack_scalars<5>(hb_t4);
        v["hb_f4_t78"] = oxdna_io::pack_scalars<5>(hb_t78);
        v["f2"] = pybind11::make_tuple(f2_k, f2_r0, f2_rc, f2_rlow, f2_rhigh, f2_rclow,
                                       f2_rchigh, f2_blow, f2_bhigh);
        v["cs_f4_t1"] = oxdna_io::pack_scalars<5>(cs_t1);
        v["cs_f4_t23"] = oxdna_io::pack_scalars<5>(cs_t23);
        v["cs_f4_t4"] = oxdna_io::pack_scalars<5>(cs_t4);
        v["cs_f4_t78"] = oxdna_io::pack_scalars<5>(cs_t78);
        return std::move(v);
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

class EvaluatorPairOxDNAHBondCross
    : public OxDNAAnisoPair<EvaluatorPairOxDNAHBondCross, OxDNAHBondCrossParams>
    {
    public:
    typedef OxDNAAnisoPair<EvaluatorPairOxDNAHBondCross, OxDNAHBondCrossParams> Base;
    using Base::Base;

    HOSTDEVICE bool evaluate_scalar(Scalar3& force,
                                    Scalar& pair_eng,
                                    bool /*energy_shift*/,
                                    Scalar3& torque_i,
                                    Scalar3& torque_j)
        {
        force = make_scalar3(0, 0, 0);
        torque_i = make_scalar3(0, 0, 0);
        torque_j = make_scalar3(0, 0, 0);
        pair_eng = Scalar(0.0);

        ForceReal rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        if (rsq >= rcutsq)
            return false;

        oxdna::BaseBaseGeom g = oxdna::base_base_geometry(dr, quat_i, quat_j, rcutsq, p.base_site);
        if (!g.valid)
            return true;

        vec3<ForceReal> f(0, 0, 0), ti(0, 0, 0), tj(0, 0, 0);
        ForceReal e = ForceReal(0.0);

        // Hydrogen bonding: f1 Morse radial, not symmetrized.
        oxdna::F1Radial hb {p.f1_a,    p.f1_r0,     p.f1_rlow, p.f1_rhigh,
                            p.f1_rclow, p.f1_rchigh, p.f1_blow, p.f1_bhigh,
                            p.eps,      p.eps * p.shift_factor};
        oxdna::base_base_term(g, p.hb_t1, p.hb_t23, p.hb_t4, p.hb_t78, /*symmetrize=*/false, hb,
                              f, e, ti, tj);

        // Cross-stacking: f2 harmonic radial, symmetrized on t4/t7/t8.
        oxdna::F2Radial cs {p.f2_k,     p.f2_r0,     p.f2_rc,   p.f2_rlow, p.f2_rhigh,
                            p.f2_rclow, p.f2_rchigh, p.f2_blow, p.f2_bhigh};
        oxdna::base_base_term(g, p.cs_t1, p.cs_t23, p.cs_t4, p.cs_t78, /*symmetrize=*/true, cs,
                              f, e, ti, tj);

        force = vec_to_scalar3(f);
        pair_eng = e;
        torque_i = vec_to_scalar3(ti);
        torque_j = vec_to_scalar3(tj);
        return true;
        }

#ifndef __HIPCC__
    static std::string getName() { return "oxdna_hbondcross"; }
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_OXDNA_HBONDCROSS_H__
