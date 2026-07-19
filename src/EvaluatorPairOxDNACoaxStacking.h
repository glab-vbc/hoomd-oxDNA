// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file EvaluatorPairOxDNACoaxStacking.h
    \brief oxDNA coaxial-stacking evaluator for AnisoPotentialPair (oxDNA1 and oxDNA2).

    Between the STACK sites of two nucleotides:
        U = f2(r) * f4(t1) f4(t4) f4(t5) f4(t6) * [f5(phi3)^2],
    with t5/t6 symmetrized. The theta1 modification and the phi3 dihedral differ by
    model version and are selected by parameters:
      * ``t1_mode`` 0 = oxDNA1 reflection f4(t)+f4(2pi-t);
                    1 = oxDNA2 pure-harmonic f4(t)+SA*(t-SB)^2;
      * ``phi3_enabled`` — oxDNA1 multiplies by f5(phi3)^2, oxDNA2 does not.
*/

#ifndef __EVALUATOR_PAIR_OXDNA_COAXSTACKING_H__
#define __EVALUATOR_PAIR_OXDNA_COAXSTACKING_H__

#include "OxDNAAnisoPair.h"
#include "OxDNANumerics.h"
#ifndef __HIPCC__
#include "OxDNAParamIO.h"
#include <string>
#endif

namespace hoomd
    {
namespace md
    {

//! Parameters for one type pair of the oxDNA coaxial-stacking term.
struct OxDNACoaxStackingParams
    {
    vec3<Scalar> stack_site;
    vec3<Scalar> stack_back_ref;
    Scalar gamma;
    Scalar f2_k, f2_r0, f2_rc, f2_rlow, f2_rhigh, f2_rclow, f2_rchigh, f2_blow, f2_bhigh;
    Scalar t1[5], t4[5], t56[5]; // theta1 (modified), theta4, theta5==theta6
    Scalar phi3[4];              // f5 phi3 (a, b, xc, xs)
    Scalar phi4[4];              // f5 phi4 (oxRNA second dihedral); unused unless rna_coax
    int t1_mode;                 // 0 = oxDNA1 reflection; 1 = oxDNA2 harmonic
    Scalar t1_sa, t1_sb;         // oxDNA2 theta1 harmonic SA*(t-SB)^2
    bool phi3_enabled;           // oxDNA1 multiplies by f5(phi3)^2
    // oxRNA coaxial: use two independent dihedrals f5(phi3)*f5(phi4) built from the
    // real BACK-BACK vector (stack_back_ref set to the RNA BACK site), phi3 from a1
    // and phi4 from b1, with the compact cross-product force. Overrides phi3_enabled.
    bool rna_coax;

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    DEVICE void load_shared(char*&, unsigned int&) { }
    HOSTDEVICE void allocate_shared(char*&, unsigned int&) const { }

    HOSTDEVICE OxDNACoaxStackingParams()
        : stack_site(0, 0, 0), stack_back_ref(0, 0, 0), gamma(0),
          f2_k(0), f2_r0(0), f2_rc(0), f2_rlow(0), f2_rhigh(0), f2_rclow(0),
          f2_rchigh(0), f2_blow(0), f2_bhigh(0), t1_mode(0), t1_sa(0), t1_sb(0),
          phi3_enabled(true), rna_coax(false)
        {
        for (int k = 0; k < 5; k++)
            t1[k] = t4[k] = t56[k] = 0;
        for (int k = 0; k < 4; k++)
            phi3[k] = phi4[k] = 0;
        }

#ifndef __HIPCC__
    OxDNACoaxStackingParams(pybind11::dict v, bool managed)
        {
        stack_site = oxdna_io::read_vec3(v["stack_site"]);
        stack_back_ref = oxdna_io::read_vec3(v["stack_back_ref"]);
        gamma = v["gamma"].cast<Scalar>();
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
        oxdna_io::read_scalars<5>(v["f4_t4"], t4);
        oxdna_io::read_scalars<5>(v["f4_t56"], t56);
        oxdna_io::read_scalars<4>(v["f5_phi3"], phi3);
        t1_mode = v["t1_mode"].cast<int>();
        t1_sa = v["t1_sa"].cast<Scalar>();
        t1_sb = v["t1_sb"].cast<Scalar>();
        phi3_enabled = v["phi3_enabled"].cast<bool>();
        rna_coax = v.contains("rna_coax") ? v["rna_coax"].cast<bool>() : false;
        if (v.contains("f5_phi4"))
            oxdna_io::read_scalars<4>(v["f5_phi4"], phi4);
        else
            for (int k = 0; k < 4; k++)
                phi4[k] = phi3[k];
        }

    pybind11::object toPython()
        {
        pybind11::dict v;
        v["stack_site"] = oxdna_io::pack_vec3(stack_site);
        v["stack_back_ref"] = oxdna_io::pack_vec3(stack_back_ref);
        v["gamma"] = gamma;
        v["f2"] = pybind11::make_tuple(f2_k, f2_r0, f2_rc, f2_rlow, f2_rhigh, f2_rclow,
                                       f2_rchigh, f2_blow, f2_bhigh);
        v["f4_t1"] = oxdna_io::pack_scalars<5>(t1);
        v["f4_t4"] = oxdna_io::pack_scalars<5>(t4);
        v["f4_t56"] = oxdna_io::pack_scalars<5>(t56);
        v["f5_phi3"] = oxdna_io::pack_scalars<4>(phi3);
        v["f5_phi4"] = oxdna_io::pack_scalars<4>(phi4);
        v["t1_mode"] = t1_mode;
        v["t1_sa"] = t1_sa;
        v["t1_sb"] = t1_sb;
        v["phi3_enabled"] = phi3_enabled;
        v["rna_coax"] = rna_coax;
        return std::move(v);
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

class EvaluatorPairOxDNACoaxStacking
    : public OxDNAAnisoPair<EvaluatorPairOxDNACoaxStacking, OxDNACoaxStackingParams>
    {
    public:
    typedef OxDNAAnisoPair<EvaluatorPairOxDNACoaxStacking, OxDNACoaxStackingParams> Base;
    using Base::Base;

    HOSTDEVICE bool evaluate_scalar(Scalar3& force_out,
                                    Scalar& pair_eng,
                                    bool /*energy_shift*/,
                                    Scalar3& torque_i,
                                    Scalar3& torque_j)
        {
        const ForceReal PI = ForceReal(3.14159265358979323846);
        force_out = make_scalar3(0, 0, 0);
        torque_i = make_scalar3(0, 0, 0);
        torque_j = make_scalar3(0, 0, 0);
        pair_eng = Scalar(0.0);

        ForceReal rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        if (rsq >= rcutsq)
            return false;

        quat<ForceReal> qi(quat_i), qj(quat_j);
        vec3<ForceReal> a1 = rotate(qi, vec3<ForceReal>(1, 0, 0));
        vec3<ForceReal> a2 = rotate(qi, vec3<ForceReal>(0, 1, 0));
        vec3<ForceReal> a3 = rotate(qi, vec3<ForceReal>(0, 0, 1));
        vec3<ForceReal> b1 = rotate(qj, vec3<ForceReal>(1, 0, 0));
        vec3<ForceReal> b3 = rotate(qj, vec3<ForceReal>(0, 0, 1));
        vec3<ForceReal> stack_site(p.stack_site), backref_site(p.stack_back_ref);
        vec3<ForceReal> stack_i = rotate(qi, stack_site);
        vec3<ForceReal> stack_j = rotate(qj, stack_site);
        vec3<ForceReal> backref_i = rotate(qi, backref_site);
        vec3<ForceReal> backref_j = rotate(qj, backref_site);

        vec3<ForceReal> Rc(-dr.x, -dr.y, -dr.z); // r_j - r_i
        vec3<ForceReal> rstack = Rc + stack_j - stack_i;
        ForceReal rstackmod = sqrt(dot(rstack, rstack));
        if (rstackmod <= ForceReal(0.0))
            return true;
        vec3<ForceReal> rstackdir = rstack / rstackmod;

        ForceReal cost1 = -dot(a1, b1);
        ForceReal cost4 = dot(a3, b3);
        ForceReal cost5 = dot(a3, rstackdir);
        ForceReal cost6 = -dot(b3, rstackdir);

        vec3<ForceReal> rbackboneref = Rc + backref_j - backref_i;
        ForceReal rbackrefmod = sqrt(dot(rbackboneref, rbackboneref));
        vec3<ForceReal> rbackbonerefdir = rbackboneref / rbackrefmod;
        ForceReal cosphi3 = dot(rstackdir, cross(rbackbonerefdir, a1));

        ForceReal t1_sa(p.t1_sa), t1_sb(p.t1_sb), gamma(p.gamma);

        ForceReal f2 = oxdna::f2_val<ForceReal>(rstackmod, p.f2_k, p.f2_r0, p.f2_rc, p.f2_rlow,
                                                p.f2_rhigh, p.f2_rclow, p.f2_rchigh, p.f2_blow,
                                                p.f2_bhigh);
        // acos is the hot transcendental; compute each angle once and reuse it for the
        // f4 value and its torque coefficient (and the reflected term via pi - angle).
        ForceReal t1_angle = acos(oxdna::clamp_cos<ForceReal>(cost1));
        ForceReal ang4 = acos(oxdna::clamp_cos<ForceReal>(cost4));
        ForceReal ang5 = acos(oxdna::clamp_cos<ForceReal>(cost5));
        ForceReal ang6 = acos(oxdna::clamp_cos<ForceReal>(cost6));
        ForceReal ang5n = PI - ang5; // acos(-cost5)
        ForceReal ang6n = PI - ang6; // acos(-cost6)

        ForceReal f4t1 = oxdna::f4_val_angle<ForceReal>(t1_angle, p.t1[0], p.t1[1], p.t1[2],
                                                        p.t1[3], p.t1[4]);
        if (p.t1_mode == 0) // oxDNA1: reflection f4(2pi - t)
            f4t1 += oxdna::f4_val_angle<ForceReal>(ForceReal(2.0) * PI - t1_angle, p.t1[0],
                                                   p.t1[1], p.t1[2], p.t1[3], p.t1[4]);
        else if (t1_angle > t1_sb) // oxDNA2: pure-harmonic SA*(t - SB)^2
            f4t1 += t1_sa * (t1_angle - t1_sb) * (t1_angle - t1_sb);
        ForceReal f4t4 = oxdna::f4_val_angle<ForceReal>(ang4, p.t4[0], p.t4[1], p.t4[2], p.t4[3], p.t4[4]);
        ForceReal f4t5 = oxdna::f4_val_angle<ForceReal>(ang5, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4])
                         + oxdna::f4_val_angle<ForceReal>(ang5n, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4]);
        ForceReal f4t6 = oxdna::f4_val_angle<ForceReal>(ang6, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4])
                         + oxdna::f4_val_angle<ForceReal>(ang6n, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4]);

        ForceReal f5cosphi3 = ForceReal(0.0), f5cosphi4 = ForceReal(0.0);
        ForceReal cosphi4 = ForceReal(0.0), phi_factor = ForceReal(1.0);
        if (p.rna_coax)
            {
            // oxRNA: second dihedral from b1, energy uses f5(phi3)*f5(phi4)
            cosphi4 = dot(rstackdir, cross(rbackbonerefdir, b1));
            f5cosphi3 = oxdna::f5_val<ForceReal>(cosphi3, p.phi3[0], p.phi3[1], p.phi3[2], p.phi3[3]);
            f5cosphi4 = oxdna::f5_val<ForceReal>(cosphi4, p.phi4[0], p.phi4[1], p.phi4[2], p.phi4[3]);
            phi_factor = f5cosphi3 * f5cosphi4;
            }
        else if (p.phi3_enabled)
            {
            f5cosphi3 = oxdna::f5_val<ForceReal>(cosphi3, p.phi3[0], p.phi3[1], p.phi3[2], p.phi3[3]);
            phi_factor = f5cosphi3 * f5cosphi3;
            }

        ForceReal energy = f2 * f4t1 * f4t4 * f4t5 * f4t6 * phi_factor;
        pair_eng = energy;
        if (energy == ForceReal(0.0))
            return true;

        ForceReal f2D = oxdna::f2_deriv<ForceReal>(rstackmod, p.f2_k, p.f2_r0, p.f2_rlow, p.f2_rhigh,
                                                   p.f2_rclow, p.f2_rchigh, p.f2_blow, p.f2_bhigh);
        ForceReal f4t1Dsin =
            -oxdna::f4_Dsin_angle<ForceReal>(t1_angle, p.t1[0], p.t1[1], p.t1[2], p.t1[3], p.t1[4]);
        if (p.t1_mode == 0)
            f4t1Dsin += -oxdna::f4_Dsin_angle<ForceReal>(ForceReal(2.0) * PI - t1_angle, p.t1[0],
                                                         p.t1[1], p.t1[2], p.t1[3], p.t1[4]);
        else if (t1_angle > t1_sb)
            {
            ForceReal sint1 = sin(t1_angle);
            ForceReal hd = (sint1 * sint1 > ForceReal(1e-8))
                               ? ForceReal(2.0) * t1_sa * (t1_angle - t1_sb) / sint1
                               : ForceReal(2.0) * t1_sa;
            f4t1Dsin += -hd;
            }
        ForceReal f4t4Dsin = oxdna::f4_Dsin_angle<ForceReal>(ang4, p.t4[0], p.t4[1], p.t4[2], p.t4[3], p.t4[4]);
        ForceReal f4t5Dsin = oxdna::f4_Dsin_angle<ForceReal>(ang5, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4])
                             - oxdna::f4_Dsin_angle<ForceReal>(ang5n, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4]);
        ForceReal f4t6Dsin = -oxdna::f4_Dsin_angle<ForceReal>(ang6, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4])
                             + oxdna::f4_Dsin_angle<ForceReal>(ang6n, p.t56[0], p.t56[1], p.t56[2], p.t56[3], p.t56[4]);

        vec3<ForceReal> force(0, 0, 0);   // force on q (=j)
        vec3<ForceReal> torquep(0, 0, 0); // torque on p (=i)
        vec3<ForceReal> torqueq(0, 0, 0); // torque on q (=j)

        force = -rstackdir * (f2D * f4t1 * f4t4 * f4t5 * f4t6 * phi_factor);

        // THETA1
        vec3<ForceReal> dir = cross(a1, b1);
        ForceReal torquemod = -f2 * f4t1Dsin * f4t4 * f4t5 * f4t6 * phi_factor;
        torquep -= dir * torquemod;
        torqueq += dir * torquemod;

        // THETA4
        dir = cross(a3, b3);
        torquemod = -f2 * f4t1 * f4t4Dsin * f4t5 * f4t6 * phi_factor;
        torquep -= dir * torquemod;
        torqueq += dir * torquemod;

        // THETA5
        ForceReal fact = f2 * f4t1 * f4t4 * f4t5Dsin * f4t6 * phi_factor;
        force += fact * (a3 - rstackdir * cost5) / rstackmod;
        dir = cross(rstackdir, a3);
        torquep -= dir * fact;

        // THETA6
        fact = f2 * f4t1 * f4t4 * f4t5 * f4t6Dsin * phi_factor;
        force += (b3 + rstackdir * cost6) * (fact / rstackmod);
        dir = cross(rstackdir, b3);
        torqueq += -dir * fact;

        // --- oxRNA coaxial: two independent dihedrals f5(phi3)*f5(phi4), built from the
        // real BACK-BACK vector, with the compact cross-product force (RNAInteraction.cpp
        // _coaxial_stacking). phi3 uses a1 (pure torque on p), phi4 uses b1 (on q); both
        // act at the STACK and BACK sites. Returns early: the oxDNA path below is untouched.
        if (p.rna_coax)
            {
            // STACK site-offset torque for the radial + theta force accumulated so far
            torquep -= cross(stack_i, force);
            torqueq += cross(stack_j, force);

            ForceReal base = f2 * f4t1 * f4t4 * f4t5 * f4t6;
            ForceReal f5Dcosphi3 = oxdna::f5_deriv<ForceReal>(cosphi3, p.phi3[0], p.phi3[1],
                                                              p.phi3[2], p.phi3[3]);
            ForceReal f5Dcosphi4 = oxdna::f5_deriv<ForceReal>(cosphi4, p.phi4[0], p.phi4[1],
                                                              p.phi4[2], p.phi4[3]);
            for (int which = 0; which < 2; which++)
                {
                vec3<ForceReal> v = (which == 0) ? a1 : b1;
                ForceReal force_c = (which == 0) ? base * f5cosphi4 * f5Dcosphi3
                                                 : base * f5cosphi3 * f5Dcosphi4;
                vec3<ForceReal> rbv = cross(rbackbonerefdir, v);
                vec3<ForceReal> vrs = cross(v, rstackdir);
                vec3<ForceReal> fstack = -(rbv - rstackdir * dot(rstackdir, rbv)) * (force_c / rstackmod);
                vec3<ForceReal> fback = -(vrs - rbackbonerefdir * dot(rbackbonerefdir, vrs))
                                        * (force_c / rbackrefmod);
                force += fstack + fback;
                torquep -= cross(stack_i, fstack) + cross(backref_i, fback);
                torqueq += cross(stack_j, fstack) + cross(backref_j, fback);
                vec3<ForceReal> pure = force_c * cross(v, cross(rstackdir, rbackbonerefdir));
                if (which == 0)
                    torquep -= pure;
                else
                    torqueq -= pure;
                }

            force_out = vec_to_scalar3(-force);
            torque_i = vec_to_scalar3(torquep);
            torque_j = vec_to_scalar3(torqueq);
            return true;
            }

        // COS PHI3 dihedral (oxDNA1 only)
        if (p.phi3_enabled)
            {
            ForceReal f5Dcosphi3 = oxdna::f5_deriv<ForceReal>(cosphi3, p.phi3[0], p.phi3[1],
                                                              p.phi3[2], p.phi3[3]);
            ForceReal gammacub = gamma * gamma * gamma;
            ForceReal rbackrefmodcub = rbackrefmod * rbackrefmod * rbackrefmod;
            ForceReal a2b1 = dot(a2, b1);
            ForceReal a3b1 = dot(a3, b1);
            ForceReal ra1 = dot(rstackdir, a1);
            ForceReal ra2 = dot(rstackdir, a2);
            ForceReal ra3 = dot(rstackdir, a3);
            ForceReal rb1 = dot(rstackdir, b1);
            ForceReal parentesi = ra3 * a2b1 - ra2 * a3b1;

            ForceReal dcdr = -gamma * parentesi * (gamma * (ra1 - rb1) + rstackmod) / rbackrefmodcub;
            ForceReal dcda1b1 = gammacub * parentesi / rbackrefmodcub;
            ForceReal dcda2b1 = gamma * ra3 / rbackrefmod;
            ForceReal dcda3b1 = -gamma * ra2 / rbackrefmod;
            ForceReal dcdra1 = -gamma * gamma * parentesi * rstackmod / rbackrefmodcub;
            ForceReal dcdra2 = -gamma * a3b1 / rbackrefmod;
            ForceReal dcdra3 = gamma * a2b1 / rbackrefmod;
            ForceReal dcdrb1 = gamma * gamma * parentesi * rstackmod / rbackrefmodcub;

            ForceReal force_c = f2 * f4t1 * f4t4 * f4t5 * f4t6 * ForceReal(2.0) * f5cosphi3 * f5Dcosphi3;

            force += -force_c
                     * (rstackdir * dcdr
                        + ((a1 - rstackdir * ra1) * dcdra1 + (a2 - rstackdir * ra2) * dcdra2
                           + (a3 - rstackdir * ra3) * dcdra3 + (b1 - rstackdir * rb1) * dcdrb1)
                              / rstackmod);

            torquep += force_c
                       * (cross(rstackdir, a1) * dcdra1 + cross(rstackdir, a2) * dcdra2
                          + cross(rstackdir, a3) * dcdra3);
            torqueq += force_c * (cross(rstackdir, b1) * dcdrb1);

            vec3<ForceReal> puretorque = force_c
                                         * (cross(a1, b1) * dcda1b1 + cross(a2, b1) * dcda2b1
                                            + cross(a3, b1) * dcda3b1);
            torquep -= puretorque;
            torqueq += puretorque;
            }

        // site-offset torque (STACK sites)
        torquep -= cross(stack_i, force);
        torqueq += cross(stack_j, force);

        // force is on q (=j); HOOMD evaluator returns force on i (=p) = -force.
        force_out = vec_to_scalar3(-force);
        torque_i = vec_to_scalar3(torquep);
        torque_j = vec_to_scalar3(torqueq);
        return true;
        }

#ifndef __HIPCC__
    static std::string getName() { return "oxdna_coaxstacking"; }
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_OXDNA_COAXSTACKING_H__
