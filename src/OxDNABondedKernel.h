// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNABondedKernel.h
    \brief Shared host+device physics for the oxDNA bonded force.

    oxdna_bonded_pair() computes one backbone bond's FENE + bonded excluded volume +
    stacking contribution: the force on B, the lab-frame torques on A and B, the
    per-term energies, and the (half) virial. Both the CPU ForceCompute and the GPU
    kernel call this, so the two paths are guaranteed identical.

    Bond convention: the bond is (A, B) with B = A's 5' neighbour. oxDNA's own p/q are
    (B, A) for stacking (it defines the pair with q = p->n3), which is handled inside.
*/

#ifndef __OXDNA_BONDED_KERNEL_H__
#define __OXDNA_BONDED_KERNEL_H__

#include "MixedPrecisionCompat.h" // ForceReal (native on the fork, == Scalar on vanilla)
#include "OxDNABondedParams.h"
#include "OxDNANumerics.h"

#ifdef __HIPCC__
#define OXDNA_BK_HOSTDEVICE __host__ __device__
#else
#define OXDNA_BK_HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {
namespace oxdna
    {

//! Result of one bonded-pair evaluation. FB is the force on B; forces on A = -FB.
//! vir is the (0.5-scaled) per-particle virial, added to BOTH A and B by the caller.
struct BondedResult
    {
    vec3<Scalar> FB;
    vec3<Scalar> tauA;
    vec3<Scalar> tauB;
    Scalar e_fene, e_bexc, e_stack;
    Scalar vir[6];
    };

//! Add a sub-interaction's contribution to the per-particle virial (0.5 split).
OXDNA_BK_HOSTDEVICE inline void bonded_accumulate_virial(Scalar vir[6],
                                                         const vec3<Scalar>& r,
                                                         const vec3<Scalar>& f)
    {
    vir[0] += Scalar(0.5) * r.x * f.x;
    vir[1] += Scalar(0.5) * r.x * f.y;
    vir[2] += Scalar(0.5) * r.x * f.z;
    vir[3] += Scalar(0.5) * r.y * f.y;
    vir[4] += Scalar(0.5) * r.y * f.z;
    vir[5] += Scalar(0.5) * r.z * f.z;
    }

//! Stacking term (DNAInteraction::_stacking). oxDNA p=B, q=A. Accumulates into
//! FB / tauA / tauB / vir (double accumulators) and returns the stacking energy.
//!
//! Templated on the working real type. The GPU/CPU paths instantiate it at ForceReal
//! (single precision on the mixed-precision fork): the stacking dihedral math is the
//! bulk of the bonded kernel's cost on a gaming GPU and has no catastrophic
//! cancellation, so single precision is safe (validated to < 1e-4). The FENE term,
//! which does cancel in float, is kept double by the caller. Angles feed the f4
//! factors once (acos is the hot transcendental) via the _angle variants.
template<class Real>
OXDNA_BK_HOSTDEVICE inline Real bonded_stacking(const oxdna_bonded_params& p,
                                                const vec3<Real>& Rc,
                                                const quat<Real>& qA,
                                                const quat<Real>& qB,
                                                unsigned int typeA,
                                                unsigned int typeB,
                                                vec3<Scalar>& FB,
                                                vec3<Scalar>& tauA,
                                                vec3<Scalar>& tauB,
                                                Scalar vir[6])
    {
    // Lab-frame body axes: a* = p's (=B), b* = q's (=A).
    vec3<Real> a1 = rotate(qB, vec3<Real>(1, 0, 0));
    vec3<Real> a2 = rotate(qB, vec3<Real>(0, 1, 0));
    vec3<Real> a3 = rotate(qB, vec3<Real>(0, 0, 1));
    vec3<Real> b1 = rotate(qA, vec3<Real>(1, 0, 0));
    vec3<Real> b2 = rotate(qA, vec3<Real>(0, 1, 0));
    vec3<Real> b3 = rotate(qA, vec3<Real>(0, 0, 1));

    vec3<Real> stack_site(p.stack_site), backref_site(p.stack_back_ref);
    vec3<Real> stackP = rotate(qB, stack_site);
    vec3<Real> stackQ = rotate(qA, stack_site);
    vec3<Real> backrefP = rotate(qB, backref_site);
    vec3<Real> backrefQ = rotate(qA, backref_site);

    vec3<Real> Rpq(-Rc.x, -Rc.y, -Rc.z); // q.pos - p.pos = posA - posB

    vec3<Real> rbackref = Rpq + backrefQ - backrefP;
    Real rbackrefmod = sqrt(dot(rbackref, rbackref));

    vec3<Real> rstack = Rpq + stackQ - stackP;
    Real rstackmod = sqrt(dot(rstack, rstack));
    vec3<Real> rstackdir = rstack / rstackmod;

    Real cost4 = dot(a3, b3);
    Real cost5 = dot(a3, rstackdir);
    Real cost6 = -dot(b3, rstackdir);
    Real cosphi1 = dot(a2, rbackref) / rbackrefmod;
    Real cosphi2 = dot(b2, rbackref) / rbackrefmod;

    Real eps = p.st_eps[typeA][typeB]; // [n3=q->type=A][n5=p->type=B]
    Real shift = eps * Real(p.st_shift_factor);

    // acos once per angle, reused for value + torque coefficient.
    Real ang4 = acos(clamp_cos<Real>(cost4));
    Real ang5 = acos(clamp_cos<Real>(-cost5));
    Real ang6 = acos(clamp_cos<Real>(cost6));

    Real f1 = f1_val<Real>(rstackmod, p.st_a, p.st_r0, p.st_rlow, p.st_rhigh,
                           p.st_rclow, p.st_rchigh, p.st_blow, p.st_bhigh, eps, shift);
    Real f4t4 = f4_val_angle<Real>(ang4, p.t4_a, p.t4_b, p.t4_t0, p.t4_ts, p.t4_tc);
    Real f4t5 = f4_val_angle<Real>(ang5, p.t5_a, p.t5_b, p.t5_t0, p.t5_ts, p.t5_tc);
    Real f4t6 = f4_val_angle<Real>(ang6, p.t5_a, p.t5_b, p.t5_t0, p.t5_ts, p.t5_tc);
    Real f5phi1 = f5_val<Real>(cosphi1, p.p1_a, p.p1_b, p.p1_xc, p.p1_xs);
    Real f5phi2 = f5_val<Real>(cosphi2, p.p1_a, p.p1_b, p.p1_xc, p.p1_xs);

    Real e_stack = f1 * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2;
    if (e_stack == Real(0.0))
        return e_stack;

    Real f1D = f1_deriv<Real>(rstackmod, p.st_a, p.st_r0, p.st_rlow, p.st_rhigh,
                              p.st_rclow, p.st_rchigh, p.st_blow, p.st_bhigh, eps);
    Real f4t4Dsin = f4_Dsin_angle<Real>(ang4, p.t4_a, p.t4_b, p.t4_t0, p.t4_ts, p.t4_tc);
    Real f4t5Dsin = f4_Dsin_angle<Real>(ang5, p.t5_a, p.t5_b, p.t5_t0, p.t5_ts, p.t5_tc);
    Real f4t6Dsin = f4_Dsin_angle<Real>(ang6, p.t5_a, p.t5_b, p.t5_t0, p.t5_ts, p.t5_tc);
    Real f5phi1D = f5_deriv<Real>(cosphi1, p.p1_a, p.p1_b, p.p1_xc, p.p1_xs);
    Real f5phi2D = f5_deriv<Real>(cosphi2, p.p1_a, p.p1_b, p.p1_xc, p.p1_xs);

    // force on q (=B)
    vec3<Real> force = -rstackdir * (f1D * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2);
    force += -(a3 - rstackdir * cost5) * (f1 * f4t4 * f4t5Dsin * f4t6 * f5phi1 * f5phi2 / rstackmod);
    force += -(b3 + rstackdir * cost6) * (f1 * f4t4 * f4t5 * f4t6Dsin * f5phi1 * f5phi2 / rstackmod);

    Real gamma = p.st_gamma;
    Real rbackrefmodcub = rbackrefmod * rbackrefmod * rbackrefmod;

    Real ra2 = dot(rstackdir, a2);
    Real ra1 = dot(rstackdir, a1);
    Real rb1 = dot(rstackdir, b1);
    Real a2b1 = dot(a2, b1);
    Real parentesi = rstackmod * ra2 - a2b1 * gamma;

    Real dcosphi1dr = (rstackmod * rstackmod * ra2 - ra2 * rbackrefmod * rbackrefmod
                       - rstackmod * (a2b1 + ra2 * (-ra1 + rb1)) * gamma
                       + a2b1 * (-ra1 + rb1) * gamma * gamma) / rbackrefmodcub;
    Real dcosphi1dra1 = rstackmod * gamma * parentesi / rbackrefmodcub;
    Real dcosphi1dra2 = -rstackmod / rbackrefmod;
    Real dcosphi1drb1 = -rstackmod * gamma * parentesi / rbackrefmodcub;
    Real dcosphi1da1b1 = -gamma * gamma * parentesi / rbackrefmodcub;
    Real dcosphi1da2b1 = gamma / rbackrefmod;

    Real force_part_phi1 = -f1 * f4t4 * f4t5 * f4t6 * f5phi1D * f5phi2;
    force += -(rstackdir * dcosphi1dr
               + ((a2 - rstackdir * ra2) * dcosphi1dra2
                  + (a1 - rstackdir * ra1) * dcosphi1dra1
                  + (b1 - rstackdir * rb1) * dcosphi1drb1) / rstackmod)
             * force_part_phi1;

    // COS PHI 2 (particle p -> b, q -> a)
    ra2 = dot(rstackdir, b2);
    ra1 = dot(rstackdir, b1);
    rb1 = dot(rstackdir, a1);
    a2b1 = dot(b2, a1);
    parentesi = rstackmod * ra2 + a2b1 * gamma;

    Real dcosphi2dr = (parentesi * (rstackmod + (rb1 - ra1) * gamma)
                       - ra2 * rbackrefmod * rbackrefmod) / rbackrefmodcub;
    Real dcosphi2dra1 = -rstackmod * gamma * (rstackmod * ra2 + a2b1 * gamma) / rbackrefmodcub;
    Real dcosphi2dra2 = -rstackmod / rbackrefmod;
    Real dcosphi2drb1 = rstackmod * gamma * parentesi / rbackrefmodcub;
    Real dcosphi2da1b1 = -gamma * gamma * parentesi / rbackrefmodcub;
    Real dcosphi2da2b1 = -gamma / rbackrefmod;

    Real force_part_phi2 = -f1 * f4t4 * f4t5 * f4t6 * f5phi1 * f5phi2D;
    force += -force_part_phi2
             * (rstackdir * dcosphi2dr
                + ((b2 - rstackdir * ra2) * dcosphi2dra2
                   + (b1 - rstackdir * ra1) * dcosphi2dra1
                   + (a1 - rstackdir * rb1) * dcosphi2drb1) / rstackmod);

    // force is on q (=A); force on p (=B) is -force.
    FB += vec3<Scalar>(-force);
    bonded_accumulate_virial(vir, vec3<Scalar>(rstack), vec3<Scalar>(force));

    // Torques (lab frame). tp = torque on p (=B), tq = torque on q (=A).
    vec3<Real> tp(0, 0, 0), tq(0, 0, 0);
    tp -= cross(stackP, force);
    tq += cross(stackQ, force);

    vec3<Real> t4dir = cross(b3, a3);
    Real torquemod = f1 * f4t4Dsin * f4t5 * f4t6 * f5phi1 * f5phi2;
    tp += -t4dir * torquemod;
    tq += t4dir * torquemod;

    vec3<Real> t5dir = cross(rstackdir, a3);
    torquemod = -f1 * f4t4 * f4t5Dsin * f4t6 * f5phi1 * f5phi2;
    tp += -t5dir * torquemod;

    vec3<Real> t6dir = cross(rstackdir, b3);
    torquemod = f1 * f4t4 * f4t5 * f4t6Dsin * f5phi1 * f5phi2;
    tq += t6dir * torquemod;

    tp += cross(rstackdir, a2) * force_part_phi1 * dcosphi1dra2
          + cross(rstackdir, a1) * force_part_phi1 * dcosphi1dra1;
    tq += cross(rstackdir, b1) * force_part_phi1 * dcosphi1drb1;
    vec3<Real> puretorque = cross(a2, b1) * force_part_phi1 * dcosphi1da2b1
                            + cross(a1, b1) * force_part_phi1 * dcosphi1da1b1;
    tp += -puretorque;
    tq += puretorque;

    tp += cross(rstackdir, a1) * force_part_phi2 * dcosphi2drb1;
    tq += cross(rstackdir, b2) * force_part_phi2 * dcosphi2dra2
          + cross(rstackdir, b1) * force_part_phi2 * dcosphi2dra1;
    puretorque = cross(a1, b2) * force_part_phi2 * dcosphi2da2b1
                 + cross(a1, b1) * force_part_phi2 * dcosphi2da1b1;
    tp += -puretorque;
    tq += puretorque;

    tauB += vec3<Scalar>(tp); // p = B
    tauA += vec3<Scalar>(tq); // q = A
    return e_stack;
    }

//! Full bonded-pair evaluation: FENE + bonded excluded volume + stacking.
//! \a Rc is the min-image COM-COM vector r_B - r_A.
OXDNA_BK_HOSTDEVICE inline void oxdna_bonded_pair(const oxdna_bonded_params& p,
                                                  const vec3<Scalar>& Rc,
                                                  const quat<Scalar>& qA,
                                                  const quat<Scalar>& qB,
                                                  unsigned int typeA,
                                                  unsigned int typeB,
                                                  BondedResult& out)
    {
    out.FB = vec3<Scalar>(0, 0, 0);
    out.tauA = vec3<Scalar>(0, 0, 0);
    out.tauB = vec3<Scalar>(0, 0, 0);
    out.e_fene = out.e_bexc = out.e_stack = Scalar(0.0);
    for (int v = 0; v < 6; v++)
        out.vir[v] = Scalar(0.0);

    // --- FENE backbone (BACK-BACK) : DOUBLE ---
    // -(eps/2) log(1 - dr^2/delta2) and its 1/(delta2 - dr^2) restoring force cancel
    // catastrophically in single precision near a stretched bond, exactly where the
    // spring must be stiff, so this term stays double for stability.
    vec3<Scalar> backA = rotate(qA, p.back_site);
    vec3<Scalar> backB = rotate(qB, p.back_site);
    vec3<Scalar> rback = Rc + backB - backA; // siteB - siteA
    Scalar rmod = sqrt(dot(rback, rback));
    Scalar dr = rmod - p.r0_backbone;
    if (fabs(dr) >= p.delta_backbone)
        {
        out.e_fene = Scalar(1.0e12); // broken bond: clamp (no dynamics contribution)
        }
    else
        {
        out.e_fene = fene_energy(dr, p.eps_backbone, p.delta2);
        vec3<Scalar> f = fene_force_prefactor(dr, rmod, p.eps_backbone, p.delta2) * rback;
        out.FB += f;
        out.tauA += cross(backA, -f);
        out.tauB += cross(backB, f);
        bonded_accumulate_virial(out.vir, rback, f);
        }

    // --- Bonded excluded volume + stacking : SINGLE precision ---
    // No cancellation traps here, and this is the bulk of the bonded kernel's cost on
    // a gaming GPU (weak FP64). Compute in ForceReal; accumulate into the double
    // force/torque accumulators. On vanilla HOOMD ForceReal == Scalar (all double).
    quat<ForceReal> qAf(qA), qBf(qB);
    vec3<ForceReal> Rcf(Rc);
    vec3<ForceReal> back_site(p.back_site), base_site(p.base_site);
    vec3<ForceReal> backAf = rotate(qAf, back_site);
    vec3<ForceReal> backBf = rotate(qBf, back_site);
    vec3<ForceReal> baseAf = rotate(qAf, base_site);
    vec3<ForceReal> baseBf = rotate(qBf, base_site);

    const vec3<ForceReal>* si[3] = {&baseAf, &baseAf, &backAf};
    const vec3<ForceReal>* sj[3] = {&baseBf, &backBf, &baseBf};
    for (int k = 0; k < 3; k++)
        {
        vec3<ForceReal> rcenter = Rcf + *sj[k] - *si[k];
        vec3<ForceReal> f;
        out.e_bexc += repulsive_lj<ForceReal>(rcenter, p.excl_eps, p.bx_sigma[k], p.bx_rstar[k],
                                              p.bx_b[k], p.bx_rc[k], f);
        out.FB += vec3<Scalar>(f);
        out.tauA += vec3<Scalar>(cross(*si[k], -f));
        out.tauB += vec3<Scalar>(cross(*sj[k], f));
        bonded_accumulate_virial(out.vir, vec3<Scalar>(rcenter), vec3<Scalar>(f));
        }

    // --- Stacking ---
    if (p.stacking_enabled)
        out.e_stack = bonded_stacking<ForceReal>(p, Rcf, qAf, qBf, typeA, typeB,
                                                 out.FB, out.tauA, out.tauB, out.vir);
    }

    } // end namespace oxdna
    } // end namespace md
    } // end namespace hoomd

#undef OXDNA_BK_HOSTDEVICE
#endif // __OXDNA_BONDED_KERNEL_H__
