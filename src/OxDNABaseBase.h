// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNABaseBase.h
    \brief Shared six-angle BASE-BASE kernel for hydrogen bonding and cross-stacking.

    oxDNA's hydrogen-bonding (_hydrogen_bonding) and cross-stacking (_cross_stacking)
    terms have identical geometry: both act between the two BASE sites and modulate a
    radial factor by f4(t1) f4(t2) f4(t3) f4(t4) f4(t7) f4(t8). They differ only in
      * the radial factor: hydrogen bonding uses f1 (Morse, sequence-dependent eps),
        cross-stacking uses f2 (harmonic stiffness K);
      * whether t4/t7/t8 are "symmetrized" (f4(c) + f4(-c)) — cross-stacking is.
    This template captures the shared energy/force/torque math; each evaluator only
    supplies a small radial functor and the symmetrize flag.

    Everything here is a nonbonded term, so the geometry and modulation math run in
    ForceReal (single precision on the mixed-precision fork; the transcendental f4
    factors are the hot path on a gaming GPU). The parameters arrive as Scalar and
    are narrowed on use. Force/torque are written back into the evaluator's Scalar3
    outputs so the AnisoPotentialPair bridge is unchanged.
*/

#ifndef __OXDNA_BASE_BASE_H__
#define __OXDNA_BASE_BASE_H__

#include "OxDNAAnisoPair.h" // HOSTDEVICE/DEVICE macros, vec3/quat, ForceReal
#include "OxDNANumerics.h"

namespace hoomd
    {
namespace md
    {
namespace oxdna
    {

//! One angular factor together with its torque coefficient (oxDNA's f4 / _custom_f4D).
struct AngFactor
    {
    ForceReal val;
    ForceReal dsin;
    };

//! f4 factor for an angle with cosine \a cost. With \a symmetrize the reflected
//! f4(-cost) is added (cross-stacking); \a sign selects oxDNA's per-angle derivative
//! sign convention. The f4 constants \a t arrive as Scalar and are narrowed on use.
//!
//! acos is the hot transcendental here (no fast HW intrinsic), so the angle is
//! computed ONCE and reused for both the value (f4_val_angle) and the torque
//! coefficient (f4_Dsin_angle). The reflected branch uses acos(-cost) = pi - acos(cost)
//! instead of a second acos. This cuts acos calls per angle from 2 (plain) / 4
//! (symmetrized) down to 1.
HOSTDEVICE inline AngFactor angular(ForceReal cost, const Scalar t[5], bool symmetrize, ForceReal sign)
    {
    const ForceReal PI = ForceReal(3.14159265358979323846);
    ForceReal ang = acos(clamp_cos<ForceReal>(cost));
    AngFactor a;
    a.val = f4_val_angle<ForceReal>(ang, t[0], t[1], t[2], t[3], t[4]);
    ForceReal d = f4_Dsin_angle<ForceReal>(ang, t[0], t[1], t[2], t[3], t[4]);
    if (symmetrize)
        {
        ForceReal ang_neg = PI - ang; // acos(-cost)
        a.val += f4_val_angle<ForceReal>(ang_neg, t[0], t[1], t[2], t[3], t[4]);
        d -= f4_Dsin_angle<ForceReal>(ang_neg, t[0], t[1], t[2], t[3], t[4]);
        }
    a.dsin = sign * d;
    return a;
    }

//! Shared BASE-BASE geometry: the two body axes of each nucleotide, the two BASE
//! sites, the site-site direction, and the six angle cosines. This is IDENTICAL for
//! hydrogen bonding and cross-stacking (both act between the same BASE sites), so a
//! fused evaluator can compute it once and feed both terms. \a valid is false when
//! the pair is beyond \a rcutsq (no rotations done) or degenerate (rmod <= 0).
struct BaseBaseGeom
    {
    bool valid;
    vec3<ForceReal> a1, a3, b1, b3;
    vec3<ForceReal> base_i, base_j;
    vec3<ForceReal> rdir;
    ForceReal rmod;
    ForceReal cost1, cost2, cost3, cost4, cost7, cost8;
    };

HOSTDEVICE inline BaseBaseGeom base_base_geometry(const ForceReal3& dr,
                                                  const Scalar4& quat_i,
                                                  const Scalar4& quat_j,
                                                  ForceReal rcutsq,
                                                  const vec3<Scalar>& base_site)
    {
    BaseBaseGeom g;
    g.valid = false;
    ForceReal rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    if (rsq >= rcutsq)
        return g;

    quat<ForceReal> qi(quat_i), qj(quat_j);
    g.a1 = rotate(qi, vec3<ForceReal>(1, 0, 0));
    g.a3 = rotate(qi, vec3<ForceReal>(0, 0, 1));
    g.b1 = rotate(qj, vec3<ForceReal>(1, 0, 0));
    g.b3 = rotate(qj, vec3<ForceReal>(0, 0, 1));
    vec3<ForceReal> site(base_site);
    g.base_i = rotate(qi, site);
    g.base_j = rotate(qj, site);

    // _computed_r (p=i, q=j) = r_j - r_i = -dr
    vec3<ForceReal> Rc(-dr.x, -dr.y, -dr.z);
    vec3<ForceReal> rvec = Rc + g.base_j - g.base_i;
    g.rmod = sqrt(dot(rvec, rvec));
    if (g.rmod <= ForceReal(0.0))
        return g;
    g.rdir = rvec / g.rmod;

    g.cost1 = -dot(g.a1, g.b1);
    g.cost2 = -dot(g.b1, g.rdir);
    g.cost3 = dot(g.a1, g.rdir);
    g.cost4 = dot(g.a3, g.b3);
    g.cost7 = -dot(g.b3, g.rdir);
    g.cost8 = dot(g.a3, g.rdir);
    g.valid = true;
    return g;
    }

//! One BASE-BASE term (hydrogen bonding or cross-stacking) given the shared geometry.
//! \a radial.eval(rmod, val, deriv) supplies the radial factor (f1 or f2); \a
//! symmetrize applies to t4/t7/t8 (cross-stacking). The force on i (oxDNA p=i, q=j;
//! lab-frame torque) and both torques are ADDED to the accumulators, so a fused
//! evaluator can sum several terms sharing one geometry. All compute is ForceReal.
template<class Radial>
HOSTDEVICE inline void base_base_term(const BaseBaseGeom& g,
                                      const Scalar t1[5],
                                      const Scalar t23[5],
                                      const Scalar t4[5],
                                      const Scalar t78[5],
                                      bool symmetrize,
                                      const Radial& radial,
                                      vec3<ForceReal>& force_acc,
                                      ForceReal& eng_acc,
                                      vec3<ForceReal>& torque_i_acc,
                                      vec3<ForceReal>& torque_j_acc,
                                      bool include_t4 = true)
    {
    AngFactor A1 = angular(g.cost1, t1, false, ForceReal(-1.0));
    AngFactor A2 = angular(g.cost2, t23, false, ForceReal(-1.0));
    AngFactor A3 = angular(g.cost3, t23, false, ForceReal(1.0));
    // oxRNA cross-stacking drops the theta4 factor entirely (f4(t4) == 1); every
    // other term keeps it. Skipping it also zeroes its torque contribution.
    AngFactor A4 = include_t4 ? angular(g.cost4, t4, symmetrize, ForceReal(1.0))
                              : AngFactor {ForceReal(1.0), ForceReal(0.0)};
    AngFactor A7 = angular(g.cost7, t78, symmetrize, ForceReal(-1.0));
    AngFactor A8 = angular(g.cost8, t78, symmetrize, ForceReal(1.0));

    ForceReal prod = A1.val * A2.val * A3.val * A4.val * A7.val * A8.val;

    ForceReal rval, rderiv;
    radial.eval(g.rmod, rval, rderiv);

    ForceReal energy = rval * prod;
    eng_acc += energy;
    if (energy == ForceReal(0.0))
        return;

    const vec3<ForceReal>&a1 = g.a1, &a3 = g.a3, &b1 = g.b1, &b3 = g.b3, &rdir = g.rdir;
    ForceReal rmod = g.rmod;

    vec3<ForceReal> force(0, 0, 0); // force on q (=j)
    vec3<ForceReal> tp(0, 0, 0);    // torque on p (=i)
    vec3<ForceReal> tq(0, 0, 0);    // torque on q (=j)

    // RADIAL
    force = -rdir * (rderiv * prod);

    // THETA4
    vec3<ForceReal> dir = cross(a3, b3);
    ForceReal tm = -rval * A1.val * A2.val * A3.val * A4.dsin * A7.val * A8.val;
    tp -= dir * tm;
    tq += dir * tm;

    // THETA1
    dir = cross(a1, b1);
    tm = -rval * A1.dsin * A2.val * A3.val * A4.val * A7.val * A8.val;
    tp -= dir * tm;
    tq += dir * tm;

    // THETA2
    ForceReal fact = rval * A1.val * A2.dsin * A3.val * A4.val * A7.val * A8.val;
    force += (b1 + rdir * g.cost2) * (fact / rmod);
    dir = cross(rdir, b1);
    tq += -dir * fact;

    // THETA3
    force += (a1 - rdir * g.cost3)
             * (rval * A1.val * A2.val * A3.dsin * A4.val * A7.val * A8.val / rmod);
    vec3<ForceReal> t3dir = cross(rdir, a1);
    tm = -rval * A1.val * A2.val * A3.dsin * A4.val * A7.val * A8.val;
    tp += t3dir * tm;

    // THETA7
    force += (b3 + rdir * g.cost7)
             * (rval * A1.val * A2.val * A3.val * A4.val * A7.dsin * A8.val / rmod);
    vec3<ForceReal> t7dir = cross(rdir, b3);
    tm = -rval * A1.val * A2.val * A3.val * A4.val * A7.dsin * A8.val;
    tq += t7dir * tm;

    // THETA8
    force += (a3 - rdir * g.cost8)
             * (rval * A1.val * A2.val * A3.val * A4.val * A7.val * A8.dsin / rmod);
    vec3<ForceReal> t8dir = cross(rdir, a3);
    tm = -rval * A1.val * A2.val * A3.val * A4.val * A7.val * A8.dsin;
    tp += t8dir * tm;

    // site-offset torque
    tp -= cross(g.base_i, force);
    tq += cross(g.base_j, force);

    // force is on q (=j); HOOMD evaluator returns force on i (=p) = -force.
    force_acc += -force;
    torque_i_acc += tp;
    torque_j_acc += tq;
    }

//! Single BASE-BASE term evaluation (the non-fused path). Computes the shared
//! geometry then one term; used by the standalone HBond / CrossStacking evaluators.
template<class Radial>
HOSTDEVICE inline bool evaluate_base_base(const ForceReal3& dr,
                                          const Scalar4& quat_i,
                                          const Scalar4& quat_j,
                                          ForceReal rcutsq,
                                          const vec3<Scalar>& base_site,
                                          const Scalar t1[5],
                                          const Scalar t23[5],
                                          const Scalar t4[5],
                                          const Scalar t78[5],
                                          bool symmetrize,
                                          const Radial& radial,
                                          Scalar3& force_out,
                                          Scalar& pair_eng,
                                          Scalar3& torque_i,
                                          Scalar3& torque_j,
                                          bool include_t4 = true)
    {
    force_out = make_scalar3(0, 0, 0);
    torque_i = make_scalar3(0, 0, 0);
    torque_j = make_scalar3(0, 0, 0);
    pair_eng = Scalar(0.0);

    BaseBaseGeom g = base_base_geometry(dr, quat_i, quat_j, rcutsq, base_site);
    ForceReal rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
    if (rsq >= rcutsq)
        return false;
    if (!g.valid)
        return true;

    vec3<ForceReal> f(0, 0, 0), ti(0, 0, 0), tj(0, 0, 0);
    ForceReal e = ForceReal(0.0);
    base_base_term(g, t1, t23, t4, t78, symmetrize, radial, f, e, ti, tj, include_t4);
    force_out = vec_to_scalar3(f);
    pair_eng = e;
    torque_i = vec_to_scalar3(ti);
    torque_j = vec_to_scalar3(tj);
    return true;
    }

//! Radial functor: oxDNA f1 (Morse) with per-pair epsilon and shift. Fields arrive
//! as Scalar (from the param struct); the evaluation runs in ForceReal.
struct F1Radial
    {
    Scalar a, r0, rlow, rhigh, rclow, rchigh, blow, bhigh, eps, shift;
    HOSTDEVICE void eval(ForceReal r, ForceReal& val, ForceReal& deriv) const
        {
        val = f1_val<ForceReal>(r, a, r0, rlow, rhigh, rclow, rchigh, blow, bhigh, eps, shift);
        deriv = f1_deriv<ForceReal>(r, a, r0, rlow, rhigh, rclow, rchigh, blow, bhigh, eps);
        }
    };

//! Radial functor: oxDNA f2 (harmonic) with stiffness k.
struct F2Radial
    {
    Scalar k, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh;
    HOSTDEVICE void eval(ForceReal r, ForceReal& val, ForceReal& deriv) const
        {
        val = f2_val<ForceReal>(r, k, r0, rc, rlow, rhigh, rclow, rchigh, blow, bhigh);
        deriv = f2_deriv<ForceReal>(r, k, r0, rlow, rhigh, rclow, rchigh, blow, bhigh);
        }
    };

    } // end namespace oxdna
    } // end namespace md
    } // end namespace hoomd

#endif // __OXDNA_BASE_BASE_H__
