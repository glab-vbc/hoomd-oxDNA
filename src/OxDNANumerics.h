// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNANumerics.h
    \brief Device-callable numeric primitives shared by the oxDNA force engine.

    This header collects the oxDNA potential building blocks (FENE backbone, the
    excluded-volume LJ core, and the f1/f2/f4/f5 modulation/smoothing factors) as
    plain host+device inline functions so the same math is reused by the CPU
    computes, the anisotropic pair evaluators, and the GPU kernels.

    Every primitive is templated on the working real type \c Real. The bonded force
    instantiates them at \c Scalar (double) for exact reference parity and to keep
    the FENE log well away from catastrophic cancellation; the nonbonded evaluators
    instantiate them at \c ForceReal (single precision on the mixed-precision fork),
    which is where the transcendental-heavy angular factors run cheaply on a gaming
    GPU. On vanilla HOOMD \c ForceReal aliases \c Scalar, so both paths stay double.

    All expressions follow oxDNA's analytic (`_nomesh`) reference in
    oxDNA/src/Interactions/DNAInteraction.cpp and oxDNA/src/model.h. Everything is
    in oxDNA simulation units; no unit conversion happens here.
*/

#ifndef __OXDNA_NUMERICS_H__
#define __OXDNA_NUMERICS_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifdef __HIPCC__
#define OXDNA_HOSTDEVICE __host__ __device__
#else
#define OXDNA_HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {
namespace oxdna
    {

//! FENE backbone energy for a given (r - r0) offset.
/*! Matches DNAInteraction::_backbone: E = -(eps/2) * ln(1 - dr^2/delta2).
    The caller is responsible for the broken-bond check (|dr| >= delta).

    Kept at double in the bonded path on purpose: near a stretched bond
    dr^2/delta2 -> 1 and 1 - dr^2/delta2 loses all significant digits in float.
*/
template<class Real>
OXDNA_HOSTDEVICE inline Real fene_energy(Real dr, Real eps, Real delta2)
    {
    return Real(-0.5) * eps * log(Real(1.0) - dr * dr / delta2);
    }

//! FENE backbone force prefactor such that force_on_q = prefactor * rback_vec.
/*! Matches DNAInteraction::_backbone: force = rback * (-(eps*dr/(delta2-dr^2))/rmod). */
template<class Real>
OXDNA_HOSTDEVICE inline Real fene_force_prefactor(Real dr, Real rmod, Real eps, Real delta2)
    {
    return -(eps * dr / (delta2 - dr * dr)) / rmod;
    }

//! Truncated, quadratically-smoothed LJ repulsion (excluded-volume core).
/*! Matches DNAInteraction::_repulsive_lj. \a r is the site-site vector
    (site_q - site_p). Returns the energy and writes \a force_out = the force on the
    "q" site (oxDNA convention: p->force -= force, q->force += force). Zero beyond rc.

    \param r        site-site vector (site_q - site_p)
    \param eps      EXCL_EPS
    \param sigma    LJ sigma for this site pair
    \param rstar    crossover radius to the quadratic smoothing branch
    \param b        smoothing amplitude
    \param rc       hard cutoff (energy and force vanish at/after rc)
*/
template<class Real>
OXDNA_HOSTDEVICE inline Real repulsive_lj(const vec3<Real>& r,
                                          Real eps,
                                          Real sigma,
                                          Real rstar,
                                          Real b,
                                          Real rc,
                                          vec3<Real>& force_out)
    {
    Real rnorm = dot(r, r);
    Real energy = Real(0.0);
    force_out = vec3<Real>(0, 0, 0);

    if (rnorm < rc * rc)
        {
        if (rnorm > rstar * rstar)
            {
            Real rmod = sqrt(rnorm);
            Real rrc = rmod - rc;
            energy = eps * b * rrc * rrc;
            force_out = (Real(-2.0) * eps * b * rrc / rmod) * r;
            }
        else
            {
            Real tmp = sigma * sigma / rnorm;
            Real lj = tmp * tmp * tmp;
            energy = Real(4.0) * eps * (lj * lj - lj);
            force_out = (Real(-24.0) * eps * (lj - Real(2.0) * lj * lj) / rnorm) * r;
            }
        }
    return energy;
    }

// --- oxDNA modulation factors f1 (Morse radial), f4 (angular), f5 (cos-phi) ---
// These follow DNAInteraction::_f1/_f1D, _f4/_f4Dsin, _f5/_f5D. The smoothing
// constants (blow/bhigh/rlow/... , b/ts/tc, b/xs/xc) are precomputed and passed in.

//! f1 radial (Morse) value. eps and shift are per-(n3,n5); shift = eps*shift_factor.
template<class Real>
OXDNA_HOSTDEVICE inline Real f1_val(Real r, Real a, Real r0, Real rlow, Real rhigh,
                                    Real rclow, Real rchigh, Real blow, Real bhigh,
                                    Real eps, Real shift)
    {
    Real val = Real(0.0);
    if (r < rchigh)
        {
        if (r > rhigh)
            val = eps * bhigh * (r - rchigh) * (r - rchigh);
        else if (r > rlow)
            {
            Real tmp = Real(1.0) - exp(-(r - r0) * a);
            val = eps * tmp * tmp - shift;
            }
        else if (r > rclow)
            val = eps * blow * (r - rclow) * (r - rclow);
        }
    return val;
    }

//! d f1 / dr.
template<class Real>
OXDNA_HOSTDEVICE inline Real f1_deriv(Real r, Real a, Real r0, Real rlow, Real rhigh,
                                      Real rclow, Real rchigh, Real blow, Real bhigh,
                                      Real eps)
    {
    Real val = Real(0.0);
    if (r < rchigh)
        {
        if (r > rhigh)
            val = Real(2.0) * bhigh * (r - rchigh);
        else if (r > rlow)
            {
            Real tmp = exp(-(r - r0) * a);
            val = Real(2.0) * (Real(1.0) - tmp) * tmp * a;
            }
        else if (r > rclow)
            val = Real(2.0) * blow * (r - rclow);
        }
    return eps * val;
    }

//! f2 radial (harmonic) value (DNAInteraction::_f2).
template<class Real>
OXDNA_HOSTDEVICE inline Real f2_val(Real r, Real k, Real r0, Real rc, Real rlow,
                                    Real rhigh, Real rclow, Real rchigh, Real blow,
                                    Real bhigh)
    {
    Real val = Real(0.0);
    if (r < rchigh)
        {
        if (r > rhigh)
            val = k * bhigh * (r - rchigh) * (r - rchigh);
        else if (r > rlow)
            val = (k / Real(2.0)) * ((r - r0) * (r - r0) - (rc - r0) * (rc - r0));
        else if (r > rclow)
            val = k * blow * (r - rclow) * (r - rclow);
        }
    return val;
    }

//! d f2 / dr (DNAInteraction::_f2D).
template<class Real>
OXDNA_HOSTDEVICE inline Real f2_deriv(Real r, Real k, Real r0, Real rlow, Real rhigh,
                                      Real rclow, Real rchigh, Real blow, Real bhigh)
    {
    Real val = Real(0.0);
    if (r < rchigh)
        {
        if (r > rhigh)
            val = Real(2.0) * k * bhigh * (r - rchigh);
        else if (r > rlow)
            val = k * (r - r0);
        else if (r > rclow)
            val = Real(2.0) * k * blow * (r - rclow);
        }
    return val;
    }

template<class Real>
OXDNA_HOSTDEVICE inline Real clamp_cos(Real c)
    {
    return c > Real(1.0) ? Real(1.0) : (c < Real(-1.0) ? Real(-1.0) : c);
    }

//! f4 angular value as a function of the angle t (DNAInteraction::_f4).
template<class Real>
OXDNA_HOSTDEVICE inline Real f4_val_angle(Real t_angle, Real a, Real b, Real t0,
                                          Real ts, Real tc)
    {
    Real t = t_angle - t0;
    if (t < Real(0.0))
        t = -t;
    if (t < tc)
        {
        if (t > ts)
            return b * (tc - t) * (tc - t);
        return Real(1.0) - a * t * t;
        }
    return Real(0.0);
    }

//! f4 torque coefficient as a function of the angle t: _f4Dsin(t) = (d f4/d theta)/sin(t).
template<class Real>
OXDNA_HOSTDEVICE inline Real f4_Dsin_angle(Real t_angle, Real a, Real b, Real t0,
                                           Real ts, Real tc)
    {
    Real tt0 = t_angle - t0;
    Real m = Real(1.0);
    if (tt0 < Real(0.0))
        {
        tt0 = -tt0;
        m = Real(-1.0);
        }
    Real val = Real(0.0);
    if (tt0 < tc)
        {
        Real sint = sin(t_angle);
        if (tt0 > ts)
            val = m * Real(2.0) * b * (tt0 - tc) / sint;
        else
            {
            if (sint * sint > Real(1e-8))
                val = -m * Real(2.0) * a * tt0 / sint;
            else
                val = -m * Real(2.0) * a;
            }
        }
    return val;
    }

//! f4 value as a function of the cosine argument (DNAInteraction::_f4 at acos(cost)).
template<class Real>
OXDNA_HOSTDEVICE inline Real f4_val(Real cost, Real a, Real b, Real t0, Real ts, Real tc)
    {
    return f4_val_angle(acos(clamp_cos(cost)), a, b, t0, ts, tc);
    }

//! f4 torque coefficient: _f4Dsin(acos(cost)).
template<class Real>
OXDNA_HOSTDEVICE inline Real f4_Dsin(Real cost, Real a, Real b, Real t0, Real ts, Real tc)
    {
    return f4_Dsin_angle(acos(clamp_cos(cost)), a, b, t0, ts, tc);
    }

//! f5 value as a function of cos(phi) (DNAInteraction::_f5).
template<class Real>
OXDNA_HOSTDEVICE inline Real f5_val(Real f, Real a, Real b, Real xc, Real xs)
    {
    if (f > xc)
        {
        if (f < xs)
            return b * (xc - f) * (xc - f);
        else if (f < Real(0.0))
            return Real(1.0) - a * f * f;
        else
            return Real(1.0);
        }
    return Real(0.0);
    }

//! d f5 / d(cos phi) (DNAInteraction::_f5D).
template<class Real>
OXDNA_HOSTDEVICE inline Real f5_deriv(Real f, Real a, Real b, Real xc, Real xs)
    {
    if (f > xc)
        {
        if (f < xs)
            return Real(2.0) * b * (f - xc);
        else if (f < Real(0.0))
            return Real(-2.0) * a * f;
        else
            return Real(0.0);
        }
    return Real(0.0);
    }

    } // end namespace oxdna
    } // end namespace md
    } // end namespace hoomd

#undef OXDNA_HOSTDEVICE

#endif // __OXDNA_NUMERICS_H__
