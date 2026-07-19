// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file EvaluatorPairOxDNADebye.h
    \brief oxDNA2 Debye-Huckel electrostatics evaluator for AnisoPotentialPair.

    DNA2Interaction::_debye_huckel, a screened Coulomb interaction between the BACK
    sites: U = prefactor * exp(-kappa r)/r for r < RHIGH, smoothed to B*(r-RC)^2 up
    to RC. Terminal nucleotides carry half charge; this is encoded as a per-particle
    charge (0.5 at strand ends, 1.0 otherwise) with cut_factor = charge_i*charge_j.
*/

#ifndef __EVALUATOR_PAIR_OXDNA_DEBYE_H__
#define __EVALUATOR_PAIR_OXDNA_DEBYE_H__

#include "OxDNAAnisoPair.h"
#ifndef __HIPCC__
#include "OxDNAParamIO.h"
#include <string>
#endif

namespace hoomd
    {
namespace md
    {

//! Parameters for one type pair of the oxDNA2 Debye-Huckel term.
struct OxDNADebyeParams
    {
    vec3<Scalar> back_site;
    Scalar minus_kappa, prefactor, b_smooth, rc, rhigh;

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    DEVICE void load_shared(char*&, unsigned int&) { }
    HOSTDEVICE void allocate_shared(char*&, unsigned int&) const { }

    HOSTDEVICE OxDNADebyeParams()
        : back_site(0, 0, 0), minus_kappa(0), prefactor(0), b_smooth(0), rc(0), rhigh(0)
        {
        }

#ifndef __HIPCC__
    OxDNADebyeParams(pybind11::dict v, bool managed)
        {
        back_site = oxdna_io::read_vec3(v["back_site"]);
        pybind11::tuple d(v["debye"]);
        minus_kappa = d[0].cast<Scalar>();
        prefactor = d[1].cast<Scalar>();
        b_smooth = d[2].cast<Scalar>();
        rc = d[3].cast<Scalar>();
        rhigh = d[4].cast<Scalar>();
        }

    pybind11::object toPython()
        {
        pybind11::dict v;
        v["back_site"] = oxdna_io::pack_vec3(back_site);
        v["debye"] = pybind11::make_tuple(minus_kappa, prefactor, b_smooth, rc, rhigh);
        return std::move(v);
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

class EvaluatorPairOxDNADebye : public OxDNAAnisoPair<EvaluatorPairOxDNADebye, OxDNADebyeParams>
    {
    public:
    typedef OxDNAAnisoPair<EvaluatorPairOxDNADebye, OxDNADebyeParams> Base;
    using Base::Base;

    // Debye-Huckel is the only oxDNA term that reads per-particle charge.
    HOSTDEVICE static bool needsCharge() { return true; }
    HOSTDEVICE void setCharge(Scalar qi, Scalar qj)
        {
        charge_i = qi;
        charge_j = qj;
        }

    HOSTDEVICE bool evaluate_scalar(Scalar3& force_out,
                                    Scalar& pair_eng,
                                    bool /*energy_shift*/,
                                    Scalar3& torque_i,
                                    Scalar3& torque_j)
        {
        force_out = make_scalar3(0, 0, 0);
        torque_i = make_scalar3(0, 0, 0);
        torque_j = make_scalar3(0, 0, 0);
        pair_eng = Scalar(0.0);

        ForceReal rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        if (rsq >= rcutsq)
            return false;

        ForceReal rc(p.rc), rhigh(p.rhigh), minus_kappa(p.minus_kappa),
            prefactor(p.prefactor), b_smooth(p.b_smooth);

        quat<ForceReal> qi(quat_i), qj(quat_j);
        vec3<ForceReal> back_site(p.back_site);
        vec3<ForceReal> back_i = rotate(qi, back_site);
        vec3<ForceReal> back_j = rotate(qj, back_site);
        vec3<ForceReal> Rc(-dr.x, -dr.y, -dr.z);
        vec3<ForceReal> rback = Rc + back_j - back_i;
        ForceReal rbackmod = sqrt(dot(rback, rback));
        if (rbackmod <= ForceReal(0.0) || rbackmod >= rc)
            return true;

        ForceReal cut_factor = ForceReal(charge_i) * ForceReal(charge_j);
        ForceReal energy;
        vec3<ForceReal> force; // force on q (=j)
        vec3<ForceReal> rbackdir = rback / rbackmod;

        if (rbackmod < rhigh)
            {
            ForceReal ex = exp(rbackmod * minus_kappa);
            energy = ex * (prefactor / rbackmod);
            force = rbackdir
                    * (ForceReal(-1.0) * (prefactor * ex)
                       * (minus_kappa / rbackmod - ForceReal(1.0) / (rbackmod * rbackmod)));
            }
        else
            {
            ForceReal drc = rbackmod - rc;
            energy = b_smooth * drc * drc;
            force = -rbackdir * (ForceReal(2.0) * b_smooth * drc);
            }
        energy *= cut_factor;
        force = force * cut_factor;

        pair_eng = energy;
        vec3<ForceReal> torquep = -cross(back_i, force);
        vec3<ForceReal> torqueq = cross(back_j, force);

        // HOOMD evaluator returns force on i (=p) = -force.
        force_out = vec_to_scalar3(-force);
        torque_i = vec_to_scalar3(torquep);
        torque_j = vec_to_scalar3(torqueq);
        return true;
        }

#ifndef __HIPCC__
    static std::string getName() { return "oxdna_debye"; }
#endif

    protected:
    Scalar charge_i = 1, charge_j = 1;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_OXDNA_DEBYE_H__
