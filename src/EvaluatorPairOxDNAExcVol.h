// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file EvaluatorPairOxDNAExcVol.h
    \brief oxDNA nonbonded excluded-volume evaluator for AnisoPotentialPair.

    oxDNA's nonbonded excluded volume is a sum of four site-site repulsions between
    the BACK and BASE interaction sites of the two nucleotides (DNAInteraction::
    _nonbonded_excluded_volume): BASE-BASE, BACK-BASE, BASE-BACK and BACK-BACK, each
    a quadratically-smoothed LJ core (_repulsive_lj) with its own (sigma, r*, b, rc).
    Site offsets and per-pair constants are parameters, so oxDNA1/oxDNA2 share it.
*/

#ifndef __EVALUATOR_PAIR_OXDNA_EXCVOL_H__
#define __EVALUATOR_PAIR_OXDNA_EXCVOL_H__

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

//! Parameters for one type pair of the oxDNA excluded-volume term.
struct OxDNAExcVolParams
    {
    Scalar eps;             //!< EXCL_EPS
    vec3<Scalar> base_site; //!< body-frame BASE offset
    vec3<Scalar> back_site; //!< body-frame BACK offset
    // Four site pairs: [0]=back-back, [1]=base-base, [2]=base(i)-back(j), [3]=back(i)-base(j).
    Scalar sigma[4];
    Scalar rstar[4];
    Scalar b[4];
    Scalar rc[4];

#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    DEVICE void load_shared(char*&, unsigned int&) { }
    HOSTDEVICE void allocate_shared(char*&, unsigned int&) const { }

    HOSTDEVICE OxDNAExcVolParams() : eps(0), base_site(0, 0, 0), back_site(0, 0, 0)
        {
        for (int k = 0; k < 4; k++)
            sigma[k] = rstar[k] = b[k] = rc[k] = 0;
        }

#ifndef __HIPCC__
    OxDNAExcVolParams(pybind11::dict v, bool managed)
        {
        eps = v["eps"].cast<Scalar>();
        base_site = oxdna_io::read_vec3(v["base_site"]);
        back_site = oxdna_io::read_vec3(v["back_site"]);
        read4(v["back_back"], 0);
        read4(v["base_base"], 1);
        read4(v["base_back"], 2);
        read4(v["back_base"], 3);
        }

    pybind11::object toPython()
        {
        pybind11::dict v;
        v["eps"] = eps;
        v["base_site"] = oxdna_io::pack_vec3(base_site);
        v["back_site"] = oxdna_io::pack_vec3(back_site);
        v["back_back"] = pack4(0);
        v["base_base"] = pack4(1);
        v["base_back"] = pack4(2);
        v["back_base"] = pack4(3);
        return std::move(v);
        }

    private:
    void read4(pybind11::object o, int k)
        {
        pybind11::tuple t(o);
        sigma[k] = t[0].cast<Scalar>();
        rstar[k] = t[1].cast<Scalar>();
        b[k] = t[2].cast<Scalar>();
        rc[k] = t[3].cast<Scalar>();
        }
    pybind11::tuple pack4(int k)
        {
        return pybind11::make_tuple(sigma[k], rstar[k], b[k], rc[k]);
        }
    public:
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

class EvaluatorPairOxDNAExcVol
    : public OxDNAAnisoPair<EvaluatorPairOxDNAExcVol, OxDNAExcVolParams>
    {
    public:
    typedef OxDNAAnisoPair<EvaluatorPairOxDNAExcVol, OxDNAExcVolParams> Base;
    using Base::Base;

    HOSTDEVICE bool evaluate_scalar(Scalar3& force,
                                    Scalar& pair_eng,
                                    bool /*energy_shift*/,
                                    Scalar3& torque_i,
                                    Scalar3& torque_j)
        {
        ForceReal rsq = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
        if (rsq >= rcutsq)
            return false;

        quat<ForceReal> qi(quat_i), qj(quat_j);
        vec3<ForceReal> base_site(p.base_site), back_site(p.back_site);
        vec3<ForceReal> base_i = rotate(qi, base_site);
        vec3<ForceReal> back_i = rotate(qi, back_site);
        vec3<ForceReal> base_j = rotate(qj, base_site);
        vec3<ForceReal> back_j = rotate(qj, back_site);

        // _computed_r (oxDNA, p=i q=j) = r_j - r_i = -dr
        vec3<ForceReal> Rc(-dr.x, -dr.y, -dr.z);

        const vec3<ForceReal>* site_i[4] = {&back_i, &base_i, &base_i, &back_i};
        const vec3<ForceReal>* site_j[4] = {&back_j, &base_j, &back_j, &base_j};

        vec3<ForceReal> F_i(0, 0, 0), tau_i(0, 0, 0), tau_j(0, 0, 0);
        ForceReal energy = ForceReal(0.0);

        for (int k = 0; k < 4; k++)
            {
            vec3<ForceReal> rcenter = Rc + *site_j[k] - *site_i[k];
            vec3<ForceReal> oxforce; // force on the j site (oxDNA convention)
            energy += oxdna::repulsive_lj<ForceReal>(rcenter, p.eps, p.sigma[k], p.rstar[k],
                                                     p.b[k], p.rc[k], oxforce);
            F_i += -oxforce;
            tau_i += cross(*site_i[k], -oxforce);
            tau_j += cross(*site_j[k], oxforce);
            }

        force = vec_to_scalar3(F_i);
        torque_i = vec_to_scalar3(tau_i);
        torque_j = vec_to_scalar3(tau_j);
        pair_eng = energy;
        return true;
        }

#ifndef __HIPCC__
    static std::string getName() { return "oxdna_excvol"; }
#endif
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_PAIR_OXDNA_EXCVOL_H__
