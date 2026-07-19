// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNABondedParams.h
    \brief Per-bond-type parameters for the oxDNA bonded terms (host + device).

    The parameter struct is a POD used on both the CPU ForceCompute and the GPU
    kernel, so it must compile under nvcc. Only the pybind11 (de)serialisation is
    host-only; everything else (the fields and the default constructor) is device
    compatible.
*/

#ifndef __OXDNA_BONDED_PARAMS_H__
#define __OXDNA_BONDED_PARAMS_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

#ifdef __HIPCC__
#define OXDNA_BP_HOSTDEVICE __host__ __device__
#else
#define OXDNA_BP_HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

//! Per-bond-type parameters for the oxDNA bonded terms (FENE + bonded exc-vol + stacking).
struct oxdna_bonded_params
    {
    // FENE backbone (DNAInteraction::_backbone)
    Scalar eps_backbone;   //!< FENE_EPS
    Scalar r0_backbone;    //!< equilibrium backbone-site separation
    Scalar delta_backbone; //!< FENE range (bond breaks at |r-r0| >= delta)
    Scalar delta2;         //!< delta_backbone^2 (derived)

    // Body-frame interaction-site offsets (oxDNA1: BACK=(-0.4,0,0), BASE=(0.4,0,0)).
    vec3<Scalar> back_site;
    vec3<Scalar> base_site;

    // Bonded excluded volume (DNAInteraction::_bonded_excluded_volume): EXCL_EPS +
    // three site pairs [0]=base-base, [1]=base(A)-back(B), [2]=back(A)-base(B).
    Scalar excl_eps;
    Scalar bx_sigma[3];
    Scalar bx_rstar[3];
    Scalar bx_b[3];
    Scalar bx_rc[3];

    // Stacking (DNAInteraction::_stacking). f1 radial x f4(t4)f4(t5)f4(t6) x f5(p1)f5(p2).
    bool stacking_enabled;
    vec3<Scalar> stack_site;     // POS_STACK * a1
    vec3<Scalar> stack_back_ref; // ungrooved POS_BACK * a1 (for rbackref)
    Scalar st_gamma;             // POS_STACK - POS_BACK
    Scalar st_a, st_r0, st_rlow, st_rhigh, st_rclow, st_rchigh, st_blow, st_bhigh, st_shift_factor;
    Scalar st_eps[4][4];         // per (n3, n5) stacking epsilon (T-dependent)
    Scalar t4_a, t4_b, t4_t0, t4_ts, t4_tc; // f4 theta4
    Scalar t5_a, t5_b, t5_t0, t5_ts, t5_tc; // f4 theta5 (== theta6)
    Scalar p1_a, p1_b, p1_xc, p1_xs;        // f5 phi1 (== phi2)

    OXDNA_BP_HOSTDEVICE oxdna_bonded_params()
        : eps_backbone(0), r0_backbone(0), delta_backbone(0), delta2(0),
          back_site(0, 0, 0), base_site(0, 0, 0), excl_eps(0),
          stacking_enabled(false), stack_site(0, 0, 0), stack_back_ref(0, 0, 0), st_gamma(0),
          st_a(0), st_r0(0), st_rlow(0), st_rhigh(0), st_rclow(0), st_rchigh(0),
          st_blow(0), st_bhigh(0), st_shift_factor(0),
          t4_a(0), t4_b(0), t4_t0(0), t4_ts(0), t4_tc(0),
          t5_a(0), t5_b(0), t5_t0(0), t5_ts(0), t5_tc(0),
          p1_a(0), p1_b(0), p1_xc(0), p1_xs(0)
        {
        for (int k = 0; k < 3; k++)
            bx_sigma[k] = bx_rstar[k] = bx_b[k] = bx_rc[k] = 0;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                st_eps[i][j] = 0;
        }

#ifndef __HIPCC__
    explicit oxdna_bonded_params(pybind11::dict params)
        {
        eps_backbone = params["epsilon"].cast<Scalar>();
        r0_backbone = params["r0"].cast<Scalar>();
        delta_backbone = params["delta"].cast<Scalar>();
        delta2 = delta_backbone * delta_backbone;
        back_site = read_vec3(params["pos_back"]);
        base_site = read_vec3(params["pos_base"]);
        excl_eps = params["excl_eps"].cast<Scalar>();
        read4(params["bexc_base_base"], 0);
        read4(params["bexc_base_back"], 1);
        read4(params["bexc_back_base"], 2);

        stacking_enabled = params.contains("stack_enabled") && params["stack_enabled"].cast<bool>();
        if (stacking_enabled)
            read_stacking(params);
        else
            {
            stack_site = vec3<Scalar>(0, 0, 0);
            stack_back_ref = vec3<Scalar>(0, 0, 0);
            st_gamma = 0;
            st_a = st_r0 = st_rlow = st_rhigh = st_rclow = st_rchigh = 0;
            st_blow = st_bhigh = st_shift_factor = 0;
            t4_a = t4_b = t4_t0 = t4_ts = t4_tc = 0;
            t5_a = t5_b = t5_t0 = t5_ts = t5_tc = 0;
            p1_a = p1_b = p1_xc = p1_xs = 0;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    st_eps[i][j] = 0;
            }
        }

    pybind11::dict asDict() const
        {
        pybind11::dict v;
        v["epsilon"] = eps_backbone;
        v["r0"] = r0_backbone;
        v["delta"] = delta_backbone;
        v["pos_back"] = pybind11::make_tuple(back_site.x, back_site.y, back_site.z);
        v["pos_base"] = pybind11::make_tuple(base_site.x, base_site.y, base_site.z);
        v["excl_eps"] = excl_eps;
        v["bexc_base_base"] = pack4(0);
        v["bexc_base_back"] = pack4(1);
        v["bexc_back_base"] = pack4(2);
        v["stack_enabled"] = stacking_enabled;
        v["stack_site"] = pybind11::make_tuple(stack_site.x, stack_site.y, stack_site.z);
        v["stack_back_ref"] = pybind11::make_tuple(stack_back_ref.x, stack_back_ref.y, stack_back_ref.z);
        v["stack_gamma"] = st_gamma;
        v["stack_f1"] = pybind11::make_tuple(st_a, st_r0, st_rlow, st_rhigh, st_rclow, st_rchigh,
                                             st_blow, st_bhigh, st_shift_factor);
        pybind11::list eps;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                eps.append(st_eps[i][j]);
        v["stack_eps"] = pybind11::tuple(eps);
        v["stack_f4_t4"] = pybind11::make_tuple(t4_a, t4_b, t4_t0, t4_ts, t4_tc);
        v["stack_f4_t5"] = pybind11::make_tuple(t5_a, t5_b, t5_t0, t5_ts, t5_tc);
        v["stack_f5_p1"] = pybind11::make_tuple(p1_a, p1_b, p1_xc, p1_xs);
        return v;
        }

    private:
    static vec3<Scalar> read_vec3(pybind11::object o)
        {
        pybind11::tuple t(o);
        return vec3<Scalar>(t[0].cast<Scalar>(), t[1].cast<Scalar>(), t[2].cast<Scalar>());
        }
    void read4(pybind11::object o, int k)
        {
        pybind11::tuple t(o);
        bx_sigma[k] = t[0].cast<Scalar>();
        bx_rstar[k] = t[1].cast<Scalar>();
        bx_b[k] = t[2].cast<Scalar>();
        bx_rc[k] = t[3].cast<Scalar>();
        }
    pybind11::tuple pack4(int k) const
        {
        return pybind11::make_tuple(bx_sigma[k], bx_rstar[k], bx_b[k], bx_rc[k]);
        }
    void read_stacking(pybind11::dict s)
        {
        stack_site = read_vec3(s["stack_site"]);
        stack_back_ref = read_vec3(s["stack_back_ref"]);
        st_gamma = s["stack_gamma"].cast<Scalar>();
        pybind11::tuple f1t(s["stack_f1"]);
        st_a = f1t[0].cast<Scalar>();
        st_r0 = f1t[1].cast<Scalar>();
        st_rlow = f1t[2].cast<Scalar>();
        st_rhigh = f1t[3].cast<Scalar>();
        st_rclow = f1t[4].cast<Scalar>();
        st_rchigh = f1t[5].cast<Scalar>();
        st_blow = f1t[6].cast<Scalar>();
        st_bhigh = f1t[7].cast<Scalar>();
        st_shift_factor = f1t[8].cast<Scalar>();
        pybind11::tuple e(s["stack_eps"]);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                st_eps[i][j] = e[i * 4 + j].cast<Scalar>();
        read5(s["stack_f4_t4"], t4_a, t4_b, t4_t0, t4_ts, t4_tc);
        read5(s["stack_f4_t5"], t5_a, t5_b, t5_t0, t5_ts, t5_tc);
        pybind11::tuple p(s["stack_f5_p1"]);
        p1_a = p[0].cast<Scalar>();
        p1_b = p[1].cast<Scalar>();
        p1_xc = p[2].cast<Scalar>();
        p1_xs = p[3].cast<Scalar>();
        }
    static void read5(pybind11::object o, Scalar& a, Scalar& b, Scalar& t0, Scalar& ts, Scalar& tc)
        {
        pybind11::tuple t(o);
        a = t[0].cast<Scalar>();
        b = t[1].cast<Scalar>();
        t0 = t[2].cast<Scalar>();
        ts = t[3].cast<Scalar>();
        tc = t[4].cast<Scalar>();
        }
#endif // __HIPCC__
    };

    } // end namespace md
    } // end namespace hoomd

#undef OXDNA_BP_HOSTDEVICE
#endif // __OXDNA_BONDED_PARAMS_H__
