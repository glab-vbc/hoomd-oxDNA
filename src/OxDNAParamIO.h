// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNAParamIO.h
    \brief Small pybind11 <-> POD helpers shared by the oxDNA parameter structs.

    Each evaluator's ``param_type`` is constructed from a Python dict and can be
    serialised back. These free functions remove the read_vec3 / read-N-tuple /
    pack-N-tuple boilerplate that would otherwise be copied into every param struct.
    Host-only (they touch pybind11); guarded out of device compilation.
*/

#ifndef __OXDNA_PARAM_IO_H__
#define __OXDNA_PARAM_IO_H__

#ifndef __HIPCC__

#include <pybind11/pybind11.h>

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

namespace hoomd
    {
namespace md
    {
namespace oxdna_io
    {

//! Read a 3-tuple into a vec3<Scalar>.
inline vec3<Scalar> read_vec3(pybind11::object o)
    {
    pybind11::tuple t(o);
    return vec3<Scalar>(t[0].cast<Scalar>(), t[1].cast<Scalar>(), t[2].cast<Scalar>());
    }

//! Pack a vec3<Scalar> into a 3-tuple.
inline pybind11::tuple pack_vec3(const vec3<Scalar>& v)
    {
    return pybind11::make_tuple(v.x, v.y, v.z);
    }

//! Read the first N elements of a tuple into a Scalar array.
template<int N> inline void read_scalars(pybind11::object o, Scalar dst[N])
    {
    pybind11::tuple t(o);
    for (int k = 0; k < N; k++)
        dst[k] = t[k].cast<Scalar>();
    }

//! Pack an N-element Scalar array into a tuple.
template<int N> inline pybind11::tuple pack_scalars(const Scalar src[N])
    {
    pybind11::list l;
    for (int k = 0; k < N; k++)
        l.append(src[k]);
    return pybind11::tuple(l);
    }

    } // end namespace oxdna_io
    } // end namespace md
    } // end namespace hoomd

#endif // __HIPCC__
#endif // __OXDNA_PARAM_IO_H__
