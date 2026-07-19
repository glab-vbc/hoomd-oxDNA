// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNAAnisoPair.h
    \brief CRTP base for the oxDNA anisotropic pair evaluators.

    Every oxDNA nonbonded term is an AnisoPotentialPair evaluator that receives the
    COM displacement plus both quaternions and returns force + two torques. The
    scaffolding around that (constructor, the ForceReal<->Scalar bridge, the trivial
    capability flags, the nullary shape_type, and the LRC/name stubs) is identical
    across terms, so it lives here once.

    A concrete evaluator derives as
        class EvaluatorX : public OxDNAAnisoPair<EvaluatorX, XParams> { ... }
    and supplies only:
      * the parameter struct ``XParams`` (the second template argument),
      * ``bool evaluate_scalar(force, pair_eng, energy_shift, torque_i, torque_j)``,
      * ``static std::string getName()``.
    Terms that need particle charge (Debye-Huckel) additionally redefine the static
    ``needsCharge()`` and instance ``setCharge()`` methods.
*/

#ifndef __OXDNA_ANISO_PAIR_H__
#define __OXDNA_ANISO_PAIR_H__

#ifndef __HIPCC__
#include <stdexcept>
#include <string>
#endif

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include "MixedPrecisionCompat.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {

//! Nullary shape type shared by every oxDNA aniso evaluator (no per-type shape data).
struct OxDNANullShape
    {
    DEVICE void load_shared(char*&, unsigned int&) { }
    HOSTDEVICE void allocate_shared(char*&, unsigned int&) const { }
    HOSTDEVICE OxDNANullShape() { }
#ifndef __HIPCC__
    OxDNANullShape(pybind11::object, bool) { }
    pybind11::object toPython() { return pybind11::none(); }
#endif
#ifdef ENABLE_HIP
    void set_memory_hint() const { }
#endif
    };

//! CRTP base providing the common AnisoPotentialPair evaluator interface.
template<class Derived, class ParamType> class OxDNAAnisoPair
    {
    public:
    typedef ParamType param_type;
    typedef OxDNANullShape shape_type;

    HOSTDEVICE OxDNAAnisoPair(const ForceReal3& _dr,
                              const Scalar4& _quat_i,
                              const Scalar4& _quat_j,
                              const ForceReal _rcutsq,
                              const param_type& _params)
        : dr(_dr), quat_i(_quat_i), quat_j(_quat_j), rcutsq(_rcutsq), p(_params)
        {
        }

    HOSTDEVICE static bool needsShape() { return false; }
    HOSTDEVICE static bool needsTags() { return false; }
    HOSTDEVICE static bool needsCharge() { return false; }
    HOSTDEVICE static constexpr bool implementsEnergyShift() { return false; }
    HOSTDEVICE void setShape(const shape_type*, const shape_type*) { }
    HOSTDEVICE void setTags(unsigned int, unsigned int) { }
    HOSTDEVICE void setCharge(Scalar, Scalar) { }

    //! AnisoPotentialPair entry point. Forwards to the derived evaluate_scalar(),
    //! which keeps a Scalar3 interface but computes in ForceReal internally (single
    //! precision on the mixed-precision fork); the Scalar3 holders here are just the
    //! boundary and cost nothing next to the term's transcendentals.
    HOSTDEVICE bool evaluate(ForceReal3& force,
                             ForceReal& pair_eng,
                             bool energy_shift,
                             ForceReal3& torque_i,
                             ForceReal3& torque_j)
        {
        Scalar3 f {}, ti {}, tj {};
        Scalar e {};
        bool ret = static_cast<Derived*>(this)->evaluate_scalar(f, e, energy_shift, ti, tj);
        force = make_forcereal3(ForceReal(f.x), ForceReal(f.y), ForceReal(f.z));
        pair_eng = ForceReal(e);
        torque_i = make_forcereal3(ForceReal(ti.x), ForceReal(ti.y), ForceReal(ti.z));
        torque_j = make_forcereal3(ForceReal(tj.x), ForceReal(tj.y), ForceReal(tj.z));
        return ret;
        }

    DEVICE Scalar evalPressureLRCIntegral() { return 0; }
    DEVICE Scalar evalEnergyLRCIntegral() { return 0; }

#ifndef __HIPCC__
    static std::string getShapeParamName() { return "Shape"; }
    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for oxDNA pair potentials.");
        }
#endif

    protected:
    ForceReal3 dr;
    Scalar4 quat_i, quat_j;
    ForceReal rcutsq;
    param_type p;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __OXDNA_ANISO_PAIR_H__
