// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNABondedForceComputeGPU.h
    \brief GPU implementation of the oxDNA bonded force.
*/

#include "OxDNABondedForceCompute.h"
#include "OxDNABondedGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __OXDNA_BONDED_FORCE_COMPUTE_GPU_H__
#define __OXDNA_BONDED_FORCE_COMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes the oxDNA bonded force on the GPU (same physics as the CPU class).
class PYBIND11_EXPORT OxDNABondedForceComputeGPU : public OxDNABondedForceCompute
    {
    public:
    OxDNABondedForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~OxDNABondedForceComputeGPU();

    virtual void setParamsPython(std::string type, pybind11::dict params);

    // Split per-term energies are reduced on demand from the per-particle device array.
    virtual Scalar getFeneEnergy() { return reduceSplit(0); }
    virtual Scalar getBondedExcVolEnergy() { return reduceSplit(1); }
    virtual Scalar getStackingEnergy() { return reduceSplit(2); }

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner;         //!< Block-size autotuner
    GPUArray<oxdna_bonded_params> m_params_gpu;    //!< Per-bond-type parameters on device
    GPUArray<Scalar4> m_split_energy;              //!< Per-particle (fene, bexc, stack, 0)

    virtual void computeForces(uint64_t timestep);
    Scalar reduceSplit(int component);
    };

namespace detail
    {
void export_OxDNABondedForceComputeGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd

#endif // __OXDNA_BONDED_FORCE_COMPUTE_GPU_H__
