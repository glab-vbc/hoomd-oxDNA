// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNABondedForceCompute.h
    \brief Orientation-aware bonded force for oxDNA 3'-5' backbone neighbours.

    oxDNA's bonded terms (FENE backbone, bonded excluded volume, stacking) act at
    interaction SITES rigidly offset from each nucleotide's centre of mass, so they
    depend on the particle orientations and produce torque. HOOMD's central
    PotentialBond (force along the COM-COM axis, no torque) cannot express them; this
    custom ForceCompute iterates the bond table and writes force + torque. The physics
    lives in OxDNABondedKernel.h so the GPU subclass reuses identical code.
*/

#pragma once

#include "OxDNABondedParams.h"

#include "hoomd/BondedGroupData.h"
#include "hoomd/ForceCompute.h"
#include "hoomd/VectorMath.h"

#include <memory>
#include <vector>

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {

//! Computes oxDNA bonded forces (FENE + bonded excluded volume + stacking).
class PYBIND11_EXPORT OxDNABondedForceCompute : public ForceCompute
    {
    public:
    OxDNABondedForceCompute(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~OxDNABondedForceCompute();

    //! Set the parameters for one bond type (from a Python dict).
    virtual void setParamsPython(std::string type, pybind11::dict params);

    //! Get the parameters for one bond type as a dict.
    pybind11::dict getParams(std::string type);

    //! Total FENE backbone energy from the most recent computeForces().
    virtual Scalar getFeneEnergy() { return m_fene_energy; }
    //! Total bonded excluded-volume energy from the most recent computeForces().
    virtual Scalar getBondedExcVolEnergy() { return m_bexc_energy; }
    //! Total stacking energy from the most recent computeForces().
    virtual Scalar getStackingEnergy() { return m_stack_energy; }

#ifdef ENABLE_MPI
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags = CommFlags(0);
        flags[comm_flag::tag] = 1;
        flags[comm_flag::orientation] = 1;
        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    protected:
    std::shared_ptr<BondData> m_bond_data;     //!< Bond connectivity (3'-5' neighbours)
    std::vector<oxdna_bonded_params> m_params; //!< Per-bond-type parameters

    Scalar m_fene_energy;  //!< Accumulated FENE energy (per computeForces call)
    Scalar m_bexc_energy;  //!< Accumulated bonded excluded-volume energy
    Scalar m_stack_energy; //!< Accumulated stacking energy

    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
void export_OxDNABondedForceCompute(pybind11::module& m);
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd
