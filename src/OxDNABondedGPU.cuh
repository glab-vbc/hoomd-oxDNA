// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file OxDNABondedGPU.cuh
    \brief GPU kernel driver declaration for the oxDNA bonded force.
*/

#ifndef __OXDNA_BONDED_GPU_CUH__
#define __OXDNA_BONDED_GPU_CUH__

#include "hoomd/BondedGroupData.cuh"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleData.cuh"
#include "MixedPrecisionCompat.h"
#include "OxDNABondedParams.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

//! Kernel driver: one thread per particle, iterating that particle's backbone bonds.
hipError_t gpu_compute_oxdna_bonded_forces(ForceReal4* d_force,
                                           ForceReal4* d_torque,
                                           ForceReal* d_virial,
                                           const size_t virial_pitch,
                                           Scalar4* d_split_energy,
                                           const unsigned int N,
                                           const Scalar4* d_pos,
                                           const Scalar4* d_orientation,
                                           const BoxDim& box,
                                           const group_storage<2>* blist,
                                           const unsigned int* bpos_list,
                                           const unsigned int pitch,
                                           const unsigned int* n_bonds_list,
                                           const oxdna_bonded_params* d_params,
                                           const unsigned int n_bond_types,
                                           int block_size);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd

#endif // __OXDNA_BONDED_GPU_CUH__
