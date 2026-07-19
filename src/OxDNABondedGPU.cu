// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"

#include "OxDNABondedGPU.cuh"
#include "OxDNABondedKernel.h"
#include "hoomd/VectorMath.h"

/*! \file OxDNABondedGPU.cu
    \brief GPU kernel for the oxDNA bonded force.

    One thread per particle. Each thread iterates the backbone bonds it participates
    in (via HOOMD's per-particle bond table), evaluates the full bonded pair with the
    shared oxdna_bonded_pair() physics, and accumulates ONLY its own force / torque /
    virial / split-energy (no atomics; each bond is evaluated by both endpoints).
    Inside oxdna_bonded_pair the FENE term is computed in double (its log/1-dr^2
    cancels in single precision near a stretched bond); the stacking and bonded
    excluded-volume terms — the bulk of the cost — are computed in ForceReal (single
    precision on the fork). The CPU path uses the same shared physics.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

__global__ void gpu_compute_oxdna_bonded_forces_kernel(ForceReal4* d_force,
                                                       ForceReal4* d_torque,
                                                       ForceReal* d_virial,
                                                       const size_t virial_pitch,
                                                       Scalar4* d_split_energy,
                                                       const unsigned int N,
                                                       const Scalar4* d_pos,
                                                       const Scalar4* d_orientation,
                                                       BoxDim box,
                                                       const group_storage<2>* blist,
                                                       const unsigned int* bpos_list,
                                                       const unsigned int pitch,
                                                       const unsigned int* n_bonds_list,
                                                       const oxdna_bonded_params* d_params)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int)N)
        return;

    int n_bonds = n_bonds_list[idx];

    Scalar4 idx_postype = d_pos[idx];
    vec3<Scalar> idx_pos(idx_postype.x, idx_postype.y, idx_postype.z);
    unsigned int idx_type = __scalar_as_int(idx_postype.w);
    quat<Scalar> q_idx(d_orientation[idx]);

    vec3<Scalar> force(0, 0, 0);  // translational force on this particle
    vec3<Scalar> torque(0, 0, 0); // torque on this particle
    Scalar energy = Scalar(0.0);
    Scalar split[3] = {Scalar(0.0), Scalar(0.0), Scalar(0.0)}; // fene, bexc, stack half-energies
    Scalar virial[6] = {0, 0, 0, 0, 0, 0};

    for (int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        group_storage<2> cur_bond = blist[pitch * bond_idx + idx];
        int other_idx = cur_bond.idx[0];
        unsigned int bond_type = cur_bond.idx[1];
        int role = bpos_list[pitch * bond_idx + idx]; // 0 => this particle is A, 1 => B

        Scalar4 other_postype = d_pos[other_idx];
        vec3<Scalar> other_pos(other_postype.x, other_postype.y, other_postype.z);
        unsigned int other_type = __scalar_as_int(other_postype.w);
        quat<Scalar> q_other(d_orientation[other_idx]);

        // Orient the bond as (A, B): role 0 => this=A, other=B; role 1 => this=B, other=A.
        vec3<Scalar> posA, posB;
        quat<Scalar> qA, qB;
        unsigned int typeA, typeB;
        if (role == 0)
            {
            posA = idx_pos; posB = other_pos;
            qA = q_idx;     qB = q_other;
            typeA = idx_type; typeB = other_type;
            }
        else
            {
            posA = other_pos; posB = idx_pos;
            qA = q_other;     qB = q_idx;
            typeA = other_type; typeB = idx_type;
            }

        Scalar3 d3 = box.minImage(make_scalar3(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z));
        vec3<Scalar> Rc(d3.x, d3.y, d3.z);

        oxdna::BondedResult r;
        oxdna::oxdna_bonded_pair(d_params[bond_type], Rc, qA, qB, typeA, typeB, r);

        // This particle's own contribution.
        if (role == 0)
            {
            force -= r.FB; // force on A = -FB
            torque += r.tauA;
            }
        else
            {
            force += r.FB; // force on B = +FB
            torque += r.tauB;
            }
        energy += Scalar(0.5) * (r.e_fene + r.e_bexc + r.e_stack);
        split[0] += Scalar(0.5) * r.e_fene;
        split[1] += Scalar(0.5) * r.e_bexc;
        split[2] += Scalar(0.5) * r.e_stack;
        for (int v = 0; v < 6; v++)
            virial[v] += r.vir[v];
        }

    d_force[idx] = make_forcereal4(ForceReal(force.x), ForceReal(force.y), ForceReal(force.z),
                                   ForceReal(energy));
    d_torque[idx] = make_forcereal4(ForceReal(torque.x), ForceReal(torque.y), ForceReal(torque.z),
                                    ForceReal(0.0));
    for (int v = 0; v < 6; v++)
        d_virial[v * virial_pitch + idx] = ForceReal(virial[v]);
    d_split_energy[idx] = make_scalar4(split[0], split[1], split[2], Scalar(0.0));
    }

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
                                           int block_size)
    {
    if (N == 0)
        return hipSuccess;

    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_oxdna_bonded_forces_kernel);
    unsigned int run_block_size = min((unsigned int)block_size, (unsigned int)attr.maxThreadsPerBlock);
    if (run_block_size == 0)
        run_block_size = 1;

    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    hipLaunchKernelGGL((gpu_compute_oxdna_bonded_forces_kernel), grid, threads, 0, 0,
                       d_force, d_torque, d_virial, virial_pitch, d_split_energy, N,
                       d_pos, d_orientation, box, blist, bpos_list, pitch, n_bonds_list,
                       d_params);
    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
