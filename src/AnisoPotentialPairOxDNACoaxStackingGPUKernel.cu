// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.cuh"
#include "EvaluatorPairOxDNACoaxStacking.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairOxDNACoaxStacking>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairOxDNACoaxStacking::param_type* d_param,
    const EvaluatorPairOxDNACoaxStacking::shape_type* d_shape_param);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
