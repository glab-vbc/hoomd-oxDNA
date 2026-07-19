// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.cuh"
#include "EvaluatorPairOxDNAExcVol.h"

namespace hoomd
    {
namespace md
    {
namespace kernel
    {

template hipError_t __attribute__((visibility("default")))
gpu_compute_pair_aniso_forces<EvaluatorPairOxDNAExcVol>(
    const a_pair_args_t& pair_args,
    const EvaluatorPairOxDNAExcVol::param_type* d_param,
    const EvaluatorPairOxDNAExcVol::shape_type* d_shape_param);

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
