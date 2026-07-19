// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "EvaluatorPairOxDNAExcVol.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AnisoPotentialPairOxDNAExcVolGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairOxDNAExcVol>(m, "AnisoPotentialPairOxDNAExcVolGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
