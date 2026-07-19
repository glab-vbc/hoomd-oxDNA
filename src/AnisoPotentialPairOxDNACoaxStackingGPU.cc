// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "EvaluatorPairOxDNACoaxStacking.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AnisoPotentialPairOxDNACoaxStackingGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairOxDNACoaxStacking>(m, "AnisoPotentialPairOxDNACoaxStackingGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
