// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "EvaluatorPairOxDNACrossStacking.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AnisoPotentialPairOxDNACrossStackingGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairOxDNACrossStacking>(m, "AnisoPotentialPairOxDNACrossStackingGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
