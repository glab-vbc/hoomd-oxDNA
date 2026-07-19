// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "EvaluatorPairOxDNAHBondCross.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AnisoPotentialPairOxDNAHBondCrossGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairOxDNAHBondCross>(m, "AnisoPotentialPairOxDNAHBondCrossGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
