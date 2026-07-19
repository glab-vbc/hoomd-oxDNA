// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPairGPU.h"
#include "EvaluatorPairOxDNAHBond.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

void export_AnisoPotentialPairOxDNAHBondGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairOxDNAHBond>(m, "AnisoPotentialPairOxDNAHBondGPU");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
