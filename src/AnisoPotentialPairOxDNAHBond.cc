// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPair.h"
#include "EvaluatorPairOxDNAHBond.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

template void
export_AnisoPotentialPair<EvaluatorPairOxDNAHBond>(pybind11::module& m, const std::string& name);

void export_AnisoPotentialPairOxDNAHBond(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairOxDNAHBond>(m, "AnisoPotentialPairOxDNAHBond");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
