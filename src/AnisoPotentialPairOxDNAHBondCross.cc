// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPair.h"
#include "EvaluatorPairOxDNAHBondCross.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

template void
export_AnisoPotentialPair<EvaluatorPairOxDNAHBondCross>(pybind11::module& m, const std::string& name);

void export_AnisoPotentialPairOxDNAHBondCross(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairOxDNAHBondCross>(m, "AnisoPotentialPairOxDNAHBondCross");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
