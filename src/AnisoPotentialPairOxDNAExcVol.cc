// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPair.h"
#include "EvaluatorPairOxDNAExcVol.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

template void
export_AnisoPotentialPair<EvaluatorPairOxDNAExcVol>(pybind11::module& m, const std::string& name);

void export_AnisoPotentialPairOxDNAExcVol(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairOxDNAExcVol>(m, "AnisoPotentialPairOxDNAExcVol");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
