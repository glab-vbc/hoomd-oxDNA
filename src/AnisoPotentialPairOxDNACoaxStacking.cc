// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPair.h"
#include "EvaluatorPairOxDNACoaxStacking.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

template void export_AnisoPotentialPair<EvaluatorPairOxDNACoaxStacking>(pybind11::module& m,
                                                                        const std::string& name);

void export_AnisoPotentialPairOxDNACoaxStacking(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairOxDNACoaxStacking>(
        m, "AnisoPotentialPairOxDNACoaxStacking");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
