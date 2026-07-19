// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include "hoomd/md/AnisoPotentialPair.h"
#include "EvaluatorPairOxDNADebye.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {

template void export_AnisoPotentialPair<EvaluatorPairOxDNADebye>(pybind11::module& m,
                                                                 const std::string& name);

void export_AnisoPotentialPairOxDNADebye(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairOxDNADebye>(m, "AnisoPotentialPairOxDNADebye");
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
