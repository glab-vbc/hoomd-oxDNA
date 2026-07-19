// Copyright (c) 2026 Goloborodko Lab.
// Released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_OxDNABondedForceCompute(pybind11::module& m);
void export_AnisoPotentialPairOxDNAExcVol(pybind11::module& m);
void export_AnisoPotentialPairOxDNAHBond(pybind11::module& m);
void export_AnisoPotentialPairOxDNACrossStacking(pybind11::module& m);
void export_AnisoPotentialPairOxDNAHBondCross(pybind11::module& m);
void export_AnisoPotentialPairOxDNACoaxStacking(pybind11::module& m);
void export_AnisoPotentialPairOxDNADebye(pybind11::module& m);
#ifdef ENABLE_HIP
void export_OxDNABondedForceComputeGPU(pybind11::module& m);
void export_AnisoPotentialPairOxDNAExcVolGPU(pybind11::module& m);
void export_AnisoPotentialPairOxDNAHBondGPU(pybind11::module& m);
void export_AnisoPotentialPairOxDNACrossStackingGPU(pybind11::module& m);
void export_AnisoPotentialPairOxDNAHBondCrossGPU(pybind11::module& m);
void export_AnisoPotentialPairOxDNACoaxStackingGPU(pybind11::module& m);
void export_AnisoPotentialPairOxDNADebyeGPU(pybind11::module& m);
#endif
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

using namespace hoomd::md::detail;

PYBIND11_MODULE(_engine, m)
    {
    export_OxDNABondedForceCompute(m);
    export_AnisoPotentialPairOxDNAExcVol(m);
    export_AnisoPotentialPairOxDNAHBond(m);
    export_AnisoPotentialPairOxDNACrossStacking(m);
    export_AnisoPotentialPairOxDNAHBondCross(m);
    export_AnisoPotentialPairOxDNACoaxStacking(m);
    export_AnisoPotentialPairOxDNADebye(m);
#ifdef ENABLE_HIP
    export_OxDNABondedForceComputeGPU(m);
    export_AnisoPotentialPairOxDNAExcVolGPU(m);
    export_AnisoPotentialPairOxDNAHBondGPU(m);
    export_AnisoPotentialPairOxDNACrossStackingGPU(m);
    export_AnisoPotentialPairOxDNAHBondCrossGPU(m);
    export_AnisoPotentialPairOxDNACoaxStackingGPU(m);
    export_AnisoPotentialPairOxDNADebyeGPU(m);
#endif
    }
