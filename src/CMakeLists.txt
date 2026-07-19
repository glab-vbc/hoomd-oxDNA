set(COMPONENT_NAME engine)

# The pure-Python package lives in the repo at ../oxdna ; install the compiled
# module next to it so `import oxdna` (with the repo on PYTHONPATH) finds it.
get_filename_component(OXDNA_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../oxdna" ABSOLUTE)

set(_${COMPONENT_NAME}_sources
    module.cc
    OxDNABondedForceCompute.cc
    AnisoPotentialPairOxDNAExcVol.cc
    AnisoPotentialPairOxDNAHBond.cc
    AnisoPotentialPairOxDNACrossStacking.cc
    AnisoPotentialPairOxDNAHBondCross.cc
    AnisoPotentialPairOxDNACoaxStacking.cc
    AnisoPotentialPairOxDNADebye.cc
    )

set(_${COMPONENT_NAME}_headers
    MixedPrecisionCompat.h
    OxDNANumerics.h
    OxDNAParamIO.h
    OxDNAAnisoPair.h
    OxDNABaseBase.h
    OxDNABondedParams.h
    OxDNABondedKernel.h
    OxDNABondedForceCompute.h
    EvaluatorPairOxDNAExcVol.h
    EvaluatorPairOxDNAHBond.h
    EvaluatorPairOxDNACrossStacking.h
    EvaluatorPairOxDNAHBondCross.h
    EvaluatorPairOxDNACoaxStacking.h
    EvaluatorPairOxDNADebye.h
    )

# GPU support: one registration (.cc) + one kernel template instantiation (.cu)
# per anisotropic evaluator; the bonded force adds a hand-written kernel. Built only
# when HOOMD was compiled with GPU support (ENABLE_HIP). The Python AnisotropicPair
# base dispatches to the "<name>GPU" class automatically on a GPU device.
if(ENABLE_HIP)
    if(NOT HOOMD_DEVICE_LANGUAGE)
        set(HOOMD_DEVICE_LANGUAGE CUDA)
    endif()
    list(APPEND _${COMPONENT_NAME}_sources
         OxDNABondedForceComputeGPU.cc
         AnisoPotentialPairOxDNAExcVolGPU.cc
         AnisoPotentialPairOxDNAHBondGPU.cc
         AnisoPotentialPairOxDNACrossStackingGPU.cc
         AnisoPotentialPairOxDNAHBondCrossGPU.cc
         AnisoPotentialPairOxDNACoaxStackingGPU.cc
         AnisoPotentialPairOxDNADebyeGPU.cc
         )
    list(APPEND _${COMPONENT_NAME}_headers
         OxDNABondedForceComputeGPU.h
         OxDNABondedGPU.cuh
         )
    set(_${COMPONENT_NAME}_cu_sources
        OxDNABondedGPU.cu
        AnisoPotentialPairOxDNAExcVolGPUKernel.cu
        AnisoPotentialPairOxDNAHBondGPUKernel.cu
        AnisoPotentialPairOxDNACrossStackingGPUKernel.cu
        AnisoPotentialPairOxDNAHBondCrossGPUKernel.cu
        AnisoPotentialPairOxDNACoaxStackingGPUKernel.cu
        AnisoPotentialPairOxDNADebyeGPUKernel.cu
        )
    # NB: --use_fast_math was measured ~20% SLOWER here. The hot transcendental is
    # acos (every f4 angular factor) which has no fast intrinsic, so fast-math cannot
    # help the bottleneck while its codegen changes (approx div/sqrt, ftz) cost
    # occupancy. The real acos lever is computing each angle once (see angular()).
    set_source_files_properties(${_${COMPONENT_NAME}_cu_sources}
        PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
endif()

hoomd_add_module(_${COMPONENT_NAME} SHARED
    ${_${COMPONENT_NAME}_sources}
    ${_${COMPONENT_NAME}_cu_sources}
    ${_${COMPONENT_NAME}_headers}
    NO_EXTRAS)

target_link_libraries(_${COMPONENT_NAME} PUBLIC HOOMD::_hoomd HOOMD::_md)

# ---------------------------------------------------------------------------
# ForceReal compatibility (see MixedPrecisionCompat.h). Same probe glab-plugins
# uses: some HOOMD builds expose a native ForceReal type without defining the
# HOOMD_HAS_FORCEREAL guard macro; detect that from the installed HOOMDMath.h.
# ---------------------------------------------------------------------------
get_target_property(_hoomd_inc_dirs HOOMD::_hoomd INTERFACE_INCLUDE_DIRECTORIES)
find_file(HOOMD_MATH_HEADER
    NAMES hoomd/HOOMDMath.h
    PATHS ${_hoomd_inc_dirs}
    NO_DEFAULT_PATH)
if (HOOMD_MATH_HEADER)
    file(READ "${HOOMD_MATH_HEADER}" _hoomd_math_src)
    string(FIND "${_hoomd_math_src}" "ForceReal" _fr_type_pos)
    string(FIND "${_hoomd_math_src}" "define HOOMD_HAS_FORCEREAL" _fr_macro_pos)
    if (NOT _fr_type_pos EQUAL -1 AND _fr_macro_pos EQUAL -1)
        message(STATUS "oxdna_engine: HOOMD provides native ForceReal without HOOMD_HAS_FORCEREAL; defining it")
        target_compile_definitions(_${COMPONENT_NAME} PRIVATE HOOMD_HAS_FORCEREAL)
    endif()
    unset(_hoomd_math_src)
else()
    message(WARNING "oxdna_engine: could not locate hoomd/HOOMDMath.h to probe for native ForceReal")
endif()

# Install the compiled module next to the pure-Python package.
install(TARGETS _${COMPONENT_NAME}
        LIBRARY DESTINATION ${OXDNA_PKG_DIR})
