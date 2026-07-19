# Register the oxDNA Python validation suite as a ctest target.
#
# The tests must run with the Python interpreter HOOMD is installed into (that env
# has both HOOMD and, after `cmake --install`, this plugin's _engine module beside
# the oxdna/ package). The repo root goes on PYTHONPATH so `import oxdna` resolves to
# the source package with the co-located compiled module.
#
# Usage (from a configured, built, and INSTALLED tree):
#     cmake --build build && cmake --install build
#     ctest --test-dir build --output-on-failure

set(OXDNA_TEST_PYTHON "${HOOMD_INSTALL_PREFIX}/bin/python" CACHE FILEPATH
    "Python interpreter used to run the oxDNA test suite (must have HOOMD installed)")

add_test(
    NAME oxdna-pytest
    COMMAND ${OXDNA_TEST_PYTHON} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -q
)

set_tests_properties(oxdna-pytest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}"
    # The physics validation runs a few hundred single-point/short simulations.
    TIMEOUT 600
)
