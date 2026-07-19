cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(oxdna_engine LANGUAGES CXX)

# oxDNA-in-HOOMD plugin.
#   Layer 1 (this build): the compiled C++ force engine, installed as the
#                         `_engine` extension module inside the `oxdna/` package.
#   Layer 2 (pure Python): the `oxdna/` package (io, params, model/, forces) is
#                          committed alongside; the compiled module is installed
#                          next to it so `import oxdna` finds both.

find_package(HOOMD REQUIRED)
message(STATUS "Found HOOMD: ${HOOMD_DIR}")

add_subdirectory(src)

# Register the Python validation suite as a `ctest` target.
enable_testing()
add_subdirectory(pytest)
