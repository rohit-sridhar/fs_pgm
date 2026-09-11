cmake_minimum_required(VERSION 3.15...3.27)

# FIX: Explicitly set the project name and designate CXX (C++) as the language
project(pgm_project LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Dynamically extract the nanobind path using Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir(), end='')"
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")

# Nanobind and Eigen
find_package(Eigen3 REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Create core C++ libraries
add_library(distributions STATIC src/distributions.cpp)
add_library(utils STATIC src/utils.cpp)

# C++ libraries linked into Python extensions must use position-independent code
set_target_properties(distributions PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Expose the include path to anyone linking to this core library
target_include_directories(distributions PUBLIC src/pgm)
target_include_directories(utils PUBLIC src/pgm)

# nanobind modules to expose them to python interface
nanobind_add_module(hmm src/hmm.cpp)

# link Eigen library
target_link_libraries(distributions PRIVATE Eigen3::Eigen)
target_link_libraries(utils PRIVATE Eigen3::Eigen)
target_link_libraries(hmm PRIVATE Eigen3::Eigen)

# link distributions and utils to hmm (main module)
target_link_libraries(hmm PRIVATE distributions)
target_link_libraries(hmm PRIVATE utils)

