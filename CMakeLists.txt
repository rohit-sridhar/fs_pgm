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

find_package(nanobind CONFIG REQUIRED)
nanobind_add_module(hmm src/hmm.cpp)

