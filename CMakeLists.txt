cmake_minimum_required(VERSION 3.15...3.27)

# FIX: Explicitly set the project name and designate CXX (C++) as the language
project(pgm_project LANGUAGES CXX)

# Set C++ Standard (Required by Eigen and nanobind)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_COMPILER g++-14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# find Python package
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
add_library(hmm_core STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/hmm.cpp)
add_library(trainer_core STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/trainer.cpp)
add_library(distributions_core STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/distributions.cpp)
add_library(data_loader_core STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/data_loader.cpp)
# utils_core is compiled as shared, so some utilities can share globals
add_library(utils_core STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/utils.cpp)

# C++ libraries linked into Python extensions must use position-independent code
set_target_properties(hmm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(trainer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(distributions_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(data_loader_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(utils_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Expose the include path to anyone linking to this core library
target_include_directories(hmm_core PUBLIC include)
target_include_directories(trainer_core PUBLIC include)
target_include_directories(distributions_core PUBLIC include)
target_include_directories(data_loader_core PUBLIC include)
target_include_directories(utils_core PUBLIC include)

# nanobind modules to expose them to python interface
nanobind_add_module(
    _pgm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/hmm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/trainer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/data_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/bindings.cpp
)

# print variables (comment out if not needed)
message(STATUS "Source directory is: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Binary (Build) directory is: ${CMAKE_CURRENT_BINARY_DIR}")

# 2. Copy python import structure files into cpp module build dir
add_custom_command(TARGET _pgm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/src/python/."  # <--- Note the "/." at the end
        "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Copying Python file import tree into the build package..."
)

# add include path for data loader
target_include_directories(_pgm PUBLIC include)

# link Eigen library
target_link_libraries(_pgm PRIVATE Eigen3::Eigen)
target_link_libraries(hmm_core PRIVATE Eigen3::Eigen)
target_link_libraries(trainer_core PRIVATE Eigen3::Eigen)
target_link_libraries(distributions_core PRIVATE Eigen3::Eigen)
target_link_libraries(data_loader_core PRIVATE Eigen3::Eigen)
target_link_libraries(utils_core PRIVATE Eigen3::Eigen)

# link distributions_core and utils_core to pgm (nb_module) and hmm_core (main module)
target_link_libraries(_pgm PRIVATE distributions_core)
target_link_libraries(_pgm PRIVATE data_loader_core)
target_link_libraries(_pgm PRIVATE utils_core)
# target_link_libraries(hmm_core PRIVATE distributions_core)
# target_link_libraries(hmm_core PRIVATE trainer_core)
# target_link_libraries(hmm_core PRIVATE data_loader_core)
# target_link_libraries(hmm_core PRIVATE utils_core)

