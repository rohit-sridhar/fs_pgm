cmake_minimum_required(VERSION 3.15...3.27)

# FIX: Explicitly set the project name and designate CXX (C++) as the language
project(pgm_project LANGUAGES CXX)

# Set C++ Standard (Required by Eigen and nanobind)
set(CMAKE_CXX_STANDARD 17)
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
add_library(hmm_core STATIC src/cpp/hmm.cpp)
add_library(distributions_core STATIC src/cpp/distributions.cpp)
add_library(data_loader_core STATIC src/cpp/data_loader.cpp)
# utils_core is compiled as shared, so some utilities can share globals
add_library(utils_core STATIC src/cpp/utils.cpp)

# C++ libraries linked into Python extensions must use position-independent code
set_target_properties(hmm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(distributions_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(data_loader_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(utils_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Expose the include path to anyone linking to this core library
target_include_directories(hmm_core PUBLIC include)
target_include_directories(distributions_core PUBLIC include)
target_include_directories(data_loader_core PUBLIC include)
target_include_directories(utils_core PUBLIC include)

# nanobind modules to expose them to python interface
nanobind_add_module(
    pgm
    src/cpp/hmm.cpp
    src/cpp/data_loader.cpp
    src/cpp/utils.cpp
    src/cpp/bindings.cpp
)

# file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/src/__init__.py"
#      DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# nanobind_add_module(
#     hmm
#     src/cpp/hmm.cpp
#     src/cpp/hmm_bindings.cpp
# )
# nanobind_add_module(
#     data_loader
#     src/cpp/data_loader.cpp
#     src/cpp/data_loader_bindings.cpp
# )
# nanobind_add_module(
#     utils
#     src/cpp/utils.cpp
#     src/cpp/utils_bindings.cpp
# )

# add include path for data loader
target_include_directories(pgm PUBLIC include)
# target_include_directories(hmm PUBLIC include)
# target_include_directories(data_loader PUBLIC include)
# target_include_directories(utils PUBLIC include)

# link Eigen library
target_link_libraries(pgm PRIVATE Eigen3::Eigen)
# target_link_libraries(hmm PRIVATE Eigen3::Eigen)
# target_link_libraries(data_loader PRIVATE Eigen3::Eigen)
# target_link_libraries(utils PRIVATE Eigen3::Eigen)
target_link_libraries(hmm_core PRIVATE Eigen3::Eigen)
target_link_libraries(distributions_core PRIVATE Eigen3::Eigen)
target_link_libraries(data_loader_core PRIVATE Eigen3::Eigen)
target_link_libraries(utils_core PRIVATE Eigen3::Eigen)

# link distributions_core and utils_core to hmm (nb_module) and hmm_core (main module)
target_link_libraries(pgm PRIVATE distributions_core)
target_link_libraries(pgm PRIVATE data_loader_core)
target_link_libraries(pgm PRIVATE utils_core)
# target_link_libraries(hmm PRIVATE distributions_core)
# target_link_libraries(hmm PRIVATE data_loader_core)
# target_link_libraries(hmm PRIVATE utils)
target_link_libraries(hmm_core PRIVATE distributions_core)
target_link_libraries(hmm_core PRIVATE data_loader_core)
target_link_libraries(hmm_core PRIVATE utils_core)

