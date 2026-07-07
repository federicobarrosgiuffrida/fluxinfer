cmake_minimum_required(VERSION 3.20)

# Statically link the MSVC runtime so a distributed fluxinfer.exe runs on a
# machine without the Visual C++ Redistributable installed -- important
# for handing out a prebuilt release binary rather than requiring everyone
# to build from source. Must be set before project() to take effect.
cmake_policy(SET CMP0091 NEW)

project(fluxinfer
    VERSION 0.3.1
    DESCRIPTION "Multiplatform autotuner/wrapper for llama.cpp"
    LANGUAGES CXX
)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(FLUXINFER_BUILD_TESTS "Build FluxInfer unit tests" ON)

# ---------------------------------------------------------------------------
# Vendored single-header dependencies (nlohmann/json, CLI11, Catch2).
# Kept as checked-in headers instead of FetchContent so builds work fully
# offline and are reproducible; see third_party/README-style comments below.
# ---------------------------------------------------------------------------
add_library(fluxinfer_nlohmann_json INTERFACE)
target_include_directories(fluxinfer_nlohmann_json INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

add_library(fluxinfer_cli11 INTERFACE)
target_include_directories(fluxinfer_cli11 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# ---------------------------------------------------------------------------
# Core library: hardware detection, process runner, llama.cpp integration,
# tuner and profile persistence. The CLI executable and the test binary both
# link against this so tests do not need to reimplement anything.
# ---------------------------------------------------------------------------
set(FLUXINFER_CORE_SOURCES
    src/hardware/cpu_probe.cpp
    src/hardware/memory_probe.cpp
    src/hardware/gpu_probe.cpp
    src/hardware/hardware_info.cpp
    src/hardware/vram_sampler.cpp
    src/process/process_runner.cpp
    src/llama/llama_locator.cpp
    src/llama/llama_runner.cpp
    src/llama/benchmark_parser.cpp
    src/llama/gguf_metadata.cpp
    src/tuner/parameter_space.cpp
    src/tuner/scoring.cpp
    src/tuner/tuner.cpp
    src/tuner/comparison.cpp
    src/profiles/profile.cpp
    src/profiles/profile_store.cpp
)

if(WIN32)
    list(APPEND FLUXINFER_CORE_SOURCES src/process/process_runner_win.cpp)
else()
    list(APPEND FLUXINFER_CORE_SOURCES src/process/process_runner_posix.cpp)
endif()

add_library(fluxinfer_core STATIC ${FLUXINFER_CORE_SOURCES})

target_include_directories(fluxinfer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fluxinfer_core PUBLIC
    fluxinfer_nlohmann_json
    fluxinfer_cli11
)

if(WIN32)
    target_link_libraries(fluxinfer_core PUBLIC pdh)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(fluxinfer_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
endif()

target_compile_features(fluxinfer_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(fluxinfer_core PRIVATE /W4 /permissive- /utf-8)
    target_compile_definitions(fluxinfer_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(fluxinfer_core PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# CLI executable
# ---------------------------------------------------------------------------
add_executable(fluxinfer src/main.cpp)
target_link_libraries(fluxinfer PRIVATE fluxinfer_core)

install(TARGETS fluxinfer RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(FLUXINFER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
