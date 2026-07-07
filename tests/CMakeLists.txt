# Catch2 is vendored as an amalgamated single header/source pair, built
# here as a small static library rather than via FetchContent so test
# builds work fully offline.
add_library(fluxinfer_catch2 STATIC ${CMAKE_SOURCE_DIR}/third_party/Catch2/catch_amalgamated.cpp)
target_include_directories(fluxinfer_catch2 PUBLIC ${CMAKE_SOURCE_DIR}/third_party)
target_compile_features(fluxinfer_catch2 PUBLIC cxx_std_17)
# tests/main.cpp provides its own main() (to intercept hidden test-helper
# CLI modes before Catch2 parses argv), so suppress the amalgamated file's
# built-in main to avoid a duplicate-symbol link error.
target_compile_definitions(fluxinfer_catch2 PRIVATE CATCH_AMALGAMATED_CUSTOM_MAIN)

add_executable(fluxinfer_tests
    main.cpp
    test_process_runner.cpp
    test_llama_flags.cpp
    test_llama_oom.cpp
    test_benchmark_parser.cpp
    test_scoring.cpp
    test_parameter_space.cpp
    test_profile.cpp
    test_gguf_metadata.cpp
    test_comparison_stats.cpp
)

target_link_libraries(fluxinfer_tests PRIVATE fluxinfer_core fluxinfer_catch2)
target_include_directories(fluxinfer_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(fluxinfer_tests PRIVATE /utf-8)
endif()

include(CTest)
add_test(NAME fluxinfer_tests COMMAND fluxinfer_tests)
