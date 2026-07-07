#pragma once

#include <string>

// Absolute path to the fluxinfer_tests binary itself, set from argv[0] in
// main() before Catch2 takes over. A handful of process-runner tests
// re-invoke this same binary with hidden "--fluxinfer-*" flags (handled in
// main.cpp before Catch2 parsing) to get a fully cross-platform, real
// child process to test against without depending on any external tool or
// a real llama.cpp build.
extern std::string g_fluxinfer_test_self_path;
