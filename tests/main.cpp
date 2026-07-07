// Custom Catch2 entrypoint. Before handing off to Catch2's test runner, this
// checks for a handful of hidden "--fluxinfer-*" modes that turn this same
// binary into a tiny stand-in for llama-bench/llama-cli, used by the
// process-runner and OOM-detection tests to spawn a real child process
// without requiring an actual llama.cpp build in the test environment.
#define CATCH_CONFIG_RUNNER
#include "Catch2/catch_amalgamated.hpp"

#include "self_path.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

std::string g_fluxinfer_test_self_path;

namespace {

// Returns the hidden-mode exit code, or -1 if argv doesn't request one.
int run_hidden_mode(int argc, char** argv) {
    const std::string mode = argv[1];

    if (mode == "--fluxinfer-echo-argv") {
        for (int i = 2; i < argc; ++i) {
            std::cout << argv[i] << "\n";
        }
        return 0;
    }

    if (mode == "--fluxinfer-print-cwd") {
        std::cout << std::filesystem::current_path().string() << "\n";
        return 0;
    }

    if (mode == "--fluxinfer-print-env") {
        if (argc < 3) return 2;
        const char* value = std::getenv(argv[2]);
        std::cout << (value ? value : "<unset>") << "\n";
        return 0;
    }

    if (mode == "--fluxinfer-sleep-ms") {
        if (argc < 3) return 2;
        std::this_thread::sleep_for(std::chrono::milliseconds(std::stoi(argv[2])));
        return 0;
    }

    if (mode == "--fluxinfer-fake-llama") {
        if (argc < 3) return 2;
        const std::string kind = argv[2];
        if (kind == "oom") {
            std::cerr << "ggml_gallocr_reserve: failed to allocate CUDA0 buffer of size 123456789\n";
            std::cerr << "CUDA error: out of memory\n";
            return 1;
        }
        if (kind == "ok") {
            std::cout << R"([
  {"n_prompt": 512, "n_gen": 0, "avg_ts": 1234.5, "stddev_ts": 12.3},
  {"n_prompt": 0, "n_gen": 128, "avg_ts": 45.6, "stddev_ts": 1.2}
])";
            return 0;
        }
        return 2;
    }

    return -1;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        const int hidden_result = run_hidden_mode(argc, argv);
        if (hidden_result != -1) {
            return hidden_result;
        }
    }

    std::error_code ec;
    std::filesystem::path exe_path = std::filesystem::absolute(argv[0], ec);
    if (!ec) {
        exe_path = std::filesystem::weakly_canonical(exe_path, ec);
    }
    g_fluxinfer_test_self_path = ec ? std::string(argv[0]) : exe_path.string();

    return Catch::Session().run(argc, argv);
}
