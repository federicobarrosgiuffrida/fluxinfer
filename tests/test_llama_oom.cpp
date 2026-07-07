#include "Catch2/catch_amalgamated.hpp"

#include "fluxinfer/llama/llama_runner.hpp"

using namespace fluxinfer::llama;

TEST_CASE("text_looks_like_oom detects known CUDA/host OOM patterns", "[oom]") {
    CHECK(text_looks_like_oom("CUDA error: out of memory"));
    CHECK(text_looks_like_oom("ggml_gallocr_reserve: failed to allocate CUDA0 buffer of size 12345"));
    CHECK(text_looks_like_oom("terminate called after throwing an instance of 'std::bad_alloc'"));
    CHECK(text_looks_like_oom("Unable to allocate backend buffer for compute graph"));
    // Case-insensitive.
    CHECK(text_looks_like_oom("CUDA ERROR: OUT OF MEMORY"));
}

TEST_CASE("text_looks_like_oom does not flag normal benchmark output", "[oom]") {
    const std::string normal_output = R"([
  {"n_prompt": 512, "n_gen": 0, "avg_ts": 1234.5, "stddev_ts": 12.3},
  {"n_prompt": 0, "n_gen": 128, "avg_ts": 45.6, "stddev_ts": 1.2}
])";
    CHECK_FALSE(text_looks_like_oom(normal_output));
    CHECK_FALSE(text_looks_like_oom(""));
}

TEST_CASE("is_crash_exit_code classifies platform-appropriate crash exit codes", "[oom]") {
#if defined(_WIN32)
    // STATUS_ACCESS_VIOLATION (0xC0000005) reinterpreted as int32.
    CHECK(is_crash_exit_code(-1073741819));
    CHECK_FALSE(is_crash_exit_code(0));
    CHECK_FALSE(is_crash_exit_code(1));
#else
    // 128 + SIGSEGV(11) as encoded by run_captured() for signal-terminated processes.
    CHECK(is_crash_exit_code(128 + 11));
    CHECK(is_crash_exit_code(128 + 6)); // SIGABRT
    CHECK_FALSE(is_crash_exit_code(0));
    CHECK_FALSE(is_crash_exit_code(1));
    CHECK_FALSE(is_crash_exit_code(127)); // ordinary "command not found"-style exit code, not a signal
#endif
}
