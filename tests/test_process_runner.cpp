#include "Catch2/catch_amalgamated.hpp"

#include "self_path.hpp"

#include "fluxinfer/llama/benchmark_parser.hpp"
#include "fluxinfer/llama/llama_runner.hpp"
#include "fluxinfer/process/process_runner.hpp"

#include <filesystem>

using namespace fluxinfer::process;

namespace {

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    std::string current;
    for (char c : text) {
        if (c == '\n') {
            if (!current.empty() && current.back() == '\r') {
                current.pop_back();
            }
            lines.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        lines.push_back(current);
    }
    return lines;
}

} // namespace

TEST_CASE("run_captured preserves argv exactly, no shell interpretation", "[process]") {
    // Arguments deliberately include spaces, embedded quotes, a trailing
    // backslash before what would become a closing quote (the classic
    // Windows argv-quoting foot-gun), an empty string, and shell
    // metacharacters that must survive as literal text since no shell is
    // ever involved.
    ProcessOptions options;
    options.executable = g_fluxinfer_test_self_path;
    options.arguments = {
        "--fluxinfer-echo-argv",
        "hello world",
        "quote\"inside",
        "C:\\path with spaces\\",
        "",
        "; rm -rf / && echo pwned",
        "a|b>c<d",
    };
    options.timeout = std::chrono::seconds(15);

    ProcessResult result = run_captured(options);
    REQUIRE(result.outcome == ProcessOutcome::Exited);
    REQUIRE(result.start_error.empty());
    REQUIRE(result.exit_code == 0);

    const std::vector<std::string> lines = split_lines(result.stdout_data);
    const std::vector<std::string> expected(options.arguments.begin() + 1, options.arguments.end());
    REQUIRE(lines.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        CHECK(lines[i] == expected[i]);
    }
}

TEST_CASE("run_captured enforces its timeout and terminates the child", "[process]") {
    ProcessOptions options;
    options.executable = g_fluxinfer_test_self_path;
    options.arguments = {"--fluxinfer-sleep-ms", "5000"};
    options.timeout = std::chrono::milliseconds(300);

    const auto start = std::chrono::steady_clock::now();
    ProcessResult result = run_captured(options);
    const auto elapsed = std::chrono::steady_clock::now() - start;

    CHECK(result.outcome == ProcessOutcome::TimedOut);
    CHECK(elapsed < std::chrono::seconds(4));
}

TEST_CASE("run_captured passes extra_environment to the child", "[process]") {
    ProcessOptions options;
    options.executable = g_fluxinfer_test_self_path;
    options.arguments = {"--fluxinfer-print-env", "FLUXINFER_TEST_VAR"};
    options.extra_environment["FLUXINFER_TEST_VAR"] = "hello-from-parent";
    options.timeout = std::chrono::seconds(10);

    ProcessResult result = run_captured(options);
    REQUIRE(result.outcome == ProcessOutcome::Exited);
    REQUIRE(result.exit_code == 0);

    const std::vector<std::string> lines = split_lines(result.stdout_data);
    REQUIRE(lines.size() == 1);
    CHECK(lines[0] == "hello-from-parent");
}

TEST_CASE("run_captured honors working_directory", "[process]") {
    ProcessOptions options;
    options.executable = g_fluxinfer_test_self_path;
    options.arguments = {"--fluxinfer-print-cwd"};
    options.working_directory = std::filesystem::temp_directory_path();
    options.timeout = std::chrono::seconds(10);

    ProcessResult result = run_captured(options);
    REQUIRE(result.outcome == ProcessOutcome::Exited);

    const std::vector<std::string> lines = split_lines(result.stdout_data);
    REQUIRE(lines.size() == 1);

    std::error_code ec;
    const auto expected = std::filesystem::weakly_canonical(*options.working_directory, ec);
    const auto actual = std::filesystem::weakly_canonical(std::filesystem::path(lines[0]), ec);
    CHECK(actual == expected);
}

TEST_CASE("run_llama_binary classifies simulated OOM output", "[llama][process]") {
    fluxinfer::llama::LlamaRunResult result = fluxinfer::llama::run_llama_binary(
        g_fluxinfer_test_self_path, {"--fluxinfer-fake-llama", "oom"}, std::chrono::seconds(10));

    CHECK(result.likely_oom);
    CHECK_FALSE(result.crashed);
    CHECK_FALSE(result.timed_out);
    CHECK_FALSE(result.usable());
}

TEST_CASE("run_llama_binary + parse_llama_bench_output parse a simulated successful run", "[llama][process]") {
    fluxinfer::llama::LlamaRunResult result = fluxinfer::llama::run_llama_binary(
        g_fluxinfer_test_self_path, {"--fluxinfer-fake-llama", "ok"}, std::chrono::seconds(10));

    REQUIRE_FALSE(result.likely_oom);
    REQUIRE_FALSE(result.crashed);
    REQUIRE(result.exit_code == 0);

    fluxinfer::llama::BenchmarkParseResult parsed = fluxinfer::llama::parse_llama_bench_output(result.stdout_data);
    REQUIRE(parsed.valid);
    REQUIRE(parsed.prompt_tokens_per_second.has_value());
    REQUIRE(parsed.generation_tokens_per_second.has_value());
    CHECK(parsed.prompt_tokens_per_second.value() == Catch::Approx(1234.5));
    CHECK(parsed.generation_tokens_per_second.value() == Catch::Approx(45.6));
}
