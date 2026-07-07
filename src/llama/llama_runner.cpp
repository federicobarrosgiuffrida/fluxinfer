#include "fluxinfer/llama/llama_runner.hpp"

#include <algorithm>
#include <cctype>
#include <regex>

namespace fluxinfer::llama {

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

const std::vector<std::string>& oom_patterns() {
    static const std::vector<std::string> patterns = {
        "out of memory",
        "cuda error",
        "cudamalloc failed",
        "cuda_error_out_of_memory",
        "failed to allocate",
        "insufficient memory",
        "std::bad_alloc",
        "ggml_gallocr_reserve",
        "not enough memory",
        "cannot allocate memory",
        "buffer type alloc buffer failed",
        "unable to allocate backend buffer",
    };
    return patterns;
}

} // namespace

std::set<std::string> extract_supported_flags(const std::string& help_text) {
    std::set<std::string> flags;

    // Matches "--flag-name" tokens anywhere in the help text; a following
    // "-x" short alias or "=VALUE"/" VALUE" is intentionally not part of
    // the match so we only capture the canonical long flag name.
    static const std::regex flag_pattern(R"(--[a-zA-Z][a-zA-Z0-9-]*)");

    auto begin = std::sregex_iterator(help_text.begin(), help_text.end(), flag_pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        flags.insert(it->str());
    }

    return flags;
}

std::set<std::string> detect_supported_flags(const std::filesystem::path& binary) {
    if (binary.empty()) {
        return {};
    }

    process::ProcessOptions options;
    options.executable = binary;
    options.arguments = {"--help"};
    options.timeout = std::chrono::seconds(10);

    process::ProcessResult result = process::run_captured(options);
    if (result.outcome == process::ProcessOutcome::FailedToStart) {
        return {};
    }

    std::set<std::string> flags = extract_supported_flags(result.stdout_data);
    std::set<std::string> stderr_flags = extract_supported_flags(result.stderr_data);
    flags.insert(stderr_flags.begin(), stderr_flags.end());
    return flags;
}

bool supports_flag(const std::set<std::string>& supported_flags, const std::string& flag) {
    return supported_flags.find(flag) != supported_flags.end();
}

std::string detect_llama_version(const std::filesystem::path& llama_bench) {
    if (llama_bench.empty()) {
        return "unknown";
    }

    process::ProcessOptions options;
    options.executable = llama_bench;
    options.arguments = {"--version"};
    options.timeout = std::chrono::seconds(5);
    process::ProcessResult result = process::run_captured(options);

    std::string combined = result.stdout_data + result.stderr_data;
    if (combined.empty()) {
        // --version is not supported by every llama.cpp build; fall back to
        // scanning --help output for a "build: <number> (<hash>)" banner.
        options.arguments = {"--help"};
        result = process::run_captured(options);
        combined = result.stdout_data + result.stderr_data;
    }

    // Different llama.cpp builds have printed this banner as either
    // "build: <number> (<hash>)" or "version: <number> (<hash>)" across
    // versions; both are accepted.
    static const std::regex build_pattern(R"((?:build|version)[:\s]+(\d+)\s*(?:\(([0-9a-fA-F]+)\))?)");
    std::smatch match;
    if (std::regex_search(combined, match, build_pattern)) {
        std::string version = "build " + match[1].str();
        if (match[2].matched) {
            version += " (" + match[2].str() + ")";
        }
        return version;
    }

    return "unknown";
}

LlamaRunResult run_llama_binary(const std::filesystem::path& binary,
                                 const std::vector<std::string>& arguments,
                                 std::chrono::milliseconds timeout) {
    process::ProcessOptions options;
    options.executable = binary;
    options.arguments = arguments;
    options.timeout = timeout;

    process::ProcessResult proc_result = process::run_captured(options);

    LlamaRunResult result;
    result.outcome = proc_result.outcome;
    result.exit_code = proc_result.exit_code;
    result.stdout_data = std::move(proc_result.stdout_data);
    result.stderr_data = std::move(proc_result.stderr_data);
    result.duration = proc_result.duration;
    result.start_error = proc_result.start_error;

    if (proc_result.outcome == process::ProcessOutcome::TimedOut) {
        result.timed_out = true;
        return result;
    }
    if (proc_result.outcome == process::ProcessOutcome::FailedToStart) {
        return result;
    }

    result.likely_oom = text_looks_like_oom(result.stdout_data + "\n" + result.stderr_data);

    if (!result.likely_oom && result.exit_code != 0 && is_crash_exit_code(result.exit_code)) {
        result.crashed = true;
    }

    return result;
}

bool text_looks_like_oom(const std::string& combined_output) {
    const std::string lower = to_lower(combined_output);
    for (const auto& pattern : oom_patterns()) {
        if (lower.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool is_crash_exit_code(int exit_code) {
#if defined(_WIN32)
    // Windows NTSTATUS exception codes (access violation, stack overflow,
    // ...) occupy the range [0xC0000000, 0xFFFFFFFF]; reinterpreted as a
    // signed 32-bit exit code that range is always negative.
    return exit_code < 0;
#else
    // WEXITSTATUS/WTERMSIG encoding used by run_captured(): 128 + signal
    // number for processes killed by a signal (SIGSEGV, SIGABRT, ...).
    return exit_code >= 128 && exit_code < 128 + 64;
#endif
}

} // namespace fluxinfer::llama
