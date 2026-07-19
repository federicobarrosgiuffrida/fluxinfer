#pragma once

#include "fluxinfer/process/process_runner.hpp"

#include <chrono>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace fluxinfer::llama {

// Pure text-processing step of flag detection, split out from
// detect_supported_flags() below so it can be unit-tested against fixture
// --help text without spawning a real llama.cpp binary. Extracts every
// "--flag-name" token found in `help_text`.
std::set<std::string> extract_supported_flags(const std::string& help_text);

// Runs `binary --help` and extracts every long option token it prints
// (e.g. "--n-gpu-layers", "--ctk", "-b, --batch-size"), independent of the
// exact help text formatting used by a given llama.cpp version. Returns an
// empty set (never throws) if the binary can't be executed.
std::set<std::string> detect_supported_flags(const std::filesystem::path& binary);

// True if `flag` (e.g. "--flash-attn") is present in a flag set returned by
// detect_supported_flags().
bool supports_flag(const std::set<std::string>& supported_flags, const std::string& flag);

// Merges FluxInfer's profile-derived argv with raw user arguments passed
// after `--` on the command line.
//
// `profile_args` is what build_config_arguments() produces: a sequence of
// [flag, value] pairs. Any flag the user also specifies is dropped from the
// profile side, so the user's value ends up alone on the final command line
// instead of being appended after FluxInfer's. llama.cpp does resolve such
// duplicates by "last one wins", but it prints a DEPRECATED warning and the
// effective configuration becomes hard to read back from the launch log.
//
// Both spellings of an option are treated as the same flag (a user passing
// `-ngl 99` also overrides a profile's `--n-gpu-layers`), as is `--flag=value`
// syntax.
//
// If `overridden_out` is non-null, it receives the profile flag tokens that
// were dropped, in the order they appeared, so callers can report them.
std::vector<std::string> merge_user_overrides(const std::vector<std::string>& profile_args,
                                               const std::vector<std::string>& user_args,
                                               std::vector<std::string>* overridden_out = nullptr);

// Best-effort extraction of a version/build string from `llama-bench
// --version` or, failing that, `--help` output. Returns "unknown" if
// nothing could be determined.
std::string detect_llama_version(const std::filesystem::path& llama_bench);

struct LlamaRunResult {
    process::ProcessOutcome outcome = process::ProcessOutcome::FailedToStart;
    int exit_code = -1;
    std::string stdout_data;
    std::string stderr_data;
    std::chrono::milliseconds duration{0};
    std::string start_error;

    // Heuristic classification, derived from exit code shape and known
    // error message substrings (CUDA OOM, bad_alloc, allocation failures).
    bool likely_oom = false;
    bool crashed = false;
    bool timed_out = false;

    bool usable() const { return outcome == process::ProcessOutcome::Exited && !likely_oom && !crashed && exit_code == 0; }
};

// Invokes `binary` with `arguments` (argv, never shell-interpreted),
// capturing output and classifying OOM / crash / timeout conditions.
LlamaRunResult run_llama_binary(const std::filesystem::path& binary,
                                 const std::vector<std::string>& arguments,
                                 std::chrono::milliseconds timeout);

// True if `combined_output` (stdout+stderr) contains a substring typical of
// a CUDA/host out-of-memory failure. Exposed for unit testing; used
// internally by run_llama_binary().
bool text_looks_like_oom(const std::string& combined_output);

// True if `exit_code` (as produced by process::run_captured) is shaped like
// a crash rather than a normal nonzero exit: on Windows, an NTSTATUS
// exception code (e.g. access violation) reinterpreted as a negative
// int32; on POSIX, 128+signal as encoded by run_captured for a process
// killed by a signal. Exposed for unit testing.
bool is_crash_exit_code(int exit_code);

} // namespace fluxinfer::llama
