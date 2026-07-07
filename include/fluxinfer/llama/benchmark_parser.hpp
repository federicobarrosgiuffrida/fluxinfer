#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fluxinfer::llama {

// One row of llama-bench output (one "test", e.g. a prompt-processing pass
// or a generation pass).
struct BenchmarkMeasurement {
    std::uint64_t n_prompt = 0;
    std::uint64_t n_gen = 0;
    std::string test_name;                 // e.g. "pp512", "tg128" (markdown mode only)
    double avg_tokens_per_second = 0.0;
    double stddev_tokens_per_second = 0.0;

    // Per-repetition tok/s samples (llama-bench's own "samples_ts" JSON
    // field, present when run with `-r N` for N > 1). Empty in markdown
    // mode or when llama-bench only ran one repetition -- callers that
    // want real per-sample statistics (mean/median/min/max/stddev) rather
    // than trusting llama-bench's own avg/stddev should check this first.
    std::vector<double> samples_tokens_per_second;
};

struct BenchmarkParseResult {
    bool valid = false;
    std::optional<double> prompt_tokens_per_second;     // from the pp (n_prompt>0) row
    std::optional<double> generation_tokens_per_second; // from the tg (n_gen>0) row
    std::vector<double> prompt_tps_samples;             // samples_tokens_per_second of the pp row, if any
    std::vector<double> generation_tps_samples;         // samples_tokens_per_second of the tg row, if any
    std::vector<BenchmarkMeasurement> measurements;
    std::string parse_error; // populated when valid == false
};

// Parses the output of `llama-bench ... -o json`. This is the primary,
// preferred path since it does not depend on any particular column layout.
BenchmarkParseResult parse_llama_bench_json(const std::string& stdout_text);

// Parses llama-bench's default GitHub-flavored markdown table output, for
// llama.cpp builds old enough not to support `-o json` (detected via
// detect_supported_flags()). Column order is discovered from the header row
// rather than assumed fixed, since llama-bench adds extra columns when
// benchmark parameters vary across rows.
BenchmarkParseResult parse_llama_bench_markdown(const std::string& stdout_text);

// Tries JSON first, falls back to markdown; convenience for callers that
// don't want to track which format they requested.
BenchmarkParseResult parse_llama_bench_output(const std::string& stdout_text);

} // namespace fluxinfer::llama
