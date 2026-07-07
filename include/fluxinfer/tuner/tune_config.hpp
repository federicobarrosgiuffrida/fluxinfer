#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fluxinfer::tuner {

struct TuneConfig {
    unsigned threads = 1;
    int gpu_layers = 0;
    unsigned batch_size = 512;
    unsigned ubatch_size = 256;

    // Only set when the located llama.cpp build supports the relevant
    // --cache-type-k/--cache-type-v flags (checked via --help).
    std::optional<std::string> kv_cache_type;

    // Additional flags detected as supported (e.g. future MoE-related
    // options) that the tuner wants to try. Stored as raw "--flag"/"value"
    // pairs so new flags don't require code changes; value may be empty for
    // boolean flags.
    std::vector<std::pair<std::string, std::string>> extra_flags;

    std::string label; // human-readable identifier for logs/progress output
};

} // namespace fluxinfer::tuner
