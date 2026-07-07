#pragma once

#include "fluxinfer/hardware/hardware_info.hpp"
#include "fluxinfer/tuner/tune_config.hpp"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace fluxinfer::tuner {

struct ParameterSpaceInput {
    hardware::HardwareInfo hardware;
    std::set<std::string> supported_flags; // from `llama-bench --help`
    std::uint64_t model_size_bytes = 0;

    // The model's real transformer layer count (GGUF
    // "<architecture>.block_count"), obtained via
    // fluxinfer::llama::parse_gguf_metadata(). std::nullopt means the
    // metadata could not be read/found: per policy, FluxInfer never
    // invents a layer count in that case, so the GPU-layers search stage
    // (see gpu_layers_candidates()) is skipped entirely rather than
    // guessing from file size.
    std::optional<std::uint64_t> real_layer_count;
};

// Stage 1: a single conservative, GPU-free baseline configuration.
TuneConfig baseline_config(const ParameterSpaceInput& input);

// Stage 2: candidates at ~0/25/50/75/100% of the model's real layer count
// (deduplicated), always including 0 (no offload) and the full layer count
// (full offload) at the extremes. Empty if there is no GPU or
// real_layer_count is unavailable -- this stage is never run on a guess.
std::vector<TuneConfig> gpu_layers_candidates(const ParameterSpaceInput& input, const TuneConfig& base);

// Stage 3: a handful of (batch, ubatch) pairs, trimmed based on available
// RAM so obviously oversized combinations aren't attempted on small
// machines.
std::vector<TuneConfig> batch_ubatch_candidates(const ParameterSpaceInput& input, const TuneConfig& base);

// Stage 4: physical core count, logical thread count, and their midpoint
// (deduplicated).
std::vector<TuneConfig> thread_candidates(const ParameterSpaceInput& input, const TuneConfig& base);

} // namespace fluxinfer::tuner
