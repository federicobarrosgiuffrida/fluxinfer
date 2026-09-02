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

    // Number of experts declared by the model's own GGUF metadata
    // ("<architecture>.expert_count"), read by
    // fluxinfer::llama::parse_gguf_metadata(). Anything greater than 1
    // means this is a mixture-of-experts model and the offload search has
    // to work differently -- see n_cpu_moe_candidates(). std::nullopt or
    // <= 1 means "treat as a dense model", never a guess.
    std::optional<std::uint64_t> expert_count;

    // VRAM deliberately left unused by the offload estimate below, on top
    // of what the KV cache and activations need. Two reasons it is not
    // zero: other processes (compositor, browser, chat clients) hold a
    // moving amount of VRAM, and on Windows/WDDM filling the last few
    // hundred MB does not fail cleanly -- the driver starts backing
    // allocations with system RAM and throughput collapses silently
    // instead of reporting OOM. 0 means "use default_vram_headroom_bytes()".
    std::uint64_t vram_headroom_bytes = 0;
};

// Platform default for ParameterSpaceInput::vram_headroom_bytes: larger on
// Windows, where the silent WDDM spill described above makes running near
// the VRAM limit actively harmful rather than merely risky.
std::uint64_t default_vram_headroom_bytes();

// Stage 1: a single conservative, GPU-free baseline configuration.
TuneConfig baseline_config(const ParameterSpaceInput& input);

// Stage 2: candidates at ~0/25/50/75/100% of the model's real layer count
// (deduplicated), always including 0 (no offload) and the full layer count
// (full offload) at the extremes, plus -- when the model size and VRAM are
// both known -- one extra candidate seeded from how many layers actually fit
// in (available VRAM - headroom). That seed only decides where to *look*
// first; as everywhere else in the tuner, what is usable is still decided by
// running the benchmark and observing real OOM/timeouts, never by the
// estimate. Empty if there is no GPU or real_layer_count is unavailable --
// this stage is never run on a guess.
std::vector<TuneConfig> gpu_layers_candidates(const ParameterSpaceInput& input, const TuneConfig& base);

// True when the model is a mixture-of-experts model *and* the located
// llama-bench build understands --n-cpu-moe, i.e. when the expert-placement
// search below can run at all.
bool is_moe_tunable(const ParameterSpaceInput& input);

// Stage 2, MoE variant: how many layers keep their experts in system RAM
// (llama.cpp's --n-cpu-moe), from "all of them" down towards "none".
//
// This replaces the gpu-layers search rather than complementing it. In a
// MoE model the expert tensors dominate the weights but only a fraction of
// them is read per token, so the useful question is not "how many whole
// layers fit on the GPU" but "which parts of each layer should live there".
// Sweeping --n-gpu-layers on such a model produces non-monotonic, misleading
// results: measured on Qwen3.6-35B-A3B, throughput peaked mid-range and then
// fell off, so the search's assumption that more offload is better until it
// fails does not hold. With expert tensors placed explicitly, every layer
// goes to the GPU (-ngl = all layers) and --n-cpu-moe becomes the axis that
// actually trades VRAM for speed monotonically.
//
// Candidates descend from all-experts-on-CPU (safe, lowest VRAM) towards
// none, so the search meets its VRAM limit walking in a known direction.
// Empty when is_moe_tunable() is false or the layer count is unknown.
std::vector<TuneConfig> n_cpu_moe_candidates(const ParameterSpaceInput& input, const TuneConfig& base);

// Stage 3: a handful of (batch, ubatch) pairs, trimmed based on available
// RAM so obviously oversized combinations aren't attempted on small
// machines.
std::vector<TuneConfig> batch_ubatch_candidates(const ParameterSpaceInput& input, const TuneConfig& base);

// Stage 4: physical core count, logical thread count, and their midpoint
// (deduplicated).
std::vector<TuneConfig> thread_candidates(const ParameterSpaceInput& input, const TuneConfig& base);

} // namespace fluxinfer::tuner
