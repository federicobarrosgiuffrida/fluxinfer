#pragma once

#include "fluxinfer/hardware/hardware_info.hpp"
#include "fluxinfer/llama/gguf_metadata.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace fluxinfer::tuner {

// What running this model at this context size is expected to demand, and
// whether the machine can plausibly meet it.
//
// This is an *estimate*, and a coarse one: it accounts for quantized
// weights, the KV cache at the requested context, and a fixed allowance
// for activations and compute buffers. It does not model per-layer size
// variation, batch-dependent buffers, or what other processes are holding.
// It exists to answer one question before a tuning run burns half an hour:
// is this combination obviously impossible, obviously comfortable, or
// somewhere in between?
struct FitEstimate {
    enum class Verdict {
        Unknown,            // metadata missing: no estimate was made, and none is guessed
        FitsInVram,         // weights + KV cache + overhead fit on the GPU with headroom to spare
        NeedsCpuOffload,    // fits the machine, but only with part of the model in system RAM
        ExceedsTotalMemory, // does not fit in VRAM and RAM combined
    };

    Verdict verdict = Verdict::Unknown;

    std::uint64_t weights_bytes = 0;
    std::uint64_t kv_cache_bytes = 0;
    std::uint64_t overhead_bytes = 0;
    std::uint64_t total_required_bytes = 0;

    // Human-readable summary, always populated, including for Unknown
    // (where it explains which metadata was missing).
    std::string explanation;

    bool is_problem() const { return verdict == Verdict::ExceedsTotalMemory; }
};

// KV cache size for `context_length` tokens, in bytes, or nullopt when the
// metadata needed to compute it is missing. Assumes f16 K and V entries
// (llama.cpp's default; quantized KV cache makes this an overestimate,
// which is the safe direction).
std::optional<std::uint64_t> estimate_kv_cache_bytes(const llama::GgufMetadata& metadata, std::uint64_t context_length);

// Combines the model file size (quantized weights as stored), the KV cache
// estimate above and a fixed overhead allowance, and compares the total to
// the machine's VRAM and RAM.
FitEstimate estimate_fit(std::uint64_t model_size_bytes, const llama::GgufMetadata& metadata,
                          std::uint64_t context_length, const hardware::HardwareInfo& hardware,
                          std::uint64_t vram_headroom_bytes);

} // namespace fluxinfer::tuner
