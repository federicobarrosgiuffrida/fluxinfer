#pragma once

#include <cstdint>
#include <memory>
#include <optional>

namespace fluxinfer::hardware {

// Polls NVML on a background thread while a benchmark subprocess runs, to
// approximate peak VRAM usage. FluxInfer has no way to hook into
// llama-bench's own allocator, so this is a coarse external sampler: it
// can miss short spikes between polls and only reflects whatever NVML
// reports as GPU-wide "used" memory (including other processes), not
// llama-bench's own allocations specifically.
class VramSampler {
public:
    VramSampler();
    ~VramSampler();

    VramSampler(const VramSampler&) = delete;
    VramSampler& operator=(const VramSampler&) = delete;

    // Starts polling every `interval_ms` (default 150ms). No-op if NVML/GPU
    // is unavailable -- peak_used_bytes() will simply stay nullopt.
    void start(unsigned interval_ms = 150);

    // Stops polling. Safe to call even if start() was never called.
    void stop();

    // Peak observed (total_vram - available_vram) across all samples taken
    // between start() and stop(), or std::nullopt if no successful GPU
    // sample was ever taken.
    std::optional<std::uint64_t> peak_used_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fluxinfer::hardware
