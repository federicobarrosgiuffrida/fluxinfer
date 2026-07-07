#include "fluxinfer/tuner/parameter_space.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace fluxinfer::tuner {

TuneConfig baseline_config(const ParameterSpaceInput& input) {
    TuneConfig config;
    config.threads = std::max(1u, input.hardware.cpu.physical_cores);
    config.gpu_layers = 0; // CPU-only: safest possible first measurement, cannot OOM the GPU
    config.batch_size = 256;
    config.ubatch_size = 128;
    config.label = "baseline (cpu-only)";
    return config;
}

std::vector<TuneConfig> gpu_layers_candidates(const ParameterSpaceInput& input, const TuneConfig& base) {
    std::vector<TuneConfig> candidates;

    // Never invent a layer count: without a GPU to offload to, or without
    // a real block_count read from the model's own GGUF metadata, this
    // entire stage is skipped rather than guessed.
    if (!input.hardware.gpu.available || !input.real_layer_count || *input.real_layer_count == 0) {
        return candidates;
    }

    const auto total_layers = static_cast<std::int64_t>(*input.real_layer_count);

    std::set<std::int64_t> layer_values;
    for (int pct : {0, 25, 50, 75, 100}) {
        std::int64_t layers = static_cast<std::int64_t>(std::llround(total_layers * (pct / 100.0)));
        layers = std::clamp<std::int64_t>(layers, 0, total_layers);
        layer_values.insert(layers);
    }
    // Guarantee both endpoints regardless of rounding above: 0 (no
    // offload) and total_layers (full offload) must always be probed.
    layer_values.insert(0);
    layer_values.insert(total_layers);

    for (std::int64_t layers : layer_values) {
        TuneConfig config = base;
        config.gpu_layers = static_cast<int>(layers);
        config.label = "gpu_layers=" + std::to_string(layers);
        candidates.push_back(config);
    }
    return candidates;
}

std::vector<TuneConfig> batch_ubatch_candidates(const ParameterSpaceInput& input, const TuneConfig& base) {
    struct Pair {
        unsigned batch;
        unsigned ubatch;
    };
    // 2048/512 mirrors llama-bench's own default batch/ubatch size.
    // Prompt-processing throughput is strongly batch-size sensitive, and a
    // small batch can leave significant performance on the table even when
    // it scores close to a larger one on a single noisy sample -- so the
    // search always tries llama-bench's own default too, not just small
    // batches, rather than assuming smaller is safer.
    std::vector<Pair> pairs = {{128, 64}, {256, 128}, {512, 256}, {2048, 512}};

    constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
    const std::uint64_t available_ram = input.hardware.memory.available_bytes;
    if (available_ram > 0 && available_ram < 4 * kGiB) {
        pairs = {{128, 64}}; // very constrained machine: only try the smallest combination
    } else if (available_ram > 0 && available_ram < 8 * kGiB) {
        pairs = {{128, 64}, {256, 128}}; // drop the larger combinations
    } else if (available_ram > 0 && available_ram < 16 * kGiB) {
        pairs = {{128, 64}, {256, 128}, {512, 256}}; // drop the largest combination
    }

    std::vector<TuneConfig> candidates;
    for (const auto& pair : pairs) {
        if (pair.batch == base.batch_size && pair.ubatch == base.ubatch_size) {
            continue; // already covered by the baseline/current best
        }
        TuneConfig config = base;
        config.batch_size = pair.batch;
        config.ubatch_size = pair.ubatch;
        config.label = "batch=" + std::to_string(pair.batch) + " ubatch=" + std::to_string(pair.ubatch);
        candidates.push_back(config);
    }
    return candidates;
}

std::vector<TuneConfig> thread_candidates(const ParameterSpaceInput& input, const TuneConfig& base) {
    std::set<unsigned> thread_values;
    const unsigned physical = std::max(1u, input.hardware.cpu.physical_cores);
    const unsigned logical = std::max(physical, input.hardware.cpu.logical_threads);
    const unsigned midpoint = physical + (logical - physical) / 2;

    thread_values.insert(physical);
    thread_values.insert(logical);
    thread_values.insert(midpoint);

    std::vector<TuneConfig> candidates;
    for (unsigned threads : thread_values) {
        TuneConfig config = base;
        config.threads = threads;
        config.label = "threads=" + std::to_string(threads);
        candidates.push_back(config);
    }
    return candidates;
}

} // namespace fluxinfer::tuner
