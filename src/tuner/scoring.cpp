#include "fluxinfer/tuner/scoring.hpp"

#include <algorithm>

namespace fluxinfer::tuner {

namespace {

double pressure_for(std::uint64_t estimated, std::uint64_t available) {
    if (available == 0) {
        return 0.0; // unknown budget (e.g. no GPU): can't judge pressure, don't penalize
    }
    const double ratio = static_cast<double>(estimated) / static_cast<double>(available);
    // Real OOM/crash/timeout is already detected authoritatively from the
    // benchmark process's actual exit behaviour (see run_llama_binary) --
    // this penalty exists only as a tie-breaking safety margin between
    // otherwise-close, already-successful configs, not as a second guess
    // at whether a config will fail. It used to ramp up from 70% of the
    // estimated budget, which combined with the VRAM estimate's own slop
    // (see Tuner::run_one) was enough to heavily penalize configurations
    // that had already completed successfully without OOM (e.g. ~75%
    // offload measuring ~40% higher generation throughput than a 50%
    // offload config, yet scoring lower purely from this estimate). Now
    // ramps from 90% instead, and callers are expected to also use a
    // smaller memory_pressure_weight (see ScoringWeights) so this stays a
    // tie-breaker rather than a dominant term.
    constexpr double kRampStart = 0.9;
    if (ratio <= kRampStart) {
        return 0.0;
    }
    const double over = std::min(ratio, 1.5) - kRampStart;
    const double normalized = over / (1.5 - kRampStart);
    return normalized * normalized * 100.0;
}

} // namespace

double compute_memory_pressure_penalty(std::uint64_t estimated_ram_bytes, std::uint64_t available_ram_bytes,
                                        std::uint64_t estimated_vram_bytes, std::uint64_t available_vram_bytes) {
    return pressure_for(estimated_ram_bytes, available_ram_bytes) + pressure_for(estimated_vram_bytes, available_vram_bytes);
}

double compute_score(const BenchmarkResult& result, std::uint64_t available_ram_bytes, std::uint64_t available_vram_bytes,
                      const ScoringWeights& weights) {
    if (!result.usable()) {
        return -weights.instability_penalty;
    }

    const double penalty = compute_memory_pressure_penalty(result.estimated_ram_bytes, available_ram_bytes,
                                                             result.estimated_vram_bytes, available_vram_bytes);

    return result.generation_tokens_per_second * weights.generation_tps_weight +
           result.prompt_tokens_per_second * weights.prompt_tps_weight -
           result.first_token_latency_ms * weights.first_token_latency_weight -
           penalty * weights.memory_pressure_weight;
}

} // namespace fluxinfer::tuner
