#pragma once

#include "fluxinfer/tuner/benchmark_result.hpp"

#include <cstdint>

namespace fluxinfer::tuner {

struct ScoringWeights {
    double generation_tps_weight = 2.0;
    double prompt_tps_weight = 0.10;
    double first_token_latency_weight = 0.01;
    // Deliberately small: real OOM/crash/timeout is already detected from
    // the benchmark process's actual exit behaviour and excluded via
    // usable(), so this only needs to be a tie-breaker between
    // already-successful configs, not a second guess that can outweigh a
    // large real throughput difference based on an imprecise estimate.
    double memory_pressure_weight = 0.3;
    double instability_penalty = 1000.0; // flat penalty for crash/oom/timeout/invalid output
};

// Penalty (0 and growing) reflecting how close estimated RAM/VRAM usage
// comes to the available budget: 0 while under ~90% usage, ramping up
// quadratically as usage approaches or exceeds 100%. Deliberately a late,
// steep ramp rather than an early one -- see compute_memory_pressure_penalty's
// definition for why.
double compute_memory_pressure_penalty(std::uint64_t estimated_ram_bytes, std::uint64_t available_ram_bytes,
                                        std::uint64_t estimated_vram_bytes, std::uint64_t available_vram_bytes);

// score = generation_tps * w.generation_tps_weight
//       + prompt_tps * w.prompt_tps_weight
//       - first_token_latency_ms * w.first_token_latency_weight
//       - memory_pressure_penalty * w.memory_pressure_weight
// Unusable results (crash/OOM/timeout/invalid output) instead score
// -w.instability_penalty, unconditionally worse than any usable result.
double compute_score(const BenchmarkResult& result, std::uint64_t available_ram_bytes, std::uint64_t available_vram_bytes,
                      const ScoringWeights& weights = {});

} // namespace fluxinfer::tuner
