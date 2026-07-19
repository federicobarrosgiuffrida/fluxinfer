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

// True when `candidate` -- a run that pushed more work onto the GPU than
// `reference` did and completed without error -- nonetheless shows the
// signature of a silent VRAM spill: the card was left with less than
// `headroom_bytes` free, and throughput fell well below the less
// GPU-heavy `reference` instead of improving.
//
// This exists because the failure it detects does not announce itself. On
// Windows, an allocation that no longer fits in VRAM is not rejected: the
// WDDM driver satisfies it from system RAM and inference keeps running at
// a fraction of the speed, with no CUDA OOM anywhere in the output. A
// search that only reacts to OOM will happily pick such a configuration on
// the grounds that it "worked".
//
// Measured on an RTX 3060 12GB with a 21GB MoE model: at the offload
// setting that left ~260MB free, prompt processing fell to roughly an
// eighth and generation to under two thirds of the next-lower setting,
// with every run reported as successful. Both `candidate` and `reference`
// must be usable runs with a measured VRAM peak; otherwise this returns
// false (no measurement, no accusation).
bool looks_like_vram_spill(const BenchmarkResult& candidate, const BenchmarkResult& reference,
                            std::uint64_t total_vram_bytes, std::uint64_t headroom_bytes);

} // namespace fluxinfer::tuner
