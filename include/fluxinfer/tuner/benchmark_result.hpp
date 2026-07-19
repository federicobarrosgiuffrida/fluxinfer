#pragma once

#include "fluxinfer/tuner/tune_config.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fluxinfer::tuner {

struct BenchmarkResult {
    TuneConfig config;

    bool ran = false; // process was launched and returned (not FailedToStart)
    int exit_code = -1;
    bool timed_out = false;
    bool crashed = false;
    bool oom = false;
    bool output_valid = false; // benchmark_parser could extract tok/s figures

    // prompt_tokens_per_second/generation_tokens_per_second are the
    // representative figures used for scoring: the median of
    // *_tps_samples when llama-bench ran more than one repetition (more
    // robust to a single noisy run than llama-bench's own average), or
    // its own avg_ts otherwise. *_tps_samples is empty for single-repetition
    // runs.
    double prompt_tokens_per_second = 0.0;
    double generation_tokens_per_second = 0.0;
    std::vector<double> prompt_tps_samples;
    std::vector<double> generation_tps_samples;
    double first_token_latency_ms = 0.0; // estimated, see Tuner::run_one()

    std::uint64_t estimated_ram_bytes = 0;
    std::uint64_t estimated_vram_bytes = 0;

    // Peak GPU-wide VRAM usage observed by polling NVML while this run was
    // in flight (see hardware::VramSampler). std::nullopt when no GPU
    // sample could be taken. Unlike estimated_vram_bytes this is measured,
    // but it is still GPU-wide: it includes whatever other processes were
    // holding at the time, which is exactly what matters for deciding
    // whether the card was full.
    std::optional<std::uint64_t> measured_peak_vram_bytes;

    // Set when a run completed "successfully" but shows the signature of a
    // silent over-VRAM spill: the card was effectively full and throughput
    // collapsed relative to a less GPU-heavy configuration. On Windows,
    // WDDM backs such allocations with system RAM instead of failing, so
    // there is no OOM to detect -- the run looks fine and is simply slow.
    // Treated as a boundary by the search, like an OOM, rather than as a
    // usable candidate.
    bool vram_spill_suspected = false;

    std::chrono::milliseconds duration{0};
    double score = 0.0;

    std::string stdout_tail; // truncated, diagnostics only
    std::string stderr_tail;

    bool usable() const { return ran && !timed_out && !crashed && !oom && output_valid; }
};

} // namespace fluxinfer::tuner
