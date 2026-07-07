#pragma once

#include "fluxinfer/tuner/benchmark_result.hpp"
#include "fluxinfer/tuner/tune_config.hpp"
#include "fluxinfer/tuner/tuner.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fluxinfer::tuner {

struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0; // sample standard deviation (n-1); 0 if fewer than 2 values

    static Stats compute(const std::vector<double>& values);
};

struct RepeatedMeasurement {
    TuneConfig config;
    // Every process-level call made to measure this config: normally just
    // one (a single `llama-bench -r <repeats>` invocation), unless
    // ensure_initialized() failed before any process could even start.
    std::vector<BenchmarkResult> measured_runs;
    Stats prompt_tps_stats;   // computed from that call's per-repetition samples_ts
    Stats generation_tps_stats;
    unsigned successful_runs = 0; // number of *repetitions* (not process calls) that produced a usable sample
    unsigned failed_runs = 0;
    std::optional<std::uint64_t> peak_vram_bytes;
};

// Measures `config` via a single `llama-bench -r <repeats>` process
// invocation (so all repetitions -- and llama-bench's own internal warmup
// pass -- share one warm CUDA context/model load, rather than FluxInfer
// launching `repeats` separate cold processes), sampling peak VRAM via
// NVML in the background while it runs (when a GPU is available).
// `warmup_runs == 0` disables llama-bench's internal warmup pass
// (`--no-warmup`); any nonzero value enables it (llama-bench doesn't
// support a configurable warmup *count*, only on/off). A failed/OOM/
// crashed/timed-out invocation is kept in measured_runs and every
// requested repetition is counted in failed_runs, since a process-level
// failure doesn't distinguish which individual repetition(s) would have
// failed.
RepeatedMeasurement measure_repeated(Tuner& tuner, const TuneConfig& config, unsigned repeats, unsigned warmup_runs);

struct ComparisonReport {
    RepeatedMeasurement baseline;
    RepeatedMeasurement selected;

    double generation_tps_improvement_pct = 0.0;
    double prompt_tps_improvement_pct = 0.0;

    // True only when every measured selected-config run was faster (higher
    // generation tok/s) than every measured baseline-config run -- i.e. the
    // two sample ranges don't overlap. This is a deliberately conservative
    // bar: percentages are still reported either way, but FluxInfer will
    // not word the report as a confirmed improvement unless this holds.
    bool improvement_confirmed = false;
};

ComparisonReport build_comparison_report(Tuner& tuner, const TuneConfig& baseline_config, const TuneConfig& selected_config,
                                          unsigned repeats, unsigned warmup_runs);

struct ReportContext {
    std::string model_path;
    std::string architecture;
    std::string quantization_label;
    std::optional<std::uint64_t> layer_count;
    std::string llama_bench_path;
    std::string llama_version;
    std::vector<std::string> baseline_arguments;
    std::vector<std::string> selected_arguments;
    std::vector<BenchmarkResult> all_search_results; // every configuration attempted during the staged search
};

// Renders a human-readable Markdown report covering everything requested
// by the milestone: model/quantization, layer count, baseline vs selected
// arguments, tok/s statistics (mean/median/min/max/stddev), peak VRAM,
// percentage improvement (only claimed when improvement_confirmed), every
// attempted configuration, and OOM/failed runs.
std::string format_comparison_report(const ComparisonReport& report, const ReportContext& context);

} // namespace fluxinfer::tuner
