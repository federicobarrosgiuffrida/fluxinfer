#include "fluxinfer/tuner/comparison.hpp"

#include "fluxinfer/hardware/vram_sampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace fluxinfer::tuner {

Stats Stats::compute(const std::vector<double>& values) {
    Stats stats;
    if (values.empty()) {
        return stats;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    stats.min = sorted.front();
    stats.max = sorted.back();
    stats.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());

    const std::size_t n = sorted.size();
    stats.median = (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;

    if (n >= 2) {
        double sum_sq_diff = 0.0;
        for (double v : sorted) {
            const double diff = v - stats.mean;
            sum_sq_diff += diff * diff;
        }
        stats.stddev = std::sqrt(sum_sq_diff / static_cast<double>(n - 1));
    }

    return stats;
}

RepeatedMeasurement measure_repeated(Tuner& tuner, const TuneConfig& config, unsigned repeats, unsigned warmup_runs) {
    RepeatedMeasurement measurement;
    measurement.config = config;

    hardware::VramSampler sampler;
    sampler.start();

    // llama-bench only supports warmup on/off, not a configurable count:
    // warmup_runs == 0 explicitly disables it (--no-warmup); any nonzero
    // value enables llama-bench's own single internal warmup pass, run
    // once before all `repeats` measured passes within the *same*
    // process/model-load. This means the measured passes -- and the
    // warmup pass itself -- share one warm CUDA context, unlike
    // FluxInfer's previous approach of launching `repeats` + 1 entirely
    // separate llama-bench processes (each paying its own cold-start
    // cost), which is both slower and not representative of how a
    // long-lived `fluxinfer run`/`serve` process actually behaves.
    const bool disable_warmup = (warmup_runs == 0);
    BenchmarkResult result = tuner.benchmark_repeated(config, repeats, disable_warmup);
    measurement.measured_runs.push_back(result);

    sampler.stop();
    measurement.peak_vram_bytes = sampler.peak_used_bytes();

    if (result.usable() && !result.prompt_tps_samples.empty() && !result.generation_tps_samples.empty()) {
        measurement.successful_runs = static_cast<unsigned>(result.generation_tps_samples.size());
        measurement.prompt_tps_stats = Stats::compute(result.prompt_tps_samples);
        measurement.generation_tps_stats = Stats::compute(result.generation_tps_samples);
    } else if (result.usable()) {
        // Ran fine but llama-bench reported no samples_ts array (e.g. -r
        // isn't supported by this build): fall back to its single
        // averaged value as a one-sample measurement.
        measurement.successful_runs = 1;
        measurement.prompt_tps_stats = Stats::compute({result.prompt_tokens_per_second});
        measurement.generation_tps_stats = Stats::compute({result.generation_tokens_per_second});
    } else {
        measurement.failed_runs = repeats;
    }

    return measurement;
}

ComparisonReport build_comparison_report(Tuner& tuner, const TuneConfig& baseline_config, const TuneConfig& selected_config,
                                          unsigned repeats, unsigned warmup_runs) {
    ComparisonReport report;
    report.baseline = measure_repeated(tuner, baseline_config, repeats, warmup_runs);
    report.selected = measure_repeated(tuner, selected_config, repeats, warmup_runs);

    const auto& baseline_stats = report.baseline.generation_tps_stats;
    const auto& selected_stats = report.selected.generation_tps_stats;

    if (baseline_stats.mean > 0.0) {
        report.generation_tps_improvement_pct = (selected_stats.mean - baseline_stats.mean) / baseline_stats.mean * 100.0;
    }
    if (report.baseline.prompt_tps_stats.mean > 0.0) {
        report.prompt_tps_improvement_pct =
            (report.selected.prompt_tps_stats.mean - report.baseline.prompt_tps_stats.mean) / report.baseline.prompt_tps_stats.mean * 100.0;
    }

    // Conservative, non-overlapping-ranges bar: only call it a confirmed
    // improvement if every selected-config sample beat every baseline
    // sample. Requires at least one successful run on each side.
    report.improvement_confirmed = report.baseline.successful_runs > 0 && report.selected.successful_runs > 0 &&
                                    selected_stats.min > baseline_stats.max;

    return report;
}

namespace {

std::string join_args(const std::vector<std::string>& args) {
    std::string out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) out += ' ';
        out += args[i];
    }
    return out;
}

void append_stats_row(std::ostringstream& out, const std::string& label, const Stats& stats) {
    out << "| " << label << " | " << stats.mean << " | " << stats.median << " | " << stats.min << " | " << stats.max
        << " | " << stats.stddev << " |\n";
}

void append_measurement_section(std::ostringstream& out, const std::string& title, const RepeatedMeasurement& measurement) {
    out << "### " << title << "\n\n";
    out << "Repetitions: " << measurement.successful_runs << " successful, " << measurement.failed_runs
        << " failed. Measured via a single warm `llama-bench -r N` process (its own internal warmup pass excluded, "
           "not a separately-launched process).\n\n";

    out << "| Metric | Mean | Median | Min | Max | Stddev |\n";
    out << "| --- | --- | --- | --- | --- | --- |\n";
    append_stats_row(out, "Prompt processing tok/s", measurement.prompt_tps_stats);
    append_stats_row(out, "Generation tok/s", measurement.generation_tps_stats);
    out << "\n";

    if (measurement.peak_vram_bytes) {
        out << "Peak VRAM observed: " << (*measurement.peak_vram_bytes / (1024.0 * 1024.0)) << " MiB\n\n";
    } else {
        out << "Peak VRAM observed: not available (no GPU or NVML could not be sampled)\n\n";
    }

    if (measurement.failed_runs > 0) {
        out << "Failed/OOM runs:\n\n";
        for (const auto& r : measurement.measured_runs) {
            if (r.usable()) continue;
            out << "- ";
            if (r.oom) out << "OOM";
            else if (r.crashed) out << "crashed (exit " << r.exit_code << ")";
            else if (r.timed_out) out << "timed out";
            else if (!r.output_valid) out << "invalid/unparseable output";
            else out << "failed";
            out << "\n";
        }
        out << "\n";
    }
}

} // namespace

std::string format_comparison_report(const ComparisonReport& report, const ReportContext& context) {
    std::ostringstream out;

    out << "# FluxInfer benchmark comparison report\n\n";
    out << "## Model\n\n";
    out << "- Path: `" << context.model_path << "`\n";
    out << "- Architecture: " << (context.architecture.empty() ? "unknown" : context.architecture) << "\n";
    out << "- Quantization: " << (context.quantization_label.empty() ? "unknown" : context.quantization_label) << "\n";
    out << "- Layer count: " << (context.layer_count ? std::to_string(*context.layer_count) : "unknown") << "\n";
    out << "- llama-bench: `" << context.llama_bench_path << "` (" << context.llama_version << ")\n\n";

    out << "## Arguments\n\n";
    out << "- Baseline: `" << join_args(context.baseline_arguments) << "`\n";
    out << "- Selected: `" << join_args(context.selected_arguments) << "`\n\n";

    out << "## Results\n\n";
    append_measurement_section(out, "Baseline", report.baseline);
    append_measurement_section(out, "Selected (FluxInfer-tuned)", report.selected);

    out << "## Improvement\n\n";
    out << "- Generation tok/s: " << report.generation_tps_improvement_pct << "%\n";
    out << "- Prompt processing tok/s: " << report.prompt_tps_improvement_pct << "%\n";
    if (report.improvement_confirmed) {
        out << "- **Confirmed**: every selected-config run outperformed every baseline-config run in this sample "
               "(non-overlapping ranges).\n\n";
    } else {
        out << "- **Not confirmed**: the measured sample ranges overlap (or too few successful runs exist), so this "
               "improvement is not conclusively demonstrated by these repeats alone.\n\n";
    }

    if (!context.all_search_results.empty()) {
        out << "## All configurations attempted during the search\n\n";
        out << "| Config | Outcome | Prompt tok/s | Gen tok/s | Score |\n";
        out << "| --- | --- | --- | --- | --- |\n";
        for (const auto& r : context.all_search_results) {
            out << "| " << r.config.label << " | ";
            if (r.oom) out << "OOM";
            else if (r.crashed) out << "crashed";
            else if (r.timed_out) out << "timed out";
            else if (!r.output_valid) out << "invalid output";
            else out << "ok";
            out << " | " << r.prompt_tokens_per_second << " | " << r.generation_tokens_per_second << " | " << r.score
                << " |\n";
        }
        out << "\n";
    }

    return out.str();
}

} // namespace fluxinfer::tuner
