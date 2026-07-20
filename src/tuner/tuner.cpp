#include "fluxinfer/tuner/tuner.hpp"

#include "fluxinfer/hardware/vram_sampler.hpp"

#include "fluxinfer/llama/benchmark_parser.hpp"
#include "fluxinfer/llama/llama_runner.hpp"
#include "fluxinfer/tuner/parameter_space.hpp"

#include <algorithm>
#include <optional>
#include <system_error>

namespace fluxinfer::tuner {

namespace {
constexpr std::size_t kTailLength = 2000;

std::string tail(const std::string& text) {
    return text.size() > kTailLength ? text.substr(text.size() - kTailLength) : text;
}

void add_if_supported(std::vector<std::string>& args, const std::set<std::string>& supported,
                       const std::string& long_flag, const std::string& short_flag, const std::string& value) {
    if (llama::supports_flag(supported, long_flag)) {
        args.push_back(short_flag);
        args.push_back(value);
    }
}

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return (n % 2 == 1) ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}
} // namespace

Tuner::Tuner(TunerOptions options) : options_(std::move(options)) {}

std::vector<std::string> Tuner::build_arguments(const TuneConfig& config, unsigned repetitions, bool disable_warmup) const {
    std::vector<std::string> args;
    args.push_back("-m");
    args.push_back(options_.model_path.string());

    add_if_supported(args, options_.supported_flags, "--n-prompt", "-p", std::to_string(options_.prompt_tokens_for_bench));
    add_if_supported(args, options_.supported_flags, "--n-gen", "-n", std::to_string(options_.gen_tokens_for_bench));
    // llama-bench has no context-size flag of its own; -d pre-fills this
    // many tokens of KV cache before measuring, which allocates a KV cache
    // comparable in size to what `-c context_length` would at run time --
    // see TunerOptions::context_length for why this matters.
    add_if_supported(args, options_.supported_flags, "--n-depth", "-d", std::to_string(options_.context_length));
    add_if_supported(args, options_.supported_flags, "--threads", "-t", std::to_string(config.threads));
    add_if_supported(args, options_.supported_flags, "--n-gpu-layers", "-ngl", std::to_string(config.gpu_layers));
    add_if_supported(args, options_.supported_flags, "--batch-size", "-b", std::to_string(config.batch_size));
    add_if_supported(args, options_.supported_flags, "--ubatch-size", "-ub", std::to_string(config.ubatch_size));
    add_if_supported(args, options_.supported_flags, "--repetitions", "-r", std::to_string(std::max(1u, repetitions)));

    if (disable_warmup && llama::supports_flag(options_.supported_flags, "--no-warmup")) {
        args.push_back("--no-warmup");
    }

    if (config.n_cpu_moe) {
        add_if_supported(args, options_.supported_flags, "--n-cpu-moe", "-ncmoe", std::to_string(*config.n_cpu_moe));
    }

    if (config.kv_cache_type) {
        if (llama::supports_flag(options_.supported_flags, "--cache-type-k")) {
            args.push_back("-ctk");
            args.push_back(*config.kv_cache_type);
        }
        if (llama::supports_flag(options_.supported_flags, "--cache-type-v")) {
            args.push_back("-ctv");
            args.push_back(*config.kv_cache_type);
        }
    }

    for (const auto& [flag, value] : config.extra_flags) {
        args.push_back(flag);
        if (!value.empty()) {
            args.push_back(value);
        }
    }

    // Ask llama-bench to announce each phase. Without this it is silent
    // until the very end in JSON mode, which makes an idle timeout
    // meaningless (it would only measure total runtime, badly).
    if (llama::supports_flag(options_.supported_flags, "--progress")) {
        args.push_back("--progress");
    }

    if (llama::supports_flag(options_.supported_flags, "--output")) {
        args.push_back("-o");
        args.push_back("json");
    }

    return args;
}

std::chrono::milliseconds Tuner::effective_idle_timeout() const {
    // Silence only means "stuck" if llama-bench would otherwise be talking.
    if (!llama::supports_flag(options_.supported_flags, "--progress")) {
        return std::chrono::milliseconds::zero();
    }
    if (options_.idle_timeout <= std::chrono::milliseconds::zero()) {
        return std::chrono::milliseconds::zero();
    }

    // The longest legitimately silent phase is the depth prefill: it
    // processes `context_length` tokens in one go, with no output until it
    // finishes. Budget it at a deliberately pessimistic floor for
    // prompt-processing throughput -- well below anything measured on real
    // hardware (the worst observed on an RTX 3060 was ~77 tok/s, and that
    // was already a machine in trouble) -- so a healthy prefill is never
    // mistaken for a hang, while a truly stuck process is still caught in
    // bounded time.
    constexpr double kPessimisticPromptTokensPerSecond = 25.0;
    const auto prefill_budget = std::chrono::milliseconds(
        static_cast<std::int64_t>(1000.0 * options_.context_length / kPessimisticPromptTokensPerSecond));
    return std::max(options_.idle_timeout, prefill_budget);
}

BenchmarkResult Tuner::run_one(const TuneConfig& config, unsigned repetitions, bool disable_warmup) {
    BenchmarkResult result;
    result.config = config;

    const std::vector<std::string> args = build_arguments(config, repetitions, disable_warmup);
    // Repeating -r times reuses the same model load/CUDA context, so the
    // dominant per-run cost (loading a possibly multi-GB model) is paid
    // once; scaling the timeout by repetitions is a generous but safe
    // ceiling rather than a tight estimate.
    auto effective_timeout = options_.per_run_timeout * std::max(1u, repetitions);
    // Scale the cap with model size: a 21GB model takes minutes just to be
    // read off disk, so a fixed budget tuned for a small model reports
    // "timed out" for runs that were never in trouble. The idle timeout
    // below is what actually catches hangs, so the cap only needs to be
    // generous enough not to fire spuriously.
    constexpr std::int64_t kTimeoutMillisPerGiB = 10000;
    constexpr std::uint64_t kGiB = 1024ULL * 1024 * 1024;
    const auto size_based = std::chrono::milliseconds(
        static_cast<std::int64_t>(model_size_bytes_ / kGiB) * kTimeoutMillisPerGiB * std::max(1u, repetitions));
    // The same prefill the idle window accounts for also has to fit inside
    // the total budget, or the cap simply replaces the idle timeout as the
    // thing that kills healthy runs.
    constexpr double kPessimisticPromptTokensPerSecond = 25.0;
    const auto prefill_budget = std::chrono::milliseconds(
        static_cast<std::int64_t>(1000.0 * options_.context_length / kPessimisticPromptTokensPerSecond));
    effective_timeout = std::max(effective_timeout, size_based + prefill_budget);

    // Measure what the card is actually doing during the run. The sampler
    // is GPU-wide and coarse (see VramSampler), which is the right scope
    // here: the question is whether the *card* ran out of room, including
    // whatever else the machine had resident at the time.
    hardware::VramSampler sampler;
    sampler.start();
    llama::LlamaRunResult run =
        llama::run_llama_binary(options_.llama_bench_path, args, effective_timeout, effective_idle_timeout());
    sampler.stop();
    result.measured_peak_vram_bytes = sampler.peak_used_bytes();

    result.ran = run.outcome != process::ProcessOutcome::FailedToStart;
    result.exit_code = run.exit_code;
    result.timed_out = run.timed_out;
    result.idle_timed_out = run.idle_timed_out;
    result.crashed = run.crashed;
    result.oom = run.likely_oom;
    result.duration = run.duration;
    result.stdout_tail = tail(run.stdout_data);
    result.stderr_tail = tail(run.stderr_data);

    if (result.ran && !result.timed_out && !result.crashed && !result.oom) {
        llama::BenchmarkParseResult parsed = llama::parse_llama_bench_output(run.stdout_data);
        result.output_valid = parsed.valid;
        if (parsed.valid) {
            result.prompt_tps_samples = parsed.prompt_tps_samples;
            result.generation_tps_samples = parsed.generation_tps_samples;
            // The median across repetitions is more robust to a single
            // noisy run than llama-bench's own average; falls back to the
            // average itself when there's only one sample (no samples_ts
            // array, or a single-repetition run).
            result.prompt_tokens_per_second = result.prompt_tps_samples.size() >= 2
                ? median_of(result.prompt_tps_samples)
                : parsed.prompt_tokens_per_second.value_or(0.0);
            result.generation_tokens_per_second = result.generation_tps_samples.size() >= 2
                ? median_of(result.generation_tps_samples)
                : parsed.generation_tokens_per_second.value_or(0.0);
        }
    }

    // Rough memory estimate (llama-bench does not report actual usage): the
    // fraction of model weights assumed resident on GPU vs host RAM, scaled
    // by gpu_layers relative to the model's real layer count (from GGUF
    // metadata), plus a fixed overhead for KV cache/activations. This is
    // still only an estimate (layers aren't necessarily equal-sized) used
    // for the scoring penalty/display; OOM itself is detected from the
    // process's real exit behaviour (see llama::run_llama_binary), not
    // from this estimate.
    constexpr std::uint64_t kFixedOverheadBytes = 512ULL * 1024 * 1024;
    const double gpu_fraction = layer_count_for_estimate_ > 0
        ? std::clamp(static_cast<double>(config.gpu_layers) / static_cast<double>(layer_count_for_estimate_), 0.0, 1.0)
        : 0.0;
    result.estimated_vram_bytes = config.gpu_layers > 0
        ? static_cast<std::uint64_t>(gpu_fraction * static_cast<double>(model_size_bytes_)) + kFixedOverheadBytes
        : 0;
    result.estimated_ram_bytes =
        static_cast<std::uint64_t>((1.0 - gpu_fraction) * static_cast<double>(model_size_bytes_)) + kFixedOverheadBytes;

    // llama-bench does not report time-to-first-token; approximate it from
    // prompt-processing throughput as a latency proxy.
    result.first_token_latency_ms = result.prompt_tokens_per_second > 0.0 ? (1000.0 / result.prompt_tokens_per_second) : 0.0;

    result.score = compute_score(result, options_.hardware.memory.available_bytes,
                                  options_.hardware.gpu.available_vram_bytes, options_.weights);

    if (options_.on_result) {
        options_.on_result(result);
    }

    return result;
}

bool Tuner::ensure_initialized(std::string* error) {
    if (initialized_) {
        return true;
    }
    std::error_code ec;
    model_size_bytes_ = std::filesystem::file_size(options_.model_path, ec);
    if (ec) {
        if (error) *error = "could not read model file size: " + ec.message();
        return false;
    }
    layer_count_for_estimate_ = options_.real_layer_count.value_or(0);
    initialized_ = true;
    return true;
}

BenchmarkResult Tuner::benchmark_once(const TuneConfig& config) {
    std::string init_error;
    if (!ensure_initialized(&init_error)) {
        BenchmarkResult result;
        result.config = config;
        result.stderr_tail = init_error;
        return result;
    }
    return run_one(config, 1, false);
}

BenchmarkResult Tuner::benchmark_repeated(const TuneConfig& config, unsigned repetitions, bool disable_warmup) {
    std::string init_error;
    if (!ensure_initialized(&init_error)) {
        BenchmarkResult result;
        result.config = config;
        result.stderr_tail = init_error;
        return result;
    }
    return run_one(config, repetitions, disable_warmup);
}

TuningOutcome Tuner::run() {
    TuningOutcome outcome;

    std::string init_error;
    if (!ensure_initialized(&init_error)) {
        outcome.error = init_error;
        return outcome;
    }

    ParameterSpaceInput psi;
    psi.hardware = options_.hardware;
    psi.supported_flags = options_.supported_flags;
    psi.model_size_bytes = model_size_bytes_;
    psi.real_layer_count = options_.real_layer_count;
    psi.vram_headroom_bytes = options_.vram_headroom_bytes;
    psi.expert_count = options_.moe_tuning_enabled ? options_.expert_count : std::nullopt;

    const std::uint64_t headroom =
        options_.vram_headroom_bytes > 0 ? options_.vram_headroom_bytes : default_vram_headroom_bytes();

    // A MoE model uses the expert-placement search *instead of* the dense
    // gpu-layers sweep: on such models the latter is not merely suboptimal
    // but misleading (non-monotonic), so running both would waste probes
    // and could hand the wrong configuration to the later stages.
    const bool moe_stage_enabled = is_moe_tunable(psi);
    const bool gpu_layers_stage_enabled =
        !moe_stage_enabled && options_.hardware.gpu.available && options_.real_layer_count && *options_.real_layer_count > 0;

    // Stage 1: baseline.
    const TuneConfig baseline = baseline_config(psi);
    outcome.baseline_config = baseline;
    BenchmarkResult best = run_one(baseline, options_.search_repetitions, false);
    outcome.all_results.push_back(best);

    // Stage 2: GPU layers search. Once a candidate OOMs, no larger value is
    // tried; an iterative binary search then refines the boundary between
    // the highest known-good value and the lowest known-OOM value, rather
    // than a single probe, to land much closer to the true offload
    // ceiling (bounded to a handful of extra probes).
    if (gpu_layers_stage_enabled) {
        std::vector<TuneConfig> candidates = gpu_layers_candidates(psi, baseline);
        std::sort(candidates.begin(), candidates.end(),
                  [](const TuneConfig& a, const TuneConfig& b) { return a.gpu_layers < b.gpu_layers; });

        int last_good_layers = -1;
        int oom_layers = -1;
        std::optional<BenchmarkResult> last_good_result;
        for (const auto& candidate : candidates) {
            if (oom_layers >= 0 && candidate.gpu_layers >= oom_layers) {
                continue;
            }
            BenchmarkResult result = run_one(candidate, options_.search_repetitions, false);
            // A successful-looking run that filled the card and collapsed
            // relative to the last good one is a silent WDDM spill, not a
            // usable candidate: treat it exactly like an OOM boundary, so
            // nothing above it is attempted and it cannot win on score.
            if (last_good_result &&
                looks_like_vram_spill(result, *last_good_result, options_.hardware.gpu.total_vram_bytes, headroom)) {
                result.vram_spill_suspected = true;
                outcome.all_results.push_back(result);
                oom_layers = candidate.gpu_layers;
                continue;
            }
            outcome.all_results.push_back(result);
            if (result.oom) {
                oom_layers = candidate.gpu_layers;
                continue;
            }
            // A timeout (or any other non-OOM failure) at this gpu_layers
            // value is inconclusive, not a confirmed success: on Windows,
            // an over-VRAM allocation is frequently reported as a hang/
            // timeout by the WDDM driver rather than a clean CUDA OOM
            // error, so treating "not explicitly OOM" as "good" would
            // wrongly promote a failed run to last_good_layers (and
            // potentially into `best`, if compute_score's -1000 for
            // unusable results were ever bypassed). It also isn't treated
            // as a hard "stop climbing" boundary the way a confirmed OOM
            // is, since an unrelated slow run (e.g. a CPU-only baseline
            // exceeding its time budget) has nothing to do with VRAM
            // capacity -- the next, larger candidate is still attempted.
            if (result.usable()) {
                last_good_layers = candidate.gpu_layers;
                last_good_result = result;
                if (result.score > best.score) {
                    best = result;
                }
            }
        }

        constexpr int kMaxRefinementProbes = 4;
        int probes = 0;
        while (oom_layers >= 0 && last_good_layers >= 0 && oom_layers - last_good_layers > 1 &&
               probes < kMaxRefinementProbes) {
            const int midpoint_layers = last_good_layers + (oom_layers - last_good_layers) / 2;
            TuneConfig midpoint = baseline;
            midpoint.gpu_layers = midpoint_layers;
            midpoint.label = "gpu_layers=" + std::to_string(midpoint_layers) + " (binary search)";
            BenchmarkResult mid_result = run_one(midpoint, options_.search_repetitions, false);
            ++probes;
            if (last_good_result &&
                looks_like_vram_spill(mid_result, *last_good_result, options_.hardware.gpu.total_vram_bytes, headroom)) {
                mid_result.vram_spill_suspected = true;
                outcome.all_results.push_back(mid_result);
                oom_layers = midpoint_layers;
                continue;
            }
            outcome.all_results.push_back(mid_result);
            if (mid_result.oom) {
                oom_layers = midpoint_layers;
            } else if (mid_result.usable()) {
                last_good_layers = midpoint_layers;
                last_good_result = mid_result;
                if (mid_result.score > best.score) {
                    best = mid_result;
                }
            } else {
                // Inconclusive (e.g. timed out without a confirmed OOM):
                // narrow the range from the top so the loop keeps making
                // progress, without promoting this candidate to
                // last_good_layers/best -- see the identical reasoning in
                // the main candidate loop above.
                oom_layers = midpoint_layers;
            }
        }
    }

    // Stage 2 (MoE variant): expert placement. Walks --n-cpu-moe down from
    // "every layer's experts in system RAM" towards "none", i.e. from the
    // lowest VRAM demand to the highest, so the search approaches its limit
    // from the safe side. The first value that OOMs or shows a silent spill
    // becomes the floor, and a bounded bisection then closes the gap
    // between it and the last good value.
    if (moe_stage_enabled) {
        std::vector<TuneConfig> candidates = n_cpu_moe_candidates(psi, baseline);

        int last_good_ncmoe = -1;
        int failed_ncmoe = -1;         // lowest value known to fail outright (OOM/spill)
        int lowest_failed_ncmoe = -1;  // lowest value that failed inconclusively (timeout/crash)
        std::optional<BenchmarkResult> last_good_result;
        for (const auto& candidate : candidates) {
            const int ncmoe = candidate.n_cpu_moe.value_or(0);
            if (failed_ncmoe >= 0 && ncmoe <= failed_ncmoe) {
                continue;
            }
            BenchmarkResult result = run_one(candidate, options_.search_repetitions, false);
            if (last_good_result &&
                looks_like_vram_spill(result, *last_good_result, options_.hardware.gpu.total_vram_bytes, headroom)) {
                result.vram_spill_suspected = true;
                outcome.all_results.push_back(result);
                failed_ncmoe = ncmoe;
                continue;
            }
            outcome.all_results.push_back(result);
            if (result.oom) {
                failed_ncmoe = ncmoe;
                continue;
            }
            if (result.usable()) {
                last_good_ncmoe = ncmoe;
                last_good_result = result;
                if (result.score > best.score) {
                    best = result;
                }
            } else {
                // Timed out or crashed: not proof of an OOM, so it does not
                // stop the sweep the way a confirmed OOM does -- but it is
                // still the lowest value known *not* to work, which is
                // exactly what the bisection below needs as a floor.
                // Without this, a Windows machine (where an over-VRAM MoE
                // configuration stalls instead of reporting OOM) never
                // refines at all: observed on an RTX 3060, where the sweep
                // stopped at the coarse grid and left the real optimum
                // between two probes unexplored.
                if (lowest_failed_ncmoe < 0 || ncmoe < lowest_failed_ncmoe) {
                    lowest_failed_ncmoe = ncmoe;
                }
            }
        }

        // Refine against whichever floor is known: a confirmed OOM/spill if
        // there is one, otherwise the lowest inconclusive failure.
        if (failed_ncmoe < 0) {
            failed_ncmoe = lowest_failed_ncmoe;
        }

        // Same bounded refinement as the dense search, mirrored: here the
        // usable side is the *higher* value and the failing side the lower.
        constexpr int kMaxRefinementProbes = 4;
        int probes = 0;
        while (failed_ncmoe >= 0 && last_good_ncmoe >= 0 && last_good_ncmoe - failed_ncmoe > 1 &&
               probes < kMaxRefinementProbes) {
            const int midpoint_ncmoe = failed_ncmoe + (last_good_ncmoe - failed_ncmoe) / 2;
            TuneConfig midpoint = baseline;
            midpoint.gpu_layers = static_cast<int>(options_.real_layer_count.value_or(0));
            midpoint.n_cpu_moe = midpoint_ncmoe;
            midpoint.label = "n_cpu_moe=" + std::to_string(midpoint_ncmoe) + " (binary search)";
            BenchmarkResult mid_result = run_one(midpoint, options_.search_repetitions, false);
            ++probes;
            if (last_good_result &&
                looks_like_vram_spill(mid_result, *last_good_result, options_.hardware.gpu.total_vram_bytes, headroom)) {
                mid_result.vram_spill_suspected = true;
                outcome.all_results.push_back(mid_result);
                failed_ncmoe = midpoint_ncmoe;
                continue;
            }
            outcome.all_results.push_back(mid_result);
            if (mid_result.usable()) {
                last_good_ncmoe = midpoint_ncmoe;
                last_good_result = mid_result;
                if (mid_result.score > best.score) {
                    best = mid_result;
                }
            } else {
                // OOM, or inconclusive (timeout/crash): either way this
                // value is not demonstrably usable, so it becomes the new
                // floor rather than being promoted -- same reasoning as the
                // dense search.
                failed_ncmoe = midpoint_ncmoe;
            }
        }
    }

    // Stage 3: batch/ubatch, based on the best configuration found so far.
    // Batch size is not VRAM-neutral: larger batches allocate larger compute
    // buffers, which on a card already close to full is enough to push the
    // configuration into the same silent spill the offload stages guard
    // against. Observed at -c 32768 on an RTX 3060, where batch=2048 halved
    // generation throughput against the very same expert placement. So the
    // guard applies here too, against the best configuration so far.
    for (const auto& candidate : batch_ubatch_candidates(psi, best.config)) {
        BenchmarkResult result = run_one(candidate, options_.search_repetitions, false);
        if (looks_like_vram_spill(result, best, options_.hardware.gpu.total_vram_bytes, headroom)) {
            result.vram_spill_suspected = true;
            outcome.all_results.push_back(result);
            continue;
        }
        outcome.all_results.push_back(result);
        if (result.score > best.score) {
            best = result;
        }
    }

    // Stage 4: thread count, based on the best configuration found so far.
    for (const auto& candidate : thread_candidates(psi, best.config)) {
        if (candidate.threads == best.config.threads) {
            continue; // already measured in an earlier stage
        }
        BenchmarkResult result = run_one(candidate, options_.search_repetitions, false);
        // Thread count is not VRAM-neutral on a MoE model either: the
        // CPU-resident experts' working buffers scale with it.
        if (looks_like_vram_spill(result, best, options_.hardware.gpu.total_vram_bytes, headroom)) {
            result.vram_spill_suspected = true;
            outcome.all_results.push_back(result);
            continue;
        }
        outcome.all_results.push_back(result);
        if (result.score > best.score) {
            best = result;
        }
    }

    outcome.best = best;
    if (!best.usable()) {
        outcome.success = false;
        outcome.error = "every benchmark configuration failed (OOM, crash, timeout or invalid output)";
        return outcome;
    }

    outcome.success = true;
    return outcome;
}

} // namespace fluxinfer::tuner
