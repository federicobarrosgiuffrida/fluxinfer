#pragma once

#include "fluxinfer/hardware/hardware_info.hpp"
#include "fluxinfer/tuner/benchmark_result.hpp"
#include "fluxinfer/tuner/scoring.hpp"
#include "fluxinfer/tuner/tune_config.hpp"

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace fluxinfer::tuner {

struct TunerOptions {
    std::filesystem::path llama_bench_path;
    std::filesystem::path model_path;
    hardware::HardwareInfo hardware;
    std::set<std::string> supported_flags; // from `llama-bench --help`
    ScoringWeights weights;

    // The model's real transformer layer count, from
    // fluxinfer::llama::parse_gguf_metadata(). std::nullopt (metadata
    // unavailable/unreadable) disables the GPU-layers search stage rather
    // than falling back to any estimate -- see parameter_space.hpp.
    std::optional<std::uint64_t> real_layer_count;

    // Expert count from GGUF metadata; > 1 selects the MoE expert-placement
    // search in place of the dense gpu-layers search. See
    // parameter_space.hpp's n_cpu_moe_candidates().
    std::optional<std::uint64_t> expert_count;

    // Set false to skip the MoE stage and tune the model as if it were
    // dense (--no-moe-tune).
    bool moe_tuning_enabled = true;

    // Passed through to ParameterSpaceInput::vram_headroom_bytes (0 =
    // platform default, see parameter_space.hpp).
    std::uint64_t vram_headroom_bytes = 0;

    // Lower bound on the per-run total time budget. The effective cap is
    // scaled up for large models (see kTimeoutMillisPerGiB in tuner.cpp),
    // because most of a benchmark run on a multi-gigabyte model is spent
    // loading weights from disk before any work starts.
    std::chrono::milliseconds per_run_timeout{60000};

    // Kill a benchmark that produces no output for this long -- but only
    // when llama-bench can actually be made to talk (--progress). In JSON
    // output mode llama-bench prints nothing at all until the whole run is
    // over, so silence carries no information: with --progress it marks
    // each phase (load, warmup, depth prefill, prompt, generation), and
    // silence *within* a phase is what a stall looks like.
    //
    // The effective window is widened for the work a phase can legitimately
    // take (see effective_idle_timeout() in tuner.cpp): prefilling a large
    // context is a single silent phase that scales with --context, and a
    // fixed 60s window fails every candidate on a big model at a big
    // context -- observed on an RTX 3060 at -c 32768, where all eleven
    // candidates were killed while working normally.
    std::chrono::milliseconds idle_timeout{60000};
    unsigned prompt_tokens_for_bench = 512; // -p
    unsigned gen_tokens_for_bench = 128;    // -n

    // The context size `fluxinfer run`/`serve` will actually launch
    // llama-cli/llama-server with (`-c`/`--ctx-size`). llama-bench has no
    // equivalent context-size flag of its own -- it only allocates as much
    // KV cache as `-p`/`-n` need, typically a few hundred tokens -- so
    // without deliberately accounting for this, a config benchmarked at a
    // tiny effective context could behave very differently (slower, or
    // even OOM) once actually run at a real context size, since a larger
    // KV cache competes with model weights for the same VRAM budget. To
    // keep tuning representative of real usage, every benchmark candidate
    // is run with `-d context_length` (llama-bench's "pre-fill this many
    // tokens of KV cache before measuring" flag), which allocates a KV
    // cache of comparable size to what `-c context_length` would at run
    // time. context_length is also persisted into the saved profile and
    // reused as `-c` by `run`/`serve`, so the benchmarked and served
    // configurations are consistent. Found via a real regression report:
    // a config tuned without this reasoning performed worse in practice
    // than in the benchmark once run with llama.cpp's own default context
    // (0 = the model's full native training context, which can be
    // enormous -- e.g. 262144+ tokens -- and was silently reserving far
    // more KV-cache VRAM at run time than the benchmark ever exercised).
    unsigned context_length = 4096;

    // Internal llama-bench repetitions (`-r`) per search-stage candidate,
    // run within a single warm process/model-load. The *median* sample is
    // used for scoring rather than llama-bench's own average, which is
    // more robust to the occasional noisy run (observed in practice) and
    // makes close candidates (e.g. two batch sizes) far less likely to be
    // decided by a coin flip. Since this only adds extra compute passes to
    // an already-loaded model (no repeated multi-GB model loads), the cost
    // is small relative to a single load+run.
    unsigned search_repetitions = 3;

    // Invoked after every completed benchmark run; useful for CLI progress
    // reporting. May be empty.
    std::function<void(const BenchmarkResult&)> on_result;
};

struct TuningOutcome {
    bool success = false;
    std::optional<BenchmarkResult> best;
    std::optional<TuneConfig> baseline_config; // stage 1's conservative CPU-only configuration
    std::vector<BenchmarkResult> all_results;
    std::string error; // populated when success == false
};

// Runs the staged search described in the project README (baseline -> GPU
// layers -> batch/ubatch -> threads), evaluating each candidate with
// llama-bench and picking the highest-scoring usable result.
class Tuner {
public:
    explicit Tuner(TunerOptions options);

    TuningOutcome run();

    // Runs a single benchmark of `config` (llama-bench `-r 1`) and returns
    // its result, without participating in the staged search. Requires
    // model_size_bytes_/layer_count_for_estimate_ to be initialized, which
    // happens on the first call (mirrors what run() does) if this is
    // invoked before run().
    BenchmarkResult benchmark_once(const TuneConfig& config);

    // Runs `config` through llama-bench with `-r repetitions` in a single
    // process/model-load (so repetitions after the first benefit from a
    // warm CUDA context, unlike separate process launches), optionally
    // disabling llama-bench's own internal warmup pass (`--no-warmup`).
    // The returned BenchmarkResult's *_tps_samples hold every repetition;
    // prompt_tokens_per_second/generation_tokens_per_second are their
    // median. Used by the repeated-measurement comparison report (see
    // comparison.hpp).
    BenchmarkResult benchmark_repeated(const TuneConfig& config, unsigned repetitions, bool disable_warmup = false);

    // Exposes the exact llama-bench argv FluxInfer would use for `config`,
    // for display in reports (see comparison.hpp).
    std::vector<std::string> build_arguments_for(const TuneConfig& config, unsigned repetitions = 1) const {
        return build_arguments(config, repetitions, false);
    }

private:
    BenchmarkResult run_one(const TuneConfig& config, unsigned repetitions, bool disable_warmup);
    std::vector<std::string> build_arguments(const TuneConfig& config, unsigned repetitions, bool disable_warmup) const;
    bool ensure_initialized(std::string* error);
    // Idle window actually applied to a run: zero (disabled) unless
    // llama-bench supports --progress, and widened to cover the silent
    // depth-prefill phase at the configured context size.
    std::chrono::milliseconds effective_idle_timeout() const;

    TunerOptions options_;
    std::uint64_t model_size_bytes_ = 0;
    std::uint64_t layer_count_for_estimate_ = 0;
    bool initialized_ = false;
};

} // namespace fluxinfer::tuner
