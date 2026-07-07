#include <CLI11/CLI11.hpp>

#include "fluxinfer/hardware/hardware_info.hpp"
#include "fluxinfer/llama/gguf_metadata.hpp"
#include "fluxinfer/llama/llama_locator.hpp"
#include "fluxinfer/llama/llama_runner.hpp"
#include "fluxinfer/process/process_runner.hpp"
#include "fluxinfer/profiles/profile.hpp"
#include "fluxinfer/profiles/profile_store.hpp"
#include "fluxinfer/tuner/comparison.hpp"
#include "fluxinfer/tuner/tuner.hpp"

#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

using namespace fluxinfer;

std::string format_gib(std::uint64_t bytes) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << gib;
    std::string s = oss.str();
    if (s.size() >= 2 && s.compare(s.size() - 2, 2, ".0") == 0) {
        s.erase(s.size() - 2);
    }
    return s + " GB";
}

std::optional<std::filesystem::path> resolve_llama_dir(const std::string& cli_value) {
    if (!cli_value.empty()) {
        return std::filesystem::path(cli_value);
    }
    if (const char* env = std::getenv("FLUXINFER_LLAMA_DIR")) {
        if (*env) {
            return std::filesystem::path(env);
        }
    }
    return std::nullopt;
}

std::string join(const std::vector<std::string>& values, const std::string& sep) {
    std::string result;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) result += sep;
        result += values[i];
    }
    return result;
}

profiles::LlamaSnapshot build_llama_snapshot(const std::filesystem::path& llama_bench,
                                              const std::set<std::string>& supported_flags) {
    profiles::LlamaSnapshot snapshot;
    snapshot.binary_path = llama_bench.string();
    snapshot.version = llama::detect_llama_version(llama_bench);
    snapshot.supported_flags.assign(supported_flags.begin(), supported_flags.end());
    return snapshot;
}

profiles::HardwareSnapshot build_hardware_snapshot(const hardware::HardwareInfo& hw) {
    profiles::HardwareSnapshot snapshot;
    snapshot.cpu = hw.cpu.name;
    snapshot.logical_threads = hw.cpu.logical_threads;
    snapshot.ram_bytes = hw.memory.total_bytes;
    snapshot.gpu = hw.gpu.available ? hw.gpu.name : std::string();
    snapshot.vram_bytes = hw.gpu.available ? hw.gpu.total_vram_bytes : 0;
    return snapshot;
}

// ---------------------------------------------------------------------
// Termination forwarding for `run` / `serve`: Ctrl+C / SIGTERM stop
// FluxInfer *and* the child llama-cli/llama-server process it launched.
// ---------------------------------------------------------------------
process::InteractiveProcess* g_active_process = nullptr;

#if defined(_WIN32)
BOOL WINAPI console_ctrl_handler(DWORD) {
    if (g_active_process) {
        g_active_process->terminate();
    }
    return TRUE;
}
#else
void posix_signal_handler(int) {
    if (g_active_process) {
        g_active_process->terminate();
    }
}
#endif

void install_termination_forwarding(process::InteractiveProcess* proc) {
    g_active_process = proc;
#if defined(_WIN32)
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    std::signal(SIGINT, posix_signal_handler);
    std::signal(SIGTERM, posix_signal_handler);
#endif
}

// ---------------------------------------------------------------------
// fluxinfer inspect
// ---------------------------------------------------------------------
int cmd_inspect(const std::string& llama_dir_opt) {
    const hardware::HardwareInfo hw = hardware::probe_hardware();

    std::cout << "CPU:\n";
    std::cout << "  Name: " << hw.cpu.name << "\n";
    std::cout << "  Physical cores: " << hw.cpu.physical_cores << "\n";
    std::cout << "  Logical threads: " << hw.cpu.logical_threads << "\n\n";

    std::cout << "Memory:\n";
    std::cout << "  Total RAM: " << format_gib(hw.memory.total_bytes) << "\n";
    std::cout << "  Available RAM: " << format_gib(hw.memory.available_bytes) << "\n\n";

    std::cout << "GPU:\n";
    if (hw.gpu.available) {
        std::cout << "  Name: " << hw.gpu.name << "\n";
        std::cout << "  Total VRAM: " << format_gib(hw.gpu.total_vram_bytes) << "\n";
        std::cout << "  Available VRAM: " << format_gib(hw.gpu.available_vram_bytes) << "\n";
        std::cout << "  Backend: " << hw.gpu.backend << "\n\n";
    } else {
        std::cout << "  Not available (" << hw.gpu.unavailable_reason << ")\n\n";
    }

    llama::LlamaLocator locator(resolve_llama_dir(llama_dir_opt));
    llama::LlamaBinaries binaries = locator.locate();

    std::cout << "llama.cpp:\n";
    auto print_binary = [](const char* name, const std::optional<std::filesystem::path>& path) {
        if (path) {
            std::cout << "  " << name << ": found (" << path->string() << ")\n";
        } else {
            std::cout << "  " << name << ": not found\n";
        }
    };
    print_binary("llama-bench", binaries.llama_bench);
    print_binary("llama-cli", binaries.llama_cli);
    print_binary("llama-server", binaries.llama_server);

    return 0;
}

// ---------------------------------------------------------------------
// fluxinfer tune <model.gguf>
// ---------------------------------------------------------------------
int cmd_tune(const std::string& model_path_str, const std::string& llama_dir_opt, const std::string& profiles_dir_opt,
             unsigned timeout_seconds, unsigned search_repetitions, unsigned context_length, unsigned compare_repeats,
             unsigned warmup_runs, const std::string& report_out_opt) {
    const std::filesystem::path model_path(model_path_str);

    std::string model_error;
    std::optional<profiles::ModelInfo> model_info = profiles::compute_model_info(model_path, &model_error);
    if (!model_info) {
        std::cerr << "error: " << model_error << "\n";
        return 1;
    }

    const hardware::HardwareInfo hw = hardware::probe_hardware();

    llama::LlamaLocator locator(resolve_llama_dir(llama_dir_opt));
    llama::LlamaBinaries binaries = locator.locate();
    if (!binaries.llama_bench) {
        std::cerr << "error: llama-bench not found. Use --llama-dir, set FLUXINFER_LLAMA_DIR, or add it to PATH.\n";
        return 1;
    }

    std::cout << "Using llama-bench: " << binaries.llama_bench->string() << "\n";
    std::set<std::string> supported_flags = llama::detect_supported_flags(*binaries.llama_bench);
    std::cout << "Detected " << supported_flags.size() << " supported llama-bench flags.\n";

    llama::GgufParseResult gguf = llama::parse_gguf_metadata(model_path);
    if (gguf.valid) {
        std::cout << "Model metadata: architecture=" << (gguf.metadata.architecture.empty() ? "unknown" : gguf.metadata.architecture);
        if (gguf.metadata.block_count) {
            std::cout << " layers=" << *gguf.metadata.block_count;
        } else {
            std::cout << " layers=unknown (no gpu-layers search will be attempted)";
        }
        if (gguf.metadata.context_length) {
            std::cout << " context_length=" << *gguf.metadata.context_length;
        }
        if (!gguf.metadata.quantization_label.empty()) {
            std::cout << " quantization=" << gguf.metadata.quantization_label;
        }
        if (gguf.metadata.expert_count && *gguf.metadata.expert_count > 0) {
            std::cout << " experts=" << *gguf.metadata.expert_count << "/" << gguf.metadata.expert_used_count.value_or(0)
                       << " active (MoE model: this MVP does not do MoE-aware tuning yet)";
        }
        std::cout << "\n";
    } else {
        std::cout << "warning: could not read GGUF metadata (" << gguf.error << "); gpu-layers search will be skipped.\n";
    }

    tuner::TunerOptions options;
    options.llama_bench_path = *binaries.llama_bench;
    options.model_path = model_path;
    options.hardware = hw;
    options.supported_flags = supported_flags;
    options.real_layer_count = gguf.valid ? gguf.metadata.block_count : std::nullopt;
    options.per_run_timeout = std::chrono::seconds(timeout_seconds);
    options.search_repetitions = std::max(1u, search_repetitions);
    options.context_length = context_length;
    std::cout << "Tuning and serving at context size " << context_length
              << " tokens (override with --context; this is what llama-bench is benchmarked against via -d, and what "
                 "run/serve will launch llama-cli/llama-server with via -c).\n";
    options.on_result = [](const tuner::BenchmarkResult& result) {
        std::cout << "  [" << result.config.label << "] ";
        if (result.oom) {
            std::cout << "OOM\n";
        } else if (result.crashed) {
            std::cout << "crashed (exit " << result.exit_code << ")\n";
        } else if (result.timed_out) {
            std::cout << "timed out\n";
        } else if (!result.output_valid) {
            std::cout << "invalid output\n";
        } else {
            std::cout << "prompt=" << result.prompt_tokens_per_second << " tok/s, gen=" << result.generation_tokens_per_second
                       << " tok/s, score=" << result.score << "\n";
        }
    };

    std::cout << "Running staged benchmark search (this can take a while)...\n";
    tuner::Tuner runner(std::move(options));
    tuner::TuningOutcome outcome = runner.run();

    if (!outcome.success || !outcome.best) {
        std::cerr << "error: " << outcome.error << "\n";
        return 1;
    }

    const tuner::BenchmarkResult& best = *outcome.best;

    profiles::Profile profile;
    profile.model = *model_info;
    profile.hardware = build_hardware_snapshot(hw);
    profile.llama = build_llama_snapshot(*binaries.llama_bench, supported_flags);
    profile.best_config.threads = best.config.threads;
    profile.best_config.gpu_layers = best.config.gpu_layers;
    profile.best_config.batch_size = best.config.batch_size;
    profile.best_config.ubatch_size = best.config.ubatch_size;
    profile.best_config.kv_cache_type = best.config.kv_cache_type;
    profile.best_config.context_length = context_length;
    profile.results.prompt_tps = best.prompt_tokens_per_second;
    profile.results.generation_tps = best.generation_tokens_per_second;
    profile.results.duration_ms = best.duration.count();
    profile.results.score = best.score;

    profiles::ProfileStore store(profiles_dir_opt);
    std::string save_error;
    if (!store.save(profile, &save_error)) {
        std::cerr << "error: could not save profile: " << save_error << "\n";
        return 1;
    }

    std::cout << "\nBest configuration: " << best.config.label << "\n";
    std::cout << "  threads=" << best.config.threads << " gpu_layers=" << best.config.gpu_layers
               << " batch=" << best.config.batch_size << " ubatch=" << best.config.ubatch_size << "\n";
    std::cout << "  prompt=" << best.prompt_tokens_per_second << " tok/s, generation=" << best.generation_tokens_per_second
               << " tok/s, score=" << best.score << "\n";
    std::cout << "Profile saved to " << store.profile_path_for(profile.model).string() << "\n";

    if (compare_repeats > 0) {
        if (!outcome.baseline_config) {
            std::cerr << "warning: no baseline configuration recorded; skipping comparison report.\n";
            return 0;
        }

        std::cout << "\nRunning reproducible comparison: " << compare_repeats
                   << " measured repetition(s) each for baseline and selected configuration, in a single warm "
                   << "llama-bench process per side (" << (warmup_runs == 0 ? "warmup disabled" : "warmup enabled")
                   << ")...\n";

        tuner::ComparisonReport comparison =
            tuner::build_comparison_report(runner, *outcome.baseline_config, best.config, compare_repeats, warmup_runs);

        tuner::ReportContext context;
        context.model_path = model_path.string();
        context.architecture = gguf.valid ? gguf.metadata.architecture : std::string();
        context.quantization_label = gguf.valid ? gguf.metadata.quantization_label : std::string();
        context.layer_count = gguf.valid ? gguf.metadata.block_count : std::nullopt;
        context.llama_bench_path = binaries.llama_bench->string();
        context.llama_version = llama::detect_llama_version(*binaries.llama_bench);
        context.baseline_arguments = runner.build_arguments_for(*outcome.baseline_config, compare_repeats);
        context.selected_arguments = runner.build_arguments_for(best.config, compare_repeats);
        context.all_search_results = outcome.all_results;

        std::string report_text = tuner::format_comparison_report(comparison, context);
        std::cout << "\n" << report_text;

        if (!report_out_opt.empty()) {
            std::ofstream report_file(report_out_opt, std::ios::binary | std::ios::trunc);
            if (report_file) {
                report_file << report_text;
                std::cout << "Report written to " << report_out_opt << "\n";
            } else {
                std::cerr << "warning: could not write report to " << report_out_opt << "\n";
            }
        }
    }

    return 0;
}

// Shared setup for `run` and `serve`: locates binaries, loads and validates
// the profile for `model_path`. Returns false (with an error already
// printed) on any failure.
bool load_profile_or_report(const std::filesystem::path& model_path, const std::string& llama_dir_opt,
                             const std::string& profiles_dir_opt, llama::LlamaBinaries& binaries_out,
                             profiles::Profile& profile_out) {
    std::string model_error;
    std::optional<profiles::ModelInfo> model_info = profiles::compute_model_info(model_path, &model_error);
    if (!model_info) {
        std::cerr << "error: " << model_error << "\n";
        return false;
    }

    const hardware::HardwareInfo hw = hardware::probe_hardware();
    llama::LlamaLocator locator(resolve_llama_dir(llama_dir_opt));
    binaries_out = locator.locate();

    profiles::LlamaSnapshot llama_snapshot;
    if (binaries_out.llama_bench) {
        std::set<std::string> supported_flags = llama::detect_supported_flags(*binaries_out.llama_bench);
        llama_snapshot = build_llama_snapshot(*binaries_out.llama_bench, supported_flags);
    } else {
        std::cerr << "warning: llama-bench not found; skipping llama.cpp version validation for the profile.\n";
    }

    profiles::ProfileStore store(profiles_dir_opt);
    std::string reason;
    std::optional<profiles::Profile> profile =
        store.load_valid(*model_info, build_hardware_snapshot(hw), llama_snapshot, &reason);
    if (!profile) {
        std::cerr << "error: no usable tuning profile for this model (" << reason << ").\n";
        std::cerr << "Run `fluxinfer tune " << model_path.string() << "` first.\n";
        return false;
    }

    profile_out = *profile;
    return true;
}

std::vector<std::string> build_config_arguments(const profiles::Profile& profile, const std::set<std::string>& supported_flags,
                                                 const std::filesystem::path& model_path) {
    std::vector<std::string> args = {"-m", model_path.string()};

    auto add_if_supported = [&](const std::string& long_flag, const std::string& short_flag, const std::string& value) {
        if (llama::supports_flag(supported_flags, long_flag)) {
            args.push_back(short_flag);
            args.push_back(value);
        }
    };

    add_if_supported("--threads", "-t", std::to_string(profile.best_config.threads));
    add_if_supported("--n-gpu-layers", "-ngl", std::to_string(profile.best_config.gpu_layers));
    add_if_supported("--batch-size", "-b", std::to_string(profile.best_config.batch_size));
    add_if_supported("--ubatch-size", "-ub", std::to_string(profile.best_config.ubatch_size));
    // Match the context size this config was actually benchmarked at (see
    // TunerOptions::context_length): without this, llama-cli/llama-server
    // fall back to their own default of the model's full native context,
    // which can be far larger than what was tested and silently changes
    // real-world VRAM usage and throughput. context_length == 0 means an
    // older profile saved before this field existed; leave -c unset in
    // that case rather than guessing.
    if (profile.best_config.context_length > 0) {
        add_if_supported("--ctx-size", "-c", std::to_string(profile.best_config.context_length));
    }
    if (profile.best_config.kv_cache_type) {
        if (llama::supports_flag(supported_flags, "--cache-type-k")) {
            args.push_back("-ctk");
            args.push_back(*profile.best_config.kv_cache_type);
        }
        if (llama::supports_flag(supported_flags, "--cache-type-v")) {
            args.push_back("-ctv");
            args.push_back(*profile.best_config.kv_cache_type);
        }
    }

    return args;
}

// ---------------------------------------------------------------------
// fluxinfer run <model.gguf> [-- extra args]
// ---------------------------------------------------------------------
int cmd_run(const std::string& model_path_str, const std::string& llama_dir_opt, const std::string& profiles_dir_opt,
            const std::vector<std::string>& extra_args) {
    const std::filesystem::path model_path(model_path_str);

    llama::LlamaBinaries binaries;
    profiles::Profile profile;
    if (!load_profile_or_report(model_path, llama_dir_opt, profiles_dir_opt, binaries, profile)) {
        return 1;
    }
    if (!binaries.llama_cli) {
        std::cerr << "error: llama-cli not found. Use --llama-dir, set FLUXINFER_LLAMA_DIR, or add it to PATH.\n";
        return 1;
    }

    std::set<std::string> supported_flags = llama::detect_supported_flags(*binaries.llama_cli);
    std::vector<std::string> args = build_config_arguments(profile, supported_flags, model_path);
    args.insert(args.end(), extra_args.begin(), extra_args.end());

    std::cout << "Launching: " << binaries.llama_cli->string() << " " << join(args, " ") << "\n";

    process::ProcessOptions options;
    options.executable = *binaries.llama_cli;
    options.arguments = args;

    std::string error_message;
    std::unique_ptr<process::InteractiveProcess> proc = process::launch_interactive(options, &error_message);
    if (!proc) {
        std::cerr << "error: failed to launch llama-cli: " << error_message << "\n";
        return 1;
    }

    install_termination_forwarding(proc.get());
    return proc->wait();
}

// ---------------------------------------------------------------------
// fluxinfer serve <model.gguf> [--host H] [--port P] [-- extra args]
// ---------------------------------------------------------------------
int cmd_serve(const std::string& model_path_str, const std::string& llama_dir_opt, const std::string& profiles_dir_opt,
              const std::string& host, unsigned port, const std::vector<std::string>& extra_args) {
    const std::filesystem::path model_path(model_path_str);

    llama::LlamaBinaries binaries;
    profiles::Profile profile;
    if (!load_profile_or_report(model_path, llama_dir_opt, profiles_dir_opt, binaries, profile)) {
        return 1;
    }
    if (!binaries.llama_server) {
        std::cerr << "error: llama-server not found. Use --llama-dir, set FLUXINFER_LLAMA_DIR, or add it to PATH.\n";
        return 1;
    }

    std::set<std::string> supported_flags = llama::detect_supported_flags(*binaries.llama_server);
    std::vector<std::string> args = build_config_arguments(profile, supported_flags, model_path);

    if (llama::supports_flag(supported_flags, "--host")) {
        args.push_back("--host");
        args.push_back(host);
    }
    if (llama::supports_flag(supported_flags, "--port")) {
        args.push_back("--port");
        args.push_back(std::to_string(port));
    }
    args.insert(args.end(), extra_args.begin(), extra_args.end());

    std::cout << "Starting llama-server on http://" << host << ":" << port << "\n";
    std::cout << "Launching: " << binaries.llama_server->string() << " " << join(args, " ") << "\n";

    process::ProcessOptions options;
    options.executable = *binaries.llama_server;
    options.arguments = args;

    std::string error_message;
    std::unique_ptr<process::InteractiveProcess> proc = process::launch_interactive(options, &error_message);
    if (!proc) {
        std::cerr << "error: failed to launch llama-server: " << error_message << "\n";
        return 1;
    }

    install_termination_forwarding(proc.get());
    return proc->wait();
}

} // namespace

int main(int argc, char** argv) {
    CLI::App app{"FluxInfer: a multiplatform autotuner/wrapper for llama.cpp"};
    app.require_subcommand(1);

    std::string llama_dir_opt;
    std::string profiles_dir_opt = "profiles";

    auto add_common_options = [&](CLI::App* sub) {
        sub->add_option("--llama-dir", llama_dir_opt, "Directory containing llama-bench/llama-cli/llama-server");
        sub->add_option("--profiles-dir", profiles_dir_opt, "Directory to read/write tuning profiles")
            ->default_val("profiles");
    };

    CLI::App* inspect_cmd = app.add_subcommand("inspect", "Show detected hardware and llama.cpp binaries");
    add_common_options(inspect_cmd);

    std::string tune_model;
    unsigned tune_timeout_seconds = 60;
    unsigned search_repetitions = 3;
    unsigned context_length = 4096;
    unsigned compare_repeats = 0;
    unsigned warmup_runs = 1;
    std::string report_out;
    CLI::App* tune_cmd = app.add_subcommand("tune", "Benchmark and select the best llama.cpp configuration for a model");
    tune_cmd->add_option("model", tune_model, "Path to a .gguf model file")->required();
    tune_cmd->add_option("--timeout", tune_timeout_seconds, "Per-benchmark timeout in seconds (per llama-bench repetition)")
        ->default_val(60);
    tune_cmd
        ->add_option("--search-repetitions", search_repetitions,
                      "llama-bench repetitions per search-stage candidate (median used for scoring; more resists "
                      "noisy single-sample decisions, at some extra time cost)")
        ->default_val(3);
    tune_cmd
        ->add_option("--context", context_length,
                      "Context size (tokens) to tune and serve at. Benchmarked via llama-bench's KV-cache pre-fill "
                      "(-d) and later launched identically by run/serve (-c), so the benchmark reflects real usage. "
                      "Larger contexts reserve more VRAM for KV cache, competing with GPU-offloaded layers.")
        ->default_val(4096);
    tune_cmd
        ->add_option("--compare-repeats", compare_repeats,
                      "After tuning, re-benchmark the baseline and selected configs this many times each and print a "
                      "reproducible comparison report (0 disables)")
        ->default_val(0);
    tune_cmd
        ->add_option("--warmup-runs", warmup_runs,
                      "0 disables llama-bench's internal warmup pass for the comparison report (--no-warmup); any "
                      "nonzero value enables it (llama-bench supports on/off only, not a configurable count)")
        ->default_val(1);
    tune_cmd->add_option("--report-out", report_out, "Optional file path to also save the comparison report to");
    add_common_options(tune_cmd);

    std::string run_model;
    std::vector<std::string> run_extra_args;
    CLI::App* run_cmd = app.add_subcommand("run", "Run llama-cli with the tuned profile for a model");
    run_cmd->add_option("model", run_model, "Path to a .gguf model file")->required();
    run_cmd->add_option("extra_args", run_extra_args, "Extra arguments forwarded to llama-cli after --");
    add_common_options(run_cmd);

    std::string serve_model;
    std::string serve_host = "127.0.0.1";
    unsigned serve_port = 8080;
    std::vector<std::string> serve_extra_args;
    CLI::App* serve_cmd = app.add_subcommand("serve", "Run llama-server with the tuned profile for a model");
    serve_cmd->add_option("model", serve_model, "Path to a .gguf model file")->required();
    serve_cmd->add_option("--host", serve_host, "Host to bind llama-server to")->default_val("127.0.0.1");
    serve_cmd->add_option("--port", serve_port, "Port to bind llama-server to")->default_val(8080);
    serve_cmd->add_option("extra_args", serve_extra_args, "Extra arguments forwarded to llama-server after --");
    add_common_options(serve_cmd);

    CLI11_PARSE(app, argc, argv);

    if (*inspect_cmd) {
        return cmd_inspect(llama_dir_opt);
    }
    if (*tune_cmd) {
        return cmd_tune(tune_model, llama_dir_opt, profiles_dir_opt, tune_timeout_seconds, search_repetitions,
                         context_length, compare_repeats, warmup_runs, report_out);
    }
    if (*run_cmd) {
        return cmd_run(run_model, llama_dir_opt, profiles_dir_opt, run_extra_args);
    }
    if (*serve_cmd) {
        return cmd_serve(serve_model, llama_dir_opt, profiles_dir_opt, serve_host, serve_port, serve_extra_args);
    }

    return 1;
}
