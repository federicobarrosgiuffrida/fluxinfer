// Interactive menu (`fluxinfer menu`).
//
// A deliberately plain, dependency-free terminal menu: numbered options
// read from stdin, no curses, no alternate screen, no cursor addressing.
// It drives the same code paths as the individual subcommands rather than
// reimplementing anything -- everything here could be typed by hand, and
// the menu prints the equivalent command before running it, so it teaches
// the CLI instead of hiding it.
//
// Note for review: the README lists "No GUI; no Electron; CLI-only" among
// the project's deliberate limits. This stays inside that constraint as
// literally as possible (it is a terminal program reading lines from
// stdin), but it does change how the tool is meant to be approached, so it
// is isolated in its own translation unit and its own commit.

#include "fluxinfer/menu.hpp"

#include "fluxinfer/hardware/hardware_info.hpp"
#include "fluxinfer/llama/gguf_metadata.hpp"
#include "fluxinfer/profiles/profile.hpp"
#include "fluxinfer/profiles/profile_store.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fluxinfer::menu {

namespace {

std::string trim(std::string text) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

// Reads a line, returning the trimmed text. Returns std::nullopt on EOF
// (piped input that ran out, or Ctrl+D/Ctrl+Z), which every caller treats
// as "quit" rather than looping forever on a closed stream.
std::optional<std::string> read_line(const std::string& prompt) {
    std::cout << prompt << std::flush;
    std::string line;
    if (!std::getline(std::cin, line)) {
        std::cout << "\n";
        return std::nullopt;
    }
    return trim(line);
}

std::string ask_with_default(const std::string& question, const std::string& fallback) {
    const std::optional<std::string> answer = read_line(question + " [" + fallback + "]: ");
    if (!answer || answer->empty()) {
        return fallback;
    }
    return *answer;
}

// `explanation` is printed above the question, so the user can decide
// without knowing llama.cpp's flags beforehand. An empty one is allowed
// for questions that need no gloss.
bool ask_yes_no(const std::string& question, const std::string& explanation, bool fallback) {
    if (!explanation.empty()) {
        std::cout << "\n  " << explanation << "\n";
    }
    const std::string suffix = fallback ? " [Y/n]: " : " [y/N]: ";
    const std::optional<std::string> answer = read_line(question + suffix);
    if (!answer || answer->empty()) {
        return fallback;
    }
    const char first = static_cast<char>(std::tolower(static_cast<unsigned char>((*answer)[0])));
    return first == 'y' || first == 's';
}

std::string format_gib(std::uint64_t bytes) {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GB";
    return out.str();
}

// Directories worth looking in for .gguf files before asking the user to
// type a path. Nothing here is authoritative -- it is a convenience, and
// the "type a path" option is always offered.
std::vector<std::filesystem::path> candidate_model_directories() {
    std::vector<std::filesystem::path> directories;
    const char* home_vars[] = {"USERPROFILE", "HOME"};
    for (const char* var : home_vars) {
        if (const char* home = std::getenv(var)) {
            const std::filesystem::path base(home);
            directories.push_back(base / "Documents" / "ai-models");
            directories.push_back(base / "models");
            directories.push_back(base / ".lmstudio" / "models");
            directories.push_back(base / ".cache" / "huggingface");
        }
    }
    directories.push_back(std::filesystem::current_path());
    return directories;
}

std::vector<std::filesystem::path> find_models(const std::filesystem::path& directory, std::size_t limit = 50) {
    std::vector<std::filesystem::path> models;
    std::error_code ec;
    if (!std::filesystem::exists(directory, ec)) {
        return models;
    }
    // Recursive, because model collections are usually one directory per
    // repository. Errors (permissions, broken links) skip the entry rather
    // than aborting the walk.
    auto iterator = std::filesystem::recursive_directory_iterator(
        directory, std::filesystem::directory_options::skip_permission_denied, ec);
    if (ec) {
        return models;
    }
    for (const auto& entry : iterator) {
        if (models.size() >= limit) {
            break;
        }
        std::error_code entry_ec;
        if (!entry.is_regular_file(entry_ec) || entry_ec) {
            continue;
        }
        const std::filesystem::path& path = entry.path();
        if (path.extension() == ".gguf") {
            // Skip the second and later shards of a split model: only the
            // first is passed to llama.cpp.
            const std::string name = path.filename().string();
            if (name.find("-00002-of-") != std::string::npos || name.find("-00003-of-") != std::string::npos) {
                continue;
            }
            models.push_back(path);
        }
    }
    return models;
}

std::optional<std::filesystem::path> choose_model() {
    std::vector<std::filesystem::path> found;
    for (const auto& directory : candidate_model_directories()) {
        for (auto& model : find_models(directory)) {
            if (std::find(found.begin(), found.end(), model) == found.end()) {
                found.push_back(std::move(model));
            }
        }
        if (found.size() >= 30) {
            break;
        }
    }

    std::cout << "\n-- Choose a model --\n";
    if (found.empty()) {
        std::cout << "No .gguf files found in the usual directories.\n";
    } else {
        for (std::size_t i = 0; i < found.size(); ++i) {
            std::error_code ec;
            const std::uintmax_t size = std::filesystem::file_size(found[i], ec);
            std::cout << "  " << (i + 1) << ") " << found[i].filename().string();
            if (!ec) {
                std::cout << "  (" << format_gib(size) << ")";
            }
            std::cout << "\n";
        }
    }
    std::cout << "  0) type a path manually\n";
    std::cout << "  b) back to the menu\n";

    const std::optional<std::string> answer = read_line("Choice (Enter = back): ");
    // Empty input, "b", or EOF all mean "cancel, return to the menu" -- not
    // an error, so nothing is printed and the caller loops back cleanly.
    if (!answer || answer->empty() || *answer == "b" || *answer == "B") {
        return std::nullopt;
    }
    if (*answer == "0" || found.empty()) {
        const std::optional<std::string> typed = read_line("Path to the .gguf file: ");
        if (!typed || typed->empty()) {
            return std::nullopt;
        }
        std::filesystem::path path(*typed);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "File not found: " << path.string() << "\n";
            return std::nullopt;
        }
        return path;
    }

    try {
        const int index = std::stoi(*answer);
        if (index >= 1 && static_cast<std::size_t>(index) <= found.size()) {
            return found[static_cast<std::size_t>(index) - 1];
        }
    } catch (const std::exception&) {
        // fall through to the error below
    }
    std::cout << "Not a valid choice.\n";
    return std::nullopt;
}

void show_command(const std::string& command) {
    std::cout << "\nEquivalent command (use it directly next time):\n  " << command << "\n\n";
}

} // namespace

// --- Actions ---------------------------------------------------------

MenuAction plan_tune(const std::filesystem::path& model) {
    MenuAction action;
    action.kind = MenuAction::Kind::Tune;
    action.model = model;

    std::cout << "\n-- Guided tuning --\n";
    std::cout << "\n  A profile is only valid at the context it was tuned for: a larger context needs more VRAM for the\n"
                 "  KV cache, which leaves less for the model. Set this to the context you will actually serve at.\n";
    action.context = ask_with_default("Context (tokens)", "4096");
    std::cout << "\n  VRAM deliberately left unused, so the tuned configuration still fits once a browser or a chat app is\n"
                 "  open. Too little here produces a profile that only works on an idle machine.\n";
    action.vram_headroom_mb = ask_with_default("VRAM headroom in MB (0 = platform default)", "0");
    if (ask_yes_no("Also write a detailed report?",
                    "Runs each finalist several more times and writes a markdown report with the spread between runs, "
                    "so a close result can be told apart from a lucky one. Adds a few minutes.",
                    false)) {
        action.report_out = "report.md";
    }
    return action;
}

MenuAction plan_serve(const std::filesystem::path& model) {
    MenuAction action;
    action.kind = MenuAction::Kind::Serve;
    action.model = model;

    std::cout << "\n-- Start server --\n";
    if (ask_yes_no("Use the model's own chat template?",
                    "--jinja: formats each turn the way this model was trained to expect, and is required for tool "
                    "calling. Leave on unless you know otherwise.",
                    true)) {
        action.extra_args.push_back("--jinja");
    }
    if (ask_yes_no("Load the whole model into RAM up front?",
                    "--no-mmap: reads the file into memory instead of mapping it. Slower to start, but faster to run "
                    "when part of the model lives in system RAM, which is the case for offloaded MoE experts.",
                    true)) {
        action.extra_args.push_back("--no-mmap");
    }
    if (ask_yes_no("Allow the web UI to use external tools (web search, MCP servers)?",
                    "--webui-mcp-proxy: lets the built-in UI reach MCP servers, which a browser cannot call directly "
                    "from a local page. Experimental in llama.cpp; fine for local use.",
                    false)) {
        action.extra_args.push_back("--webui-mcp-proxy");
    }
    if (ask_yes_no("Keep the model's reasoning between turns?",
                    "--reasoning-preserve: carries the model's thinking into later turns. Valuable for multi-step "
                    "agentic work; it also consumes context and makes short replies noticeably slower.",
                    false)) {
        action.extra_args.push_back("--reasoning-preserve");
    }
    return action;
}

void list_profiles(const std::filesystem::path& profiles_directory) {
    std::cout << "\n-- Saved profiles --\n";
    std::error_code ec;
    if (!std::filesystem::exists(profiles_directory, ec)) {
        std::cout << "No profiles in " << profiles_directory.string() << "\n";
        return;
    }

    bool any = false;
    for (const auto& entry : std::filesystem::directory_iterator(profiles_directory, ec)) {
        if (entry.path().extension() != ".json") {
            continue;
        }
        std::ifstream file(entry.path());
        if (!file) {
            continue;
        }
        nlohmann::json document;
        try {
            file >> document;
        } catch (const std::exception&) {
            continue;
        }
        std::string error;
        const std::optional<profiles::Profile> profile = profiles::profile_from_json(document, &error);
        if (!profile) {
            continue;
        }
        any = true;

        std::filesystem::path model_path(profile->model.path);
        std::cout << "\n  " << model_path.filename().string() << "\n";
        std::cout << "    threads=" << profile->best_config.threads << " gpu_layers=" << profile->best_config.gpu_layers
                  << " batch=" << profile->best_config.batch_size << "/" << profile->best_config.ubatch_size;
        if (profile->best_config.n_cpu_moe) {
            std::cout << " n_cpu_moe=" << *profile->best_config.n_cpu_moe;
        }
        std::cout << " context=" << profile->best_config.context_length << "\n";
        std::cout << "    measured: " << profile->results.generation_tps << " tok/s generation, "
                  << profile->results.prompt_tps << " tok/s prompt processing\n";
        std::cout << "    hardware: " << profile->hardware.gpu << "\n";
    }
    if (!any) {
        std::cout << "No valid profiles in " << profiles_directory.string() << "\n";
    }
}

MenuAction run_menu(const std::filesystem::path& profiles_directory) {
    MenuAction none;

    for (;;) {
        std::cout << "\n===============================\n";
        std::cout << " FluxInfer\n";
        std::cout << "===============================\n";
        std::cout << "  1) Tune a model (find the best configuration)\n";
        std::cout << "  2) Start the server with an already-tuned model\n";
        std::cout << "  3) Show saved profiles\n";
        std::cout << "  4) Diagnostics (doctor)\n";
        std::cout << "  5) Quit\n";

        const std::optional<std::string> choice = read_line("\nChoice: ");
        if (!choice || *choice == "5" || *choice == "q") {
            return none;
        }

        if (*choice == "1") {
            const std::optional<std::filesystem::path> model = choose_model();
            if (!model) {
                continue;
            }
            std::cout << "\nAbout to tune " << model->filename().string()
                      << ". This can take from minutes to about an hour.\n";
            if (!ask_yes_no("Continue?", "", true)) {
                continue; // back to the menu, nothing started
            }
            MenuAction action = plan_tune(*model);
            std::string command = "fluxinfer tune \"" + model->string() + "\" --context " + action.context;
            if (action.vram_headroom_mb != "0") {
                command += " --vram-headroom-mb " + action.vram_headroom_mb;
            }
            if (!action.report_out.empty()) {
                command += " --report-out " + action.report_out;
            }
            show_command(command);
            return action;
        }
        if (*choice == "2") {
            const std::optional<std::filesystem::path> model = choose_model();
            if (!model) {
                continue;
            }
            MenuAction action = plan_serve(*model);
            std::string command = "fluxinfer serve \"" + model->string() + "\"";
            if (!action.extra_args.empty()) {
                command += " --";
                for (const std::string& argument : action.extra_args) {
                    command += " " + argument;
                }
            }
            show_command(command);
            return action;
        }
        if (*choice == "3") {
            list_profiles(profiles_directory);
            continue;
        }
        if (*choice == "4") {
            MenuAction action;
            action.kind = MenuAction::Kind::Doctor;
            show_command("fluxinfer doctor");
            return action;
        }
        std::cout << "Unrecognised choice.\n";
    }
}

} // namespace fluxinfer::menu
