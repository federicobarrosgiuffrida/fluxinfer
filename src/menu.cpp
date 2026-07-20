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

bool ask_yes_no(const std::string& question, bool fallback) {
    const std::string suffix = fallback ? " [S/n]: " : " [s/N]: ";
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

    std::cout << "\n-- Scegli un modello --\n";
    if (found.empty()) {
        std::cout << "Nessun .gguf trovato nelle cartelle solite.\n";
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
    std::cout << "  0) inserisci un percorso a mano\n";

    const std::optional<std::string> answer = read_line("Scelta: ");
    if (!answer) {
        return std::nullopt;
    }
    if (*answer == "0" || found.empty()) {
        const std::optional<std::string> typed = read_line("Percorso del file .gguf: ");
        if (!typed || typed->empty()) {
            return std::nullopt;
        }
        std::filesystem::path path(*typed);
        std::error_code ec;
        if (!std::filesystem::exists(path, ec)) {
            std::cout << "File non trovato: " << path.string() << "\n";
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
    std::cout << "Scelta non valida.\n";
    return std::nullopt;
}

void show_command(const std::string& command) {
    std::cout << "\nComando equivalente (puoi usarlo direttamente la prossima volta):\n  " << command << "\n\n";
}

} // namespace

// --- Actions ---------------------------------------------------------

MenuAction plan_tune(const std::filesystem::path& model) {
    MenuAction action;
    action.kind = MenuAction::Kind::Tune;
    action.model = model;

    std::cout << "\n-- Tuning guidato --\n";
    std::cout << "Il profilo vale solo per il context a cui lo tuni: usa quello a cui servirai davvero.\n";
    action.context = ask_with_default("Context (token)", "4096");
    action.vram_headroom_mb = ask_with_default("VRAM da lasciare libera in MB (0 = predefinito di sistema)", "0");
    if (ask_yes_no("Generare anche un report dettagliato?", false)) {
        action.report_out = "report.md";
    }
    return action;
}

MenuAction plan_serve(const std::filesystem::path& model) {
    MenuAction action;
    action.kind = MenuAction::Kind::Serve;
    action.model = model;

    std::cout << "\n-- Avvio server --\n";
    if (ask_yes_no("Template della chat (--jinja)?", true)) {
        action.extra_args.push_back("--jinja");
    }
    if (ask_yes_no("Disabilitare mmap (consigliato con esperti in RAM)?", true)) {
        action.extra_args.push_back("--no-mmap");
    }
    if (ask_yes_no("Abilitare il proxy MCP per la web UI?", false)) {
        action.extra_args.push_back("--webui-mcp-proxy");
    }
    if (ask_yes_no("Mantenere il ragionamento tra i turni?", false)) {
        action.extra_args.push_back("--reasoning-preserve");
    }
    return action;
}

void list_profiles(const std::filesystem::path& profiles_directory) {
    std::cout << "\n-- Profili salvati --\n";
    std::error_code ec;
    if (!std::filesystem::exists(profiles_directory, ec)) {
        std::cout << "Nessun profilo in " << profiles_directory.string() << "\n";
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
        std::cout << "    misurato: " << profile->results.generation_tps << " tok/s generazione, "
                  << profile->results.prompt_tps << " tok/s prompt\n";
        std::cout << "    hardware: " << profile->hardware.gpu << "\n";
    }
    if (!any) {
        std::cout << "Nessun profilo valido in " << profiles_directory.string() << "\n";
    }
}

MenuAction run_menu(const std::filesystem::path& profiles_directory) {
    MenuAction none;

    for (;;) {
        std::cout << "\n===============================\n";
        std::cout << " FluxInfer\n";
        std::cout << "===============================\n";
        std::cout << "  1) Ottimizza un modello (tune)\n";
        std::cout << "  2) Avvia il server con un modello gia' ottimizzato\n";
        std::cout << "  3) Mostra i profili salvati\n";
        std::cout << "  4) Diagnostica (doctor)\n";
        std::cout << "  5) Esci\n";

        const std::optional<std::string> choice = read_line("\nScelta: ");
        if (!choice || *choice == "5" || *choice == "q") {
            return none;
        }

        if (*choice == "1") {
            const std::optional<std::filesystem::path> model = choose_model();
            if (!model) {
                continue;
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
        std::cout << "Scelta non riconosciuta.\n";
    }
}

} // namespace fluxinfer::menu
