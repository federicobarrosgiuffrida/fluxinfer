#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fluxinfer::menu {

// What the menu decided to do. The menu itself never launches anything:
// it collects a choice and hands it back to main(), which dispatches to
// exactly the same functions the individual subcommands use. That keeps
// the menu a front-end over the CLI rather than a second implementation
// of it.
struct MenuAction {
    enum class Kind { None, Tune, Serve, Doctor };

    Kind kind = Kind::None;
    std::filesystem::path model;

    // Tune options, as strings because they are handed straight to the
    // same parsing the CLI does.
    std::string context = "4096";
    std::string vram_headroom_mb = "0";
    std::string report_out;

    // Serve options passed through to llama-server after `--`.
    std::vector<std::string> extra_args;
};

// Draws the menu and returns the chosen action (Kind::None if the user
// quit). Reads from stdin, writes to stdout; no terminal control
// sequences, so it behaves under any terminal, in CI, or piped.
MenuAction run_menu(const std::filesystem::path& profiles_directory);

} // namespace fluxinfer::menu
