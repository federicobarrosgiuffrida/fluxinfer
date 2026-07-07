#pragma once

#include <filesystem>
#include <optional>
#include <vector>

namespace fluxinfer::llama {

struct LlamaBinaries {
    std::optional<std::filesystem::path> llama_bench;
    std::optional<std::filesystem::path> llama_cli;
    std::optional<std::filesystem::path> llama_server;

    bool all_found() const { return llama_bench && llama_cli && llama_server; }
};

// Locates the llama-bench / llama-cli / llama-server executables. Search
// order:
//   1. an explicit directory (--llama-dir CLI flag or FLUXINFER_LLAMA_DIR)
//   2. every directory on PATH
//   3. common local build output directories (external/llama.cpp/build/bin,
//      ./llama.cpp/build/bin, ./build/bin, relative to the current
//      working directory)
// FluxInfer never builds or links llama.cpp itself; it only looks for
// already-built binaries.
class LlamaLocator {
public:
    explicit LlamaLocator(std::optional<std::filesystem::path> explicit_dir = std::nullopt);

    LlamaBinaries locate() const;

    // Exposed for tests / `fluxinfer inspect` diagnostics.
    std::vector<std::filesystem::path> search_directories() const;

private:
    std::optional<std::filesystem::path> find_binary(const std::string& base_name) const;

    std::optional<std::filesystem::path> explicit_dir_;
};

} // namespace fluxinfer::llama
